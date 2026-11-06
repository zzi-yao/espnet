"""Inference-only, pinned-checkpoint Dev alpha sweep using ASR stages 12/13."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
import yaml

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def percent(path):
    for line in path.read_text().splitlines():
        if 'Sum/Avg' in line:
            columns = line.split('|')
            return float(columns[3].split()[4])
    raise ValueError(f'No sclite Sum/Avg in {path}')


def model_digest(model):
    h = hashlib.sha256()
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def preflight(root, identity, device, decode):
    """Run real Dev speech through the production scorer with 0/.5/1 weights."""
    from .inference import Speech2Text
    from .task import DualASRTask
    result = root / 'preflight.json'
    if result.exists():
        saved = json.loads(result.read_text())
        if saved['identity'] != identity:
            raise ValueError('Existing preflight belongs to another experiment')
        return saved
    smoke = root / f'smoke_{time.time_ns()}'
    smoke.mkdir(exist_ok=False)
    source = Path('dump/raw/org/valid')
    shape = Path('exp/asr_stats_raw_zh_whisper_multilingual/valid/speech_shape')
    lengths = [(int(line.split()[1].split(',')[0]), line.split()[0])
               for line in shape.read_text().splitlines()]
    keys = [key for _, key in sorted(lengths)[:2]]
    keyfile = smoke / 'keys.scp'
    keyfile.write_text('\n'.join(keys) + '\n')
    recognizer = Speech2Text(asr_train_config=identity['config_path'],
        asr_model_file=identity['checkpoint_path'], device=device, **decode)
    model = recognizer.asr_model
    assert model.decoder.alpha == 0.5, 'Training fusion weight must remain 0.5'
    assert not model.training
    trained = torch.load(identity['checkpoint_path'], map_location='cpu', weights_only=False)
    current = model.state_dict()
    assert set(trained) == set(current)
    assert all(torch.equal(v.cpu(), trained[k]) for k, v in current.items())
    del trained, current
    pretrained = torch.load(identity['small_path'], map_location='cpu', weights_only=False)['model_state_dict']
    for prefix, module in [('encoder.', model.encoder.small.encoders),
                           ('decoder.', model.decoder.small.decoders)]:
        assert all(torch.equal(v.cpu(), pretrained[prefix + k]) for k, v in module.state_dict().items())
    del pretrained
    initial = model_digest(model)
    samples = list(DualASRTask.build_streaming_iterator(
        [(str(source / 'wav.scp'), 'speech', 'kaldi_ark')], dtype='float32',
        batch_size=1, num_workers=0, key_file=str(keyfile), inference=True,
        preprocess_fn=DualASRTask.build_preprocess_fn(recognizer.asr_train_args, False),
        collate_fn=DualASRTask.build_collate_fn(recognizer.asr_train_args, False)))
    speech = samples[0][1]['speech'].to(device)
    with torch.no_grad():
        memory, _ = model.encode(speech, torch.tensor([speech.size(1)], device=device))
        prefix = torch.tensor([list(model.primer)], device=device)
        values, grad_flags = {}, []
        def hook(name):
            def collect(module, inputs, output):
                values[name] = output[0]
                grad_flags.append(torch.is_grad_enabled() or output[0].requires_grad)
            return collect
        # Hooks observe the actual per-step decoder path, not teacher-forcing.
        # forward_one_step is a method, so wrap it rather than Module.forward hooks.
        from unittest.mock import patch
        small_step = model.decoder.small.forward_one_step
        tiny_step = model.decoder.tiny.forward_one_step
        def capture(name, fn):
            def call(*args, **kwargs):
                output = fn(*args, **kwargs)
                hook(name)(None, None, output)
                return output
            return call
        logits = {}
        with patch.object(model.decoder.small, 'forward_one_step', capture('small', small_step)), \
             patch.object(model.decoder.tiny, 'forward_one_step', capture('tiny', tiny_step)):
            model.decoder.set_inference_fusion_alpha(None)
            original, _ = model.decoder.batch_score(prefix, [(None, None)], memory)
            for alpha in [0.0, 0.5, 1.0]:
                model.decoder.set_inference_fusion_alpha(alpha)
                actual, _ = model.decoder.batch_score(prefix, [(None, None)], memory)
                fused = alpha * values['small'] + (1 - alpha) * values['tiny']
                torch.testing.assert_close(actual, fused.log_softmax(-1))
                logits[alpha] = fused
                if alpha == 0.0:
                    torch.testing.assert_close(fused, values['tiny'])
                elif alpha == 1.0:
                    torch.testing.assert_close(fused, values['small'])
                else:
                    torch.testing.assert_close(actual, original, atol=0, rtol=0)
        delta = float((logits[0.0] - logits[1.0]).abs().max())
        assert delta > 0 and not any(grad_flags)
    texts = {}
    for alpha in [0.0, 0.5, 1.0]:
        output = smoke / f'alpha_{alpha:.1f}'
        output.mkdir(exist_ok=False)
        model.decoder.set_inference_fusion_alpha(alpha)
        lines = []
        for names, batch in samples:
            result_text = recognizer(batch['speech'][0])[0][0] or ''
            lines.append(names[0] + ' ' + result_text)
        (output / 'text').write_text('\n'.join(lines) + '\n')
        texts[str(alpha)] = lines
    assert all(p.grad is None for p in model.parameters())
    assert model_digest(model) == initial, 'Inference changed parameters or buffers'
    saved = dict(identity=identity, smoke_dir=str(smoke), keys=keys, tiny_checkpoint_exact_match=True,
        frozen_small_pretrained_exact_match=True, no_gradients=True, weights_unchanged=True,
        alpha_half_matches_original=True, endpoint_logits_exact=True,
        endpoint_logits_max_difference=delta, texts=texts)
    write_json(result, saved)
    return saved


def stage_command(directory, alpha, device, split):
    return ['./asr.sh', '--python', str(Path('local/dual_python.sh').resolve()),
        '--stage', '12', '--stop_stage', '13', '--skip_train', 'true',
        '--skip_data_prep', 'true', '--use_proxy_tuning', 'false',
        '--lang', 'zh', '--ngpu', '1' if device == 'cuda' else '0',
        '--token_type', 'whisper_multilingual', '--inference_nj', '1',
        '--gpu_inference', 'true' if device == 'cuda' else 'false',
        '--feats_normalize', '', '--audio_format', 'flac.ark', '--feats_type', 'raw',
        '--use_lm', 'false', '--inference_config', str(directory / 'decode.yaml'),
        '--inference_asr_model', 'model.pth', '--asr_exp', str(directory),
        '--inference_tag', 'decode', '--eval_valid_set', 'false',
        '--train_set', 'train', '--valid_set', 'valid', '--test_sets', split,
        '--lm_train_text', 'data/train/text', '--bpe_train_text', 'data/train/text',
        '--inference_args', f'--inference_fusion_alpha {alpha} --num_workers 0']



def pipeline_preflight(root, identity, device, decode, check):
    result = root / 'pipeline_preflight.json'
    if result.exists():
        assert json.loads(result.read_text())['identity'] == identity
        return
    directory = root / f'pipeline_smoke_{time.time_ns()}'
    directory.mkdir()
    (directory / 'config.yaml').symlink_to(identity['config_path'])
    (directory / 'model.pth').symlink_to(identity['checkpoint_path'])
    (directory / 'decode.yaml').write_text(yaml.safe_dump(decode))
    data = directory / 'dump/raw/org/valid'
    data.mkdir(parents=True)
    keys = set(check['keys'])
    for name in ['wav.scp', 'text', 'utt2spk']:
        lines = [line for line in (Path('dump/raw/org/valid') / name).read_text().splitlines()
                 if line.split()[0] in keys]
        (data / name).write_text('\n'.join(lines) + '\n')
    for name in ['feats_type', 'audio_format']:
        (data / name).write_bytes((Path('dump/raw/org/valid') / name).read_bytes())
    command = stage_command(directory, 0.5, device, 'org/valid') + ['--dumpdir', str(directory / 'dump')]
    write_json(directory / 'command.json', command)
    with (directory / 'stages.log').open('x') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, DUAL_PYTHON=sys.executable))
    scored = directory / 'decode/org/valid'
    job_log = (scored / 'logdir/asr_inference.1.log').read_text()
    assert 'inference_alpha=0.5' in job_log
    assert 'Stage 11:' not in (directory / 'stages.log').read_text()
    write_json(result, dict(identity=identity, directory=str(directory),
        sample_cer_percent=percent(scored / 'score_cer/result.txt'),
        sample_wer_percent=percent(scored / 'score_wer/result.txt'), cli_alpha_verified=True))


def summarize(root, rows, complete=False):
    fields = ['alpha', 'small_weight', 'tiny_weight', 'dev_cer_percent',
              'dev_wer_percent', 'decode_seconds', 'decode_result_dir']
    ordered = sorted(rows, key=lambda row: (row['dev_cer_percent'], row['alpha']))
    temp = root / 'dev_results.tmp'
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    temp.replace(root / 'dev_results.csv')
    if ordered:
        best = dict(ordered[0], complete=complete, tiny_fft_dev_cer_percent=29.1,
                    difference_from_tiny_fft_pp=ordered[0]['dev_cer_percent'] - 29.1)
        write_json(root / 'best_alpha.json', best)
        if complete:
            zero = next(row for row in ordered if row['alpha'] == 0.0)
            lines = ['| alpha | Dev CER (%) | Dev WER (%) | seconds |',
                     '| --- | --- | --- | --- |']
            lines += [f"| {row['alpha']:.1f} | {row['dev_cer_percent']:.1f} | {row['dev_wer_percent']:.1f} | {row['decode_seconds']:.0f} |" for row in ordered]
            lines += ['', f"Best alpha={best['alpha']:.1f}, Dev CER={best['dev_cer_percent']:.1f}%.",
                      f"Difference from standalone tiny FFT Dev 29.1%: {best['difference_from_tiny_fft_pp']:+.1f} percentage points (lower is better).",
                      f"Jointly trained tiny alone (alpha=0): {zero['dev_cer_percent']:.1f}%."]
            if best['alpha'] > 0 and best['dev_cer_percent'] < zero['dev_cer_percent']:
                lines += ['Frozen small improves inference relative to this jointly trained tiny alone.']
            else:
                lines += ['The scanned weights show no inference improvement from adding frozen small over alpha=0.']
            lines += ['This inference sweep alone does not establish a causal benefit of small during training. Test has not been decoded by the Dev sweep.']
            (root / 'dev_analysis.md').write_text('\n'.join(lines) + '\n')
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='exp/asr_whisper_dual_small_tiny')
    parser.add_argument('--checkpoint', default='valid.loss.ave.pth')
    parser.add_argument('--output', default='exp/dual_whisper_alpha_sweep')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--baseline-tolerance-pp', type=float, default=0.5)
    test = parser.add_mutually_exclusive_group()
    test.add_argument('--test-alpha', type=float, help='Explicit manual Test decode; never swept or used for selection')
    test.add_argument('--test-best', action='store_true', help='Explicit manual Test decode using completed Dev best alpha')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # OS file lock prevents two sweep processes from writing the same directories.
    import fcntl
    with (root / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args, root)
        except Exception as error:
            write_json(root / 'failure.json', dict(error=repr(error), unix_time=time.time()))
            raise


def execute(args, root):
    if args.test_best:
        best = json.loads((root / 'best_alpha.json').read_text())
        if not best['complete']:
            raise RuntimeError('Dev sweep is incomplete; no Test decoding allowed with --test-best')
        args.test_alpha = best['alpha']
    experiment = Path(args.experiment).resolve()
    config = (experiment / 'config.yaml').resolve(strict=True)
    checkpoint = (experiment / args.checkpoint).resolve(strict=True)
    conf = yaml.safe_load(config.read_text())
    assert conf['model'] == 'dual_whisper'
    assert conf['model_conf'].get('fusion_alpha', 0.5) == 0.5
    assert conf['model_conf']['small_model_name'] == 'small'
    assert conf['model_conf']['tiny_model_name'] == 'tiny'
    small = Path(conf.get('encoder_conf', {}).get('download_dir') or '~/.cache/whisper').expanduser() / 'small.pt'
    small = small.resolve(strict=True)
    decode = yaml.safe_load(Path('conf/whisper_decode_dual.yaml').read_text())
    assert decode == dict(beam_size=3, ctc_weight=0.0, lm_weight=0.0,
                         maxlenratio=-128, minlenratio=0.0, penalty=0.0)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or rerun on a GPU host')
    identity = dict(config_path=str(config), config_sha256=digest(config),
        checkpoint_path=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        small_path=str(small), small_sha256=digest(small), decode=decode,
        device=device, dev_data_sha256={name: digest(Path('dump/raw/org/valid') / name)
                                      for name in ['wav.scp', 'text', 'utt2spk']})
    manifest = root / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Manifest differs: select a fresh output directory; no existing results overwritten')
    else:
        write_json(manifest, identity)
    print(f'Pinned checkpoint: {checkpoint}\nDevice: {device}', flush=True)
    write_json(root / 'status.json', dict(state='preflight', pid=os.getpid()))
    check = preflight(root, identity, device, decode)
    pipeline_preflight(root, identity, device, decode, check)
    print('Preflight passed: endpoints, original 0.5, exact checkpoint, no gradients/updates.', flush=True)
    if args.preflight_only:
        return
    if args.test_alpha is not None and not 0 <= args.test_alpha <= 1:
        raise ValueError('test-alpha must be in [0, 1]')
    split = 'test' if args.test_alpha is not None else 'org/valid'
    alphas = [args.test_alpha] if args.test_alpha is not None else [0.5] + [a for a in ALPHAS if a != 0.5]
    rows = []
    if args.test_alpha is None:
        summarize(root, rows)
    for alpha in alphas:
        group = 'test' if args.test_alpha is not None else 'dev'
        directory = root / group / f'alpha_{alpha:.1f}'
        completed = directory / 'result.json'
        if completed.exists():
            row = json.loads(completed.read_text())
        else:
            # Refuse to overwrite incomplete results; user can move them for a retry.
            directory.mkdir(parents=True, exist_ok=False)
            (directory / 'config.yaml').symlink_to(config)
            (directory / 'model.pth').symlink_to(checkpoint)
            (directory / 'decode.yaml').write_text(yaml.safe_dump(decode))
            command = stage_command(directory, alpha, device, split)
            write_json(directory / 'command.json', command)
            write_json(root / 'status.json', dict(state='decoding', split=split, alpha=alpha,
                       started_unix=time.time(), directory=str(directory), pid=os.getpid()))
            print(f'Decoding {split}: small={alpha:.1f}, tiny={1-alpha:.1f}: {directory}', flush=True)
            start = time.monotonic()
            env = dict(os.environ, DUAL_PYTHON=sys.executable)
            with (directory / 'stages.log').open('x') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
            total_elapsed = time.monotonic() - start
            scored = directory / 'decode' / split
            job_log = (scored / 'logdir/asr_inference.1.log').read_text()
            match = re.search(r'elapsed time ([0-9.]+) seconds', job_log)
            if not match:
                raise RuntimeError('Missing successful decoder timing in job log')
            elapsed = float(match.group(1))
            write_json(directory / 'timing.json', dict(decode_seconds=elapsed, stages_seconds=total_elapsed))
            cer = percent(scored / 'score_cer/result.txt')
            wer = percent(scored / 'score_wer/result.txt')
            assert digest(checkpoint) == identity['checkpoint_sha256']
            assert digest(config) == identity['config_sha256']
            assert digest(small) == identity['small_sha256']
            row = dict(alpha=alpha, small_weight=alpha, tiny_weight=round(1-alpha, 10),
                dev_cer_percent=cer, dev_wer_percent=wer, decode_seconds=elapsed,
                decode_result_dir=str(scored))
            if args.test_alpha is not None:
                row['cer_percent'] = row.pop('dev_cer_percent')
                row['wer_percent'] = row.pop('dev_wer_percent')
            write_json(completed, row)
        if args.test_alpha is not None:
            write_json(root / 'test_result.json', dict(alpha=alpha, cer_percent=row['cer_percent'],
                       wer_percent=row['wer_percent'], decode_seconds=row['decode_seconds'],
                       result_dir=row['decode_result_dir']))
            continue
        rows.append(row)
        summarize(root, rows)
        if alpha == 0.5 and abs(row['dev_cer_percent'] - 30.6) > args.baseline_tolerance_pp:
            write_json(root / 'status.json', dict(state='baseline_mismatch', observed=row['dev_cer_percent'], expected=30.6))
            raise RuntimeError('alpha=0.5 did not reproduce Dev CER 30.6%; stopped before selecting alpha')
    if args.test_alpha is None:
        ordered = summarize(root, rows, complete=True)
        print(json.dumps(ordered, indent=2), flush=True)
        print(f"Best Dev alpha={ordered[0]['alpha']}; CER={ordered[0]['dev_cer_percent']}%", flush=True)
    write_json(root / 'status.json', dict(state='complete', split=split))


if __name__ == '__main__':
    main()
