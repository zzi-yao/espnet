"""One real batch: aux-CE identity, gradients, update and legacy compatibility."""
import argparse
import json
from pathlib import Path
import tempfile

import torch
import yaml
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from .task import DualASRTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--output', default='exp/dual_whisper_auxce_checks/results.json')
    options = parser.parse_args()
    if options.amp and options.device != 'cuda':
        parser.error('--amp requires CUDA')
    old = yaml.safe_load(Path('conf/tuning/train_asr_whisper_dual.yaml').read_text())
    new = yaml.safe_load(Path('conf/tuning/train_asr_whisper_dual_auxce.yaml').read_text())
    stripped = dict(new, model_conf=dict(new['model_conf']))
    assert stripped['model_conf'].pop('loss_mode') == 'tiny_ce_plus_fusion_ce'
    assert stripped['model_conf'].pop('lambda_fusion') == 0.5
    assert stripped == old, 'A training option other than the loss was changed'
    torch.manual_seed(new['seed'])
    args = DualASRTask.get_parser().parse_args([
        '--config', 'conf/tuning/train_asr_whisper_dual_auxce.yaml',
        '--token_list', 'data/zh_token_list/whisper_multilingual/tokens.txt',
        '--token_type', 'whisper_multilingual', '--bpemodel', 'whisper_multilingual',
        '--output_dir', '/tmp/dual_auxce_check'])
    assert args.init is None and not args.init_param
    model = DualASRTask.build_model(args).to(options.device).train()
    optimizer = DualASRTask.build_optimizers(args, model)[0]
    keys = [line.split()[0] for line in Path('dump/raw/train/wav.scp').read_text().splitlines()[:2]]
    with tempfile.TemporaryDirectory(prefix='auxce-check-') as temporary:
        keyfile = Path(temporary) / 'keys'
        keyfile.write_text('\n'.join(keys) + '\n')
        iterator = DualASRTask.build_streaming_iterator(
            [('dump/raw/train/wav.scp', 'speech', 'kaldi_ark'), ('dump/raw/train/text', 'text', 'text')],
            dtype='float32', batch_size=2, num_workers=0, key_file=str(keyfile),
            preprocess_fn=DualASRTask.build_preprocess_fn(args, False),
            collate_fn=DualASRTask.build_collate_fn(args, False))
        _, batch = next(iter(iterator))
    batch = {k: v.to(options.device) for k, v in batch.items()}
    assert batch['text_lengths'][0] != batch['text_lengths'][1]  # Actual padding.
    assert all(not p.requires_grad for n, p in model.named_parameters() if model.frozen_key(n))
    assert all(p.requires_grad for n, p in model.named_parameters() if not model.frozen_key(n))
    assert not model.encoder.small.training and not model.decoder.small.training
    before = model.encoder.tiny.encoders.conv1.weight.clone()
    frozen_before = model.encoder.small.encoders.conv1.weight.clone()
    if options.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    captured = {}
    def hook(name):
        def capture(module, inputs, output):
            captured[name] = output[0]
        return capture
    hooks = [part.register_forward_hook(hook(name)) for name, part in
             [('small', model.decoder.small), ('tiny', model.decoder.tiny), ('fusion', model.decoder)]]
    with torch.autocast(options.device, enabled=options.amp, dtype=torch.float16):
        loss, stats, weight = model(**batch)
    for item in hooks:
        item.remove()
    assert not captured['small'].requires_grad and captured['tiny'].requires_grad
    assert captured['fusion'].requires_grad
    _, targets = add_sos_eos(batch['text'], model.sos, model.eos, model.ignore_id)
    targets[:, :len(model.primer)-1] = model.ignore_id
    with torch.no_grad(), torch.autocast(options.device, enabled=options.amp, dtype=torch.float16):
        denom = targets.ne(model.ignore_id).sum() if model.length_normalized_loss else targets.size(0)
        tiny = model.criterion_att(captured['tiny'].reshape(-1, model.vocab_size), targets.reshape(-1)) / denom
        fused = model.criterion_att(captured['fusion'].reshape(-1, model.vocab_size), targets.reshape(-1)) / denom
        torch.testing.assert_close(loss.reshape(()), tiny + 0.5 * fused)
        torch.testing.assert_close(stats['loss_total'], stats['loss_tiny'] + 0.5 * stats['loss_fusion'])
        model.loss_mode = 'fusion_ce_only'
        legacy, _, _ = model(**batch)
        torch.testing.assert_close(legacy.reshape(()), fused)
        model.loss_mode = 'tiny_ce_plus_fusion_ce'
        model.eval()
        validation, validation_stats, _ = model(**batch)
        torch.testing.assert_close(validation.reshape(()), fused)
        assert 'loss_tiny' not in validation_stats and 'cer_tf' in validation_stats
    model.train()
    assert torch.isfinite(loss)
    scaler = torch.cuda.amp.GradScaler(enabled=options.amp, init_scale=128)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(p.grad is None for n, p in model.named_parameters() if model.frozen_key(n))
    for module in (model.encoder.tiny, model.decoder.tiny):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, model.encoder.tiny.encoders.conv1.weight)
    assert torch.equal(frozen_before, model.encoder.small.encoders.conv1.weight)
    assert model._auxiliary_stats == {}  # No retained per-batch graphs or stats.
    peak = dict(allocated=None, reserved=None)
    if options.device == 'cuda':
        torch.cuda.synchronize()
        peak = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
    result = dict(device=options.device, amp=options.amp, real_utterances=keys,
        loss_tiny=float(stats['loss_tiny']), loss_fusion=float(stats['loss_fusion']),
        loss_total=float(loss), legacy_loss=float(legacy), validation_loss=float(validation),
        acc_tiny=float(stats['acc_tiny']), acc_fusion=float(stats['acc_fusion']),
        same_targets_and_normalization=True, small_grad_none=True,
        tiny_gradients_finite=True, tiny_parameter_updated=True,
        old_mode_matches=True, validation_unchanged=True, config_only_loss_changed=True,
        cuda_peak_bytes=peak)
    path = Path(options.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
