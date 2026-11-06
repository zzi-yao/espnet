"""Read-only initialization, strict-load and optimizer audit for FFT-small dual."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from .task import DualASRTask


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = DualASRTask.get_parser().parse_args([
        '--config', 'conf/tuning/train_asr_whisper_dual_smallfft.yaml',
        '--token_list', 'data/zh_token_list/whisper_multilingual/tokens.txt',
        '--token_type', 'whisper_multilingual', '--bpemodel', 'whisper_multilingual',
        '--output_dir', 'exp/asr_whisper_dual_smallfft_tiny'])
    assert args.init is None and args.init_param == []
    load_reports = []
    original = torch.nn.Module.load_state_dict
    def record(module, state_dict, strict=True, **kwargs):
        result = original(module, state_dict, strict=strict, **kwargs)
        if type(module).__name__ in ('OpenAIWhisperEncoder', 'OpenAIWhisperDecoder'):
            load_reports.append(dict(module=type(module).__name__, strict=strict,
                tensor_count=len(state_dict), missing_keys=result.missing_keys,
                unexpected_keys=result.unexpected_keys))
        return result
    with patch.object(torch.nn.Module, 'load_state_dict', record):
        model = DualASRTask.build_model(args)
    assert len(load_reports) == 2
    assert all(r['strict'] and not r['missing_keys'] and not r['unexpected_keys'] for r in load_reports)
    source = Path(model.small_model_file)
    state = torch.load(source, map_location='cpu', weights_only=False)
    extras = [k for k in state if not k.startswith(('encoder.', 'decoder.'))]
    small_equal = {}
    for prefix, module in [('encoder.', model.encoder.small), ('decoder.', model.decoder.small)]:
        expected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        actual = module.state_dict()
        assert set(actual) == set(expected)
        assert all(torch.equal(v, expected[k]) for k, v in actual.items())
        small_equal[prefix] = len(actual)
    del state, expected, actual
    tiny_file = Path.home() / '.cache/whisper/tiny.pt'
    tiny = torch.load(tiny_file, map_location='cpu', weights_only=False)['model_state_dict']
    for prefix, module in [('encoder.', model.encoder.tiny.encoders), ('decoder.', model.decoder.tiny.decoders)]:
        assert all(torch.equal(v, tiny[prefix+k]) for k, v in module.state_dict().items())
    del tiny
    model.train()
    assert not model.encoder.small.training and not model.decoder.small.training
    optimizer = DualASRTask.build_optimizers(args, model)[0]
    expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
    actual_params = [p for group in optimizer.param_groups for p in group['params']]
    assert {id(p) for p in actual_params} == expected_ids
    assert all(p.requires_grad for p in actual_params)
    assert all(not p.requires_grad for n, p in model.named_parameters() if model.frozen_key(n))
    own_checkpoint = model.state_dict()
    assert not any(model.frozen_key(k) for k in own_checkpoint)
    reloaded = model.load_state_dict(own_checkpoint, strict=True)
    assert not reloaded.missing_keys and not reloaded.unexpected_keys
    report = dict(source_checkpoint=str(source), source_sha256=sha256(source),
        source_config=str(source.parent/'config.yaml'), strict_loads=load_reports,
        outside_encoder_decoder_keys=extras, all_small_tensors_exact_match=small_equal,
        tiny_pretrained_file=str(tiny_file), tiny_pretrained_sha256=sha256(tiny_file),
        all_tiny_tensors_exact_match=True, parameter_report=model.parameter_report(),
        trainable_tensor_count=len(expected_ids), optimizer_tensor_count=len(actual_params),
        optimizer_exactly_trainable=True, optimizer_state_empty=not bool(optimizer.state),
        small_frozen_eval=True, own_checkpoint_strict_reload_passed=True,
        proposed_experiment_dir=str(Path(args.output_dir).resolve()),
        proposed_dir_exists=Path(args.output_dir).exists(), init_param=args.init_param,
        runtime_config={k:getattr(args,k) for k in ('optim','optim_conf','scheduler','scheduler_conf',
            'max_epoch','batch_size','valid_batch_size','accum_grad','grad_clip','grad_clip_type',
            'use_amp','train_dtype','seed','best_model_criterion','early_stopping_criterion',
            'patience','keep_nbest_models','num_workers','sort_batch','sort_in_batch')})
    output = Path('exp/dual_whisper_smallfft_checks/audit.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
