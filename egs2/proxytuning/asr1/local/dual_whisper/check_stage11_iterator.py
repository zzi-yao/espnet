"""Reparse the actual Stage 11 invocation and read real batches, without a model."""
import json
from pathlib import Path
import shlex

import yaml
from espnet2.train.distributed_utils import DistributedOption
from .task import DualASRTask


def main():
    root = Path('exp/asr_whisper_dual_smallfft_aligned')
    line = (root / 'train.log').read_text().splitlines()[0]
    command = shlex.split(line.removeprefix('# '))
    cli = command[command.index('-m') + 2:]
    args = DualASRTask.get_parser().parse_args(cli)
    assert args.batch_type == 'folded'
    assert args.fold_length == [80000, 150], args.fold_length
    expected_train = ['exp/asr_stats_raw_zh_whisper_multilingual/train/speech_shape',
                      'exp/asr_stats_raw_zh_whisper_multilingual/train/text_shape.whisper_multilingual']
    expected_valid = ['exp/asr_stats_raw_zh_whisper_multilingual/valid/speech_shape',
                      'exp/asr_stats_raw_zh_whisper_multilingual/valid/text_shape.whisper_multilingual']
    assert args.train_shape_file == expected_train and args.valid_shape_file == expected_valid
    reference = yaml.safe_load(Path('exp/asr_train_asr_whisper_small_finetune_raw_zh_whisper_multilingual_8:1:1_1e-4_small/config.yaml').read_text())
    assert args.fold_length == reference['fold_length']
    assert args.train_shape_file == reference['train_shape_file']
    assert args.valid_shape_file == reference['valid_shape_file']
    failed = yaml.safe_load((root/'config.yaml').read_text())
    old_yaml = yaml.safe_load(Path('conf/tuning/train_asr_whisper_dual_smallfft_aligned.yaml').read_text())
    # Verify no other effective Stage 11 option changed. Defaults and actual CLI
    # are both represented in the previously saved failed-run configuration.
    equal_fields = ['batch_type','batch_size','accum_grad','max_epoch','optim','optim_conf',
        'scheduler','scheduler_conf','grad_clip','grad_clip_type','use_amp','train_dtype',
        'seed','model_conf','best_model_criterion','early_stopping_criterion','patience',
        'valid_batch_size','num_workers','keep_nbest_models']
    assert all(getattr(args,k)==failed[k] for k in equal_fields)
    assert 'fold_length' not in old_yaml
    # Avoid GPU pinning and worker IPC restrictions during the read-only check.
    # Preserve the configured worker count in args; only this test loader uses 0.
    args.ngpu = 0
    info = {}
    for mode in ['train','valid']:
        factory = DualASRTask.build_iter_factory(args, DistributedOption(distributed=False,ngpu=0), mode)
        assert factory.num_workers == 4
        factory.num_workers = 0
        keys, batch = next(iter(factory.build_iter(1)))
        assert batch['speech'].shape[0] == len(keys) == batch['text'].shape[0]
        assert batch['speech_lengths'].gt(0).all() and batch['text_lengths'].gt(0).all()
        info[mode] = dict(shape_files=getattr(args,mode+'_shape_file'), fold_length=args.fold_length,
            batch_count=len(factory.sampler), first_batch_utterances=keys,
            speech_shape=list(batch['speech'].shape), text_shape=list(batch['text'].shape),
            speech_lengths=batch['speech_lengths'].tolist(),text_lengths=batch['text_lengths'].tolist())
    assert info['train']['batch_count']==4683 and info['valid']['batch_count']==583
    report = dict(old_failed_fold_length=failed['fold_length'], fixed_effective_fold_length=args.fold_length,
        matches_reference_small=True, other_effective_settings_unchanged=equal_fields,
        test_only_overrides=dict(ngpu=0,loader_workers=0), configured_training_workers=4,
        iterators=info, model_constructed=False, training_started=False)
    output=Path('exp/dual_whisper_smallfft_aligned_checks/iterator_check.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
