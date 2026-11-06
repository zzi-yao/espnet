"""Verify the controlled small-FFT protocol against native ESPnet, without training."""
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet2.samplers.folded_batch_sampler import FoldedBatchSampler
from .task import DualASRTask

REFERENCE = Path('exp/asr_train_asr_whisper_small_finetune_raw_zh_whisper_multilingual_8:1:1_1e-4_small')


class ReferenceDecoder(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.ys_in = None

    def forward(self, memory, lengths, ys_in, ys_lengths):
        self.ys_in = ys_in.clone()
        self.ys_lengths = ys_lengths.clone()
        return self.logits, ys_lengths


def main():
    torch.manual_seed(2022)
    baseline = yaml.safe_load((REFERENCE/'config.yaml').read_text())
    args = DualASRTask.get_parser().parse_args([
        '--config', 'conf/tuning/train_asr_whisper_dual_smallfft_aligned.yaml',
        '--token_list', 'data/zh_token_list/whisper_multilingual/tokens.txt',
        '--token_type', 'whisper_multilingual', '--bpemodel', 'whisper_multilingual',
        '--fold_length', '80000', '--fold_length', '150',
        '--output_dir', '/tmp/dual_aligned_check'])
    shared = ['optim','optim_conf','scheduler','scheduler_conf','grad_clip','grad_clip_type',
        'max_epoch','batch_size','accum_grad','batch_type','fold_length','sort_batch','sort_in_batch',
        'seed','use_amp','train_dtype','num_workers','valid_batch_size','valid_batch_type','patience',
        'early_stopping_criterion','best_model_criterion','keep_nbest_models','val_scheduler_criterion',
        'num_iters_per_epoch','drop_last_iter','grad_noise','use_tf32','cudnn_enabled',
        'cudnn_deterministic','cudnn_benchmark','speech_volume_normalize','rir_scp','noise_scp']
    assert all(getattr(args, key) == baseline[key] for key in shared)
    assert args.init is None and args.init_param == []
    model = DualASRTask.build_model(args).train()
    assert model.training_protocol == 'small_fft'
    assert isinstance(model.criterion_att, LabelSmoothingLoss)
    assert model.criterion_att.smoothing == 0.1 and not model.criterion_att.normalize_length
    assert not model.encoder.small.training and not model.decoder.small.training
    assert all(not p.requires_grad for n,p in model.named_parameters() if model.frozen_key(n))
    # Read two REAL speech/label pairs, of different lengths with padding.
    _, batch = next(iter(DualASRTask.build_streaming_iterator(
        [('exp/dual_whisper_smoke/train.wav.scp','speech','kaldi_ark'),
         ('exp/dual_whisper_smoke/train.text','text','text')], dtype='float32', batch_size=2,
        num_workers=0, preprocess_fn=DualASRTask.build_preprocess_fn(args, False),
        collate_fn=DualASRTask.build_collate_fn(args, False))))
    labels = batch['text'].clone()
    lengths = batch['text_lengths'].clone()
    seen = {}
    def capture_decoder(module, inputs, output):
        seen['ys_in'] = inputs[2].clone()
        seen['ys_lengths'] = inputs[3].clone()
        seen['logits'] = output[0]
    def capture_criterion(module, inputs):
        seen['target'] = inputs[1].clone()
    hooks = [model.decoder.register_forward_hook(capture_decoder),
             model.criterion_att.register_forward_pre_hook(capture_criterion)]
    with patch('dual_whisper.model.BeamSearch', side_effect=AssertionError('Unexpected beam search')):
        loss, stats, _ = model(**batch)
    for hook in hooks: hook.remove()
    # A native single-model loss instance with the SAME logits isolates and tests
    # protocol equivalence from the intentional difference in model architecture.
    reference = ESPnetASRModel.__new__(ESPnetASRModel)
    torch.nn.Module.__init__(reference)
    for name in ['sos','eos','ignore_id','vocab_size','error_calculator','lang_token_id']:
        setattr(reference, name, getattr(model, name))
    reference.decoder = ReferenceDecoder(seen['logits'])
    reference.criterion_att = LabelSmoothingLoss(model.vocab_size, model.ignore_id,
        baseline['model_conf']['lsm_weight'], baseline['model_conf']['length_normalized_loss'])
    reference.train()
    target_seen = []
    ref_hook = reference.criterion_att.register_forward_pre_hook(
        lambda m, inputs: target_seen.append(inputs[1].clone()))
    with torch.no_grad():
        expected, ref_acc, _, _ = ESPnetASRModel._calc_att_loss(
            reference, torch.empty(2,1,1), torch.ones(2,dtype=torch.long), labels, lengths)
    ref_hook.remove()
    torch.testing.assert_close(loss.reshape(()), expected, atol=0, rtol=0)
    assert torch.equal(seen['ys_in'], reference.decoder.ys_in)
    assert torch.equal(seen['ys_lengths'], reference.decoder.ys_lengths)
    assert torch.equal(seen['target'], target_seen[0])
    target = seen['target']
    prompt_len = len(model.primer)-1
    assert torch.equal(target[:, :prompt_len], target.new_tensor(model.primer[1:]).expand(2,-1))
    assert target.ne(model.ignore_id).sum().item() == int((lengths+1).sum())
    for i, length in enumerate(lengths.tolist()):
        assert target[i, length] == model.eos
        assert seen['ys_in'][i, 0] == model.sos
        assert target[i, length+1:].eq(model.ignore_id).all()
    assert abs(float(stats['acc']) - ref_acc) < 1e-7
    pred = seen['logits'].argmax(-1)
    mask = target.ne(model.ignore_id)
    numerator = int(((pred == target) & mask).sum())
    denominator = int(mask.sum())
    assert abs(ref_acc - numerator/denominator) < 1e-12
    probes = {}
    with torch.no_grad():
        for name, position in [('prompt', 0), ('eos', int(lengths[0]))]:
            probe = torch.full_like(seen['logits'], -30)
            probe.scatter_(2, target.masked_fill(~mask, 0).unsqueeze(-1), 30)
            correct_id = int(target[0, position])
            probe[0, position].fill_(-30)
            probe[0, position, (correct_id+1) % model.vocab_size] = 30
            reference.decoder.logits = probe
            _, native_acc, _, _ = ESPnetASRModel._calc_att_loss(
                reference, torch.empty(2,1,1), torch.ones(2,dtype=torch.long), labels, lengths)
            with patch.object(model.decoder, 'forward', return_value=(probe, lengths+1)):
                _, dual_acc, _, _ = model._calc_att_loss(
                    torch.empty(2,1,1), torch.ones(2,dtype=torch.long), labels, lengths)
            assert native_acc == dual_acc == (denominator-1)/denominator
            probes[name] = dict(native_acc=native_acc, dual_acc=dual_acc)
    reference.decoder.logits = seen['logits']
    loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if model.frozen_key(n))
    for module in [model.encoder.tiny, model.decoder.tiny]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert any(p.grad.abs().max() > 0 for p in module.parameters())
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), patch('dual_whisper.model.BeamSearch', side_effect=AssertionError('TF searched')):
        val_loss, val_stats, _ = model(**batch)
    torch.testing.assert_close(val_loss, loss, atol=0, rtol=0)
    assert abs(float(val_stats['acc']) - ref_acc) < 1e-7
    # Reconstruct reference batch lists from current stats; no dataset iteration
    # or optimizer updates. Folded sizes are variable, not a constant batch=40.
    sampler_args = dict(batch_size=args.batch_size, fold_lengths=args.fold_length,
        sort_in_batch=args.sort_in_batch, sort_batch=args.sort_batch)
    train = FoldedBatchSampler(shape_files=baseline['train_shape_file'], **sampler_args)
    valid = FoldedBatchSampler(shape_files=baseline['valid_shape_file'], **sampler_args)
    assert len(train) == 4683 and len(valid) == 583
    assert sum(len(b) for b in train) == 44675
    # Map the old training checkpoint without copying its multi-GB model/optimizer
    # tensors into RAM, and read only the scalar step counters.
    old_checkpoint = torch.load(REFERENCE/'checkpoint.pth',map_location='cpu',weights_only=False,mmap=True)
    old_steps = sorted({int(v['step']) for v in old_checkpoint['optimizers'][0]['state'].values()})
    assert old_steps == [8190] and old_checkpoint['schedulers'][0]['last_epoch'] == 8190
    result = dict(device='cpu', native_protocol_loss=float(expected), dual_loss=float(loss),
        decoder_input_exact_match=True, target_and_padding_exact_match=True,
        prompt_supervised=True, eos_supervised=True, loss_denominator='batch_size',
        label_smoothing=0.1, criterion='native ESPnet LabelSmoothingLoss',
        acc_native=ref_acc, acc_dual=float(val_stats['acc']), accuracy_numerator=numerator,
        accuracy_denominator=denominator, valid_acc_definition_matches=True,
        prompt_and_eos_error_probes=probes,
        small_grad_none=True, tiny_gradients_finite=True, tf_validation_no_search=True,
        shared_runtime_settings_match=shared, train_samples=44675,
        train_batches=4683, valid_batches=583, nominal_max_effective_batch_size=40,
        scheduled_update_boundaries_per_epoch=len(train)//args.accum_grad,
        reference_actual_adamw_steps=old_steps, total_scheduled_update_boundaries=8190,
        native_loss_method_sha256=hashlib.sha256(inspect.getsource(ESPnetASRModel._calc_att_loss).encode()).hexdigest(),
        no_formal_training_started=True)
    output = Path('exp/dual_whisper_smallfft_aligned_checks/results.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
