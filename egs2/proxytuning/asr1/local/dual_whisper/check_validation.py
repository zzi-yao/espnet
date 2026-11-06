"""Real data regression: gradients, fast TF metrics, and opt-in AR validation."""
import argparse
import json
import logging
from pathlib import Path
import time
from unittest.mock import patch

import torch
from espnet.nets.beam_search import BeamSearch
from .task import DualASRTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='exp/dual_whisper_validation_checks.json')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train_args = DualASRTask.get_parser().parse_args([
        '--config', 'conf/tuning/train_asr_whisper_dual.yaml',
        '--token_list', 'data/zh_token_list/whisper_multilingual/tokens.txt',
        '--token_type', 'whisper_multilingual', '--bpemodel', 'whisper_multilingual',
        '--output_dir', '/tmp/dual_validation_check'])
    model = DualASRTask.build_model(train_args)
    iterator = DualASRTask.build_streaming_iterator(
        [('exp/dual_whisper_smoke/train.wav.scp', 'speech', 'kaldi_ark'),
         ('exp/dual_whisper_smoke/train.text', 'text', 'text')],
        dtype='float32', batch_size=2, num_workers=0,
        preprocess_fn=DualASRTask.build_preprocess_fn(train_args, train=False),
        collate_fn=DualASRTask.build_collate_fn(train_args, train=False))
    ids, batch = next(iter(iterator))
    reference = batch['text'].clone()
    assert len(set(batch['text_lengths'].tolist())) == 2  # Exercise real padding.
    model.train()
    assert not model.encoder.small.training and not model.decoder.small.training
    assert model.encoder.tiny.training and model.decoder.tiny.training
    with patch('dual_whisper.model.BeamSearch', side_effect=AssertionError('Unexpected TF beam search')):
        train_loss, _, weight = model(**batch)
        assert torch.isfinite(train_loss) and weight.item() == 2
        train_loss.backward()
        assert all(p.grad is None for n, p in model.named_parameters() if model.frozen_key(n))
        gradients = {}
        for name, part in [('encoder', model.encoder.tiny), ('decoder', model.decoder.tiny)]:
            params = list(part.named_parameters())
            assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
                       for _, p in params)
            major = [(n, p) for n, p in params if 'weight' in n and
                     ('conv1' in n or 'token_embedding' in n or 'attn.query' in n)]
            assert major and all(p.grad.abs().max() > 0 for _, p in major)
            gradients[name] = {n: float(p.grad.norm()) for n, p in major}
        model.zero_grad(set_to_none=True)
        model.eval()
        captured = []
        hook = model.decoder.register_forward_hook(lambda m, inputs, output: captured.append(output[0]))
        start = time.perf_counter()
        with torch.no_grad():
            valid_loss, stats, weight = model(**batch)
        tf_seconds = time.perf_counter() - start
        hook.remove()
        assert len(captured) == 1  # Exactly one parallel teacher-forcing decoder call.
        assert torch.isfinite(valid_loss)
        assert 'cer_tf' in stats and 'wer_tf' in stats
        assert 'cer' not in stats and 'wer' not in stats
        n_prefix = len(model.primer) - 1
        expected_cer, expected_wer = model.error_calculator(
            captured[0].argmax(-1)[:, n_prefix:].cpu(), reference[:, n_prefix:].cpu())
        assert abs(float(stats['cer_tf']) - expected_cer) < 1e-6
        assert abs(float(stats['wer_tf']) - expected_wer) < 1e-6
        assert weight.item() == 2
        torch.testing.assert_close(train_loss, valid_loss)  # No dropout or updates.
    model.validation_decode_mode = 'autoregressive'
    model.validation_max_tokens = 8
    with patch('dual_whisper.model.BeamSearch', wraps=BeamSearch) as beam:
        with torch.no_grad():
            _, ar_stats, _ = model(**{k: v[:1] for k, v in batch.items()})
        assert beam.call_count == 1
        assert 'cer' in ar_stats and 'wer' in ar_stats
        assert 'cer_tf' not in ar_stats and 'wer_tf' not in ar_stats
    report = dict(device='cpu', real_utterances=ids, train_loss=float(train_loss),
        valid_loss=float(valid_loss), cer_tf=float(stats['cer_tf']), wer_tf=float(stats['wer_tf']),
        valid_batch_weight=int(weight), tf_seconds=tf_seconds, tf_beam_calls=0,
        tf_decoder_calls=1, gradients=gradients, small_grad_none=True,
        upstream_error_calculator_matches=True,
        autoregressive_mode=dict(cer=float(ar_stats['cer']), wer=float(ar_stats['wer']), beam_calls=1))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
