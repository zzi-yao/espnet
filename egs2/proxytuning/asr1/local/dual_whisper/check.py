"""Real pretrained model forward/backward and numerical/checkpoint diagnostics."""
import argparse
import json
from pathlib import Path

import torch
from .task import DualASRTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", default="exp/dual_whisper_checks.json")
    args = parser.parse_args()
    if args.amp and args.device != "cuda":
        parser.error("--amp requires --device cuda")
    if args.steps < 2:
        parser.error("--steps must be at least 2 for the loss descent check")
    torch.manual_seed(2022)
    train_args = DualASRTask.get_parser().parse_args([
        "--config", "conf/tuning/train_asr_whisper_dual.yaml",
        "--token_list", "data/zh_token_list/whisper_multilingual/tokens.txt",
        "--token_type", "whisper_multilingual", "--bpemodel", "whisper_multilingual",
        "--output_dir", "/tmp/dual_check"])
    model = DualASRTask.build_model(train_args).to(args.device).train()
    assert not model.encoder.small.training and not model.decoder.small.training
    assert model.encoder.tiny.training and model.decoder.tiny.training
    for n, p in model.named_parameters():
        assert p.requires_grad != model.frozen_key(n), n
    optim = DualASRTask.build_optimizers(train_args, model)[0]
    # Fixed synthetic waveform and Chinese labels: exercises real encoder/decoder,
    # not an ASR quality evaluation. Use a short sequence to keep CPU cost bounded.
    speech = torch.randn(1, 4000, device=args.device) * 0.01
    ids = list(model.primer[1:]) + model.whisper_tokenizer.encode("你好", add_special_tokens=False)
    text = torch.tensor([ids], device=args.device)
    batch = dict(speech=speech, speech_lengths=torch.tensor([4000], device=args.device),
                 text=text, text_lengths=torch.tensor([len(ids)], device=args.device))
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    captured = {}
    def capture(name):
        def hook(module, inputs, output):
            captured[name] = output[0]
        return hook
    hooks = [part.register_forward_hook(capture(name)) for name, part in
             (("small", model.decoder.small), ("tiny", model.decoder.tiny),
              ("final", model.decoder))]
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    losses = []
    gradients = {}
    for step in range(args.steps):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=args.device, enabled=args.amp, dtype=torch.float16):
            loss, _, _ = model(**batch)
        if step == 0:
            assert not captured["small"].requires_grad
            assert captured["tiny"].requires_grad and captured["final"].requires_grad
            assert tuple(captured["final"].shape) == (1, len(ids) + 1, 51865)
            with torch.no_grad():
                torch.testing.assert_close(captured["final"],
                    0.5 * captured["small"] + 0.5 * captured["tiny"])
            for hook in hooks:
                hook.remove()
            captured.clear()
        assert torch.isfinite(loss), loss
        losses.append(float(loss))
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        assert all(p.grad is None for n, p in model.named_parameters() if model.frozen_key(n))
        for name, part in (("encoder", model.encoder.tiny), ("decoder", model.decoder.tiny)):
            params = [(n, p) for n, p in part.named_parameters()]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in params), name
            major = [(n, p) for n, p in params if "weight" in n and ("conv1" in n or "token_embedding" in n or "attn.query" in n)]
            assert major and all(p.grad.abs().max() > 0 for _, p in major), name
            gradients[name] = {n: float(p.grad.norm()) for n, p in major}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
    with torch.no_grad(), torch.autocast(device_type=args.device, enabled=args.amp, dtype=torch.float16):
        final_loss = float(model(**batch)[0])
    assert final_loss < losses[0], (losses, final_loss)
    # The same prefix must produce identical teacher-forcing last-step and
    # autoregressive fused log probabilities for serial and batch scorers.
    training_shapes = model.decoder.last_shapes
    model.eval()
    with torch.no_grad():
        memory, lengths = model.encode(speech, batch["speech_lengths"])
        prefix = text.new_tensor([list(model.primer)])
        logits, _ = model.decoder(memory, lengths, prefix, text.new_tensor([prefix.size(1)]))
        expected = logits[:, -1].log_softmax(-1)
        serial, state = model.decoder.score(prefix[0], (None, None), memory[0])
        batched, states = model.decoder.batch_score(prefix, [(None, None)], memory)
        torch.testing.assert_close(serial, expected[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(batched, expected, atol=1e-5, rtol=1e-5)
        assert state == (None, None) and states == [(None, None)]
    autoregressive_shapes = model.decoder.last_shapes
    state = model.state_dict()
    assert not any(model.frozen_key(k) for k in state)
    assert any(k.startswith("encoder.tiny.") for k in state)
    assert any(k.startswith("decoder.tiny.") for k in state)
    # A missing tiny key must remain an error, even though frozen keys are omitted.
    damaged = state.copy()
    del damaged[next(k for k in damaged if k.startswith("decoder.tiny."))]
    try:
        model.load_state_dict(damaged)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing tiny checkpoint key was silently accepted")
    model.load_state_dict(state)
    peak = dict(allocated=None, reserved=None)
    if args.device == "cuda":
        torch.cuda.synchronize()
        peak = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
    report = dict(device=args.device, amp=args.amp, parameters=model.parameter_report(),
                  losses=losses, final_loss=final_loss, gradient_norms=gradients,
                  training_shapes=training_shapes, autoregressive_shapes=autoregressive_shapes, cuda_peak_bytes=peak,
                  checkpoint_tensor_count=len(state), small_grad_none=True,
                  autoregressive_logits_match=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
