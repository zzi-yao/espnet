"""Logits-only side tuning, using the recipe's existing Whisper implementations."""
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.beam_search import BeamSearch
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos


class FrozenSmallMixin:
    def train(self, mode=True):
        super().train(mode)
        self.small.eval()
        return self

    def freeze_small(self):
        self.small.requires_grad_(False)
        self.tiny.requires_grad_(True)
        self.small.eval()


class DualWhisperEncoder(FrozenSmallMixin, AbsEncoder):
    def __init__(self, input_size=1, small_model_name="small", tiny_model_name="tiny",
                 download_dir=None):
        super().__init__()
        self.small = OpenAIWhisperEncoder(input_size=input_size,
            whisper_model=small_model_name, download_dir=download_dir)
        self.tiny = OpenAIWhisperEncoder(input_size=input_size,
            whisper_model=tiny_model_name, download_dir=download_dir)
        self.freeze_small()

    def output_size(self):
        return self.small.output_size() + self.tiny.output_size()

    def forward(self, xs_pad, ilens, prev_states=None):
        with torch.no_grad():
            large, large_lens, _ = self.small(xs_pad, ilens)
        tiny, tiny_lens, _ = self.tiny(xs_pad, ilens)
        if large.shape[:2] != tiny.shape[:2] or not torch.equal(large_lens, tiny_lens):
            raise ValueError("Both Whisper encoders must have the same time axis")
        # Transport container for ESPnet's tensor-based beam search, NOT feature fusion.
        # The decoder splits this back into the two independent memories immediately.
        return torch.cat((large, tiny), dim=-1), tiny_lens, None


class DualWhisperDecoder(FrozenSmallMixin, AbsDecoder, BatchScorerInterface):
    def __init__(self, vocab_size, encoder_output_size, small_model_name="small",
                 tiny_model_name="tiny", fusion_alpha=0.5, download_dir=None):
        super().__init__()
        self.small = OpenAIWhisperDecoder(vocab_size, encoder_output_size,
            whisper_model=small_model_name, download_dir=download_dir,
            use_proxy_tuning=True)
        self.tiny = OpenAIWhisperDecoder(vocab_size, encoder_output_size,
            whisper_model=tiny_model_name, download_dir=download_dir,
            use_proxy_tuning=True)
        self.small_dim = self.small.decoders.token_embedding.embedding_dim
        self.tiny_dim = self.tiny.decoders.token_embedding.embedding_dim
        self.alpha = fusion_alpha
        self.inference_fusion_alpha = None
        self.last_shapes = None
        self.freeze_small()

    def split_memory(self, memory):
        if memory.size(-1) != self.small_dim + self.tiny_dim:
            raise ValueError("Unexpected packed encoder dimension")
        return memory.split((self.small_dim, self.tiny_dim), dim=-1)

    def fuse(self, small, tiny, alpha=None):
        if small.shape != tiny.shape:
            raise ValueError(f"Logits differ: {small.shape} vs {tiny.shape}")
        alpha = self.alpha if alpha is None else alpha
        final = alpha * small + (1.0 - alpha) * tiny
        self.last_shapes = tuple(small.shape), tuple(tiny.shape), tuple(final.shape)
        return final

    def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens, *, return_tiny_logits=False):
        small_memory, tiny_memory = self.split_memory(hs_pad)
        with torch.no_grad():
            small, _ = self.small(small_memory, hlens, ys_in_pad, ys_in_lens)
        tiny, _ = self.tiny(tiny_memory, hlens, ys_in_pad, ys_in_lens)
        fused = self.fuse(small, tiny)
        if return_tiny_logits:
            return fused, ys_in_lens, tiny
        return fused, ys_in_lens

    def forward_one_step(self, tgt, tgt_mask, memory, *, cache=None):
        # Existing Whisper wrappers recompute the complete prefix and return no KV
        # cache. Preserve that correct reference path; never mix decoder states.
        small_memory, tiny_memory = self.split_memory(memory)
        small_state, tiny_state = (None, None) if cache is None else cache
        with torch.no_grad():
            small, small_state = self.small.forward_one_step(
                tgt, tgt_mask, small_memory, cache=small_state)
        tiny, tiny_state = self.tiny.forward_one_step(
            tgt, tgt_mask, tiny_memory, cache=tiny_state)
        logits = self.fuse(small, tiny, alpha=self.inference_fusion_alpha)
        return F.log_softmax(logits, dim=-1), (small_state, tiny_state)

    def set_inference_fusion_alpha(self, alpha):
        """Override only autoregressive fusion; None restores training weight."""
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("inference_fusion_alpha must be in [0, 1]")
        self.inference_fusion_alpha = alpha

    def init_state(self, x):
        return (None, None)

    def select_state(self, state, i, new_id=None):
        return state[i] if isinstance(state, list) else state

    def score(self, ys, state, x):
        scores, state = self.forward_one_step(
            ys.unsqueeze(0), None, x.unsqueeze(0), cache=state)
        return scores.squeeze(0), state

    def batch_score(self, ys, states, xs):
        # No cache is retained by either underlying wrapper.
        scores, _ = self.forward_one_step(ys, None, xs)
        return scores, [(None, None) for _ in range(ys.size(0))]


class DualWhisperASRModel(ESPnetASRModel):
    def __init__(self, *args, dual_model=True, small_model_name="small",
                 tiny_model_name="tiny", fusion_alpha=0.5, freeze_small=True,
                 train_tiny=True, use_ffm=False, validation_beam_size=3,
                 validation_max_tokens=128, validation_decode_mode="teacher_forcing",
                 loss_mode="fusion_ce_only", lambda_fusion=0.5,
                 small_model_file=None, training_protocol="legacy", **kwargs):
        if training_protocol not in ("legacy", "small_fft"):
            raise ValueError("training_protocol must be legacy or small_fft")
        if training_protocol == "small_fft":
            if loss_mode != "fusion_ce_only" or fusion_alpha != 0.5 or small_model_file is None:
                raise ValueError("small_fft protocol requires FFT small, fusion_ce_only and alpha=0.5")
            if kwargs.get("lsm_weight", 0.0) != 0.1 or kwargs.get("length_normalized_loss", False):
                raise ValueError("small_fft protocol requires smoothing=0.1 and normalize_length=false")
            if validation_decode_mode != "teacher_forcing":
                raise ValueError("small_fft protocol uses teacher-forcing epoch validation")
        if loss_mode not in ("fusion_ce_only", "tiny_ce_plus_fusion_ce"):
            raise ValueError("Unknown dual Whisper loss_mode")
        if not math.isfinite(lambda_fusion) or lambda_fusion < 0:
            raise ValueError("lambda_fusion must be finite and nonnegative")
        if validation_decode_mode not in ("teacher_forcing", "autoregressive"):
            raise ValueError("validation_decode_mode must be teacher_forcing or autoregressive")
        if not dual_model or not freeze_small or not train_tiny or use_ffm:
            raise ValueError("dual mode requires frozen small, full tiny, and no FFM")
        if not 0.0 < fusion_alpha < 1.0:
            raise ValueError("fusion_alpha must be strictly between 0 and 1")
        for key in ("ctc_weight", "interctc_weight"):
            if kwargs.get(key, 0.0) != 0.0:
                raise ValueError(f"Minimal logits CE requires {key}=0")
        if kwargs.get("lang_token_id", -1) != -1:
            raise ValueError("Whisper preprocessor already supplies the language/task prefix")
        if training_protocol == "legacy" and kwargs.get("lsm_weight", 0.0) != 0.0:
            raise ValueError("Legacy protocol requires lsm_weight=0")
        kwargs.update(ctc_weight=0.0)
        super().__init__(*args, **kwargs)
        if not isinstance(self.encoder, DualWhisperEncoder) or not isinstance(self.decoder, DualWhisperDecoder):
            raise TypeError("Dual model requires its paired encoder and decoder")
        self.length_normalized_loss = self.criterion_att.normalize_length
        self.training_protocol = training_protocol
        if training_protocol == "legacy":
            self.criterion_att = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_id, reduction="sum")
        # small_fft retains the original LabelSmoothingLoss constructed by ESPnet.
        self.validation_beam_size = validation_beam_size
        self.validation_max_tokens = validation_max_tokens
        self.validation_decode_mode = validation_decode_mode
        self.loss_mode = loss_mode
        self.lambda_fusion = lambda_fusion
        self._auxiliary_stats = {}
        logging.info("Dual training_protocol=%s; validation mode=%s; error metrics=%s",
                     training_protocol, validation_decode_mode,
                     "cer_tf/wer_tf" if validation_decode_mode == "teacher_forcing" else "cer/wer")
        self.dual_config = dict(small_model_name=small_model_name,
            tiny_model_name=tiny_model_name, fusion_alpha=fusion_alpha)
        self.decoder.alpha = fusion_alpha
        self.primer = None  # Set and verified by the Task before training/inference.
        self.small_model_file = None
        if small_model_file is not None:
            self.load_small_checkpoint(small_model_file)

    def load_small_checkpoint(self, model_file):
        """Load an ESPnet small FFT encoder/decoder without touching tiny."""
        path = Path(model_file).resolve(strict=True)
        config = yaml.safe_load((path.parent / "config.yaml").read_text())
        if any(config.get(key) != "whisper" for key in ("encoder", "decoder")):
            raise ValueError("Frozen small checkpoint must use Whisper encoder/decoder")
        if any(config.get(key, {}).get("whisper_model") != "small"
               for key in ("encoder_conf", "decoder_conf")):
            raise ValueError("Expected a Whisper-small FFT checkpoint")
        tokens = config["token_list"]
        if isinstance(tokens, str):
            token_path = Path(tokens)
            if not token_path.exists():
                token_path = path.parent / token_path
            tokens = token_path.read_text().splitlines()
        if [t.rstrip() for t in tokens] != [t.rstrip() for t in self.token_list]:
            raise ValueError("Small FFT token IDs differ from the dual model")
        prompt = config.get("preprocessor_conf", {})
        if prompt.get("whisper_language") != "zh" or prompt.get("whisper_task") != "transcribe":
            raise ValueError("Small FFT must use zh/transcribe")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        if config.get("use_adapter", False):
            if config.get("adapter") != "lora" or config.get("save_strategy") != "required_grad_only":
                raise ValueError("Only adapter-only LoRA checkpoints are supported")
            adapter = config["adapter_conf"]
            rank = adapter["rank"]
            scale = adapter["alpha"] / rank
            # Reconstruct official base + trained LoRA delta. Frozen eval LoRA
            # is equivalent to this merge; dropout is inactive during inference.
            merged = {}
            consumed = set()
            for prefix, module in (("encoder.", self.encoder.small),
                                   ("decoder.", self.decoder.small)):
                for key, value in module.state_dict().items():
                    full_key = prefix + key
                    merged[full_key] = state.get(full_key, value)
                    if full_key in state:
                        consumed.add(full_key)
                for name, layer in module.named_modules():
                    if not any(name.endswith(t) for t in adapter["target_modules"]):
                        continue
                    if not isinstance(layer, torch.nn.Linear):
                        raise ValueError(f"Unsupported LoRA target: {prefix}{name}")
                    stem = prefix + name
                    a_key, b_key = stem + ".lora_A", stem + ".lora_B"
                    a, b = state[a_key], state[b_key]
                    if a.shape != (rank, layer.in_features) or b.shape != (layer.out_features, rank):
                        raise ValueError(f"Invalid LoRA dimensions: {stem}")
                    merged[stem + ".weight"] = merged[stem + ".weight"] + (b @ a) * scale
                    consumed.update((a_key, b_key))
            if consumed != set(state):
                raise ValueError(f"Unconsumed LoRA keys: {sorted(set(state) - consumed)}")
            logging.info("Merged frozen small LoRA: rank=%s alpha=%s; consumed %s tensors",
                         rank, adapter["alpha"], len(consumed))
            state = merged
        # Strict loading checks every parameter/buffer name and dimension on
        # each branch. Do not apply the complete single-model state to tiny.
        for prefix, module in (("encoder.", self.encoder.small),
                               ("decoder.", self.decoder.small)):
            branch = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            expected = module.state_dict()
            if set(branch) != set(expected) or any(branch[k].shape != expected[k].shape for k in expected):
                raise ValueError(f"Incompatible small FFT checkpoint branch: {prefix}")
        self.encoder.small.load_state_dict(
            {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}, strict=True)
        self.decoder.small.load_state_dict(
            {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}, strict=True)
        self.encoder.small.requires_grad_(False).eval()
        self.decoder.small.requires_grad_(False).eval()
        self.small_model_file = str(path)
        logging.info("Frozen small FFT checkpoint loaded: %s; tiny unchanged", path)

    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        self._auxiliary_stats = {}
        loss, stats, weight = super().forward(
            speech, speech_lengths, text, text_lengths, **kwargs)
        stats.update(force_gatherable(self._auxiliary_stats, loss.device))
        self._auxiliary_stats = {}
        if self.validation_decode_mode == "teacher_forcing":
            # Keep ESPnet's loss, sample weighting and gatherable stats unchanged;
            # never expose teacher-forcing error rates under autoregressive names.
            stats["cer_tf"] = stats.pop("cer")
            stats["wer_tf"] = stats.pop("wer")
        return loss, stats, weight

    def _calc_att_loss(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens):
        n_prefix = len(self.primer) - 1
        prefix = ys_pad.new_tensor(self.primer[1:]).expand(ys_pad.size(0), -1)
        if not torch.equal(ys_pad[:, :n_prefix], prefix):
            raise ValueError("Labels must start with the configured Whisper language/task/no-timestamps prefix")
        if self.training_protocol == "small_fft":
            # Delegate target construction, EOS supervision, padding, smoothing,
            # normalization, th_accuracy and TF error calculation to the SAME
            # implementation as the reference single-model small FFT.
            return ESPnetASRModel._calc_att_loss(
                self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        ys_in, ys_out = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        # Prefix is supplied by ESPnet's Whisper preprocessor, never prepended twice.
        ys_out[:, :n_prefix] = self.ignore_id
        use_auxiliary = self.training and self.loss_mode == "tiny_ce_plus_fusion_ce"
        if use_auxiliary:
            logits, _, tiny_logits = self.decoder(
                encoder_out, encoder_out_lens, ys_in, ys_pad_lens + 1,
                return_tiny_logits=True)
        else:
            logits, _ = self.decoder(encoder_out, encoder_out_lens, ys_in, ys_pad_lens + 1)
        loss = self.criterion_att(logits.reshape(-1, self.vocab_size), ys_out.reshape(-1))
        denom = (ys_out != self.ignore_id).sum().clamp_min(1) if self.length_normalized_loss else ys_pad.size(0)
        loss = loss / denom
        acc = th_accuracy(logits.reshape(-1, self.vocab_size), ys_out,
                          ignore_label=self.ignore_id)
        if use_auxiliary:
            # Same criterion, targets, prefix/padding mask and denominator as the
            # original fused CE. Reuse the tiny forward; retain its full graph.
            loss_fusion = loss
            loss_tiny = self.criterion_att(
                tiny_logits.reshape(-1, self.vocab_size), ys_out.reshape(-1)) / denom
            loss = loss_tiny + self.lambda_fusion * loss_fusion
            self._auxiliary_stats = {
                "loss_tiny": loss_tiny.detach(),
                "loss_fusion": loss_fusion.detach(),
                "loss_total": loss.detach(),
                "acc_tiny": logits.new_tensor(th_accuracy(
                    tiny_logits.reshape(-1, self.vocab_size), ys_out,
                    ignore_label=self.ignore_id)),
                "acc_fusion": logits.new_tensor(acc),
            }
        cer, wer = None, None
        if not self.training:
            if self.validation_decode_mode == "autoregressive":
                cer, wer = self.autoregressive_errors(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
            elif self.error_calculator is not None:
                # Drop the fixed prompt positions on both sides. The original
                # ESPnet calculator then truncates predictions to reference length
                # at padding, omitting the extra EOS prediction just as upstream.
                ys_hat = logits.argmax(dim=-1)[:, n_prefix:]
                reference = ys_pad[:, n_prefix:]
                reference = reference.masked_fill(reference == self.ignore_id, -1)
                cer, wer = self.error_calculator(ys_hat.cpu(), reference.cpu())
        return loss, acc, cer, wer

    @torch.no_grad()
    def autoregressive_errors(self, memory, lengths, labels, label_lengths):
        import editdistance
        char_errors = char_total = word_errors = word_total = 0
        for i, length in enumerate(lengths.tolist()):
            beam = BeamSearch(beam_size=self.validation_beam_size,
                vocab_size=self.vocab_size, weights={"decoder": 1.0},
                scorers={"decoder": self.decoder}, sos=self.sos, eos=self.eos,
                token_list=self.token_list, pre_beam_score_key="full")
            beam.set_hyp_primer(list(self.primer))
            # Negative ratio is ESPnet's absolute maximum output length.
            max_tokens = min(self.validation_max_tokens,
                self.decoder.tiny.decoders.positional_embedding.size(0) - len(self.primer))
            hypotheses = beam(memory[i, :length], maxlenratio=-max_tokens)
            if not hypotheses:
                raise RuntimeError("Autoregressive validation returned no hypothesis")
            hyp = hypotheses[0].yseq.tolist()[len(self.primer):]
            ref = labels[i, len(self.primer)-1:int(label_lengths[i])].tolist()
            hyp_text = self.text_from_ids(hyp)
            ref_text = self.text_from_ids(ref)
            hchars, rchars = hyp_text.replace(" ", ""), ref_text.replace(" ", "")
            char_errors += editdistance.eval(hchars, rchars)
            char_total += len(rchars)
            word_errors += editdistance.eval(hyp_text.split(), ref_text.split())
            word_total += len(ref_text.split())
        # Batch weighting is handled by ESPnet; final corpus CER comes from stage 13.
        return char_errors / max(char_total, 1), word_errors / max(word_total, 1)

    def text_from_ids(self, ids):
        ids = [i for i in ids if 0 <= i < self.sos - 1]
        return self.whisper_tokenizer.decode(ids, skip_special_tokens=True).strip()

    @staticmethod
    def frozen_key(key):
        return key.startswith(("encoder.small.", "decoder.small."))

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        for key in list(state):
            if self.frozen_key(key[len(prefix):]):
                del state[key]
        return state

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # The constructor has already reloaded the frozen pretrained small.
        # Fill ONLY its omitted keys, preserving strict checks on every tiny key.
        complete = state_dict.copy()
        current = super().state_dict()
        missing_tiny = [k for k in current if not self.frozen_key(k) and k not in complete]
        if missing_tiny:
            raise RuntimeError(f"Missing trainable checkpoint keys: {missing_tiny}")
        for key, value in current.items():
            if self.frozen_key(key):
                complete.setdefault(key, value)
        result = super().load_state_dict(complete, strict=strict, **kwargs)
        self.encoder.small.eval()
        self.decoder.small.eval()
        return result

    def parameter_report(self):
        report = {}
        for name in ("small", "tiny"):
            params = list(getattr(self.encoder, name).parameters()) + list(getattr(self.decoder, name).parameters())
            report[name] = dict(total=sum(p.numel() for p in params),
                trainable=sum(p.numel() for p in params if p.requires_grad))
        logging.info("Dual Whisper parameters: %s", report)
        return report
