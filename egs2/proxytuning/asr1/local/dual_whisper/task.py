"""Recipe-local extension of ASRTask; all data/Trainer machinery stays upstream."""
import copy
from pathlib import Path

import torch
from espnet2.tasks.asr import ASRTask, encoder_choices, decoder_choices, model_choices
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from .model import DualWhisperEncoder, DualWhisperDecoder, DualWhisperASRModel
from .trainer import DualTrainer

encoder_choices.classes["dual_whisper"] = DualWhisperEncoder
decoder_choices.classes["dual_whisper"] = DualWhisperDecoder
model_choices.classes["dual_whisper"] = DualWhisperASRModel


class DualASRTask(ASRTask):
    trainer = DualTrainer

    @classmethod
    def build_model(cls, args):
        if getattr(args, "model", None) != "dual_whisper":
            return super().build_model(args)
        if args.init is not None:
            raise ValueError("Use init: null to preserve pretrained Whisper weights")
        if args.frontend is not None or args.normalize is not None or args.specaug is not None or args.preencoder is not None or args.postencoder is not None:
            raise ValueError("Minimal dual experiment uses raw speech without auxiliary modules")
        if getattr(args, "use_adapter", False) or getattr(args, "freeze_param", []):
            raise ValueError("No adapters or extra parameter freezing in the minimal experiment")
        if args.token_type != "whisper_multilingual" or args.bpemodel != "whisper_multilingual":
            raise ValueError("Both models require whisper_multilingual tokenization")
        conf = args.model_conf
        names = dict(small_model_name=conf.get("small_model_name", "small"),
                     tiny_model_name=conf.get("tiny_model_name", "tiny"))
        if names != dict(small_model_name="small", tiny_model_name="tiny"):
            raise ValueError("This experiment requires multilingual small + tiny pretrained models")
        language = args.preprocessor_conf.get("whisper_language", "zh")
        task = args.preprocessor_conf.get("whisper_task", "transcribe")
        if language != "zh" or task != "transcribe":
            raise ValueError("This experiment requires zh/transcribe")
        converter = OpenAIWhisperTokenIDConverter("whisper_multilingual", language=language, task=task)
        tokenizer = converter.tokenizer
        token_list = args.token_list
        if isinstance(token_list, (str, Path)):
            token_list = Path(token_list).read_text(encoding="utf-8").splitlines()
        expected = tokenizer.tokenizer.convert_ids_to_tokens(list(range(converter.get_num_vocabulary_size())))
        # Match whisper_export_vocabulary: byte-level space marker is exported as a space.
        expected = [t.replace("Ġ", " ") for t in expected]
        if [t.rstrip() for t in token_list] != [t.rstrip() for t in expected]:
            raise ValueError("ESPnet token list does not match pretrained Whisper token IDs")
        if len(expected) != 51865:
            raise ValueError("Unexpected multilingual small/tiny vocabulary size")
        local_args = copy.deepcopy(args)
        local_args.encoder = local_args.decoder = "dual_whisper"
        local_args.encoder_conf = dict(names, **args.encoder_conf)
        local_args.decoder_conf = dict(names, fusion_alpha=conf.get("fusion_alpha", 0.5), **args.decoder_conf)
        # Standard ASRTask constructs these wrappers and the selected ASRModel.
        model = super().build_model(local_args)
        args.token_list = local_args.token_list  # Save portable vocabulary in config.yaml.
        model.primer = tuple(tokenizer.sot_sequence_including_notimestamps)
        model.whisper_tokenizer = tokenizer.tokenizer
        if model.sos != tokenizer.sot or model.eos != tokenizer.eot:
            raise ValueError("SOS/EOS IDs differ from pretrained Whisper")
        if any(model.frozen_key(n) == p.requires_grad for n, p in model.named_parameters()):
            raise AssertionError("Unexpected gradient flags")
        model.parameter_report()
        return model

    @classmethod
    def build_optimizers(cls, args, model):
        if not isinstance(model, DualWhisperASRModel):
            return super().build_optimizers(args, model)
        # Reuse ESPnet's optimizer selection and weight-decay handling, exposing
        # only the existing trainable modules (no copies and no extra parameters).
        trainable = torch.nn.ModuleDict({"encoder": model.encoder.tiny,
                                        "decoder": model.decoder.tiny})
        optimizers = super().build_optimizers(args, trainable)
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        actual = {id(p) for o in optimizers for g in o.param_groups for p in g["params"]}
        if actual != expected:
            raise AssertionError("Optimizer must contain exactly the trainable tiny parameters")
        return optimizers
