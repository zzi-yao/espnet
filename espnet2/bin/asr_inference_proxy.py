#!/usr/bin/env python3
import argparse
import copy
import logging
import sys
from distutils.version import LooseVersion
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.quantization
from typeguard import typechecked

from espnet2.asr.decoder.hugging_face_transformers_decoder import (
    get_hugging_face_model_lm_head,
    get_hugging_face_model_network,
)
from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.asr.partially_AR_model import PartiallyARInference
from espnet2.asr.transducer.beam_search_transducer import (
    BeamSearchTransducer,
)
from espnet2.asr.transducer.beam_search_transducer import (
    ExtendedHypothesis as ExtTransHypothesis,
)
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.hugging_face_token_id_converter import HuggingFaceTokenIDConverter
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.batch_beam_search_proxy_tuning import BatchBeamSearch
from espnet.nets.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet.nets.beam_search_proxy_tuning import BeamSearch, Hypothesis
from espnet.nets.beam_search_timesync import BeamSearchTimeSync
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args

try:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    from transformers.file_utils import ModelOutput

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


ListOfHypothesis = List[
    Tuple[
        Optional[str],
        List[str],
        List[int],
        Union[Hypothesis, ExtTransHypothesis, TransHypothesis],
    ]
]

logger = logging.getLogger(__name__)


class Speech2Text:
    """Speech2Text class for Whisper Proxy Tuning."""

    @typechecked
    def __init__(
        self,
        asr_train_config: Union[Path, str, None] = None,
        asr_model_file: Union[Path, str, None] = None,
        transducer_conf: Optional[Dict] = None,
        lm_train_config: Union[Path, str, None] = None,
        lm_file: Union[Path, str, None] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str, None] = None,
        token_type: Optional[str] = None,
        bpemodel: Optional[str] = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 20,
        ctc_weight: float = 0.0,
        proxy_weight: float = 0.5,  # 🌟 显式引入 Proxy Tuning 的控制因子 alpha
        lm_weight: float = 1.0,
        ngram_weight: float = 0.9,
        penalty: float = 0.0,
        nbest: int = 1,
        normalize_length: bool = False,
        streaming: bool = False,
        enh_s2t_task: bool = False,
        quantize_asr_model: bool = False,
        quantize_lm: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        hugging_face_decoder: bool = False,
        hugging_face_decoder_conf: Dict[str, Any] = {},
        time_sync: bool = False,
        multi_asr: bool = False,
        lid_prompt: bool = False,
        lang_prompt_token: Optional[str] = None,
        nlp_prompt_token: Optional[str] = None,
        prompt_token_file: Optional[str] = None,
        partial_ar: bool = False,
        threshold_probability: float = 0.99,
        max_seq_len: int = 5,
        max_mask_parallel: int = -1,
    ):
        task = ASRTask if not enh_s2t_task else EnhS2TTask

        if quantize_asr_model or quantize_lm:
            if quantize_dtype == "float16" and torch.__version__ < LooseVersion("1.5.0"):
                raise ValueError(
                    "float16 dtype for dynamic quantization is not supported with "
                    "torch version < 1.5.0. Switch to qint8 dtype instead."
                )

        qconfig_spec = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype: torch.dtype = getattr(torch, quantize_dtype)

        # 1. 构建并加载 ASR 主模型
        scorers = {}
        proxy_scorers = {}
        asr_model, asr_train_args = task.build_model_from_file(
            asr_train_config, asr_model_file, device
        )

        asr_model.to(dtype=getattr(torch, dtype)).eval()

        if quantize_asr_model:
            logger.info("Use quantized asr model for decoding.")
            asr_model = torch.quantization.quantize_dynamic(
                asr_model, qconfig_spec=qconfig_spec, dtype=quantize_dtype
            )

        token_list = asr_model.token_list

        # 2. 注册主 Scorer (Base Large Decoder)
        scorers.update(
            decoder=asr_model.decoder,
            length_bonus=LengthBonus(len(token_list)),
        )
        if ctc_weight > 0.0 and hasattr(asr_model, "ctc"):
            scorers["ctc"] = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)

        # 🌟 3. 注册 Proxy Tuning 代理模型（对齐 expert 与 anti 键名）
        # 注册已微调的 Expert 小模型
        if hasattr(asr_model, "expert_decoder") and asr_model.expert_decoder is not None:
            proxy_scorers["expert_decoder"] = asr_model.expert_decoder
        elif hasattr(asr_model, "proxy_decoder") and asr_model.proxy_decoder is not None:
            proxy_scorers["expert_decoder"] = asr_model.proxy_decoder

        # 🌟 核心修复：正确传入 vocab_size 构建官方未经微调的 Whisper-tiny Decoder
        if hasattr(asr_model, "anti_decoder") and asr_model.anti_decoder is not None:
            proxy_scorers["anti_decoder"] = asr_model.anti_decoder
        else:
            logger.info("正在从 OpenAI 官方权重恢复纯净 Whisper-tiny Decoder 作为 anti_decoder...")
            try:
                import copy
                import whisper
                from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
                
                exp_dec = proxy_scorers.get("expert_decoder", None)
                whisper_model_name = getattr(exp_dec, "whisper_model", "tiny") if exp_dec is not None else "tiny"
                enc_out_size = getattr(exp_dec, "encoder_output_size", 384 if "tiny" in whisper_model_name else 768)

                # 1. 实例化结构
                anti_decoder = OpenAIWhisperDecoder(
                    vocab_size=len(token_list),
                    encoder_output_size=enc_out_size,
                    whisper_model=whisper_model_name,
                    dropout_rate=0.0,
                )
                
                # 🌟 2. 必须载入官方原生权重，绝不能裸跑随机数！
                official_whisper = whisper.load_model(whisper_model_name, device="cpu")
                # 将官方 whisper.decoder 的权重加载给 anti_decoder.decoder
                anti_decoder.decoders.load_state_dict(official_whisper.decoder.state_dict())
                del official_whisper  # 释放内存

                anti_decoder.to(device=device, dtype=getattr(torch, dtype)).eval()
                proxy_scorers["anti_decoder"] = anti_decoder
                logger.info(f"成功将官方预训练权重载入 {whisper_model_name} anti_decoder！")
            except Exception as e:
                logger.error(f"构建并载入官方 anti_decoder 权重失败: {e}", exc_info=True)

        # 4. 语言模型与 N-gram (保持原样)
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            if quantize_lm:
                logger.info("Use quantized lm for decoding.")
                lm = torch.quantization.quantize_dynamic(
                    lm, qconfig_spec=qconfig_spec, dtype=quantize_dtype
                )
            scorers["lm"] = lm.lm

        if ngram_file is not None:
            if ngram_scorer == "full":
                from espnet.nets.scorers.ngram import NgramFullScorer
                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                from espnet.nets.scorers.ngram import NgramPartScorer
                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None
        scorers["ngram"] = ngram

        # 5. 构建加权字典与搜索实例
        weights = dict(
            decoder=1.0 - ctc_weight,
            ctc=ctc_weight,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
        )

        beam_search = BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            proxy_scorers=proxy_scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "full",
            normalize_length=normalize_length,
            proxy_logits_weight=proxy_weight,  # 🌟 将 alpha 传入底层
        )

        # 🌟 核心修复：动态切换时必须指向 BatchBeamSearchProxyTuning
        if batch_size == 1:
            non_batch = [
                k for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                # 切成 ProxyTuning 专用批处理类，防止类覆盖失效
                beam_search.__class__ = BatchBeamSearch
                logger.info("BatchBeamSearchProxyTuning implementation is selected.")
            else:
                logger.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )

        # 🌟 6. 核心设备同步修复：同时同步 scorers 与 proxy_scorers
        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in list(scorers.values()) + list(proxy_scorers.values()):
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()

        logger.info(f"Beam_search: {beam_search}")
        logger.info(f"Decoding device={device}, dtype={dtype}")

        # 7. 构建文本转换器 (Whisper Tokenizer & Prompt 设置)
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        preprocessor_conf = getattr(asr_train_args, "preprocessor_conf", {})
        whisper_language = preprocessor_conf.get("whisper_language", None)
        whisper_task = preprocessor_conf.get("whisper_task", None)

        if token_type is None:
            tokenizer = None
        elif token_type in ["bpe", "hugging_face"]:
            tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel) if bpemodel else None
        elif "whisper" in token_type:
            tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                whisper_language=whisper_language,
                whisper_task=whisper_task,
                non_linguistic_symbols=prompt_token_file,
            )
        else:
            tokenizer = build_tokenizer(token_type=token_type)

        if bpemodel not in ["whisper_en", "whisper_multilingual"]:
            converter = TokenIDConverter(token_list=token_list)
        else:
            converter = OpenAIWhisperTokenIDConverter(
                model_type=bpemodel,
                added_tokens_txt=prompt_token_file,
                language=whisper_language or "en",
                task=whisper_task or "transcribe",
                sot="speaker_change_symbol" in preprocessor_conf,
            )
            beam_search.set_hyp_primer(
                list(converter.tokenizer.sot_sequence_including_notimestamps)
            )
            if lang_prompt_token is not None:
                a1 = converter.tokenizer.tokenizer.convert_ids_to_tokens(
                    converter.tokenizer.sot_sequence_including_notimestamps
                )
                a1 = a1[:1] + lang_prompt_token.split() + a1[3:]
                beam_search.set_hyp_primer(
                    list(converter.tokenizer.tokenizer.convert_tokens_to_ids(a1))
                )
            elif nlp_prompt_token is not None:
                a1 = converter.tokenizer.tokenizer.convert_ids_to_tokens(
                    converter.tokenizer.sot_sequence_including_notimestamps
                )
                prompt_tokens = tokenizer.text2tokens(nlp_prompt_token)
                a1 = a1[:2] + prompt_tokens + a1[3:]
                beam_search.set_hyp_primer(
                    list(converter.tokenizer.tokenizer.convert_tokens_to_ids(a1))
                )
            elif lid_prompt:
                a1 = converter.tokenizer.tokenizer.convert_ids_to_tokens(
                    converter.tokenizer.sot_sequence_including_notimestamps
                )
                a1 = a1[:1]
                beam_search.set_hyp_primer(
                    list(converter.tokenizer.tokenizer.convert_tokens_to_ids(a1))
                )

        logger.info(f"Text tokenizer: {tokenizer}")

        # 8. 挂载实例属性
        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.beam_search_transducer = None
        self.hugging_face_model = None
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.enh_s2t_task = enh_s2t_task
        self.multi_asr = multi_asr
    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray]):
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        batch = to_device(batch, device=self.device)

        # 编码声学特征：主模型与 Proxy 模型分别输出编码特征
        enc_out = self.asr_model.encode(**batch)
        if len(enc_out) == 4:
            enc, enc_olens, proxy_enc, proxy_enc_olens = enc_out
        else:
            enc, enc_olens = enc_out
            proxy_enc = None

        # 单句解码
        results = self._decode_single_sample(
            enc=enc[0], 
            proxy_enc=proxy_enc[0] if proxy_enc is not None else None
        )
        return results

    def _decode_single_sample(
        self, enc: torch.Tensor, proxy_enc: Optional[torch.Tensor] = None
    ) -> ListOfHypothesis:
        # 清理掉多余无用的 HuggingFace 和 S4 分支，直接调用 BeamSearch
        beam_kwargs = dict(
            x=enc,
            maxlenratio=self.maxlenratio,
            minlenratio=self.minlenratio,
        )
        # 如果存在 proxy 编码表征，透传给 Proxy-Tuning BeamSearch
        if proxy_enc is not None:
            beam_kwargs["p_x"] = proxy_enc

        nbest_hyps = self.beam_search(**beam_kwargs)
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            # 后处理：剥除起始/结束标记并还原文字
            token_int = hyp.yseq[1:-1] if isinstance(hyp.yseq, list) else hyp.yseq[1:-1].tolist()
            token_int = list(filter(lambda x: x != 0, token_int))
            token = self.converter.ids2tokens(token_int)
            text = self.tokenizer.tokens2text(token) if self.tokenizer is not None else None
            results.append((text, token, token_int, hyp))

        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Speech2Text instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: Speech2Text instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2Text(**kwargs)


@typechecked
def inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    ngram_weight: float,
    penalty: float,
    nbest: int,
    normalize_length: bool,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    transducer_conf: Optional[dict],
    streaming: bool,
    enh_s2t_task: bool,
    quantize_asr_model: bool,
    quantize_lm: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
    hugging_face_decoder: bool,
    hugging_face_decoder_conf: Dict[str, Any],
    time_sync: bool,
    multi_asr: bool,
    lang_prompt_token: Optional[str],
    nlp_prompt_token: Optional[str],
    prompt_token_file: Optional[str],
    partial_ar: bool,
    threshold_probability: float,
    max_seq_len: int,
    max_mask_parallel: int,
    proxy_weight: float = 0.5,  
):
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        transducer_conf=transducer_conf,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        normalize_length=normalize_length,
        streaming=streaming,
        enh_s2t_task=enh_s2t_task,
        multi_asr=multi_asr,
        quantize_asr_model=quantize_asr_model,
        quantize_lm=quantize_lm,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        hugging_face_decoder=hugging_face_decoder,
        hugging_face_decoder_conf=hugging_face_decoder_conf,
        time_sync=time_sync,
        prompt_token_file=prompt_token_file,
        lang_prompt_token=lang_prompt_token,
        nlp_prompt_token=nlp_prompt_token,
        partial_ar=partial_ar,
        threshold_probability=threshold_probability,
        max_seq_len=max_seq_len,
        max_mask_parallel=max_mask_parallel,
        proxy_weight=proxy_weight,  # 🌟 显式打入 kwargs
    )
    speech2text = Speech2Text.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["<space>"], [2], hyp]] * nbest
                if enh_s2t_task:
                    num_spk = getattr(speech2text.asr_model.enh_model, "num_spk", 1)
                    results = [results for _ in range(num_spk)]

            # Only supporting batch_size==1
            key = keys[0]
            if enh_s2t_task or multi_asr:
                # Enh+ASR joint task
                for spk, ret in enumerate(results, 1):
                    for n, (text, token, token_int, hyp) in zip(
                        range(1, nbest + 1), ret
                    ):
                        # Create a directory: outdir/{n}best_recog_spk?
                        ibest_writer = writer[f"{n}best_recog"]

                        # Write the result to each file
                        ibest_writer[f"token_spk{spk}"][key] = " ".join(token)
                        ibest_writer[f"token_int_spk{spk}"][key] = " ".join(
                            map(str, token_int)
                        )
                        ibest_writer[f"score_spk{spk}"][key] = str(hyp.score)

                        if text is not None:
                            ibest_writer[f"text_spk{spk}"][key] = text

            else:
                # Normal ASR
                encoder_interctc_res = None
                if isinstance(results, tuple):
                    results, encoder_interctc_res = results

                for n, (text, token, token_int, hyp) in zip(
                    range(1, nbest + 1), results
                ):
                    # Create a directory: outdir/{n}best_recog
                    ibest_writer = writer[f"{n}best_recog"]

                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)

                    if text is not None:
                        ibest_writer["text"][key] = text

                # Write intermediate predictions to
                # encoder_interctc_layer<layer_idx>.txt
                ibest_writer = writer["1best_recog"]
                if encoder_interctc_res is not None:
                    for idx, text in encoder_interctc_res.items():
                        ibest_writer[f"encoder_interctc_layer{idx}.txt"][key] = (
                            " ".join(text)
                        )


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--asr_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--asr_model_file",
        type=str,
        help="ASR model parameter file",
    )
    group.add_argument(
        "--lm_train_config",
        type=str,
        help="LM training configuration",
    )
    group.add_argument(
        "--lm_file",
        type=str,
        help="LM parameter file",
    )
    group.add_argument(
        "--word_lm_train_config",
        type=str,
        help="Word LM training configuration",
    )
    group.add_argument(
        "--word_lm_file",
        type=str,
        help="Word LM parameter file",
    )
    group.add_argument(
        "--ngram_file",
        type=str,
        help="N-gram parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )
    group.add_argument(
        "--enh_s2t_task",
        type=str2bool,
        default=False,
        help="Whether we are using an enhancement and ASR joint model",
    )
    group.add_argument(
        "--multi_asr",
        type=str2bool,
        default=False,
        help="Whether we are using a monolithic multi-speaker ASR model "
        "(This flag should be False if a speech separation model is used before ASR)",
    )
    group = parser.add_argument_group("Quantization related")
    group.add_argument(
        "--quantize_asr_model",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to ASR model.",
    )
    group.add_argument(
        "--quantize_lm",
        type=str2bool,
        default=False,
        help="Apply dynamic quantization to LM.",
    )
    group.add_argument(
        "--quantize_modules",
        type=str,
        nargs="*",
        default=["Linear"],
        help="""List of modules to be dynamically quantized.
        E.g.: --quantize_modules=[Linear,LSTM,GRU].
        Each specified module should be an attribute of 'torch.nn', e.g.:
        torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU, ...""",
    )
    group.add_argument(
        "--quantize_dtype",
        type=str,
        default="qint8",
        choices=["float16", "qint8"],
        help="Dtype for dynamic quantization.",
    )

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
    group.add_argument(
        "--proxy_weight",
        type=float,
        default=1.0,
        help="Weight factor (alpha) for Proxy-Tuning logits steering.",
    )
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
        "If maxlenratio=0.0 (default), it uses a end-detect "
        "function "
        "to automatically find maximum hypothesis lengths."
        "If maxlenratio<0.0, its absolute value is interpreted"
        "as a constant max output length",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.5,
        help="CTC weight in joint decoding",
    )
    group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
    group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
    group.add_argument("--streaming", type=str2bool, default=False)
    group.add_argument("--hugging_face_decoder", type=str2bool, default=False)
    group.add_argument(
        "--hugging_face_decoder_conf",
        type=NestedDictAction,
        default=dict(),
        help="Custom kwargs for the HF .generate()",
    )

    group.add_argument(
        "--transducer_conf",
        default=None,
        help="The keyword arguments for transducer beam search.",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--time_sync",
        type=str2bool,
        default=False,
        help="Time synchronous beam search.",
    )
    group.add_argument(
        "--lang_prompt_token",
        type=str,
        default=None,
        help="Prompt token for mulitlingual prompting",
    )
    group.add_argument(
        "--nlp_prompt_token",
        type=str,
        default=None,
        help="Prompt token for natural language phrases as prompting",
    )
    group.add_argument(
        "--prompt_token_file",
        type=str,
        default=None,
        help="Prompt token file",
    )
    group.add_argument(
        "--normalize_length",
        type=str2bool,
        default=False,
        help="If true, best hypothesis is selected by length-normalized scores",
    )

    group = parser.add_argument_group("Partially AR related")
    group.add_argument(
        "--partial_ar",
        type=str2bool,
        default=False,
        help="Flag to use the partially AR decoding",
    )
    group.add_argument(
        "--threshold_probability",
        type=float,
        default=0.99,
        help="Threshold for probability of the token to be masked",
    )
    group.add_argument(
        "--max_seq_len",
        type=int,
        default=5,
        help="Maximum sequence length for each hypothesis."
        + "Will stop beam_search after max_seq_len iteration in partially AR decoding.",
    )
    group.add_argument(
        "--max_mask_parallel",
        type=int,
        default=-1,
        help="Maximum number of masks to predict in parallel."
        + "If you got OOM error, try to decrease this value."
        + "Default to -1, which means always predict all masks simultaneously.",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()