import copy
from typing import Any, List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.scorer_interface import BatchScorerInterface
import torch.nn as nn


class ExpandedTokenEmbedding(torch.nn.Module):
    def __init__(self, ori_emebedding, additional_size):
        super().__init__()
        self.ori_emb = ori_emebedding

        orig_emb_std, orig_emb_mean = torch.std_mean(ori_emebedding.weight)
        self.add_emb = torch.nn.Embedding(additional_size, ori_emebedding.embedding_dim)
        torch.nn.init.normal_(
            self.add_emb.weight,
            orig_emb_mean.item(),
            orig_emb_std.item(),
        )
        self.num_embeddings = ori_emebedding.num_embeddings + additional_size

    @property
    def weight(self):
        return torch.cat([self.ori_emb.weight, self.add_emb.weight], dim=0)

    def forward(self, input):
        return torch.nn.functional.embedding(
            input,
            self.weight,
            self.ori_emb.padding_idx,
            self.ori_emb.max_norm,
            self.ori_emb.norm_type,
            self.ori_emb.scale_grad_by_freq,
            self.ori_emb.sparse,
        )


class OpenAIWhisperDecoder(AbsDecoder, BatchScorerInterface):
    """Transformer-based Speech-to-Text Decoder from OpenAI's Whisper Model for Proxy Tuning."""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: Optional[str] = None,
        load_origin_token_embedding=False,
        use_proxy_tuning=False,  # 核心开关：开启后返回原始 Logits 供外部做减法
    ):
        try:
            import whisper
        except Exception as e:
            print("Error: whisper is not properly installed.")
            raise e

        super().__init__()

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(
            whisper_model, download_root=download_dir, device="cpu"
        )
        self.decoders = copy.deepcopy(_model.decoder)
        attention_dim = self.decoders.token_embedding.embedding_dim

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.load_origin_token_embedding = load_origin_token_embedding

        if vocab_size != self.decoders.token_embedding.num_embeddings:
            if self.load_origin_token_embedding:
                assert (
                    vocab_size > self.decoders.token_embedding.num_embeddings
                ), "expanded vocab_size should be larged than the origin"
                self.decoders.token_embedding = ExpandedTokenEmbedding(
                    self.decoders.token_embedding,
                    vocab_size - self.decoders.token_embedding.num_embeddings,
                )
            else:
                orig_emb_std, orig_emb_mean = torch.std_mean(
                    self.decoders.token_embedding.weight
                )
                self.decoders.token_embedding = torch.nn.Embedding(
                    vocab_size, attention_dim
                )
                torch.nn.init.normal_(
                    self.decoders.token_embedding.weight,
                    orig_emb_mean.item(),
                    orig_emb_std.item(),
                )

        self.decoders.train()
        del _model
        
        self.use_proxy_tuning = use_proxy_tuning

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder for training loss calculation."""
        tgt, memory = ys_in_pad, hs_pad
        tgt = (
            self.decoders.token_embedding(tgt)
            + self.decoders.positional_embedding[: tgt.size(1)]
        )
        tgt = self.dropout(tgt)
        x = tgt.to(memory.dtype)

        for layer, block in enumerate(self.decoders.blocks):
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        x = (
            x @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return x, ys_in_lens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        *,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step for Beam Search inference."""
        x = (
            self.decoders.token_embedding(tgt)
            + self.decoders.positional_embedding[: tgt.size(1)]
        )
        # x = self.dropout(x)
        x = x.to(memory.dtype)

        for layer, block in enumerate(self.decoders.blocks):
            x = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        y = x[:, -1]
        y = (
            y @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        if getattr(self, "use_proxy_tuning", False):
            return y, None
        else:
            y = torch.log_softmax(y, dim=-1)
            return y, None




    def init_state(self, x: torch.Tensor) -> Any:
        """Initialize decoder state for serial beam search."""
        return None

    def batch_init_state(self, x: torch.Tensor) -> Any:
        """Initialize decoder state for batch beam search."""
        return None

    def select_state(self, state: Any, i: int) -> Any:
        """Select state for hypothesis i."""
        return None

    def score(self, ys, state, x):
        """Score single token."""
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), torch.empty(0), x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score token batch for Beam Search."""
        logp, states = self.forward_one_step(ys, torch.empty(0), xs, cache=None)
        return logp, None