"""Beam search module for Proxy Tuning."""

import logging
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Tuple, Union, Optional

import torch

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorer_interface import PartialScorerInterface, ScorerInterface

logger = logging.getLogger(__name__)


import logging
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from espnet.nets.beam_search import end_detect
from espnet.nets.scorer_interface import PartialScorerInterface, ScorerInterface

logger = logging.getLogger(__name__)


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: torch.Tensor
    score: Union[float, torch.Tensor] = 0
    scores: Dict[str, Union[float, torch.Tensor]] = dict()
    states: Dict[str, Any] = dict()
    hs: List[torch.Tensor] = []

    def asdict(self) -> dict:
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()


class BeamSearch(torch.nn.Module):
    """Beam search implementation with Logits Proxy Tuning support."""

    def __init__(
        self,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        proxy_scorers: Dict[str, ScorerInterface] = None,
        token_list: List[str] = None,
        pre_beam_ratio: float = 1.5,
        pre_beam_score_key: str = None,
        return_hs: bool = False,
        hyp_primer: List[int] = None,
        normalize_length: bool = False,
        proxy_logits_weight: float = 1.0,  # 默认遵循自然基准 alpha = 1.0
    ):
        super().__init__()
        self.weights = weights
        self.scorers = dict()
        self.proxy_scorers = proxy_scorers if proxy_scorers is not None else dict()
        self.full_scorers = dict()
        self.part_scorers = dict()
        self.nn_dict = torch.nn.ModuleDict()

        # 注册主打分器
        for k, v in scorers.items():
            w = weights.get(k, 0)
            if w == 0 or v is None:
                continue
            assert isinstance(
                v, ScorerInterface
            ), f"{k} ({type(v)}) does not implement ScorerInterface"
            self.scorers[k] = v
            if isinstance(v, PartialScorerInterface):
                self.part_scorers[k] = v
            else:
                self.full_scorers[k] = v
            if isinstance(v, torch.nn.Module):
                self.nn_dict[k] = v

        # 注册 Proxy-Tuning 辅助打分器（托管权重，便于统一 .to(device) 和 .eval()）
        for k, v in self.proxy_scorers.items():
            if isinstance(v, torch.nn.Module):
                self.nn_dict[f"proxy_{k}"] = v

        self.sos = sos
        self.eos = eos
        self.hyp_primer = hyp_primer
        self.token_list = token_list
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.beam_size = beam_size
        self.n_vocab = vocab_size

        if (
            pre_beam_score_key is not None
            and pre_beam_score_key != "full"
            and pre_beam_score_key not in self.full_scorers
        ):
            raise KeyError(f"{pre_beam_score_key} is not found in {self.full_scorers}")
        self.pre_beam_score_key = pre_beam_score_key
        self.do_pre_beam = (
            self.pre_beam_score_key is not None
            and self.pre_beam_size < self.n_vocab
            and len(self.part_scorers) > 0
        )
        self.return_hs = return_hs
        self.normalize_length = normalize_length
        self.proxy_logits_weight = proxy_logits_weight

    def set_hyp_primer(self, hyp_primer: List[int] = None) -> None:
        self.hyp_primer = hyp_primer

    def init_hyp(
        self, 
        x: torch.Tensor, 
        proxy_x: Optional[torch.Tensor] = None
    ) -> List[Hypothesis]:
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.init_state(x)
            init_scores[k] = 0.0

        for k, d in self.proxy_scorers.items():
            # 小模型如果提供专用特征则送入 proxy_x，否则回退到 x
            cur_x = proxy_x if proxy_x is not None else x
            init_states[f"proxy_{k}"] = d.init_state(cur_x)

        primer = [self.sos] if self.hyp_primer is None else self.hyp_primer

        return [
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                hs=[],
                yseq=torch.tensor(primer, device=x.device),
            )
        ]

    @staticmethod
    def append_token(xs: torch.Tensor, x: int) -> torch.Tensor:
        x = torch.tensor([x], dtype=xs.dtype, device=xs.device)
        return torch.cat((xs, x))

    def score_full(
        self, 
        hyp: Hypothesis, 
        x: torch.Tensor, 
        proxy_x: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        scores = dict()
        states = dict()

        # 1. 计算主模型的各个分值
        for k, d in self.full_scorers.items():
            if "decoder" in k and self.return_hs:
                scores[k], hs, states[k] = d.score(
                    hyp.yseq, hyp.states[k], x, return_hs=self.return_hs
                )
            else:
                # 剔除已作废的 pre_x，恢复标准的 3 参数调用
                scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x)

        # 2. 🌟 Proxy Tuning 核心差分引导
        if "expert_decoder" in self.proxy_scorers and "anti_decoder" in self.proxy_scorers:
            expert_d = self.proxy_scorers["expert_decoder"]
            anti_d = self.proxy_scorers["anti_decoder"]

            expert_state = hyp.states.get("proxy_expert_decoder", None)
            anti_state = hyp.states.get("proxy_anti_decoder", None)

            cur_px = proxy_x if proxy_x is not None else x

            logits_exp, states["proxy_expert_decoder"] = expert_d.score(hyp.yseq, expert_state, cur_px)
            logits_anti, states["proxy_anti_decoder"] = anti_d.score(hyp.yseq, anti_state, cur_px)

            # 标准公式：final_logits = base + alpha * (tuned_small - base_small)
            base_logits = scores["decoder"]
            final_logits = base_logits + self.proxy_logits_weight * (logits_exp - logits_anti)
            
            # 转回对数概率供 Beam 评分
            scores["decoder"] = torch.log_softmax(final_logits, dim=-1)

        elif "decoder" in scores and getattr(self.scorers["decoder"], "use_proxy_tuning", False):
            scores["decoder"] = torch.log_softmax(scores["decoder"], dim=-1)

        if self.return_hs:
            return hs, scores, states
        return scores, states

    def score_partial(
        self, hyp: Hypothesis, ids: torch.Tensor, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.score_partial(hyp.yseq, ids, hyp.states[k], x)
        return scores, states

    def beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if weighted_scores.size(0) == ids.size(0):
            top_ids = weighted_scores.topk(self.beam_size)[1]
            return top_ids, top_ids

        tmp = weighted_scores[ids]
        weighted_scores[:] = -float("inf")
        weighted_scores[ids] = tmp
        top_ids = weighted_scores.topk(self.beam_size)[1]
        local_ids = weighted_scores[ids].topk(self.beam_size)[1]
        return top_ids, local_ids

    @staticmethod
    def merge_scores(
        prev_scores: Dict[str, float],
        next_full_scores: Dict[str, torch.Tensor],
        full_idx: int,
        next_part_scores: Dict[str, torch.Tensor],
        part_idx: int,
    ) -> Dict[str, torch.Tensor]:
        new_scores = dict()
        for k, v in next_full_scores.items():
            new_scores[k] = prev_scores.get(k, 0.0) + v[full_idx]
        for k, v in next_part_scores.items():
            new_scores[k] = prev_scores.get(k, 0.0) + v[part_idx]
        return new_scores

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, d in self.part_scorers.items():
            new_states[k] = d.select_state(part_states[k], part_idx)
        return new_states

    def search(
        self,
        running_hyps: List[Hypothesis],
        x: torch.Tensor,
        proxy_x: Optional[torch.Tensor] = None,
    ) -> List[Hypothesis]:
        best_hyps = []
        part_ids = torch.arange(self.n_vocab, device=x.device)
        for hyp in running_hyps:
            weighted_scores = torch.zeros(self.n_vocab, dtype=x.dtype, device=x.device)
            if self.return_hs:
                hs, scores, states = self.score_full(hyp, x, proxy_x=proxy_x)
            else:
                scores, states = self.score_full(hyp, x, proxy_x=proxy_x)

            for k in self.full_scorers:
                weighted_scores += self.weights[k] * scores[k]

            if self.do_pre_beam:
                pre_beam_scores = (
                    weighted_scores
                    if self.pre_beam_score_key == "full"
                    else scores[self.pre_beam_score_key]
                )
                part_ids = torch.topk(pre_beam_scores, self.pre_beam_size)[1]

            part_scores, part_states = self.score_partial(hyp, part_ids, x)
            for k in self.part_scorers:
                weighted_scores[part_ids] += self.weights[k] * part_scores[k]
            weighted_scores += hyp.score

            for j, part_j in zip(*self.beam(weighted_scores, part_ids)):
                new_hs = hyp.hs + [hs.squeeze(0)] if self.return_hs else []
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(
                            hyp.scores, scores, j, part_scores, part_j
                        ),
                        states=self.merge_states(states, part_states, part_j),
                        hs=new_hs,
                    )
                )

            best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
                : min(len(best_hyps), self.beam_size)
            ]
        return best_hyps

    def forward(
        self,
        x: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        p_x: Optional[torch.Tensor] = None,        # 兼容上层 Speech2Text 传参
        proxy_x: Optional[torch.Tensor] = None,    # 兼容显式传参
        **kwargs,
    ) -> List[Hypothesis]:
        # 兼容 p_x 与 proxy_x 传参别名
        px = p_x if p_x is not None else proxy_x

        if maxlenratio == 0:
            maxlen = x.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))

        if minlenratio < 0:
            minlen = -1 * int(minlenratio)
        else:
            minlen = int(minlenratio * x.size(0))

        logger.info("decoder input length: " + str(x.shape[0]))
        logger.info("max output length: " + str(maxlen))
        logger.info("min output length: " + str(minlen))

        running_hyps = self.init_hyp(x, proxy_x=px)
        ended_hyps = []

        for i in range(maxlen):
            logger.debug("position " + str(i))
            best = self.search(running_hyps, x, proxy_x=px)
            running_hyps = self.post_process(
                i, maxlen, minlen, maxlenratio, best, ended_hyps
            )
            if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                logger.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logger.info("no hypothesis. Finish decoding.")
                break

        if self.normalize_length:
            nbest_hyps = sorted(
                ended_hyps, key=lambda x: x.score / (len(x.yseq) - 1), reverse=True
            )
        else:
            nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)

        if len(nbest_hyps) == 0:
            logger.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(
                    x, maxlenratio, max(0.0, minlenratio - 0.1), p_x=px
                )
            )

        best = nbest_hyps[0]
        logger.info(f"total log probability: {best.score:.2f}")
        logger.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        return nbest_hyps

    def post_process(
        self,
        i: int,
        maxlen: int,
        minlen: int,
        maxlenratio: float,
        running_hyps: List[Hypothesis],
        ended_hyps: List[Hypothesis],
    ) -> List[Hypothesis]:
        if i == maxlen - 1:
            running_hyps = [
                h._replace(yseq=self.append_token(h.yseq, self.eos))
                for h in running_hyps
            ]

        remained_hyps = []
        for hyp in running_hyps:
            if hyp.yseq[-1] == self.eos:
                for k, d in chain(self.full_scorers.items(), self.part_scorers.items()):
                    s = d.final_score(hyp.states[k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
                if i >= minlen:
                    ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)
        return remained_hyps