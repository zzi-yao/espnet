"""Parallel beam search module for Proxy Tuning."""

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
from packaging.version import parse as V
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.beam_search_proxy_tuning import BeamSearch, Hypothesis

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

logger = logging.getLogger(__name__)


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    yseq: torch.Tensor = torch.tensor([])  # (batch, maxlen)
    score: torch.Tensor = torch.tensor([])  # (batch,)
    length: torch.Tensor = torch.tensor([])  # (batch,)
    scores: Dict[str, torch.Tensor] = dict()  # values: (batch,)
    states: Dict[str, Dict] = dict()
    hs: List[torch.Tensor] = []  # (batch, maxlen, adim)

    def __len__(self) -> int:
        return len(self.length)


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation for Proxy Tuning."""

    def batchfy(self, hyps: List[Hypothesis]) -> BatchHypothesis:
        if len(hyps) == 0:
            return BatchHypothesis()

        hs = [h.hs for h in hyps] if self.return_hs else []

        all_scorer_keys = list(self.scorers.keys()) + [
            f"proxy_{k}" for k in self.proxy_scorers.keys()
        ]

        return BatchHypothesis(
            yseq=pad_sequence(
                [h.yseq for h in hyps], batch_first=True, padding_value=self.eos
            ),
            length=torch.tensor([len(h.yseq) for h in hyps], dtype=torch.int64),
            score=torch.tensor([h.score for h in hyps]),
            scores={k: torch.tensor([h.scores.get(k, 0.0) for h in hyps]) for k in self.scorers},
            states={k: [h.states.get(k, None) for h in hyps] for k in all_scorer_keys},
            hs=hs,
        )

    def _batch_select(self, hyps: BatchHypothesis, ids: torch.Tensor) -> BatchHypothesis:
        ids_list = ids.tolist()
        hs = [hyps.hs[i] for i in ids_list] if self.return_hs else []

        new_states = dict()
        for k, v in hyps.states.items():
            if k.startswith("proxy_"):
                orig_k = k.replace("proxy_", "")
                scorer = self.proxy_scorers.get(orig_k)
            else:
                scorer = self.scorers.get(k)
            
            if scorer is not None and hasattr(scorer, "select_state"):
                new_states[k] = [scorer.select_state(v, i) for i in ids_list]
            else:
                new_states[k] = [v[i] for i in ids_list]

        # 🌟 修复关键：保证切片索引与被切片张量在同一设备上
        # 1. 对齐 yseq、score、length 的索引设备
        yseq_ids = ids.to(hyps.yseq.device)
        score_ids = ids.to(hyps.score.device)
        length_ids = ids.to(hyps.length.device) if hyps.length is not None else None

        # 2. 对齐 scores 字典中每个分数的索引设备
        new_scores = {}
        for k, v in hyps.scores.items():
            if isinstance(v, torch.Tensor):
                new_scores[k] = v[ids.to(v.device)]
            else:
                new_scores[k] = [v[i] for i in ids_list]

        return BatchHypothesis(
            yseq=hyps.yseq[yseq_ids],
            score=hyps.score[score_ids],
            length=hyps.length[length_ids] if length_ids is not None else None,
            scores=new_scores,
            states=new_states,
            hs=hs,
        )

    def _select(self, hyps: BatchHypothesis, i: int) -> Hypothesis:
        new_states = dict()
        for k, v in hyps.states.items():
            if k.startswith("proxy_"):
                orig_k = k.replace("proxy_", "")
                scorer = self.proxy_scorers.get(orig_k)
            else:
                scorer = self.scorers.get(k)

            if scorer is not None and hasattr(scorer, "select_state"):
                new_states[k] = scorer.select_state(v, i)
            else:
                new_states[k] = v[i] if v is not None else None

        return Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states=new_states,
            hs=hyps.hs[i] if self.return_hs else [],
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        return [self._select(batch_hyps, i) for i in range(len(batch_hyps.length))]

    def batch_beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        top_ids = weighted_scores.view(-1).topk(self.beam_size)[1]
        if is_torch_1_9_plus:
            prev_hyp_ids = torch.div(top_ids, self.n_vocab, rounding_mode="trunc")
        else:
            prev_hyp_ids = top_ids // self.n_vocab
        new_token_ids = top_ids % self.n_vocab
        return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

    def init_hyp(
        self, 
        x: torch.Tensor, 
        proxy_x: Optional[torch.Tensor] = None,
        anti_x: Optional[torch.Tensor] = None,
    ) -> BatchHypothesis:
        init_states = dict()
        init_scores = dict()
        
        # Base 大模型使用 x
        for k, d in self.scorers.items():
            init_states[k] = d.batch_init_state(x)
            init_scores[k] = 0.0

        # Proxy 特征路由：微调小模型用 proxy_x，基底未微调小模型用 anti_x（缺省则自动回退）
        cur_px = proxy_x if proxy_x is not None else x
        cur_ax = anti_x if anti_x is not None else cur_px

        for k, d in self.proxy_scorers.items():
            feat = cur_ax if "anti" in k else cur_px
            init_states[f"proxy_{k}"] = d.batch_init_state(feat)

        # 依次为: <|startoftranscript|>(50258), <|zh|>(50260), <|transcribe|>(50359), <|notimestamps|>(50363)
        if self.hyp_primer is not None:
            primer = self.hyp_primer
        elif hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            lang_ids = self.lang_token_id.view(-1).tolist()
            primer = [self.sos] + lang_ids
        else:
            primer = [50258, 50260, 50359, 50363]

        return self.batchfy(
            [
                Hypothesis(
                    score=0.0,
                    scores=init_scores,
                    states=init_states,
                    hs=[],
                    yseq=torch.tensor(primer, device=x.device, dtype=torch.long),
                )
            ]
        )

    def score_full(
        self,
        hyp: BatchHypothesis,
        x: torch.Tensor,
        proxy_x: Optional[torch.Tensor] = None,
        anti_x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:

        scores = dict()
        states = dict()

        # 1. Base 模型打分
        for k, d in self.full_scorers.items():
            scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)

        # 2. Proxy 模型打分与概率空间引导
        if "expert_decoder" in self.proxy_scorers and "anti_decoder" in self.proxy_scorers:
            expert_d = self.proxy_scorers["expert_decoder"]
            anti_d = self.proxy_scorers["anti_decoder"]

            expert_state = hyp.states.get("proxy_expert_decoder", None)
            anti_state = hyp.states.get("proxy_anti_decoder", None)

            cur_px = proxy_x if proxy_x is not None else x
            cur_ax = anti_x if anti_x is not None else cur_px

            # 对应模型吃各自对应的声学特征
            logits_exp, states["proxy_expert_decoder"] = expert_d.batch_score(hyp.yseq, expert_state, cur_px)
            logits_anti, states["proxy_anti_decoder"] = anti_d.batch_score(hyp.yseq, anti_state, cur_ax)

            base_out = scores["decoder"]

            # 🌟 修复 1：强制无条件转为对数概率空间，彻底消除尺度不一致
            log_p_base = torch.log_softmax(base_out, dim=-1)
            log_p_exp  = torch.log_softmax(logits_exp, dim=-1)
            log_p_anti = torch.log_softmax(logits_anti, dim=-1)

            diff = log_p_exp - log_p_anti
            if diff.size(-1) > 50257:
                diff[:, 50257:] = 0.0

            final_log_p = log_p_base + self.proxy_logits_weight * diff
            scores["decoder"] = torch.log_softmax(final_log_p, dim=-1)

            # 🌟 修复 2：必须执行二次 log_softmax，确保概率严格合法 (<= 0.0)，防止数值爆炸
            scores["decoder"] = torch.log_softmax(final_log_p, dim=-1)

        elif "decoder" in scores and getattr(self.scorers["decoder"], "use_proxy_tuning", False):
            scores["decoder"] = torch.log_softmax(scores["decoder"], dim=-1)

        return scores, states

    def score_partial(
        self,
        hyp: BatchHypothesis,
        ids: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.batch_score_partial(hyp.yseq, ids, hyp.states[k], x)
        return scores, states

    def search(
        self,
        running_hyps: BatchHypothesis,
        x: torch.Tensor,
        proxy_x: Optional[torch.Tensor] = None,
        anti_x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BatchHypothesis:
        n_batch = len(running_hyps)
        part_ids = None
        weighted_scores = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device
        )

        # 沿 batch 维度广播特征
        expanded_x = x.expand(n_batch, *x.shape)
        expanded_px = proxy_x.expand(n_batch, *proxy_x.shape) if proxy_x is not None else None
        expanded_ax = anti_x.expand(n_batch, *anti_x.shape) if anti_x is not None else expanded_px

        # 透传 anti_x 至 score_full
        scores, states = self.score_full(
            running_hyps,
            expanded_x,
            proxy_x=expanded_px,
            anti_x=expanded_ax,
        )

        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]

        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]

        part_scores, part_states = self.score_partial(running_hyps, part_ids, x)
        for k in self.part_scorers:
            weighted_scores[part_ids] += self.weights[k] * part_scores[k]

        weighted_scores += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)
        for (
            full_prev_hyp_id,
            full_new_token_id,
            part_prev_hyp_id,
            part_new_token_id,
        ) in zip(*self.batch_beam(weighted_scores, part_ids)):
            prev_hyp = prev_hyps[full_prev_hyp_id]
            best_hyps.append(
                Hypothesis(
                    score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                    yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                    scores=self.merge_scores(
                        prev_hyp.scores,
                        {k: v[full_prev_hyp_id] for k, v in scores.items()},
                        full_new_token_id,
                        {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
                        part_new_token_id,
                    ),
                    states=self.merge_states(
                        {
                            k: (
                                self.proxy_scorers[k[6:]].select_state(v, full_prev_hyp_id)
                                if k.startswith("proxy_") and k[6:] in self.proxy_scorers and hasattr(self.proxy_scorers[k[6:]], "select_state")
                                else self.full_scorers[k].select_state(v, full_prev_hyp_id)
                                if k in self.full_scorers and hasattr(self.full_scorers[k], "select_state")
                                else v
                            )
                            for k, v in states.items()
                        },
                        {
                            k: self.part_scorers[k].select_state(
                                v, part_prev_hyp_id, part_new_token_id
                            )
                            for k, v in part_states.items()
                        },
                        part_new_token_id,
                    ),
                    hs=[],
                )
            )
        return self.batchfy(best_hyps)

    def forward(
        self, 
        x: torch.Tensor, 
        maxlenratio: float = 0.0, 
        minlenratio: float = 0.0,
        p_x: Optional[torch.Tensor] = None,       # 兼容 Speech2Text 常用入参
        proxy_x: Optional[torch.Tensor] = None,   # 兼容显式命名
        x_expert: Optional[torch.Tensor] = None,  # 兼容微调小模型声学特征
        x_anti: Optional[torch.Tensor] = None,    # 🌟 修复：基底小模型声学特征显式路由
        **kwargs,
    ) -> List[Hypothesis]:
        # 1. 统一声学特征路由映射
        # 微调小模型特征：优先取显式传入的 x_expert，其次取 proxy_x / p_x
        px_expert = x_expert if x_expert is not None else (proxy_x if proxy_x is not None else p_x)
        # 基底未微调小模型特征：若未单独指定 x_anti，则复用 px_expert
        px_anti = x_anti if x_anti is not None else px_expert

        max_limit = getattr(self, "maxlen", 448)

        # 确定 Prompt 前缀长度（Whisper 中文转录通常为 4）
        n_prefix = len(self.hyp_primer) if self.hyp_primer is not None else 4

        if maxlenratio == 0:
            maxlen = min(x.shape[0], max_limit)
        elif maxlenratio < 0:
            maxlen = min(-1 * int(maxlenratio), max_limit)
        else:
            maxlen = min(max(1, int(maxlenratio * x.size(0))), max_limit)

        # 🌟 长度补偿：自回归步数必须计入预先填入的前缀长度
        maxlen = min(maxlen + n_prefix, max_limit)
        minlen = int(minlenratio * x.size(0)) if minlenratio >= 0 else -1 * int(minlenratio)
        minlen = minlen + n_prefix

        logger.info("decoder input length: " + str(x.size(0)))
        logger.info("max output length: " + str(maxlen))
        logger.info("min output length: " + str(minlen))

        # 🌟 初始化状态：同步传入 Base、Expert 和 Anti 所需的声学特征
        running_hyps = self.init_hyp(
            x, 
            proxy_x=px_expert, 
            anti_x=px_anti
        )
        ended_hyps = []

        for i in range(maxlen):
            logger.debug("position " + str(i))
            running_hyps = self.search(
                running_hyps, 
                x, 
                proxy_x=px_expert,
                anti_x=px_anti,
            )
            running_hyps = self.post_process(
                i, maxlen, minlen, maxlenratio, running_hyps, ended_hyps
            )
            if len(running_hyps) == 0:
                logger.info("no hypothesis. finish searching.")
                break

        if len(ended_hyps) == 0 and len(running_hyps) > 0:
            ended_hyps = self.unbatchfy(running_hyps)

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        if len(nbest_hyps) > 0:
            best = nbest_hyps[0]
            logger.info(f"total log probability: {best.score:.2f}")
            logger.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
            if self.token_list is not None:
                # 🌟 日志输出：剥离前缀 Prompt 和 EOS 字符，只打印有效文本
                yseq_valid = best.yseq[n_prefix:].tolist()
                token_clean = [
                    self.token_list[idx] 
                    for idx in yseq_valid 
                    if idx != self.eos and idx < len(self.token_list)
                ]
                token_str = "".join(token_clean)
                logger.info(f"best hypo: {token_str}")
        else:
            logger.warning("there is no ended hypothesis.")
        return nbest_hyps

    def post_process(
        self,
        i: int,
        maxlen: int,
        minlen: int,
        maxlenratio: float,
        running_hyps: BatchHypothesis,
        ended_hyps: List[Hypothesis],
    ) -> BatchHypothesis:
        n_batch = running_hyps.yseq.shape[0]
        device = running_hyps.yseq.device

        # 1. 达到最大长度时，为尚未结束的序列追加 EOS
        if i == maxlen - 1:
            yseq_eos = torch.cat(
                (
                    running_hyps.yseq,
                    torch.full(
                        (n_batch, 1),
                        self.eos,
                        device=device,
                        dtype=torch.int64,
                    ),
                ),
                dim=1,
            )
            # 🌟 修复关键：重新构造 BatchHypothesis 对象，避免 can't set attribute 报错
            running_hyps = BatchHypothesis(
                yseq=yseq_eos,
                score=running_hyps.score,
                length=(
                    running_hyps.length + 1 
                    if running_hyps.length is not None 
                    else torch.full((n_batch,), yseq_eos.shape[1], device=device, dtype=torch.int64)
                ),
                scores=running_hyps.scores,
                states=running_hyps.states,
                hs=running_hyps.hs,
            )

        # 2. 正确计算当前末尾 Token 是否为 EOS
        batch_idx = torch.arange(n_batch, device=device)
        length_idx = (running_hyps.length - 1).to(device)
        token_is_eos = (running_hyps.yseq[batch_idx, length_idx] == self.eos)

        # 只有步数达到了 minlen，生成的 EOS 才算有效终止
        valid_eos = token_is_eos if i >= minlen else torch.zeros_like(token_is_eos)

        # 3. 将正常终止的假设加入 ended_hyps
        for b in torch.nonzero(valid_eos, as_tuple=False).view(-1):
            hyp = self._select(running_hyps, b.item())
            ended_hyps.append(hyp)

        # 4. 过滤未结束的假设继续自回归
        remained_ids = torch.nonzero(valid_eos == 0, as_tuple=False).view(-1)
        return self._batch_select(running_hyps, remained_ids)