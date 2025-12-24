from __future__ import annotations

from operator import attrgetter
import re

import torch.nn as nn
from torch.optim import Optimizer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


# def create_moelora_optimizer(
#     model, optimizer_cls: type[Optimizer], **kwargs
# ) -> Optimizer:
    
#     decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
#     decay_parameters = [name for name in decay_parameters if "bias" not in name]
    

#     param_groups = {
#         "moelora_O": {},          # O层MoE-LoRA参数（lora_A/lora_B）
#         "moelora_O_no_decay": {}, # O层MoE-LoRA偏置/1维参数（无权重衰减）
#         "single_lora": {},         # 其他层单LoRA参数（Q/K/V/MLP）
#         "single_lora_no_decay": {},# 单LoRA偏置/1维参数
#         "gate_network": {},        # 门控网络参数（新增）
#         "gate_network_no_decay": {},# 门控网络偏置（新增）
#     }

#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # 跳过已冻结的参数
        
#         if "gate" in name:  # 假设门控参数名含"gate"，如"attn.out.gate.weight"
#             if name in decay_parameters:
#                 param_groups["gate_network"][name] = param
#             else:
#                 param_groups["gate_network_no_decay"][name] = param
#         # 2. 匹配O层MoE-LoRA的lora_A/B
#         elif "attn.out" in name and ("lora_A" in name or "lora_B" in name):
#             if name in decay_parameters:
#                 param_groups["moelora_O"][name] = param
#             else:
#                 param_groups["moelora_O_no_decay"][name] = param
#         # 3. 匹配其他层普通LoRA的lora_A/B
#         elif "lora_A" in name or "lora_B" in name:
#             if name in decay_parameters:
#                 param_groups["single_lora"][name] = param
#             else:
#                 param_groups["single_lora_no_decay"][name] = param
#         else:
#             continue

#     base_lr = kwargs.pop("lr", 1e-4)  # 单LoRA基础学习率
#     moelora_lr_coeff = kwargs.pop("moelora_lr_coeff", 0.5)  # MoE-LoRA系数（下调至0.5）
#     gate_lr_coeff = kwargs.pop("gate_lr_coeff", 0.3)  # 门控网络系数（MoE-LoRA的0.6倍）
#     weight_decay = kwargs.pop("weight_decay", 1e-5)
#     optimizer_grouped_parameters = [
#         # 1. O层MoE-LoRA（带衰减）
#         {
#             "params": list(param_groups["moelora_O"].values()),
#             "weight_decay": weight_decay,
#             "lr": base_lr * moelora_lr_coeff,  # 普通LoRA的0.5倍
#         },
#         # 2. O层MoE-LoRA（无衰减）
#         {
#             "params": list(param_groups["moelora_O_no_decay"].values()),
#             "weight_decay": 0.0,
#             "lr": base_lr * moelora_lr_coeff,
#         },
#         # 3. 普通LoRA（带衰减）
#         {
#             "params": list(param_groups["single_lora"].values()),
#             "weight_decay": weight_decay,
#             "lr": base_lr,
#         },
#         # 4. 普通LoRA（无衰减）
#         {
#             "params": list(param_groups["single_lora_no_decay"].values()),
#             "weight_decay": 0.0,
#             "lr": base_lr,
#         },
#         # 5. 门控网络（带衰减）：学习率=普通LoRA*0.15（2e-5*0.5*0.3=3e-6）
#         {
#             "params": list(param_groups["gate_network"].values()),
#             "weight_decay": weight_decay,
#             "lr": base_lr * moelora_lr_coeff * gate_lr_coeff,
#         },
#         # 6. 门控网络（无衰减）
#         {
#             "params": list(param_groups["gate_network_no_decay"].values()),
#             "weight_decay": 0.0,
#             "lr": base_lr * moelora_lr_coeff * gate_lr_coeff,
#         },
#     ]
#     optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
#     optimizer = optimizer_cls(optimizer_grouped_parameters, **kwargs)
#     return optimizer


def create_moelora_optimizer(
    model, optimizer_cls: type[Optimizer], **kwargs
) -> Optimizer:
    # 1. 原逻辑：获取衰减参数
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    # 2. 极简参数组：按专家ID拆分MoELoRA（核心改动）
    expert_params = {0: {"decay": [], "nodecay": []}, 1: {"decay": [], "nodecay": []}, 2: {"decay": [], "nodecay": []}}
    single_lora = {"decay": [], "nodecay": []}
    gate = {"decay": [], "nodecay": []}

    # 3. 遍历参数（极简匹配逻辑）
    moe_pattern = re.compile(r"attn\.out\.(lora_A|lora_B)\.(\d+)")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 门控参数
        if "gate" in name:
            (gate["decay"] if name in decay_parameters else gate["nodecay"]).append(param)
        # O层MoELoRA：按ParameterList索引分专家
        elif "attn.out" in name and ("lora_A" in name or "lora_B" in name):
            match = moe_pattern.match(name)
            if match:
                eid = int(match.group(2))
                (expert_params[eid]["decay"] if name in decay_parameters else expert_params[eid]["nodecay"]).append(param)
        # 普通LoRA
        elif "lora_A" in name or "lora_B" in name:
            (single_lora["decay"] if name in decay_parameters else single_lora["nodecay"]).append(param)

    # 4. 超参数+专家lr系数（19:16:9 → 1.0:0.84:0.47）
    base_lr = kwargs.pop("lr", 1e-4)
    moe_coeff = kwargs.pop("moelora_lr_coeff", 0.5)
    # gate_coeff = kwargs.pop("gate_lr_coeff", 0.3)
    wd = kwargs.pop("weight_decay", 1e-5)
    expert_lr_scale = {0: 1.0, 1: 0.84, 2: 0.47}  # 样本数比例换算

    # 5. 构建优化器组（极简拼接）
    opt_groups = []
    # 5.1 MoELoRA：各专家差异化lr
    for eid in [0,1,2]:
        lr = base_lr * moe_coeff * expert_lr_scale[eid]
        if expert_params[eid]["decay"]:
            opt_groups.append({"params": expert_params[eid]["decay"], "weight_decay": wd, "lr": lr})
        if expert_params[eid]["nodecay"]:
            opt_groups.append({"params": expert_params[eid]["nodecay"], "weight_decay": 0.0, "lr": lr})
    # 5.2 普通LoRA（原逻辑）
    if single_lora["decay"]:
        opt_groups.append({"params": single_lora["decay"], "weight_decay": wd, "lr": base_lr})
    if single_lora["nodecay"]:
        opt_groups.append({"params": single_lora["nodecay"], "weight_decay": 0.0, "lr": base_lr})
    # # 5.3 门控网络（原逻辑）
    # gate_lr = base_lr * moe_coeff * gate_coeff
    # if gate["decay"]:
    #     opt_groups.append({"params": gate["decay"], "weight_decay": wd, "lr": gate_lr})
    # if gate["nodecay"]:
    #     opt_groups.append({"params": gate["nodecay"], "weight_decay": 0.0, "lr": gate_lr})

    # 6. 创建优化器
    return optimizer_cls([g for g in opt_groups if g["params"]],** kwargs)

