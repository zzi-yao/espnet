from __future__ import annotations

from operator import attrgetter

import torch.nn as nn
from torch.optim import Optimizer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


def create_loraplus_optimizer(
    model, optimizer_cls: type[Optimizer],   **kwargs
) -> Optimizer:
    """
    Creates a LoraPlus optimizer.
    """
    
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        # "embedding": {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # module = attrgetter(name)(model)
        # if isinstance(module, Embedding):
        #     param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    lr = kwargs.pop("lr", 1.0e-5)
    loraplus_lr_ratio = kwargs.pop("loraplus_lr_ratio", 16.0)#kwargs["lr"] = lr
    loraplus_weight_decay = kwargs.pop("loraplus_weight_decay", 0.0)
    #r = kwargs.pop("r", 16)  # 添加一个参数 r，默认值为 16
    # loraplus_lr_embedding = kwargs.pop("loraplus_lr_embedding", 1e-6)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": loraplus_weight_decay,
            "lr": lr,
        },
        # {
        #     "params": list(param_groups["embedding"].values()),
        #     "weight_decay": loraplus_weight_decay,
        #     "lr": loraplus_lr_embedding,
        # },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": loraplus_weight_decay,
            "lr": lr * loraplus_lr_ratio,  # 修改学习率计算公式     "lr": lr * (loraplus_lr_ratio / (r ** 0.5)),
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
             "lr": lr * loraplus_lr_ratio, # 修改学习率计算公式      "lr": lr * (loraplus_lr_ratio / (r ** 0.5)),
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **kwargs)
    #eight_bit_names = ["Adam8bit", "AdamW8bit", "PagedAdam8bit", "PagedAdamW8bit"]
    # if optimizer_cls.__name__ in eight_bit_names:
    #     import bitsandbytes

    #     manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
    #     for module in model.modules():
    #         if isinstance(module, nn.Embedding):
    #             manager.register_module_override(module, "weight", {"optim_bits": 32})
    return optimizer 