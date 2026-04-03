from __future__ import annotations
import torch.nn as nn
from torch.optim import Optimizer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


def create_vera_optimizer(
    model, optimizer_cls: type[Optimizer],   **kwargs
) -> Optimizer:
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupd": {},
        "groupB1": {},
        "groupB1_no_decay": {},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif "vera_d" in name :
            param_groups["groupd"][name] = param            
        else:
            if name in decay_parameters:
                param_groups["groupB1"][name] = param
            else:
                param_groups["groupB1_no_decay"][name] = param

    lr = kwargs.pop("lr", 5.0e-4)
    loraplus_lr_ratio = kwargs.pop("loraplus_lr_ratio", 1.0e-2)
    loraplus_weight_decay = kwargs.pop("loraplus_weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupd"].values()),
            "weight_decay": loraplus_weight_decay,   #0.1
            "lr": loraplus_lr_ratio,
        },
        {
            "params": list(param_groups["groupB1"].values()),
            "weight_decay": loraplus_weight_decay,
            "lr": lr,  
        },
        {
            "params": list(param_groups["groupB1_no_decay"].values()),
            "weight_decay": 0.0,
             "lr": lr, 
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **kwargs)
    return optimizer 