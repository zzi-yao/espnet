#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import json
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

import os

##
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)

class VeRALayer():
    def __init__(
        self, 
        r: int, 
        vera_alpha: int, 
        vera_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.vera_alpha = vera_alpha
        # Optional dropout
        if vera_dropout > 0.:
            self.vera_dropout = nn.Dropout(p=vera_dropout)
        else:
            self.vera_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
class VeRALinear(nn.Linear, VeRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        vera_alpha: int = 1, 
        vera_dropout: float = 0.,
        shared_A: torch.Tensor = None,
        shared_B: torch.Tensor = None,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        VeRALayer.__init__(self, r=r, vera_alpha=vera_alpha, vera_dropout=vera_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            assert shared_A is not None and shared_B is not None
            self.register_buffer('vera_A',shared_A)
            self.register_buffer('vera_B',shared_B)
            self.vera_d = nn.Parameter(torch.full((r,), 0.1))  # 或 1e-1
            self.vera_b = nn.Parameter(torch.zeros(out_features))
            self.scaling = self.vera_alpha / self.r
            #self.scaling = self.lora_alpha / (self.r ** 0.5)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(torch.diag(self.vera_b) @ self.vera_B @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(torch.diag(self.vera_b) @ self.vera_B @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += F.linear(self.vera_dropout(x), (torch.diag(self.vera_b) @ self.vera_B @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling)
            #result += F.linear(self.vera_dropout(x), torch.diag(self.vera_b) @ self.vera_B @ torch.diag(self.vera_d) @ self.vera_A) 
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class MeLoRALayer():
    def __init__(self, r: int, 
        melora_alpha: int, 
        melora_dropout: float,
        merge_weights: bool, 
        n:int, ):
        self.n = n
        self.merge_weights = merge_weights
        self.melora_alpha = melora_alpha
        self.r = r

        if melora_dropout > 0.:
            self.melora_dropout = nn.Dropout(p=melora_dropout)
        else:
            self.melora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights
class MeLinear(nn.Linear, MeLoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        melora_alpha: int = 1, 
        melora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        n: int = 1,  # 新增参数，表示 MeLoRA 的小模块数量
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MeLoRALayer.__init__(self, r=r, melora_alpha=melora_alpha, melora_dropout=melora_dropout,
                           merge_weights=merge_weights, n=n)
        self.fan_in_fan_out = fan_in_fan_out
        self.r_per_module = r // n
        self.in_features_per_module = in_features // n
        self.out_features_per_module = out_features // n

        if r > 0:
            # 创建 n 个小 LoRA 模块
            self.melora_A_list = nn.ParameterList([
                nn.Parameter(torch.empty(self.r_per_module, self.in_features_per_module, requires_grad=True))
                for _ in range(n)
            ])
            self.melora_B_list = nn.ParameterList([
                nn.Parameter(torch.empty(self.out_features_per_module, self.r_per_module, requires_grad=True))
                for _ in range(n)
            ])
            # self.scaling = self.r / self.n
            self.scaling = 2  #self.melora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)


    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'melora_A_list') and hasattr(self, 'melora_B_list'):
            for A, B in zip(self.melora_A_list, self.melora_B_list):
                #nn.init.normal_(A, mean=0, std=0.02)
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                nn.init.zeros_(B)


    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    for i in range(self.n):
                        Ai = self.melora_A_list[i]
                        Bi = self.melora_B_list[i]
                        melora_weight = T(Bi @ Ai)
                        self.weight.data[i * self.out_features_per_module:(i + 1) * self.out_features_per_module,
                                         i * self.in_features_per_module:(i + 1) * self.in_features_per_module] -= melora_weight * self.scaling
                    self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # 创建一个与 self.weight 形状相同的零张量
                    weight_update = torch.zeros_like(self.weight)
            
                    for i in range(self.n):
                        Ai = self.melora_A_list[i]
                        Bi = self.melora_B_list[i]
                        # 计算每个 LoRA 模块的权重贡献
                        melora_weight = T(Bi @ Ai)
                        # 将权重贡献加到相应的位置
                        weight_update[i * self.out_features_per_module:(i + 1) * self.out_features_per_module,
                                      i * self.in_features_per_module:(i + 1) * self.in_features_per_module] += melora_weight * self.scaling
            
                    # 将扩展后的权重加到 self.weight 上
                    self.weight.data += weight_update
                    self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if len(self.melora_A_list) > 0 and not self.merged:
            x_split = x.chunk(self.n, dim=-1)  # 按最后一维均分输入
            outputs = []
            for i in range(self.n):
                xi = x_split[i]
                Ai = self.melora_A_list[i]
                Bi = self.melora_B_list[i]
                outputs.append(xi @ Ai.T @ Bi.T)
            lora_out = torch.cat(outputs, dim=-1) * self.scaling
            lora_out = self.melora_dropout(lora_out)
            return F.linear(x, T(self.weight), bias=self.bias) + lora_out
        else:
            return F.linear(x, T(self.weight), bias=self.bias)      


# class MoELoRALinear(nn.Linear, LoRALayer):
#     def __init__(
#         self,
#         in_features: int,          
#         out_features: int,         
#         r: int = 0,                
#         lora_alpha: int = 1,       
#         lora_dropout: float = 0.,  
#         expert_num: int = 4,       
#         gate_temp: float = 6.0,    
#         top_k: int = 1,  # 新增：默认k=2
#         fan_in_fan_out: bool = False,  
#         merge_weights: bool = True,    
#         load_balance_coeff: float = 0.005,   #0.01
#         is_o_layer: bool = False,
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
#         self.fan_in_fan_out = fan_in_fan_out
#         self.expert_num = expert_num
#         self.gate_temp = gate_temp
#         self.top_k = top_k
#         self.load_balance_coeff = load_balance_coeff
#         self.load_balance_loss = torch.tensor(0.0, device=self.weight.device)
#         self.is_o_layer = is_o_layer

#         self.per_expert_r = r // expert_num if (r > 0 and expert_num > 0) else 0
#         if r > 0:
#             assert r % expert_num == 0
#             self.scaling = self.lora_alpha / self.r  
#             self.lora_A = nn.ParameterList([
#                 nn.Parameter(self.weight.new_zeros((self.per_expert_r, in_features)))
#                 for _ in range(self.expert_num)
#             ])
#             self.lora_B = nn.ParameterList([
#                 nn.Parameter(self.weight.new_zeros((out_features, self.per_expert_r)))
#                 for _ in range(self.expert_num)
#             ])
#             self.gate_linear = nn.Linear(in_features, self.expert_num)  # 仅一层线性层
#             self.gate_linear.weight.requires_grad = True
#             if self.gate_linear.bias is not None:
#                 self.gate_linear.bias.requires_grad = True
#             self.weight.requires_grad = False
#         self.pretrained_expert_paths = [
#             "exp/asr_train_asr_whisper_small_moelora_raw_zh_whisper_multilingual_yu/pretrained_moelora_experts/expert_0.pth",
#             "exp/asr_train_asr_whisper_small_moelora_raw_zh_whisper_multilingual_yu/pretrained_moelora_experts/expert_1.pth",
#             "exp/asr_train_asr_whisper_small_moelora_raw_zh_whisper_multilingual_yu/pretrained_moelora_experts/expert_2.pth",
#             "exp/asr_train_asr_whisper_small_moelora_raw_zh_whisper_multilingual_yu/pretrained_moelora_experts/expert_3.pth"
#         ]
#         self.reset_parameters()
#         if r > 0 and self.is_o_layer:  # 仅当LoRA秩>0时加载
#             self.load_pretrained_experts()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.transpose(0, 1)  

#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'r') and self.r > 0:
#             if hasattr(self, 'gate_linear'):
#                 nn.init.xavier_uniform_(self.gate_linear.weight, gain=0.1)
#                 if self.gate_linear.bias is not None:
#                     nn.init.zeros_(self.gate_linear.bias)

#     def load_pretrained_experts(self):
#         assert len(self.pretrained_expert_paths) == self.expert_num, "专家数不匹配"
#         for e, ckpt_path in enumerate(self.pretrained_expert_paths):
#             ckpt = torch.load(ckpt_path, map_location=self.weight.device)
#             self.lora_A[e].data = ckpt['lora_A'].to(self.weight.device)
#             self.lora_B[e].data = ckpt['lora_B'].to(self.weight.device)

#     def _get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
#         x_reshaped = x.reshape(-1, x.shape[-1])  # [B*L, D] → 展平序列维度，便于gate_mlp计算
#         # gate_weights = self.gate_mlp(x_reshaped)  # gate_mlp: D → gate_hidden_dim → expert_num
#         gate_weights = self.gate_linear(x_reshaped)
#         gate_weights = gate_weights.reshape(*x.shape[:-1], self.expert_num)  # 核心：匹配输入的前N维
#         gate_weights = torch.softmax(gate_weights / self.gate_temp, dim=-1)
#         # if self.training and hasattr(self, 'target_expert_id'):
#         #     target_expert = self.target_expert_id  # 直接用提前设置的专家ID
#         #     gate_weights = gate_weights.clone()  # 复制张量，脱离原计算图
#         #     gate_weights[..., target_expert] = gate_weights[..., target_expert] + 1.5
#         #     gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)
#         # 条件1：训练模式；条件2：有target_expert_ids；条件3：输入不是dummy_x（batch维度匹配）
#         is_real_train_sample = (
#             self.training 
#             and hasattr(self, 'target_expert_ids') 
#             and len(self.target_expert_ids) > 0  # 宽松判断
#         )
#         if is_real_train_sample:
#             target_ids = torch.tensor(self.target_expert_ids, device=gate_weights.device)  # [B]
#             if gate_weights.dim() == 3:
#                 target_ids = target_ids.unsqueeze(1).repeat(1, gate_weights.shape[1])  # [B, L]
#             gate_weights = gate_weights.clone()
#             gate_weights = gate_weights.scatter_add_(
#                 dim=-1,
#                 index=target_ids.unsqueeze(-1),  # [B, L, 1] 或 [B, 1]
#                 src=torch.ones_like(gate_weights) * 0.5  # 偏置强度0.5
#             )
#             gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)
#             # ========== 修正：使用全局gate_logger，补充batch序号 ==========
#             current_batch_idx = getattr(self, 'moe_batch_idx', 0)
#             if current_batch_idx % 50 == 0:
#                 gate_weights_mean = gate_weights.mean(dim=1) if gate_weights.dim() ==3 else gate_weights
#                 sample_num = min(3, len(self.target_expert_ids))
#                 for idx in range(sample_num):
#                     target_id = self.target_expert_ids[idx]
#                     weight = gate_weights_mean[idx].detach().cpu().numpy().round(3)
#                     gate_logger.info(
#                         f"Batch {current_batch_idx} - "
#                         f"Sample {idx} - Target expert: {target_id} - "
#                         f"Gate weights: {weight}"
#                     )
#         return gate_weights
    
#     def train(self, mode: bool = True):
#         def T(w): 
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         nn.Linear.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 if self.r == 0:
#                     return
#                 dummy_x = torch.randn(1, self.in_features, device=self.weight.device)
#                 gate_weights = self._get_gate_weights(dummy_x)  
#                 for e in range(self.expert_num):
#                     delta_W = T(self.lora_B[e] @ self.lora_A[e]) 
#                     self.weight.data -= delta_W * self.scaling * gate_weights[0, e]
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 if self.r == 0:
#                     return
#                 dummy_x = torch.randn(1, self.in_features, device=self.weight.device)
#                 gate_weights = self._get_gate_weights(dummy_x)  
#                 for e in range(self.expert_num):
#                     delta_W = T(self.lora_B[e] @ self.lora_A[e])  
#                     self.weight.data += delta_W * self.scaling * gate_weights[0, e]
#                 self.merged = True

#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         result = F.linear(x, T(self.weight), bias=self.bias)
#         if self.r > 0 and not self.merged:
#             x_dropout = self.lora_dropout(x)  
#             gate_weights = self._get_gate_weights(x_dropout)
#             x_shape = x_dropout.shape
#             x_flat = x_dropout.reshape(-1, x_shape[-1])  
#             gate_weights_flat = gate_weights.reshape(-1, self.expert_num)
#             topk_vals, topk_indices = torch.topk(
#                 gate_weights_flat, 
#                 k=self.top_k, 
#                 dim=-1,  # 沿专家维度筛选
#                 largest=True  # 选权重最大的k个
#             )
#             gate_weights_masked = torch.zeros_like(gate_weights_flat)
#             gate_weights_masked.scatter_(
#                 dim=-1, 
#                 index=topk_indices, 
#                 src=topk_vals
#             )
#             gate_weights_flat = gate_weights_masked
#             # 5. 计算负载均衡损失
#             expert_selected = (gate_weights_masked > 0).float().sum(dim=0)
#             if expert_selected.sum() > 0:
#                 expert_selected_ratio = expert_selected / (expert_selected.sum() + 1e-8)  # 分母防0
#                 expert_selected_ratio = expert_selected_ratio.clamp(min=1e-8)  # 分子防0，避免log(0)
#                 target_ratio = torch.ones_like(expert_selected_ratio) / self.expert_num
#                 target_ratio = target_ratio.clamp(min=1e-8)  # 目标分布也防0（兜底）
#                 ratio_loss = F.kl_div(
#                     expert_selected_ratio.log(),  # 输入1：对数概率
#                     target_ratio,                # 输入2：目标概率
#                     reduction="mean",            # 替换batchmean→mean（无批次维度）
#                     log_target=False             # 显式声明target不是对数概率（默认False，可省略）
#                 )
#                 diversity_loss = 0.0
#                 count = 0
#                 with torch.no_grad():
#                     expert_outs = []
#                     for e in range(self.expert_num):
#                         a_out = x_flat @ self.lora_A[e].T
#                         b_out = a_out @ self.lora_B[e].T
#                         b_out_mean = b_out.mean(dim=0)  # 仅保留特征维度均值
#                         expert_outs.append(b_out_mean)
#                         del a_out, b_out
#                         torch.cuda.empty_cache()  # 清理显存碎片
#                     for e1 in range(self.expert_num):
#                         for e2 in range(e1+1, self.expert_num):
#                             sim = F.cosine_similarity(expert_outs[e1], expert_outs[e2], dim=-1).item()  # 标量输出
#                             diversity_loss += sim
#                             count += 1
#                 diversity_loss = diversity_loss / count if count > 0 else 0.0
#                 diversity_loss = torch.tensor(diversity_loss, device=x.device, dtype=torch.float32).clamp(min=0.0, max=1.0)
#                 self.load_balance_loss = 0.01 * ratio_loss + 0.99 * diversity_loss
#                 # ========== 打印负载均衡损失到独立日志 ==========
#                 current_batch_idx = getattr(self, 'moe_batch_idx', 0)
#                 if current_batch_idx % 50 == 0:
#                     gate_logger.info(
#                         f"Batch {current_batch_idx} - "  # 统一用current_batch_idx，避免显示unknown
#                         f"Load balance loss: {self.load_balance_loss.item():.6f} | "
#                         f"Ratio loss: {ratio_loss.item():.6f} | "
#                         f"Diversity loss: {diversity_loss.item():.6f}"
#                     ) 
#             else:
#                 self.load_balance_loss = torch.tensor(0.0, device=x.device)
            
#             explore_prob = 0.0 #0.01  
#             explore_mask = (torch.rand_like(gate_weights_flat[:,0]) < explore_prob).unsqueeze(-1)
#             if explore_mask.any():  
#                 expert_selected_ratio = expert_selected / (expert_selected.sum() + 1e-8)
#                 explore_probs = 1 - expert_selected_ratio
#                 explore_probs = explore_probs / (explore_probs.sum() + 1e-8)
#                 explore_probs_expanded = explore_probs.unsqueeze(0).repeat(gate_weights_flat.shape[0], 1)
#                 mask_squeezed = explore_mask.squeeze(-1)
#                 random_indices = torch.multinomial(
#                     explore_probs_expanded[mask_squeezed],
#                     num_samples=1,
#                     replacement=True,
#                 )
#                 random_indices_full = torch.zeros( (gate_weights_flat.shape[0], 1), dtype=torch.long, device=x.device)
#                 random_indices_full[mask_squeezed, :] = random_indices
                
#                 random_vals = torch.zeros_like(gate_weights_flat)
#                 random_vals.scatter_(
#                     dim=-1,  # 在最后一维（专家维度）填充
#                     index=random_indices_full,  # [B*L, 1]（2D）
#                     src=torch.ones_like(random_indices_full, dtype=random_vals.dtype)  # 权重设为1，维度匹配
#                 )
#                 gate_weights_flat = torch.where(explore_mask, random_vals, gate_weights_flat)

#             lora_output = torch.zeros(x_flat.shape[0], self.out_features, device=x_flat.device, dtype=x_flat.dtype)
#             for e in range(self.expert_num):
#                 a_out = x_flat @ self.lora_A[e].T  
#                 b_out = a_out @ self.lora_B[e].T  
#                 b_out = b_out * self.scaling * gate_weights_flat[:, e: e+1]  
#                 lora_output += b_out
#             lora_output = lora_output.reshape(*x_shape[:-1], self.out_features) 
#             result += lora_output
#             return result
#         else:
#             return F.linear(x, T(self.weight), bias=self.bias)
#     def get_load_balance_loss(self) -> torch.Tensor:
#         return self.load_balance_loss * self.load_balance_coeff
class MoELoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,          
        out_features: int,         
        r: int = 16,                
        lora_alpha: int = 1,       
        lora_dropout: float = 0.,  
        expert_num: int = 3,       
        fan_in_fan_out: bool = False,  
        merge_weights: bool = True,    
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.expert_num = expert_num
        self.expert_id = 0
        self.per_expert_r = [5, 5, 6] if expert_num == 3 else (r // expert_num if (r > 0 and expert_num > 0) else 0)
    
        if r > 0:
           self.scaling = self.lora_alpha / self.r  
           self.lora_A = nn.ParameterList()
           self.lora_B = nn.ParameterList()
           for e in range(self.expert_num):
               curr_r = self.per_expert_r[e] if expert_num == 3 else self.per_expert_r
               self.lora_A.append(nn.Parameter(self.weight.new_zeros((curr_r, in_features))))
               self.lora_B.append(nn.Parameter(self.weight.new_zeros((out_features, curr_r))))
           self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)  

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'r') and self.r > 0:
            if hasattr(self, 'lora_A') and hasattr(self, 'lora_B'):
                base_coeff = math.sqrt(5) * 0.1  # 匹配语音特征±10的量级
                noise_level = 1e-3                # 匹配差分特征±1的量级
                param_clip = 0.05                # 语音特征的合理参数范围
                
                # 3专家的差异化初始化（按你的分配方案：0=轻度，1=中度，2=重度）
                for e in range(self.expert_num):
                    seed = 42 + e * 100
                    torch.manual_seed(seed)
                    if self.expert_num == 3:
                        if e == 0:  # 专家0（轻度病理）：强鲁棒性
                            nn.init.kaiming_uniform_(self.lora_A[e], a=base_coeff * 2)
                            nn.init.zeros_(self.lora_B[e])
                        elif e == 1:  # 专家1（中度病理）：聚焦细节
                            nn.init.kaiming_normal_(self.lora_A[e], a=base_coeff * 0.5)
                            noise = torch.randn_like(self.lora_A[e]) * noise_level
                            self.lora_A[e].data = self.lora_A[e].data + noise
                            nn.init.normal_(self.lora_B[e], mean=0, std=noise_level / 10)
                        else:  # 专家2（重度病理）：高容错性
                            nn.init.orthogonal_(self.lora_A[e], gain=0.08)
                            noise = torch.randn_like(self.lora_A[e]) * (noise_level * 1.5)
                            self.lora_A[e].data = self.lora_A[e].data + noise
                            nn.init.uniform_(self.lora_B[e], a=-noise_level/8, b=noise_level/8)
                    self.lora_A[e].data = self.lora_A[e].data.clamp(-param_clip, param_clip)
                torch.manual_seed(torch.initial_seed())
    def train(self, mode: bool = True):
        def T(w): 
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r == 0:
                    return
                delta_W = T(self.lora_B[self.expert_id] @ self.lora_A[self.expert_id]) 
                self.weight.data -= delta_W * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r == 0:
                    return
                delta_W = T(self.lora_B[self.expert_id] @ self.lora_A[self.expert_id]) 
                self.weight.data += delta_W * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0 and not self.merged:
            x_dropout = self.lora_dropout(x)  
            x_shape = x_dropout.shape
            x_flat = x_dropout.reshape(-1, x_shape[-1])  

            expert_id = self.expert_id
            lora_output = torch.zeros(x_flat.shape[0], self.out_features, device=x_flat.device)
            a_out = x_flat @ self.lora_A[expert_id].T
            b_out = a_out @ self.lora_B[expert_id].T
            b_out = b_out * self.scaling
            lora_output += b_out
            lora_output = lora_output.reshape(*x_shape[:-1], self.out_features) 
            result += lora_output
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
class GoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        gora_init_method: str = 'vanilla',  # vanilla/grad_compress/grad_svd
        gora_rank_stablize: bool = False,   # 是否对秩做平方根缩放
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.gora_init_method = gora_init_method
        self.gora_rank_stablize = gora_rank_stablize
        self.grad_stored = None  # 存储累积的梯度
        self.grad_steps = 0      # 梯度累积步数
        self.lora_A = None
        self.lora_B = None
        self.scaling = 0.
        self._forward_output = None  # 新增：存储前向输出张量
        self._output_hook_handle = None  # 钩子句柄初始化为None
        self._grad_hook_func = None      # 复用的钩子函数初始化为None
        if r > 0:
            self._init_lora_params()
            self.weight.requires_grad = False
        
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def _init_lora_params(self):
        """初始化 LoRA 参数（抽离便于后续动态更新）"""
        self.lora_A = nn.Parameter(self.weight.new_zeros((self.r, self.in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, self.r)))
        scale_rank = self.r
        if self.gora_rank_stablize:
            scale_rank = math.sqrt(scale_rank)
        self.scaling = self.lora_alpha / scale_rank

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A') and self.lora_A is not None:
            if self.gora_init_method == 'vanilla':
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            # 其他 GoRA 初始化方式需先存储梯度，在 dynamic_init 中执行

    # def register_gradient_hook(self):      #记录权重梯度
    #     def grad_hook(grad):
    #         if self.grad_stored is None:
    #             self.grad_stored = grad.detach().clone()
    #         else:
    #             self.grad_stored += grad.detach().clone()
    #         self.grad_steps += 1
    #         return grad
    #     hook_handle = self.weight.register_hook(grad_hook)
    #     return hook_handle
    def register_gradient_hook(self):      
        def output_grad_hook(grad):
            if hasattr(self, '_forward_x') and self._forward_x is not None:
                x = self._forward_x.detach()
                grad = grad.detach()
                if x.dim() == 3:
                    x_flat = x.reshape(-1, x.shape[-1])
                    grad_flat = grad.reshape(-1, grad.shape[-1])
                elif x.dim() == 2:
                    x_flat = x
                    grad_flat = grad
                else:
                    print(f"不支持的张量维度:x.dim()={x.dim()}")
                    return grad
                weight_grad = torch.matmul(grad_flat.mT, x_flat) / x_flat.shape[0]
                if self.grad_stored is None:
                    self.grad_stored = weight_grad
                else:
                    self.grad_stored += weight_grad
                self.grad_steps += 1
            return grad
        # self._output_hook_handle = None
        if not hasattr(self, '_grad_hook_func'):
            self._grad_hook_func = output_grad_hook
        return output_grad_hook

    def compute_importance(self, importance_type: str = 'union_mean') -> float:
        """计算层重要性（梯度驱动秩分配的核心）"""
        if self.grad_stored is None:
            print(f"层 {self.name} 未存储梯度，返回重要性0")
            return 0.0
        if self.grad_steps == 0:
            print(f"层 {self.name} 梯度累积步数为0，返回重要性0")
            return 0.0
        if self.grad_steps > 1:
            grad = self.grad_stored / self.grad_steps  # 累积梯度和 → 均值
        else:
            grad = self.grad_stored  # 仅1步梯度，无需归一化
        param = self.weight.data
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print(f"层 {self.name} 梯度包含NaN/Inf，替换为0")
            grad = torch.nan_to_num(grad, nan=0.0, posinf=1e-6, neginf=-1e-6)
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"层 {self.name} 参数包含NaN/Inf，替换为0")
            param = torch.nan_to_num(param, nan=0.0, posinf=1e-6, neginf=-1e-6)
        try:
            if importance_type == 'union_mean':
                param_mean = torch.mean(torch.abs(param)).clamp(min=1e-8)  # 兜底：避免0
                grad_mean = torch.mean(torch.abs(grad)).clamp(min=1e-8)
                importance = (param_mean * grad_mean * 1e8).item()
                # importance = (param_mean * grad_mean).item()
            elif importance_type == 'grad_frobenius':
                grad_norm = torch.linalg.matrix_norm(grad).item()
                importance = grad_norm / grad.numel()  # 除以元素数，避免大矩阵范数过大
            elif importance_type == 'grad_nuc':
                grad_norm = torch.linalg.matrix_norm(grad, ord='nuc').item()
                importance = grad_norm / grad.numel()
            else:
                print(f"不支持的重要性类型：{importance_type}，返回0")
                return 0.0
    
        except Exception as e:
            print(f"层 {self.name} 计算重要性失败：{str(e)}，返回0")
            return 0.0
        importance = max(importance, 1e-6)
        print(f"层 {self.name} 重要性({importance_type})：{importance:.6f}")
        return importance

    def dynamic_init(self, target_rank: int, stable_gamma: float = 16.):
        """
        动态初始化 LoRA 参数(GoRA 核心）
        :param target_rank: 自适应分配的目标秩
        :param stable_gamma: 稳定化系数（防止梯度初始化值过大）
        """
        if target_rank <= 0:
            self.r = 0
            self.lora_A = None
            self.lora_B = None
            return
        self.r = target_rank
        self._init_lora_params()
        
        if self.grad_stored is None or self.grad_steps == 0:
            self.reset_parameters()
            return

        grad = self.grad_stored / self.grad_steps
        grad = grad.to(self.weight.device, dtype=self.weight.dtype)
        
        if self.gora_init_method == 'grad_compress':
            # GoRA 核心：梯度压缩初始化 B = G @ (A^T A + εI)^{-1} A^T
            # 1. 初始化 A（保持原有 kaiming_uniform）
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # 2. 计算 A 的伪逆
            A_T = self.lora_A.T
            AAT = torch.matmul(self.lora_A, A_T)
            AAT_inv = torch.linalg.pinv(AAT + 1e-8 * torch.eye(self.r).to(self.weight.device))
            AAT_inv_AT = torch.matmul(A_T, AAT_inv)
            # 3. 计算 B（梯度压缩）
            self.lora_B.data = torch.matmul(grad, AAT_inv_AT)
            # 4. 稳定化缩放
            self.lora_B.data *= stable_gamma / self.lora_alpha
            
        elif self.gora_init_method == 'grad_svd':
            U, S, V = torch.svd_lowrank(grad.float(), q=4*self.r, niter=4)
            V = V.T
            # 取前 r 个奇异向量初始化
            self.lora_B.data = U[:, :self.r].to(self.weight.dtype)
            self.lora_A.data = V[self.r:2*self.r, :].to(self.weight.dtype)
            # 稳定化缩放
            scale = math.pow(self.out_features, 0.25) / math.sqrt(stable_gamma)
            self.lora_A.data *= scale
            self.lora_B.data *= scale
        
        # 清除已存储的梯度（避免重复使用）
        self.grad_stored = None
        self.grad_steps = 0

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        """保持原有 forward 逻辑不变"""
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        self._forward_x = x
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
        if self.training:
            if self._output_hook_handle is None:
                hook_func = self.register_gradient_hook()
                self._output_hook_handle = result.register_hook(hook_func)
        return result
        # if self.r > 0 and not self.merged:
        #     result = F.linear(x, T(self.weight), bias=self.bias)            
        #     result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        #     if self._output_hook_handle is None and self.training:
        #         hook_func = self.register_gradient_hook()
        #         self._output_hook_handle = result.register_hook(hook_func)
        #     return result
        # else:
        #     # return F.linear(x, T(self.weight), bias=self.bias)
        #     result = F.linear(x, T(self.weight), bias=self.bias)
        #     if self._output_hook_handle is None and self.training:
        #         hook_func = self.register_gradient_hook()
        #         self._output_hook_handle = result.register_hook(hook_func)
        #     return result

# ---------------------- 辅助函数：全局秩分配 ----------------------
def allocate_ranks_by_importance(
    model: nn.Module,
    total_param_budget: int,
    importance_type: str = 'union_mean',
    min_rank: int = 1,
    max_rank: int = 64,
    param_tolerance: float = 0.05  # 新增：参数量浮动容忍度（5%）
) -> dict:
    # 1. 收集层信息（重要性+维度系数）
    layer_info = {}
    total_weighted_importance = 0.
    for name, module in model.named_modules():
        if isinstance(module, GoRALinear) and module.r > 0:
            imp = module.compute_importance(importance_type)
            dim_coeff = module.in_features + module.out_features
            weighted_imp = imp * dim_coeff
            layer_info[name] = {
                "imp": imp,
                "dim_coeff": dim_coeff,
                "weighted_imp": weighted_imp
            }
            total_weighted_importance += weighted_imp
    
    # 2. 按加权重要性分配秩（先不微调）
    named_ranks = {}
    current_total_params = 0
    for name, info in layer_info.items():
        layer_param_budget = (info["weighted_imp"] / total_weighted_importance) * total_param_budget
        rank = round(layer_param_budget / info["dim_coeff"])
        rank = max(min(rank, max_rank), min_rank)
        named_ranks[name] = rank
        current_total_params += rank * info["dim_coeff"]
    
    # 3. 计算偏差，仅当超过容忍度时微调
    param_diff = total_param_budget - current_total_params
    abs_diff_ratio = abs(param_diff) / total_param_budget  # 偏差比例
    
    if abs_diff_ratio > param_tolerance:
        # 偏差超过5%，才进行微调（逻辑和之前一致，但只微调到容忍度内）
        sorted_layers = sorted(layer_info.keys(), key=lambda k: layer_info[k]["weighted_imp"], reverse=True)
        for layer_name in sorted_layers:
            if abs_diff_ratio <= param_tolerance:
                break
            info = layer_info[layer_name]
            current_rank = named_ranks[layer_name]
            param_per_rank = info["dim_coeff"]
            
            if param_diff > 0 and current_rank < max_rank:
                named_ranks[layer_name] += 1
                current_total_params += param_per_rank
            elif param_diff < 0 and current_rank > min_rank:
                named_ranks[layer_name] -= 1
                current_total_params -= param_per_rank
            
            # 更新偏差比例
            param_diff = total_param_budget - current_total_params
            abs_diff_ratio = abs(param_diff) / total_param_budget
    # 偏差≤5%，直接放行
    
    # 可选：打印偏差信息（调试用）
    final_diff_ratio = abs(total_param_budget - current_total_params) / total_param_budget
    print(f"参数量偏差比例：{final_diff_ratio:.2%}（容忍度：{param_tolerance:.2%}）")
    
    return named_ranks
