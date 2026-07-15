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
            # self.scaling = self.lora_alpha / 7
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
# class VeRALinear(nn.Linear, VeRALayer):
#     # LoRA implemented in a dense layer
#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         r: int = 0, 
#         vera_alpha: int = 1, 
#         vera_dropout: float = 0.,
#         shared_A: torch.Tensor = None,
#         shared_B: torch.Tensor = None,
#         fan_in_fan_out: bool = False, 
#         merge_weights: bool = True,
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         VeRALayer.__init__(self, r=r, vera_alpha=vera_alpha, vera_dropout=vera_dropout,
#                            merge_weights=merge_weights)
#         self.fan_in_fan_out = fan_in_fan_out
#         # Actual trainable parameters
#         if r > 0:
#             assert shared_A is not None and shared_B is not None
#             self.register_buffer('vera_A',shared_A)
#             self.register_buffer('vera_B',shared_B)
#             self.vera_d = nn.Parameter(torch.full((r,), 1.0))  # 向量
#             # self.vera_d = nn.Parameter(torch.eye(r) * 1.0)  # 初始化为0.1倍单位矩阵
#             # self.vera_d1 = nn.Parameter(torch.full((r,), 1.0))
#             # self.vera_d2 = nn.Parameter(torch.full((r,), 0.1))
#             # self.vera_b = nn.Parameter(torch.zeros(out_features))
#             # self.vera_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
#             self.vera_B1 = nn.Parameter(self.weight.new_zeros((out_features, r)))
#             self.scaling = self.vera_alpha / self.r
#             #self.scaling = self.lora_alpha / (self.r ** 0.5)
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.transpose(0, 1)

#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'vera_B1'):
#             # nn.init.kaiming_uniform_(self.vera_A, a=math.sqrt(5))
#             nn.init.zeros_(self.vera_B1)

        
#     def train(self, mode: bool = True):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         nn.Linear.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 if self.r > 0:
#                     # self.weight.data -= T(torch.diag(self.vera_b) @ self.vera_B @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling
#                     self.weight.data -= T(self.vera_B1 @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling
#                     # self.weight.data -= T(torch.diag(self.vera_b) @ self.vera_B @ self.vera_d @ self.vera_A) * self.scaling
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 if self.r > 0:
#                     # self.weight.data += T(torch.diag(self.vera_b) @ self.vera_B @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling
#                     self.weight.data += T(self.vera_B1 @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling
#                     # self.weight.data += T(torch.diag(self.vera_b) @ self.vera_B @ self.vera_d @ self.vera_A) * self.scaling
#                 self.merged = True       

#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         if self.r > 0 and not self.merged:
#             result = F.linear(x, T(self.weight), bias=self.bias)            
#             # result += F.linear(self.vera_dropout(x), (torch.diag(self.vera_b) @ self.vera_B1 @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling)
#             result += F.linear(self.vera_dropout(x), (self.vera_B1 @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling)
#             # result += F.linear(self.vera_dropout(x), (torch.diag(self.vera_b) @ self.vera_B @ self.vera_d @ self.vera_A) * self.scaling)
#             return result
#         else:
#             return F.linear(x, T(self.weight), bias=self.bias)
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
            assert shared_A is not None 
            self.register_buffer('vera_A',shared_A)
            # self.vera_A = shared_A 
            # self.vera_A = nn.Parameter(
            #     torch.empty(r, in_features), requires_grad=False
            # )
            # nn.init.kaiming_uniform_(self.vera_A, a=math.sqrt(5))
            # self.register_buffer('vera_A', torch.empty(r, in_features))
            # nn.init.kaiming_uniform_(self.vera_A, a=math.sqrt(5))
            # self.vera_d = nn.Parameter(torch.full((r,), 1.0))  # 向量
            # W = self.weight.data.detach()          # (out_features, in_features)
            # _, _, Vh = torch.linalg.svd(W, full_matrices=False)  # Vh: (in_features, in_features)
            # A_init = Vh[:r].contiguous()           # 取 top-r 左奇异向量 -> (r, in_features)
            # self.register_buffer('vera_A', A_init)
            self.vera_d = nn.Parameter(torch.full((r,), 1.0))  # 向量
            self.vera_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.vera_alpha / self.r
            #self.scaling = self.lora_alpha / (self.r ** 0.5)
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'vera_B'):
            # nn.init.kaiming_uniform_(self.vera_A, a=math.sqrt(5))
            nn.init.zeros_(self.vera_B)

        
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.vera_B @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.vera_B @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += F.linear(self.vera_dropout(x), (self.vera_B @ torch.diag(self.vera_d) @ self.vera_A) * self.scaling)
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


class DEELoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,          
        out_features: int,         
        r: int = 16,                
        lora_alpha: int = 1,       
        lora_dropout: float = 0.,        
        fan_in_fan_out: bool = False,  
        merge_weights: bool = True,    
        **kwargs
    ):
        # ====================== 新增：读取预分配秩文件 ======================
        rank_file_path = "/home/q/espnet/egs2/cdsdsb/asr1/exp/asr_train_asr_whisper_small_gora_raw_zh_whisper_multilingual96rank/gora_rank_allocation.json"  #32rank分秩
        # gradG_path = "/home/q/espnet/egs2/cdsds/asr1/exp/asr_train_asr_whisper_small_gora_raw_zh_whisper_multilingual48rankg/all_gradG.pt"
        matched_r = r
        # grad_G = None  #添加
        self.grad_G = None
        if os.path.exists(rank_file_path):
            with open(rank_file_path, "r", encoding="utf-8") as f:
                rank_dict = json.load(f)
            layer_name = kwargs.pop("layer_name", "")
            self.name = layer_name  # 绑定层名
            if layer_name in rank_dict:
                matched_r = rank_dict[layer_name]
                matched_r = matched_r
                # print(f"Linear [{layer_name}] 加载预分配秩：{matched_r}（原秩：{r})")
            # if os.path.exists(gradG_path):
            #     gradG_all = torch.load(gradG_path, map_location="cpu")  
            #     grad_G = gradG_all.get(layer_name, None)
            #     del gradG_all  # 立即删除大字典，释放内存！！！
            #     if grad_G is not None:
            #         grad_G = grad_G.float()  # 转回float32匹配模型精度
            # self.grad_G = grad_G
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=matched_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        # ====================== 修改结束 ======================
        # nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out 
        # if r > 0:
        if matched_r > 0:
            # self.lora_A1 = nn.Parameter(self.weight.new_zeros((r, in_features)))
            # self.lora_B1 = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # self.lora_A2 = nn.Parameter(self.weight.new_zeros((r, in_features)))
            # self.lora_B2 = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # self.lora_expert = nn.Parameter(torch.zeros(2))
            # self.lora_gate = nn.Linear(in_features, 2, bias=False)
            self.lora_A2 = nn.Parameter(self.weight.new_zeros((matched_r, in_features)))
            self.lora_B2 = nn.Parameter(self.weight.new_zeros((out_features,matched_r)))
            # self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)), requires_grad=False)
            # self.scaling = 2
            # self.scaling = self.lora_alpha / self.r
            self.scaling = self.lora_alpha / matched_r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)  

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # if hasattr(self, 'lora_gate'):
        #     nn.init.normal_(self.lora_gate.weight, std=0.01)
        if hasattr(self, 'lora_A2'):
            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B2)
        # if hasattr(self, 'lora_A'):
        #     nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        #     nn.init.zeros_(self.lora_B)
        #     if hasattr(self, 'grad_G') and self.grad_G is not None:
        #         try:
        #             grad_G = self.grad_G.to(self.lora_A2.device)
        #             weight_device = self.lora_A2.device
        #             weight_dtype = self.lora_A2.dtype
        #             grad_G = torch.nan_to_num(grad_G, nan=0.0, posinf=1e3, neginf=-1e3)
        #             grad_norm = torch.norm(grad_G, p='fro', dim=(0, 1))
        #             grad_norm = torch.clamp(grad_norm, min=1e-8)  # 防止范数为0
        #             grad_G = grad_G / grad_norm
        #             rank = self.lora_A2.shape[0]
        #             AT = self.lora_A2.T  # 核心：转置后维度是 768×36，而非36×768
        #             AAT = torch.matmul(self.lora_A2, AT)
        #             AAT += 1e-8 * torch.eye(rank, device=weight_device, dtype=weight_dtype)
        #             AAT_inv = torch.linalg.pinv(AAT)
        #             AAT_inv_AT = torch.matmul(AT, AAT_inv)  # 原错误：AAT_inv @ AT
        #             lora_B_val = -0.01 * torch.matmul(grad_G, AAT_inv_AT)
        #             lora_B_val = torch.clamp(lora_B_val, min=-10.0, max=10.0)
        #             self.lora_B2.data = lora_B_val
        #             del grad_G, AT, AAT, AAT_inv, AAT_inv_AT, lora_B_val  # 释放所有临时变量
        #         except Exception as e:
        #             print(f" {self.name} GoRA初始化lora_B失败:{e}")
        #             nn.init.zeros_(self.lora_B)
        #             pass
        #     if hasattr(self, 'grad_G'):
        #         del self.grad_G

        
    def train(self, mode: bool = True):
        def T(w): 
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    # s = torch.softmax(self.lora_expert, dim=0)  # [w1, w2]
                    # self.weight.data -= T((s[0] * (self.lora_B1 @ self.lora_A1)) + (s[1] * (self.lora_B2 @ self.lora_A2))) * self.scaling
                    self.weight.data -= T(self.lora_B2 @ self.lora_A2) * self.scaling
                    # self.weight.data -= T(self.lora_B2 @ self.lora_A2) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    # s = torch.softmax(self.lora_expert, dim=0)  # [w1, w2]
                    # self.weight.data += T((s[0] * (self.lora_B1 @ self.lora_A1)) + (s[1] * (self.lora_B2 @ self.lora_A2))) * self.scaling#方法二的公式1
                    self.weight.data += T(self.lora_B2 @ self.lora_A2) * self.scaling
                    # self.weight.data += T(self.lora_B2 @ self.lora_A2) * self.scaling
                self.merged = True  

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        # if self.r > 0 and not self.merged:
        #     result = F.linear(x, T(self.weight), bias=self.bias)
        #     gate_logits = self.lora_gate(x)
        #     gate_weights = F.softmax(gate_logits, dim=-1) 
        #     x_dropout = self.lora_dropout(x)
        #     delta_1 = (x_dropout @ self.lora_A1.t()) @ self.lora_B1.t()
        #     delta_2 = (x_dropout @ self.lora_A2.t()) @ self.lora_B2.t()
        #     lora_output = gate_weights[:, :, 0:1] * (delta_1 * self.scaling) + \
        #                   gate_weights[:, :, 1:2] * (delta_2 * self.scaling)    
        #     result += lora_output
        #     return result
        # else:
        #     return F.linear(x, T(self.weight), bias=self.bias)

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)  
            # s = torch.softmax(self.lora_expert, dim=0)  # [w1, w2]
            # delta = (s[0] * (self.lora_A1.transpose(0, 1) @ self.lora_B1.transpose(0, 1))) + (s[1] * (self.lora_A2.transpose(0, 1) @ self.lora_B2.transpose(0, 1))) 
            # result += (self.lora_dropout(x) @ delta) * self.scaling
            result += (self.lora_dropout(x) @ self.lora_A2.transpose(0, 1) @ self.lora_B2.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class DEEGoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,          
        out_features: int,         
        r: int = 16,                
        lora_alpha: int = 1,       
        lora_dropout: float = 0.,        
        fan_in_fan_out: bool = False,  
        merge_weights: bool = True,    
        **kwargs
    ):
        # ====================== 新增：读取预分配秩文件 ======================
        rank_file_path1 = "/home/q/espnet/egs2/MSDM/asr1/exp/asr_train_asr_whisper_small_gora_raw_zh_whisper_multilingual32rankb/gora_rank_allocation.json"  #32rank分秩
        rank_file_path2 = "/home/q/espnet/egs2/MSDM/asr1/exp/asr_train_asr_whisper_small_gora_raw_zh_whisper_multilingual48rankb/gora_rank_allocation.json"
        matched_r1 = r
        matched_r2 = r
        if os.path.exists(rank_file_path1):
            with open(rank_file_path1, "r", encoding="utf-8") as f:
                rank_dict1 = json.load(f)
            layer_name = kwargs.pop("layer_name", "")
            self.name = layer_name  # 绑定层名
            if layer_name in rank_dict1:
                matched_r1 = rank_dict1[layer_name]
                matched_r1 = matched_r1
        if os.path.exists(rank_file_path2):
            with open(rank_file_path2, "r", encoding="utf-8") as f:
                rank_dict2 = json.load(f)
            # layer_name = kwargs.pop("layer_name", "")
            self.name = layer_name  # 绑定层名
            if layer_name in rank_dict2:
                matched_r2 = rank_dict2[layer_name]
                matched_r2 = matched_r2
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=matched_r2, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out 
        if matched_r1 > 0:
            self.lora_A1 = nn.Parameter(self.weight.new_zeros((matched_r1, in_features)))
            self.lora_B1 = nn.Parameter(self.weight.new_zeros((out_features, matched_r1)))
            self.lora_A2 = nn.Parameter(self.weight.new_zeros((matched_r2, in_features)))
            self.lora_B2 = nn.Parameter(self.weight.new_zeros((out_features, matched_r2)))
            self.lora_expert = nn.Parameter(torch.zeros(2))
            self.scaling = 2
            # # self.scaling = self.lora_alpha / self.r
            # self.scaling = self.lora_alpha / matched_r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)  

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # if hasattr(self, 'lora_A1'):
        #     nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
        #     nn.init.zeros_(self.lora_B1)

        
    def train(self, mode: bool = True):
        def T(w): 
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    s = torch.softmax(self.lora_expert, dim=0)  # [w1, w2]
                    self.weight.data -= T((s[0] * (self.lora_B1 @ self.lora_A1)) + (s[1] * (self.lora_B2 @ self.lora_A2))) * self.scaling
                    # self.weight.data -= T(self.lora_B1 @ self.lora_A1) * self.scaling
                    # self.weight.data -= T(self.lora_B2 @ self.lora_A2) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    s = torch.softmax(self.lora_expert, dim=0)  # [w1, w2]
                    self.weight.data += T((s[0] * (self.lora_B1 @ self.lora_A1)) + (s[1] * (self.lora_B2 @ self.lora_A2))) * self.scaling#方法二的公式1
                    # self.weight.data += T(self.lora_B1 @ self.lora_A1) * self.scaling
                    # self.weight.data += T(self.lora_B2 @ self.lora_A2) * self.scaling
                self.merged = True  

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)  
            s = torch.softmax(self.lora_expert, dim=0)  # [w1, w2]
            delta = (s[0] * (self.lora_A1.transpose(0, 1) @ self.lora_B1.transpose(0, 1))) + (s[1] * (self.lora_A2.transpose(0, 1) @ self.lora_B2.transpose(0, 1))) 
            result += (self.lora_dropout(x) @ delta) * self.scaling
            # result += (self.lora_dropout(x) @ self.lora_A1.transpose(0, 1) @ self.lora_B1.transpose(0, 1)) * self.scaling
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
        # # ====================== 新增：读取预分配秩文件 ======================
        # import json
        # import os
        # # rank_file_path = "/home/q/espnet/egs2/cdsds/asr1/exp/exp0/asr_train_asr_whisper_small_gora_raw_zh_whisper_multilingual16rank/gora_rank_allocation.json"
        # rank_file_path = "/home/q/espnet/egs2/cdsds/asr1/exp/asr_train_asr_whisper_small_gora_raw_zh_whisper_multilingual1/gora_rank_allocation.json"
        # matched_r = r
        # if os.path.exists(rank_file_path):
        #     with open(rank_file_path, "r", encoding="utf-8") as f:
        #         rank_dict = json.load(f)
        #     layer_name = kwargs.pop("layer_name", "")
        #     self.name = layer_name  # 绑定层名
        #     if layer_name in rank_dict:
        #         matched_r = rank_dict[layer_name]
        #         # print(f"GoRALinear [{layer_name}] 加载预分配秩：{matched_r}（原秩：{r}）")
        # nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # LoRALayer.__init__(self, r=matched_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        #                    merge_weights=merge_weights)
        # # ====================== 修改结束 ======================
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.gora_init_method = gora_init_method
        self.gora_rank_stablize = gora_rank_stablize
        self.grad_stored = None  # 存储累积的梯度
        self.grad_steps = 0      # 梯度累积步数   
        # self.name = ""   
        # self.name = layer_name
        if r > 0:
        # if matched_r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            scale_rank = self.r 
            # self.lora_A = nn.Parameter(self.weight.new_zeros((matched_r, in_features)))  # 改用 matched_r
            # self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, matched_r)))  # 改用 matched_r
            # scale_rank = matched_r 
            self.scaling = self.lora_alpha / scale_rank
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)


    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            # 其他 GoRA 初始化方式需先存储梯度，在 dynamic_init 中执行
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
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
        return result

    def compute_importance(self, importance_type: str = 'union_mean') -> float:
        """计算层重要性（梯度驱动秩分配的核心）"""
        if self.grad_stored is None or self.grad_steps == 0:
            return 1e-6  # 兜底，避免0
        grad = self.grad_stored / self.grad_steps  #计算平均梯度（矩阵）
        grad = torch.nan_to_num(grad, nan=1e-6, posinf=1e-6, neginf=-1e-6)  #对矩阵进行数据清洗
        param = torch.nan_to_num(self.weight.data, nan=1e-6, posinf=1e-6, neginf=-1e-6)
        if importance_type == 'union_mean':
            param_mean = torch.mean(torch.abs(param)).clamp(min=1e-8)  # 变成标量，是一个数字
            grad_mean = torch.mean(torch.abs(grad)).clamp(min=1e-8)
            # importance = (param_mean * grad_mean * 1e8).item()
            importance = (param_mean * grad_mean).item()
        elif importance_type == 'grad_frobenius':
            grad_norm = torch.linalg.matrix_norm(grad).item()
            importance = grad_norm / grad.numel()  # 除以元素数，避免大矩阵范数过大
        elif importance_type == 'grad_nuc':
            grad_norm = torch.linalg.matrix_norm(grad, ord='nuc').item()
            importance = grad_norm / grad.numel()
        else:
            print(f"不支持的重要性类型：{importance_type}返回0")
            return 0.0
        importance = max(importance, 1e-6)
        print(f"层 {self.name} 重要性({importance_type}):{importance:.6f}")
        return importance
def allocate_ranks_by_importance(
    model: nn.Module,
    total_param_budget: int,
    importance_type: str = 'union_mean',
    min_rank: int = 1,
    max_rank: int = 64,
    param_tolerance: float = 0.05  # 新增：参数量浮动容忍度（5%）0.05
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
    # min_allowed_params = total_param_budget * (1 - param_tolerance)
    # 3. 计算偏差，仅当超过容忍度时微调
    param_diff = total_param_budget - current_total_params
    abs_diff_ratio = abs(param_diff) / total_param_budget  # 偏差比例
    # if current_total_params > total_param_budget:
    #     # 情况1：总参数量超标，必须减秩（优先减重要性低的层，影响更小）
    #     sorted_layers = sorted(layer_info.keys(), key=lambda k: layer_info[k]["weighted_imp"], reverse=True)
    #     # sorted_layers = sorted(layer_info.keys(), key=lambda k: layer_info[k]["weighted_imp"], reverse=False)
    #     for layer_name in sorted_layers:
    #         if current_total_params <= total_param_budget:
    #             break  # 已降到目标值以下，停止调整
    #         info = layer_info[layer_name]
    #         current_rank = named_ranks[layer_name]
    #         param_per_rank = info["dim_coeff"]
            
    #         if current_rank > min_rank:  # 仅当秩>最小值时才能减
    #             named_ranks[layer_name] -= 1
    #             current_total_params -= param_per_rank
    # elif current_total_params < min_allowed_params:
    #     # 情况2：总参数量过低（低于容忍度下限），小幅加秩（但不超过目标值）
    #     sorted_layers = sorted(layer_info.keys(), key=lambda k: layer_info[k]["weighted_imp"], reverse=True)
    #     for layer_name in sorted_layers:
    #         if current_total_params >= total_param_budget or current_total_params >= min_allowed_params:
    #             break  # 达到目标值 或 达到容忍度下限，停止调整
    #         info = layer_info[layer_name]
    #         current_rank = named_ranks[layer_name]
    #         param_per_rank = info["dim_coeff"]
            
    #         if current_rank < max_rank:  # 仅当秩<最大值时才能加
    #             named_ranks[layer_name] += 1
    #             current_total_params += param_per_rank
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
            param_diff = total_param_budget - current_total_params
            abs_diff_ratio = abs(param_diff) / total_param_budget
    # 偏差≤5%，直接放行
    # 可选：打印偏差信息（调试用）
    final_diff_ratio = abs(total_param_budget - current_total_params) / total_param_budget
    print(f"参数量偏差比例：{final_diff_ratio:.2%}（容忍度：{param_tolerance:.2%})")
    
    return named_ranks
class AdaLoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # -------------------------- 改动点1：替换lora_A/lora_B为P/Lambda/Q --------------------------
        if r > 0:
            # AdaLoRA核心参数：P(左奇异向量)、Lambda(奇异值)、Q(右奇异向量)
            self.lora_P = nn.Parameter(self.weight.new_zeros((out_features, r)))  # [out, r]
            self.lora_Lambda = nn.Parameter(self.weight.new_zeros((r,)))          # [r,] 奇异值（初始为0）
            self.lora_Q = nn.Parameter(self.weight.new_zeros((r, in_features)))   # [r, in]
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_P'):
            nn.init.normal_(self.lora_P, std=1/math.sqrt(self.r))
            nn.init.zeros_(self.lora_Lambda)
            nn.init.normal_(self.lora_Q, std=1/math.sqrt(self.r))

    # -------------------------- 新增：极简版奇异值裁剪（核心） --------------------------
    def prune_lambda(self, k: int):
        """裁剪奇异值:保留top-k个绝对值最大的奇异值,其余置0"""
        if self.r == 0 or k >= self.r:
            return
        # 取top-k奇异值的索引
        top_k_idx = torch.topk(self.lora_Lambda.data.abs(), k=k).indices
        # 构建掩码，仅top-k保留
        mask = torch.zeros_like(self.lora_Lambda.data)
        mask[top_k_idx] = 1.0
        self.lora_Lambda.data = self.lora_Lambda.data * mask

    # -------------------------- 新增：基础正交正则化（保证P/Q正交） --------------------------
    def ortho_reg(self):
        if self.r == 0:
            return torch.tensor(0.0, device=self.weight.device)
        # 正交约束：||P^T P - I|| + ||Q Q^T - I||
        P_T_P = self.lora_P.T @ self.lora_P
        Q_Q_T = self.lora_Q @ self.lora_Q.T
        I = torch.eye(self.r, device=self.lora_P.device)
        return torch.norm(P_T_P - I, p='fro') + torch.norm(Q_Q_T - I, p='fro')

    # -------------------------- 改动点2：适配AdaLoRA的train()方法（权重合并） --------------------------
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # 解合并：权重 -= PΛQ * scaling
                if self.r > 0:
                    delta = self.lora_P @ torch.diag(self.lora_Lambda) @ self.lora_Q
                    self.weight.data -= T(delta) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # 合并：权重 += PΛQ * scaling
                if self.r > 0:
                    delta = self.lora_P @ torch.diag(self.lora_Lambda) @ self.lora_Q
                    self.weight.data += T(delta) * self.scaling
                self.merged = True       

    # -------------------------- 改动点3：适配AdaLoRA的forward()方法 --------------------------
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            # 替换原LoRA的A/B计算为AdaLoRA的P/Λ/Q
            result += (self.lora_dropout(x) @ self.lora_Q.T @ torch.diag(self.lora_Lambda) @ self.lora_P.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
class DoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.m = nn.Parameter(torch.ones(out_features, 1, device=self.weight.device, dtype=self.weight.dtype))#self.m = nn.Parameter(torch.ones(out_features, 1))
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.V = nn.Parameter(self.weight.clone(), requires_grad=False)  # 方向基底（冻结）
            self.scaling = self.lora_alpha / self.r
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
            nn.init.ones_(self.m)# 初始化幅度向量
    def get_weight_norm(self, weight, lora_weight, scaling):
        if self.fan_in_fan_out:
            weight = weight.transpose(0, 1)
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    # delta_V = (self.lora_B @ self.lora_A) * self.scaling  # 低秩方向增量
                    # V_updated = self.V + delta_V  # 更新后的方向
                    # V_updated_norma = torch.norm(V_updated, dim=0, keepdim=True)
                    # V_normalized = V_updated / V_updated_norma
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                    # self.weight.data -= T((self.m * V_normalized)-self.V)
                    self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                    # delta_V = (self.lora_B @ self.lora_A) * self.scaling  # 低秩方向增量
                    # V_updated = self.V + delta_V  # 更新后的方向
                    # V_updated_norma = torch.norm(V_updated, dim=0, keepdim=True)
                    # V_normalized = V_updated / V_updated_norma
                    # self.weight.data += T((self.m * V_normalized)-self.V)
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            original_weight_norm = self.get_weight_norm(self.weight, torch.zeros_like(self.weight), 1.0)
            adjusted_weight_norm = self.get_weight_norm(self.weight, T(self.lora_B @ self.lora_A) * self.scaling, self.scaling)
            adjusted_weight = self.weight * (original_weight_norm / adjusted_weight_norm).view(-1, 1)#adjusted_weight = self.weight * (original_weight_norm / adjusted_weight_norm)
            adjusted_weight += T(self.lora_B @ self.lora_A) * self.scaling
            magnitude = self.m * (original_weight_norm / adjusted_weight_norm).view(-1, 1)#magnitude = self.m * (original_weight_norm / adjusted_weight_norm)
            result = F.linear(x, T(adjusted_weight), bias=self.bias)      
            magnitude = magnitude.view(1, 1, -1)  # Ensure magnitude has shape (1, 1, out_features)
            result *= magnitude  # Apply magnitude
            # base_output = F.linear(x, T(self.weight), bias=self.bias)  # 原始方向基底的输出
            # delta_V = (self.lora_B @ self.lora_A) * self.scaling  # 低秩方向增量
            # V_updated = self.V + delta_V  # 更新后的方向
            # V_updated_norma = torch.norm(V_updated, dim=1, keepdim=True)
            # V_normalized = V_updated / V_updated_norma
            # delta_weight = (self.m * V_normalized)-self.V
            # lora_output = F.linear(self.lora_dropout(x), delta_weight)
            # result = base_output + lora_output
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
class MoELoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,          
        out_features: int,         
        r: int = 16,                
        lora_alpha: int = 1,       
        lora_dropout: float = 0.,        
        fan_in_fan_out: bool = False,  
        merge_weights: bool = True,    
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out 
        num_experts = 2
        self.num_experts = num_experts

        if r > 0:
            self.lora_A = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((r, in_features))) for _ in range(num_experts)
            ])
            self.lora_B = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((out_features, r))) for _ in range(num_experts)
            ])
            self.lora_gate = nn.Linear(in_features, num_experts, bias=False)
            self.scaling = self.lora_alpha / r
            self.weight.requires_grad = False
            
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            for i in range(self.num_experts):
                nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[i])
            nn.init.normal_(self.lora_gate.weight, std=0.01)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            gate_logits = self.lora_gate(x)
            gate_weights = F.softmax(gate_logits, dim=-1) 
            lora_output = 0
            x_dropout = self.lora_dropout(x)
            
            for i in range(self.num_experts):
                expert_weight = gate_weights[:, :, i:i+1]
                expert_delta = (x_dropout @ self.lora_A[i].transpose(0, 1) @ self.lora_B[i].transpose(0, 1))
                lora_output += expert_weight * (expert_delta * self.scaling)
            result += lora_output
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)