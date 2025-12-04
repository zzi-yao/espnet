#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

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

  # 单任务版MoELoRA（无任务嵌入/任务ID，数据驱动门控）
class MoELoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,          # 输入维度
        out_features: int,         # 输出维度
        r: int = 0,                # 总LoRA秩
        lora_alpha: int = 1,       # LoRA alpha（缩放系数）
        lora_dropout: float = 0.,  # Dropout率
        expert_num: int = 4,       # 专家数量
        gate_temp: float = 6.0,    # 门控Softmax温度系数（经验值）
        gate_hidden_dim: int = 64, # 门控网络隐藏维度（数据驱动门控用）
        fan_in_fan_out: bool = False,  # 权重存储格式（适配LLM）
        merge_weights: bool = True,    # 是否合并权重
        **kwargs
    ):
        # 初始化父类
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        # 原有属性保留
        self.fan_in_fan_out = fan_in_fan_out

        # MoELoRA核心参数（单任务版）
        self.expert_num = expert_num
        self.gate_temp = gate_temp
        self.gate_hidden_dim = gate_hidden_dim

        # 计算每个专家的秩（必须整除）
        self.per_expert_r = r // expert_num if (r > 0 and expert_num > 0) else 0
        if r > 0:
            assert r % expert_num == 0, f"总秩r={r}必须能被专家数expert_num={expert_num}整除"
            self.scaling = self.lora_alpha / self.r  # LoRA缩放系数（和原始LoRA一致）

            # 1. 拆分LoRA A/B为多个专家（替代Expert类）
            # A_e: 每个专家的A矩阵 → 维度 [per_expert_r, in_features]
            self.lora_A = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((self.per_expert_r, in_features)))
                for _ in range(expert_num)
            ])
            # B_e: 每个专家的B矩阵 → 维度 [out_features, per_expert_r]
            self.lora_B = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((out_features, self.per_expert_r)))
                for _ in range(expert_num)
            ])

            # 2. 数据驱动的门控网络（替代任务嵌入+门控，单任务场景）
            # 门控网络：输入特征的全局统计 → 专家权重（无任务ID）
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_features, gate_hidden_dim),  # 输入特征维度→隐藏维度
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, expert_num)    # 隐藏维度→专家权重
            )

            # 冻结预训练权重（只训练LoRA和门控）
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)  # 适配权重存储格式

    def reset_parameters(self):
        """重置参数:对齐原始LoRA的初始化逻辑"""
        # 重置预训练Linear层参数
        nn.Linear.reset_parameters(self)

        # 初始化MoELoRA的专家矩阵（和原始LoRA一致）
        if self.r > 0:
            # A矩阵：kaiming_uniform初始化（原始LoRA逻辑）
            for a in self.lora_A:
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            # B矩阵：全0初始化（原始LoRA逻辑）
            for b in self.lora_B:
                nn.init.zeros_(b)
            # 门控网络：xavier初始化
            for m in self.gate_mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        """数据驱动的门控权重计算（单任务场景核心）"""
        # 计算输入特征的全局统计（均值池化，适配任意维度输入）
        # x: [batch_size, ..., in_features] → [batch_size, in_features]
        x_pooled = x.mean(dim=tuple(range(1, len(x.shape)-1)))  # 除了batch和最后一维，其余维度求均值
        # 门控输出：[batch_size, expert_num] → Softmax归一化
        gate_logits = self.gate_mlp(x_pooled) / self.gate_temp
        gate_weights = F.softmax(gate_logits, dim=-1)
        return gate_weights
    def T(w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                self.unmerge()
                self.merged = False
        else:
            # 评估模式：合并权重（如果未合并）
            if self.merge_weights and not self.merged:
                self.merge()
                self.merged = True

    def merge(self):
        """合并MoELoRA权重到预训练权重(评估加速)"""
        if self.r == 0:
            return
        # 单任务场景：用默认均值权重合并（或随机选一个样本的权重）
        dummy_x = torch.randn(1, self.in_features, device=self.weight.device)
        gate_weights = self._get_gate_weights(dummy_x)  # [1, expert_num]
        # 合并每个专家的LoRA增量
        for e in range(self.expert_num):
            delta_W = T(self.lora_B[e] @ self.lora_A[e])  # [out_features, in_features]
            self.weight.data += delta_W * self.scaling * gate_weights[0, e]

    def unmerge(self):
        """解合并MoELoRA权重"""
        if self.r == 0:
            return
        dummy_x = torch.randn(1, self.in_features, device=self.weight.device)
        gate_weights = self._get_gate_weights(dummy_x)  # [1, expert_num]
        for e in range(self.expert_num):
            delta_W = T(self.lora_B[e] @ self.lora_A[e])  # [out_features, in_features]
            self.weight.data -= delta_W * self.scaling * gate_weights[0, e]
    
    def forward(self, x: torch.Tensor):
        """
        前向传播:单任务MoELoRA核心逻辑(无需task_id)
        Args:
            x: 输入特征 → [batch_size, ..., in_features]
        """
        # 1. 基础预训练Linear层输出
        result = F.linear(x, self._transpose_weight(self.weight), bias=self.bias)

        # 2. MoELoRA核心逻辑（仅当LoRA启用且未合并时生效）
        if self.r > 0 and not self.merged:
            # 2.1 LoRA Dropout（和原始LoRA一致）
            x_dropout = self.lora_dropout(x)  # [batch_size, ..., in_features]
            # 2.2 计算数据驱动的专家权重
            gate_weights = self._get_gate_weights(x)  # [batch_size, expert_num]
            # 2.3 展平输入维度（方便计算）
            x_shape = x_dropout.shape
            x_flat = x_dropout.reshape(-1, x_shape[-1])  # [total_tokens, in_features]
            gate_weights_flat = gate_weights.repeat_interleave(
                x_flat.shape[0] // gate_weights.shape[0], dim=0
            )  # [total_tokens, expert_num]

            

            # 2.4 遍历每个专家计算LoRA输出，再加权求和（MoELoRA核心）
            lora_output = 0.0
            for e in range(self.expert_num):
                # 专家A：x → [total_tokens, per_expert_r]
                a_out = x_flat @ self.lora_A[e].T  # A_e.T: [in_features, per_expert_r]
                # 专家B：a_out → [total_tokens, out_features]
                b_out = a_out @ self.lora_B[e].T  # B_e.T: [per_expert_r, out_features]
                # 乘以当前专家的权重 + LoRA缩放系数
                b_out = b_out * self.scaling * gate_weights_flat[:, e: e+1]  # [total_tokens, out_features]
                # 累加所有专家的输出
                lora_output += b_out

            # 2.5 恢复维度并叠加到基础输出
            lora_output = lora_output.reshape(*x_shape[:-1], self.out_features)  # [batch_size, ..., out_features]
            result += lora_output

        return result