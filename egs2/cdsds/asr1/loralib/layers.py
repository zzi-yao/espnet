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


class MoELoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,          
        out_features: int,         
        r: int = 0,                
        lora_alpha: int = 1,       
        lora_dropout: float = 0.,  
        expert_num: int = 4,       
        gate_temp: float = 6.0,    
        top_k: int = 2,  # 新增：默认k=2
        fan_in_fan_out: bool = False,  
        merge_weights: bool = True,    
        load_balance_coeff: float = 0.005,   #0.01
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.expert_num = expert_num
        self.gate_temp = gate_temp
        # 新增：保存top_k参数
        self.top_k = top_k
        self.load_balance_coeff = load_balance_coeff
        self.load_balance_loss = torch.tensor(0.0, device=self.weight.device)

        self.per_expert_r = r // expert_num if (r > 0 and expert_num > 0) else 0
        if r > 0:
            assert r % expert_num == 0
            self.scaling = self.lora_alpha / self.r  
            self.lora_A = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((self.per_expert_r, in_features)))
                for _ in range(self.expert_num)
            ])
            self.lora_B = nn.ParameterList([
                nn.Parameter(self.weight.new_zeros((out_features, self.per_expert_r)))
                for _ in range(self.expert_num)
            ])
            # self.gate_mlp = nn.Sequential(
            #     nn.Linear(in_features, self.gate_hidden_dim),  
            #     nn.ReLU(),
            #     nn.Dropout(0.1),  # 新增dropout
            #     nn.Linear(self.gate_hidden_dim, expert_num)    
            # )
            # 优化后（小样本专用）
            # self.gate_mlp = nn.Sequential(
            #     nn.Linear(in_features, self.expert_num, bias=False),  # 直接映射，无隐藏层
            #     nn.Softmax(dim=-1)
            # )g
            self.gate_linear = nn.Linear(in_features, self.expert_num)  # 仅一层线性层
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)  

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'r') and self.r > 0:
            if hasattr(self, 'lora_A') and hasattr(self, 'lora_B'):   # and hasattr(self, 'gate_mlp'):
                # for a in self.lora_A:
                #     nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                # for b in self.lora_B:
                #     nn.init.zeros_(b)
                # for e in range(self.expert_num):
                #     # 不同专家用不同初始化种子
                #     torch.manual_seed(42 + e)
                #     nn.init.kaiming_uniform_(self.lora_A[e], a=math.sqrt(5) * (e+1))
                #     nn.init.zeros_(self.lora_B[e])
                # torch.manual_seed(torch.initial_seed())  # 恢复随机种子
                base_coeff = math.sqrt(5) * 0.1  # 匹配语音特征±10的量级
                noise_level = 1e-3                # 匹配差分特征±1的量级
                param_clip = 0.05                # 语音特征的合理参数范围
                for e in range(self.expert_num):
                    seed = 42 + e * 100
                    torch.manual_seed(seed)
                    if self.expert_num == 2:
                        if e == 0:
                            nn.init.kaiming_uniform_(self.lora_A[e], a=base_coeff * 2)  # 稍大系数，增强鲁棒性
                            nn.init.zeros_(self.lora_B[e])  # 无偏移，保证稳定性
                        else:
                            nn.init.kaiming_normal_(self.lora_A[e], a=base_coeff * 0.5)  # 小系数，聚焦细节
                            noise = torch.randn_like(self.lora_A[e]) * noise_level
                            self.lora_A[e].data = self.lora_A[e].data + noise
                            nn.init.normal_(self.lora_B[e], mean=0, std=noise_level / 10)  # 更小的B层噪声      
                    elif self.expert_num == 4:
                        if e == 0:
                            nn.init.kaiming_uniform_(self.lora_A[e], a=base_coeff * 2)
                            nn.init.zeros_(self.lora_B[e])
                        elif e == 1:
                            nn.init.kaiming_normal_(self.lora_A[e], a=base_coeff * 0.5)
                            noise = torch.randn_like(self.lora_A[e]) * noise_level
                            self.lora_A[e].data = self.lora_A[e].data + noise
                            nn.init.normal_(self.lora_B[e], mean=0, std=noise_level / 10)
                        elif e == 2:
                            nn.init.orthogonal_(self.lora_A[e], gain=0.1)  # 正交适配时序连续性
                            nn.init.uniform_(self.lora_B[e], a=-noise_level/5, b=noise_level/5)
                        else:
                            nn.init.constant_(self.lora_A[e], val=0.01 * (e+1))  # 固定常数偏移
                            noise = torch.randn_like(self.lora_A[e]) * noise_level * 2
                            self.lora_A[e].data = self.lora_A[e].data + noise
                            nn.init.constant_(self.lora_B[e], val=noise_level / 20)
                    self.lora_A[e].data = self.lora_A[e].data.clamp(-param_clip, param_clip)
                torch.manual_seed(torch.initial_seed())
                # for m in self.gate_mlp:
                #     if isinstance(m, nn.Linear):
                #         nn.init.xavier_uniform_(m.weight, gain=0.1)  # 降低初始化增益
                #         if m.bias is not None:
                #             nn.init.zeros_(m.bias)

    def _get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        x_reshaped = x.reshape(-1, x.shape[-1])  # [B*L, D] → 展平序列维度，便于gate_mlp计算
        # gate_weights = self.gate_mlp(x_reshaped)  # gate_mlp: D → gate_hidden_dim → expert_num
        gate_weights = self.gate_linear(x_reshaped)
        gate_weights = gate_weights.reshape(*x.shape[:-1], self.expert_num)  # 核心：匹配输入的前N维
        gate_weights = torch.softmax(gate_weights / self.gate_temp, dim=-1)
        return gate_weights
    
    def train(self, mode: bool = True):
        def T(w): 
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r == 0:
                    return
                dummy_x = torch.randn(1, self.in_features, device=self.weight.device)
                gate_weights = self._get_gate_weights(dummy_x)  
                for e in range(self.expert_num):
                    delta_W = T(self.lora_B[e] @ self.lora_A[e]) 
                    self.weight.data -= delta_W * self.scaling * gate_weights[0, e]
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r == 0:
                    return
                dummy_x = torch.randn(1, self.in_features, device=self.weight.device)
                gate_weights = self._get_gate_weights(dummy_x)  
                for e in range(self.expert_num):
                    delta_W = T(self.lora_B[e] @ self.lora_A[e])  
                    self.weight.data += delta_W * self.scaling * gate_weights[0, e]
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0 and not self.merged:
            x_dropout = self.lora_dropout(x)  
            gate_weights = self._get_gate_weights(x_dropout)
            x_shape = x_dropout.shape
            x_flat = x_dropout.reshape(-1, x_shape[-1])  
            gate_weights_flat = gate_weights.reshape(-1, self.expert_num)
            # a. 筛选前k个专家的权重和索引
            topk_vals, topk_indices = torch.topk(
                gate_weights_flat, 
                k=self.top_k, 
                dim=-1,  # 沿专家维度筛选
                largest=True  # 选权重最大的k个
            )
            # b. 构建掩码：仅保留前k个专家的权重，其余置0
            gate_weights_masked = torch.zeros_like(gate_weights_flat)
            # scatter_：将topk_vals填充到topk_indices对应的位置
            gate_weights_masked.scatter_(
                dim=-1, 
                index=topk_indices, 
                src=topk_vals
            )
            # c. 替换原有gate_weights_flat为掩码后的版本
            gate_weights_flat = gate_weights_masked
            
            
            # 5. 计算负载均衡损失
            expert_selected = (gate_weights_masked > 0).float().sum(dim=0)
            if expert_selected.sum() > 0:
                # expert_selected_ratio = expert_selected / expert_selected.sum()
                # target_ratio = torch.ones_like(expert_selected_ratio) / self.expert_num
                # self.load_balance_loss = F.mse_loss(expert_selected_ratio, target_ratio)
                expert_selected_ratio = expert_selected / (expert_selected.sum() + 1e-8)  # 分母防0
                expert_selected_ratio = expert_selected_ratio.clamp(min=1e-8)  # 分子防0，避免log(0)
                target_ratio = torch.ones_like(expert_selected_ratio) / self.expert_num
                target_ratio = target_ratio.clamp(min=1e-8)  # 目标分布也防0（兜底）
                # self.load_balance_loss = F.kl_div(
                #     expert_selected_ratio.log(),  # 输入1：对数概率
                #     target_ratio,                # 输入2：目标概率
                #     reduction="mean",            # 替换batchmean→mean（无批次维度）
                #     log_target=False             # 显式声明target不是对数概率（默认False，可省略）
                # )
                ratio_loss = F.kl_div(
                    expert_selected_ratio.log(),  # 输入1：对数概率
                    target_ratio,                # 输入2：目标概率
                    reduction="mean",            # 替换batchmean→mean（无批次维度）
                    log_target=False             # 显式声明target不是对数概率（默认False，可省略）
                )
                # # 新增：专家输出差异损失（鼓励专家学不同特征）
                # expert_outputs = []
                # for e in range(self.expert_num):
                #     a_out = x_flat @ self.lora_A[e].T
                #     b_out = a_out @ self.lora_B[e].T
                #     expert_outputs.append(b_out)
                # expert_outputs = torch.stack(expert_outputs, dim=1)  # [B*L, expert_num, out_features]
                # # 计算专家间的余弦相似度（越小越好）
                # sim_matrix = F.cosine_similarity(expert_outputs.unsqueeze(1), expert_outputs.unsqueeze(2), dim=-1)
                # diversity_loss = (sim_matrix.sum() - self.expert_num) / (self.expert_num * (self.expert_num - 1))
                # # 总负载均衡损失：比例均衡（低权重） + 特征多样性（高权重）
                # self.load_balance_loss = 0.1 * ratio_loss + 0.9 * diversity_loss
                diversity_loss = 0.0
                count = 0

                # for e1 in range(self.expert_num):
                #     for e2 in range(e1+1, self.expert_num):  
                #         a_out1 = x_flat @ self.lora_A[e1].T
                #         b_out1 = a_out1 @ self.lora_B[e1].T
                #         a_out2 = x_flat @ self.lora_A[e2].T
                #         b_out2 = a_out2 @ self.lora_B[e2].T
                #         sim = F.cosine_similarity(b_out1, b_out2, dim=-1).mean()
                #         diversity_loss += sim
                #         count += 1
                # diversity_loss = diversity_loss / count if count > 0 else 0.0
                # diversity_loss = diversity_loss.clamp(min=0.0, max=1.0)
                with torch.no_grad():
                    expert_outs = []
                    for e in range(self.expert_num):
                        a_out = x_flat @ self.lora_A[e].T
                        b_out = a_out @ self.lora_B[e].T
                        b_out_mean = b_out.mean(dim=0)  # 仅保留特征维度均值
                        expert_outs.append(b_out_mean)
                        del a_out, b_out
                        torch.cuda.empty_cache()  # 清理显存碎片
                    for e1 in range(self.expert_num):
                        for e2 in range(e1+1, self.expert_num):
                            sim = F.cosine_similarity(expert_outs[e1], expert_outs[e2], dim=-1).item()  # 标量输出
                            diversity_loss += sim
                            count += 1
                diversity_loss = diversity_loss / count if count > 0 else 0.0
                diversity_loss = torch.tensor(diversity_loss, device=x.device, dtype=torch.float32).clamp(min=0.0, max=1.0)
                self.load_balance_loss = 0.01 * ratio_loss + 0.99 * diversity_loss  
            else:
                self.load_balance_loss = torch.tensor(0.0, device=x.device)
            
            explore_prob = 0.1 #0.01  
            explore_mask = (torch.rand_like(gate_weights_flat[:,0]) < explore_prob).unsqueeze(-1)
            if explore_mask.any():  
                # random_indices = torch.randint(
                #     0, self.expert_num, 
                #     size=gate_weights_flat.shape[:-1],  
                #     device=x.device
                # )
                expert_selected_ratio = expert_selected / (expert_selected.sum() + 1e-8)
                explore_probs = 1 - expert_selected_ratio
                explore_probs = explore_probs / (explore_probs.sum() + 1e-8)
                explore_probs_expanded = explore_probs.unsqueeze(0).repeat(gate_weights_flat.shape[0], 1)
                mask_squeezed = explore_mask.squeeze(-1)
                random_indices = torch.multinomial(
                    explore_probs_expanded[mask_squeezed],
                    num_samples=1,
                    replacement=True,
                )
                random_indices_full = torch.zeros( (gate_weights_flat.shape[0], 1), dtype=torch.long, device=x.device)
                random_indices_full[mask_squeezed, :] = random_indices
                
                random_vals = torch.zeros_like(gate_weights_flat)
                random_vals.scatter_(
                    dim=-1,  # 在最后一维（专家维度）填充
                    index=random_indices_full,  # [B*L, 1]（2D）
                    src=torch.ones_like(random_indices_full, dtype=random_vals.dtype)  # 权重设为1，维度匹配
                )
                gate_weights_flat = torch.where(explore_mask, random_vals, gate_weights_flat)

            lora_output = torch.zeros(x_flat.shape[0], self.out_features, device=x_flat.device, dtype=x_flat.dtype)
            for e in range(self.expert_num):
                a_out = x_flat @ self.lora_A[e].T  
                b_out = a_out @ self.lora_B[e].T  
                b_out = b_out * self.scaling * gate_weights_flat[:, e: e+1]  
                lora_output += b_out
            lora_output = lora_output.reshape(*x_shape[:-1], self.out_features) 
            result += lora_output
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
    def get_load_balance_loss(self) -> torch.Tensor:
        return self.load_balance_loss * self.load_balance_coeff
