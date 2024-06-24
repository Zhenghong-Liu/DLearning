"""
Math2Latex LayerNormalization.py
2024年06月11日
by Zhenghong Liu
"""
import torch
from torch import nn
import torch.functional as F
import math

#Batch Normalization是对于同一个批次中，相同通道的数据进行归一化，
#而Layer Normalization是对于同一个通道的数据进行归一化

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) #gamma初始全1
        self.beta = nn.Parameter(torch.zeros(d_model)) #beta初始全0
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
