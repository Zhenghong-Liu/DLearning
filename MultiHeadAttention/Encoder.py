"""
Math2Latex Encoder.py
2024年06月11日
by Zhenghong Liu
"""
import torch
from torch import nn
import torch.functional as F
import math

from MultiHeadAttention import multi_head_attention
from FeedForwardNetwork import PositionwiseFeedForward
from LayerNormalization import LayerNorm
from Embedding import TransformerEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, hidden, drop_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = multi_head_attention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask = None):
        _x = x #因为需要用到残差链接，因此做一个备份
        x = self.attention(x, x, x, mask)
        x = self.drop1(x)
        x = self.norm1(x + _x) #add & norm

        _x = x
        x = self.ffn(x)
        x = self.drop2(x)
        x = self.norm2(x + _x) #add & norm
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_layers, n_head, hidden, drop_prob, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, hidden, drop_prob) for _ in range(n_layers)]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x
