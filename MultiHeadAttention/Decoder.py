"""
Math2Latex Decoder.py
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

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, hidden, drop_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.attention1 = multi_head_attention(d_model, n_head)
        self.drop1 = nn.Dropout(drop_prob)
        self.norm1 = LayerNorm(d_model)

        self.cross_attention = multi_head_attention(d_model, n_head)
        self.drop2 = nn.Dropout(drop_prob)
        self.norm2 = LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, hidden, drop_prob)
        self.drop3 = nn.Dropout(drop_prob)
        self.norm3 = LayerNorm(d_model)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec #备份，用于残差链接
        x = self.attention1(dec, dec, dec, t_mask) #下三角掩码
        x = self.drop1(x)
        x = self.norm1(x + _x) #add & norm

        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc, s_mask) #填充掩码
            x = self.drop2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.drop3(x)
        x = self.norm3(x + _x) #add & norm
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_layers, n_head, hidden, drop_prob, device):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, hidden, drop_prob) for _ in range(n_layers)]
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, dec, enc, t_mask, s_mask):
        x = self.embedding(dec)
        for layer in self.layers:
            x = layer(x, enc, t_mask, s_mask)
        x = self.fc(x)
        return x
