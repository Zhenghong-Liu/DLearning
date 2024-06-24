"""
Math2Latex Transformer.py
2024年06月11日
by Zhenghong Liu
"""
import torch
from torch import nn
import torch.functional as F
import math

from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, max_len, d_model, n_heads, ffn_hidden,n_layers, drop_prob, device):
        super(Transformer, self).__init__()

        self.encoder = Encoder(enc_voc_size, max_len, d_model, n_layers, n_heads, ffn_hidden, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, n_layers, n_heads, ffn_hidden, drop_prob, device)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_padding_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        # (batch, Time, len_q, len_k)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3) #ne: not equal, (batch, len_q) -> (batch, 1, len_q, 1)
        q = q.repeat(1, 1, 1, len_k) # (batch, 1, len_q, 1) -> (batch, 1, len_q, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2) #ne: not equal, (batch, len_k) -> (batch, 1, 1, len_k)
        k = k.repeat(1, 1, len_q, 1) # (batch, 1, 1, len_k) -> (batch, 1, len_q, len_k)

        mask = q & k #只要有一个是pad_idx，就是要mask的
        return mask

    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask

    def forward(self, src, trg):
        src_mask = self.make_padding_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_padding_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)
        causal_mask = self.make_casual_mask(trg, trg)
        trg_mask = trg_mask & causal_mask
        src_trg_mask = self.make_padding_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, src_mask)
        output = self.decoder(trg, enc, trg_mask, src_trg_mask)
        return output

if __name__ == '__main__':
    device = torch.device('cpu')
    model = Transformer(0, 0, 10, 10, 10, 512, 8, 2048, 6, 0.1, device)
    src = torch.randint(0, 10, (2, 10)).to(device)
    trg = torch.randint(0, 10, (2, 10)).to(device)
    print(model(src, trg).shape)

