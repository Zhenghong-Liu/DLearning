"""
Math2Latex Embedding.py
2024年06月10日
by Zhenghong Liu
"""
import torch
from torch import nn
import torch.functional as F
import math

# 设置CPU的随机种子
torch.manual_seed(0)

# 如果使用CUDA，也设置CUDA的随机种子
# torch.cuda.manual_seed(0)

#生成测试的数据
X = torch.randn(8, 64, 512) #batch, time, dimension
# print(X.shape)

d_model = 512 #映射到qkv空间中的维度
n_head = 8 #头的数量

#=======Token Embedding========

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


#=======Position Embedding========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, device = torch.device('cpu')):
        super(PositionalEncoding, self).__init__()
        self.embedding = torch.zeros(max_len, d_model, device=device)
        #这个编码创建之后是不需要梯度的
        self.embedding.requires_grad_(False)

        pos = torch.arange(0, max_len, device=device) #生成从0到max_len-1的序列
        print(f'pos -1 = {pos[-1]}')
        print(f'pos shape = {pos.shape}')
        pos = pos.float().unsqueeze(1) #增加一个维度
        print(f'pos shape = {pos.shape}')
        _2i = torch.arange(0, d_model, 2, device=device).float() #生成0,2,4,6,...,d_model-2

        #首先计算偶数信息
        self.embedding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        #然后计算奇数信息
        self.embedding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))

    def forward(self, x):
        seq_len = x.shape[1] #x : batch, time, dimension
        return self.embedding[:seq_len, :]

#=====total Embedding========
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512, drop_prob = 0.1, device=torch.device('cpu')):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(x)
        return self.dropout(tok_emb + pos_emb)

#=======测试========
pos_embedding = PositionalEncoding(d_model, 512)

pos = pos_embedding(X)
print(pos.shape)