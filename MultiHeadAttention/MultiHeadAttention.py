"""
Math2Latex MultiHeadAttention.py.py
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

class multi_head_attention(nn.Module):

    def __init__(self, d_model, n_head):
        super(multi_head_attention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.wq(q), self.wk(k), self.wv(v)

        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        # score = q @ k.permute(0, 1, 3, 2) / math.sqrt(n_d) #两种方式都可以实现转置。
        score = q @ k.transpose(2, 3) / math.sqrt(n_d)

        if mask is not None:
            #mask attention
            # mask = torch.tril(torch.ones(time, time, dtype=bool)) #生成下三角矩阵
            score = score.masked_fill(mask == 0, float('-inf')) #将mask为0的位置设置为负无穷, 使得softmax后为0
        score = self.softmax(score) @ v #注意softmax的位置
        # 执行contigous()后，score的内存是连续的，才能进行view操作
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, self.d_model) #合并多头,contiguous()保证内存连续
        output = self.combine(score)
        return output

attention = multi_head_attention(d_model, n_head)
output = attention(X, X, X)
print(output.shape, '\n', output)

