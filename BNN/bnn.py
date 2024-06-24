"""
Math2Latex bnn.py
2024年06月13日
by Zhenghong Liu

实现一个拟合正弦函数的贝叶斯神经网络
参考方法：bayes by backprop
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class BayesLayer(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0, prior_sigma=1):
        super(BayesLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_rho, -3)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3)

    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        epsilon_weight = torch.randn_like(self.weight_mu)
        epsilon_bias = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias
        return F.linear(x, weight, bias), self.kl_divergence()

    def kl_divergence(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        kl = 0.5 * (self.prior_sigma ** 2 / weight_sigma ** 2 + (self.weight_mu - self.prior_mu) ** 2 / self.prior_sigma ** 2 - 1 + torch.log(weight_sigma ** 2 / self.prior_sigma ** 2))
        kl += 0.5 * (self.prior_sigma ** 2 / bias_sigma ** 2 + (self.bias_mu - self.prior_mu) ** 2 / self.prior_sigma ** 2 - 1 + torch.log(bias_sigma ** 2 / self.prior_sigma ** 2))
        return kl.sum()

    def sample(self, x):
        weight = torch.randn_like(self.weight_mu) * torch.log1p(torch.exp(self.weight_rho)) + self.weight_mu
        bias = torch.randn_like(self.bias_mu) * torch.log1p(torch.exp(self.bias_rho)) + self.bias_mu
        return F.linear(x, weight, bias)


class BayesianNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, prior_mu=0, prior_sigma=1):
        super(BayesianNN, self).__init__()
        self.layer1 = BayesLayer(in_features, hidden_features, prior_mu, prior_sigma)
        self.layer2 = BayesLayer(hidden_features, hidden_features, prior_mu, prior_sigma)
        self.layer3 = BayesLayer(hidden_features, out_features, prior_mu, prior_sigma)

    def forward(self, x):
        x, kl = self.layer1(x)
        x = F.relu(x)
        x, kl1 = self.layer2(x)
        x = F.relu(x)
        x, kl2 = self.layer3(x)
        return x, kl + kl1 + kl2

    def kl_divergence(self):
        return self.layer1.kl_divergence() + self.layer2.kl_divergence() + self.layer3.kl_divergence()

    def sample(self, x):
        x = F.relu(self.layer1.sample(x))
        x = F.relu(self.layer2.sample(x))
        return self.layer3.sample(x)


def train(model, optimizer, criterion, x, y, kl_weight=1e-2):
    optimizer.zero_grad()
    y_pred, kl = model(x)
    loss = criterion(y_pred, y) + kl_weight * kl
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, criterion, x, y):
    with torch.no_grad():
        y_pred, _ = model(x)
        loss = criterion(y_pred, y)
    return loss.item()


def main():
    # 超参数
    in_features = 1
    hidden_features = 100
    out_features = 1
    prior_mu = 0
    prior_sigma = 1
    lr = 1e-3
    epochs = 1000
    batch_size = 32
    kl_weight = 1e-2

    # 数据
    x = torch.linspace(-math.pi, math.pi, 1000).view(-1, 1)
    y = torch.sin(x)
    y += 0.1 * torch.randn_like(y)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型、优化器、损失函数
    model = BayesianNN(in_features, hidden_features, out_features, prior_mu, prior_sigma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            loss = train(model, optimizer, criterion, x_batch, y_batch, kl_weight)
        if epoch % 100 == 0:
            model.eval()
            loss = test(model, criterion, x, y)
            print(f'Epoch: {epoch}, Loss: {loss:.6f}')

    # 测试
    model.eval()
    y_pred, _ = model(x)
    plt.plot(x.numpy(), y.numpy(), label='True')
    plt.plot(x.numpy(), y_pred.numpy(), label='Predict')
    plt.legend()
    plt.show()