"""
Math2Latex mine_bnn.py
2024年06月13日
by Zhenghong Liu
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class BayesLayer(nn.Module):
    def __init__(self, input_feature, output_feature, device):
        super(BayesLayer, self).__init__()

        self.weight_mean = nn.Parameter(torch.randn(output_feature, input_feature).clamp_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.randn(output_feature, input_feature).clamp_(-5, -4))

        self.bias_mean = nn.Parameter(torch.randn(output_feature).clamp_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.randn(output_feature).clamp_(-5, -4))

        self.device = device


    def forward(self, x):
        '''
        x : [batch, dim]  dim == input_feature
        '''
        if True or self.training:
            weight_sigma = torch.log(1.0 + torch.exp(self.weight_rho))
            epsilon = torch.randn_like(weight_sigma, device=self.device)
            weight = self.weight_mean + weight_sigma * epsilon

            bias_sigma = torch.log(1.0 + torch.exp(self.bias_rho))
            epsilon = torch.randn_like(bias_sigma,device=self.device)
            bias = self.bias_mean + bias_sigma * epsilon
        else:
            weight = self.weight_mean
            bias = self.bias_mean
        # print(f"x :{x.device}, weight :{weight.device}, bias :{bias.device}")
        # print(f"x :{x.shape}, weight :{weight.shape}, bias :{bias.shape}")
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        weight_sigma = torch.log1p_(torch.exp(self.weight_rho))
        KL = 0.5 * torch.sum(
            2 * torch.log( 1.0 / weight_sigma) +  weight_sigma **2 + self.weight_mean ** 2 - 1
        )
        bias_sigma = torch.log1p_(torch.exp(self.bias_rho))
        KL += 0.5 * torch.sum(
            2 * torch.log(1.0 / bias_sigma) + bias_sigma ** 2 + self.bias_mean ** 2 - 1
        )
        return KL


class BayesNN(nn.Module):
    def __init__(self, input_channel, hidden_channel, ouput_channel, device):
        super(BayesNN, self).__init__()
        self.bLinear1 = BayesLayer(input_channel, hidden_channel, device)
        self.bLinear2 = BayesLayer(hidden_channel, hidden_channel, device)
        self.bLinear3 = BayesLayer(hidden_channel, hidden_channel, device)
        self.bLinear4 = BayesLayer(hidden_channel, ouput_channel, device)

    def forward(self, x):
        x = F.relu(self.bLinear1(x))
        x = F.relu(self.bLinear2(x))
        x = F.relu(self.bLinear3(x))
        x = self.bLinear4(x)
        return x

    def kl_divergence(self):
        kl = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kl += m.kl_divergence()
        return kl


# 3. 定义损失函数
def loss_function(y_pred, y_true, kl_divergence, beta=1.0):
    mse_loss = torch.nn.functional.mse_loss(y_pred, y_true, reduction='sum')
    return mse_loss + beta * kl_divergence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device : {device}")


# 1. 生成带噪声的sin函数数据
def generate_data(n=100):
    X = np.linspace(-2*np.pi, 2*np.pi, n)
    y = np.sin(X) + 0.1 * np.random.normal(size=X.shape)
    return X, y

X, y = generate_data(1000)
X = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(1)
y = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)


# 4. 训练模型
model = BayesNN(1, 15, 1, device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 3000
beta = 0.1

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X)
    kl_div = model.kl_divergence()
    loss = loss_function(y_pred, y, kl_div, beta)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 评估模型和计算不确定性
model.eval()
num_samples = 50
predictions = []

# 5. 生成测试数据
X_test = np.linspace(-3*np.pi, 3*np.pi, 200)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device).unsqueeze(1)
print(f"x_test shape : {X_test.shape}")

with torch.no_grad():
    for _ in range(num_samples):
        predictions.append(model(X_test).detach().cpu().numpy())

predictions = np.array(predictions)
mean_prediction = predictions.mean(axis=0)
std_prediction = predictions.std(axis=0)
print(predictions.shape)
print(mean_prediction.shape)
print(std_prediction.shape)

# 可视化结果
plt.figure(figsize=(10, 6))
X = X.cpu()
y = y.cpu()
X_test = X_test.cpu()
plt.scatter(X.numpy(), y.numpy(), label='Data')
plt.plot(X_test.numpy(), mean_prediction, label='Prediction', color='r')
plt.fill_between(X_test.numpy().squeeze(),
                 mean_prediction.squeeze() - 2 * std_prediction.squeeze(),
                 mean_prediction.squeeze() + 2 * std_prediction.squeeze(),
                 color='r', alpha=0.5, label='Uncertainty')
plt.legend()
plt.show()




