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
from torch.utils.data import Dataset, DataLoader

from BNN.异方差.DataSet import SineDataset
from mine_bnn import BayesNN, loss_function




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#apple device mac device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device : {device}")


# 1. 生成带噪声的sin函数数据
def generate_data():
    np.random.seed(32)
    x = np.linspace(0, 1 * 2 * np.pi, 300)
    y1 = 3 * np.sin(x)
    y1 = np.concatenate((np.zeros(60), y1 + np.random.normal(0, 0.15 * np.abs(y1), 300), np.zeros(60)))
    x = np.concatenate((np.linspace(-3, 0, 60), np.linspace(0, 3 * 2 * np.pi, 300),
                        np.linspace(3 * 2 * np.pi, 3 * 2 * np.pi + 3, 60)))
    y2 = 0.1 * x + 1
    y = y1 + y2
    return x, y



dataset = SineDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 4. 训练模型
model = BayesNN(1, 15, 1, device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000
beta = 0.1

model.train()
for epoch in range(num_epochs):
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

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
X, y = generate_data()
# X_test = np.linspace(-3*np.pi, 3*np.pi, 200)
X_test = np.linspace(-3, 3*2*np.pi + 3, 500)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device).unsqueeze(1)
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
X = (X - X.min()) / (X.max() - X.min())
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
X = X
y = y
X_test = X_test.cpu()
plt.scatter(X, y, label='Data')
plt.plot(X_test.numpy(), mean_prediction, label='Prediction', color='r')
plt.fill_between(X_test.numpy().squeeze(),
                 mean_prediction.squeeze() - 2 * std_prediction.squeeze(),
                 mean_prediction.squeeze() + 2 * std_prediction.squeeze(),
                 color='r', alpha=0.5, label='Uncertainty')
plt.legend()
plt.show()




