import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from BNN.异方差.DataSet import SineDataset
from MLP import MLP, CustomLoss
# from BayesNN import BayesNN

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device : {device}")
loss_fn = CustomLoss().to(device)
model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(1000):
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_mean, y_log_sigma = model(x)
        loss = loss_fn(y_mean, y_log_sigma, y)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')


model.eval()
x,y = dataset.x, dataset.y
x = (x - x.min()) / (x.max() - x.min())
x = x.to(device)
with torch.no_grad():
    y_mean, y_log_sigma = model(x)
    y_pred = y_mean.cpu().numpy()
    y_sigma = torch.exp(y_log_sigma).cpu().numpy()

plt.figure(figsize=(10, 6))

x = x.cpu().numpy()
plt.scatter(x, y, s=5, label='True')
plt.plot(x, y_pred, label='Predict')
print(f"x shape : {x.shape}, y_pred shape : {y_pred.shape}, y_sigma shape : {y_sigma.shape}")
x = x.reshape(-1)
y_pred = y_pred.reshape(-1)
y_sigma = y_sigma.reshape(-1)
plt.fill_between(x, y_pred - y_sigma, y_pred + y_sigma, alpha=0.8, label="1 sigma", color='orange')
plt.fill_between(x, y_pred - 2 * y_sigma, y_pred + 2 * y_sigma, alpha=0.2, label="2 sigma", color='blue')
plt.legend()
plt.show()


