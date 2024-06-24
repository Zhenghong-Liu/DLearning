"""
Math2Latex vae.py
2024年06月14日
by Zhenghong Liu

实现VAE网络，并生成minist图片
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


class VAE(nn.Module):
    def __init__(self, device):
        super(VAE, self).__init__()
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(64 * 7 * 7, 128)
        self.fc3 = nn.Linear(128, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        miu = self.fc1(x)
        rho = self.fc2(x)

        sigma = torch.log(1.0 + torch.exp(rho))
        epsilon = torch.randn_like(sigma)

        z = miu + sigma * epsilon
        z = self.fc3(z).view(-1, 64, 7, 7)

        x = self.decoder(z)
        # print(f"new img shape : {x.shape}")
        return x , self.kl_divergence(miu, sigma)

    def kl_divergence(self, miu, sigma):

        return 0.5 * torch.sum(
            2 * torch.log(1 / sigma) + sigma ** 2 + miu ** 2 - 1
        )

    def sample(self):
        z = torch.randn((1, 128), device=self.device)
        z = self.fc3(z).view(-1, 64, 7, 7)
        return self.decoder(z).detach().cpu().numpy().squeeze()

def loss_function(img, new_img, KL, beta=1):
    return F.mse_loss(img, new_img, reduction='sum') + beta * KL

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using device : {device}")

    # Hyper parameters
    num_epochs = 15
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)


    model = VAE(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            img = batch[0].to(device)
            # print(f"minist img shape : {img.shape}")
            new_img, KL = model(img)  # batch[0]是图像, batch[1]是label
            loss = loss_function(img, new_img, KL)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                fack_image = model.sample()
                plt.title(f"epoch = {epoch}")
                plt.imshow(fack_image, cmap="gray")
                plt.show()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    print("测试性能==================")
    model.eval()
    plt.figure(figsize=(15, 15))
    with torch.no_grad():
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(model.sample(), cmap="gray")
        plt.show()




