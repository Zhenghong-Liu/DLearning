"""
Math2Latex conditionalVAE.py
2024年07月07日
by Zhenghong Liu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets

import os
if not os.path.exists('images'):
    os.makedirs('images')


# 定义超参数
num_epochs = 200
batch_size = 64
learning_rate = 0.001
img_size = 28
channels = 1
latent_dim = 100
n_classes = 10 #类别数

img_shape = (channels, img_size, img_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")


dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape) + n_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = torch.randn(mu.size(0), latent_dim).to(device)
        z = mu + std * sampled_z
        return z

    def KL_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, img, label_embedding):
        img = img.view(img.size(0), -1) #保留batch_size，将后面的维度展平合并为一个维度
        x = torch.cat((img, label_embedding), -1)
        x = self.model(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, self.KL_divergence(mu, logvar)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, np.prod(img_shape)),
            nn.Tanh()
        )

    def forward(self, z, label_embedding):

        x = torch.cat((z, label_embedding), -1)
        img = self.model(x)
        img = img.view(img.size(0), *img_shape)
        return img


class ConditionalVAE(nn.Module):
    def __init__(self):
        super(ConditionalVAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.label_emb = nn.Embedding(n_classes, n_classes) #n_classes个类别，每个类别n_classes维的embedding


    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        z, KL_div = self.encoder(img, label_embedding)
        img_recon = self.decoder(z, label_embedding)
        return img_recon, KL_div

    def sample(self, z, labels):
        label_embedding = self.label_emb(labels)
        return self.decoder(z, label_embedding)


# 定义模型、损失函数和优化器
cvae_model = ConditionalVAE().to(device)
optimizer = torch.optim.Adam(cvae_model.parameters(), lr=learning_rate)

# 损失函数
def loss_fn(img, img_recon, KL_div, beta=1):
    BCE = F.mse_loss(img, img_recon, reduction='sum')
    return BCE + beta * KL_div

def sample_image(n_row, file_name):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.randn(n_row * n_classes, latent_dim).to(device)
    labels = torch.LongTensor([num for _ in range(n_row) for num in range(n_classes)]).to(device) #batch_size=n_row*n_classes
    gen_imgs = cvae_model.sample(z, labels)
    torchvision.utils.save_image(gen_imgs.data, file_name, nrow=n_classes, normalize=True)

# 训练模型
for epoch in range(num_epochs):
    cvae_model.train()
    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        img_recon, KL_div = cvae_model(imgs, labels)
        loss = loss_fn(imgs, img_recon, KL_div, beta=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}"
    )
    cvae_model.eval()
    with torch.no_grad():
        sample_image(n_row=8, file_name=f"images/epoch{epoch}.png")