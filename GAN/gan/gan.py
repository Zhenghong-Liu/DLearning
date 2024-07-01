"""
Math2Latex gan.py
2024年07月02日
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
from torch.autograd import Variable #variable可以反向传播。不过已经不需要了，只需要把requires_grad属性设置成True就可以了。

import os

# os.mkdir('images', exist_ok=True)
if not os.path.exists('images'):
    os.makedirs('images')

# 定义超参数
num_epochs = 200
batch_size = 64
learning_rate = 0.0002
img_size = 28
channels = 1
latent_dim = 100
sample_interval = 400

img_shape = (channels, img_size, img_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feature, out_feature, normalize=True):
            layers = [nn.Linear(in_feature, out_feature)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feature, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True)) #inplace=True的意思是进行原地操作，例如：x=x+5是对x的原地操作 y=x+5,x=y不是对x的原地操作
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))), #np.prod计算数组中所有元素的乘积
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size()[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.model(x)
        return validity


adversarial_loss = nn.BCELoss() #二进制交叉熵损失函数

generator = Generator()
discriminator = Discriminator()

generator.to(device)
discriminator.to(device)

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


#优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))


#训练
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        # valid = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).to(device) #真实图片的标签
        # fake = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(device) #生成图片的标签
        valid = torch.FloatTensor(imgs.size(0), 1).fill_(1.0).to(device) #真实图片的标签
        fake = torch.FloatTensor(imgs.size(0), 1).fill_(0.0).to(device) #生成图片的标签


        real_imgs = imgs.to(device)

        # -----------------
        #  训练 Generator
        # -----------------
        optimizer_G.zero_grad()
        #使用正态分布生成噪声，然后生成图片
        z = torch.FloatTensor(np.random.normal(0, 1, (imgs.size(0), latent_dim))).to(device)
        gen_imgs = generator(z)
        # 让生成器尽可能生成真实图片，即让判别器尽量把生成的图片判别为真实图片
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  训练 Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # 让判别器尽可能把真实图片判别为真实图片，把生成的图片判别为生成图片
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batch_done = epoch * len(dataloader) + i
        if batch_done % sample_interval == 0:
            torchvision.utils.save_image(gen_imgs.data[:25], "images/%d.png" % batch_done, nrow=5, normalize=True)