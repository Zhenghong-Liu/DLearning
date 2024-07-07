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
n_classes = 10 #类别数

img_shape = (channels, img_size, img_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes) #n_classes个类别，每个类别n_classes维的embedding

        def block(in_feature, out_feature, normalize=True):
            layers = [nn.Linear(in_feature, out_feature)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feature, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True)) #inplace=True的意思是进行原地操作，例如：x=x+5是对x的原地操作 y=x+5,x=y不是对x的原地操作
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False), #输入是噪声和标签embedding
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))), #np.prod计算数组中所有元素的乘积
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes) #n_classes个类别，每个类别n_classes维的embedding

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        img = img.view(img.size(0), -1)
        label_embedding = self.label_embedding(labels)
        d_in = torch.cat((img, label_embedding), -1)
        validity = self.model(d_in)
        return validity


adversarial_loss = nn.MSELoss() #这里的分类器没有使用sigmoid激活函数，所以使用MSE损失函数，是一个回归问题

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


def sample_image(n_row, file_name):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.randn(n_row * n_classes, latent_dim).to(device)
    labels = torch.LongTensor([num for _ in range(n_row) for num in range(n_classes)]).to(device) #batch_size=n_row*n_classes
    gen_imgs = generator(z, labels)
    torchvision.utils.save_image(gen_imgs.data, file_name, nrow=n_classes, normalize=True)


#训练
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        # Adversarial ground truths
        # valid = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).to(device) #真实图片的标签
        # fake = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(device) #生成图片的标签
        valid = torch.FloatTensor(imgs.size(0), 1).fill_(1.0).to(device) #真实图片的标签
        fake = torch.FloatTensor(imgs.size(0), 1).fill_(0.0).to(device) #生成图片的标签


        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # -----------------
        #  训练 Generator
        # -----------------
        optimizer_G.zero_grad()
        #采样噪声和标签作为生成器的输入
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_labels = torch.randint(0, n_classes, (imgs.size(0),)).to(device)
        #使用生成器生成图片
        gen_imgs = generator(z, gen_labels)

        # loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  训练 Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # 让判别器尽可能把真实图片判别为真实图片，把生成的图片判别为生成图片
        validity_real = discriminator(real_imgs, labels)
        real_loss = adversarial_loss(validity_real, valid)

        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        fake_loss = adversarial_loss(validity_fake, fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
    )

    sample_image(n_row=8, file_name=f'images/{epoch}.png')