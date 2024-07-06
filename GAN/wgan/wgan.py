import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets

import os

# os.mkdir('images', exist_ok=True)
if not os.path.exists('images'):
    os.makedirs('images')


# 定义超参数
num_epochs = 200
batch_size = 64
learning_rate = 0.00005
img_size = 28
channels = 1
latent_dim = 100
sample_interval = 400

clip_value = 0.01 # WGAN的一个重要改进，对权重进行截断
n_critic = 5 # WGAN的一个重要参数，训练5次判别器，训练1次生成器

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
            nn.Linear(256, 1), #与GAN相比，这里不需要使用sigmoid激活函数
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.model(x)
        return validity


# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

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

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=learning_rate) #与GAN相比，这里使用RMSprop优化器，或者SGD也可以，但是不要使用Adam
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate)



# ---------- Training ----------

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        real_imgs = imgs.to(device)

        #====================
        # Train Discriminator
        #====================
        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], latent_dim).to(device)
        # Generate a batch of images
        fake_imgs = generator(z)
        # Adversarial loss
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)
        loss_D = -torch.mean(real_validity) + torch.mean(fake_validity)

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # Train the generator every n_critic iterations, n_critic是WGAN的一个重要参数, 通常为5
        if i % n_critic == 0:
            #====================
            # Train Generator
            #====================
            optimizer_G.zero_grad()
            z = torch.randn(imgs.shape[0], latent_dim).to(device)
            gen_imgs = generator(z)
            #adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, num_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
    )
    torchvision.utils.save_image(gen_imgs.data[:25], "images/epoch%d.png" % epoch, nrow=5, normalize=True)
















