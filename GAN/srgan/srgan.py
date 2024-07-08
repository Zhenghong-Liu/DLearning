"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools #itertools模块提供了很多有用的用于操作迭代对象的函数

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
import os
import time
from PIL import Image

#利用现成的生成器和判别器
from models import *
from datasets import *


if not os.path.exists('images'):
    os.makedirs('images')

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# 定义超参数
start_epoch = 0
num_epochs = 200
decay_epoch = 100
batch_size = 1 #cyclegan通常使用1作为batch_size，这样训练单张图像可以更好的保持细节和特征，而且增加了backward的次数。
learning_rate = 0.0002
hr_height = 256
hr_width = 256
channels = 3
dataset_name = 'img_align_celeba'
save_model = False

hr_shape = (hr_height, hr_width)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#initialize generator and discriminator
generator = GeneratorResNet().to(device)
discriminator = Discriminator(input_shape=(channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

#set feature extractor to inference mode
feature_extractor.eval() #vgg在这里的作用就是提取特征，用于计算lr和hr之间的特征差异

#losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss() #用于计算content loss


if start_epoch != 0:
    #load models
    generator.load_state_dict(torch.load('saved_models/generator_%d.pth' % start_epoch))
    discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth' % start_epoch))

#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

dataloader = DataLoader(
    ImageDataset("../../data/%s" % dataset_name, hr_shape=hr_shape),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)


def sample_images(imgs_lr, gen_hr, file_name):
    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
    img_grid = torch.cat((imgs_lr, gen_hr), -1)
    save_image(img_grid, file_name, normalize=False)


#Training
for epoch in range(start_epoch, num_epochs):
    for i, batch in enumerate(dataloader):

        imgs_lr = batch['lr'].to(device)
        imgs_hr = batch['hr'].to(device)

        #Adversarial ground truths
        valid = torch.ones((imgs_lr.size(0), *discriminator.output_shape), requires_grad=False).to(device)
        fake = torch.zeros((imgs_lr.size(0), *discriminator.output_shape), requires_grad=False).to(device)

        #=====================
        #Train Generator
        #=====================
        optimizer_G.zero_grad()
        gen_hr = generator(imgs_lr)

        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        #Perceptual loss
        # 感知损失使用预训练的卷积神经网络（例如 VGG）来提取高层次特征。
        # 这些特征能够更好地捕捉图像的语义信息和人类感知的质量。
        # 直接在图像像素上计算损失（如 MAE 或 MSE）往往只能捕捉到低层次的细节
        # 而感知损失能确保生成的图像在感知上更加逼真。
        # 通过在特征空间中计算损失，模型被迫生成的图像在高级特征上与真实图像匹配。
        # 这种方法能更好地重现图像的纹理和其他细节，使得超分辨率图像看起来更加自然和真实。
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach()) #detach()用于阻断反向传播，这里将real_features作为常数

        #total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()


        #=====================
        #Train Discriminator
        #=====================
        optimizer_D.zero_grad()

        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake) #detach()用于阻断反向传播，这里将gen_hr作为常数

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

    print(
        f"[Epoch {epoch}/{num_epochs}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]"
    )

    sample_images(imgs_lr, gen_hr, f"images/epoch{epoch}.png")

    if save_model:
        torch.save(generator.state_dict(), 'saved_models/generator.pth')
        torch.save(discriminator.state_dict(), 'saved_models/discriminator.pth')






















