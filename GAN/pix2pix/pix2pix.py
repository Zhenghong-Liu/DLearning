import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
import os
import time
from PIL import Image

from UNet.UNet import Unet
from datasets import ImageDataset

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalization:
                block.append(nn.InstanceNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)), # ZeroPad2d((left, right, top, bottom))
            nn.Conv2d(512, 1, 4, padding=1, bias=False), #判别器最后一层不用bias，可以和bn层配合使用，保持数值稳定和简化实现
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# os.mkdir('images', exist_ok=True)
if not os.path.exists('images'):
    os.makedirs('images')

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# 定义超参数
num_epochs = 200
batch_size = 64
learning_rate = 0.0002
img_height = 256
img_width = 256
channels = 3
decay_epoch = 100
dataset_name = 'facades'
save_model = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


#loss function
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

#loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

#calculate output of image discriminator (PatchGAN)
patch = (1, img_height // 2 ** 4, img_width // 2 ** 4) # 256 / 2**4 = 16, 对应了discriminator的4次下采样

# Initialize generator and discriminator
generator = Unet(input_channels=channels, output_channels=channels)
discriminator = Discriminator()

generator.to(device)
discriminator.to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

#Optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Configure dataloaders
transforms = [
    transforms.Resize((img_height, img_width), Image.BICUBIC), #Image.BICUBIC:双三次插值
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(f"../../data/{dataset_name}", transforms_=transforms),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)

val_dataloader = DataLoader(
    ImageDataset(f"../../data/{dataset_name}", transforms_=transforms, mode='val'),
    batch_size=10,
    shuffle=True,
    num_workers=1,
)


def sample_images(file_name):
    imgs = next(iter((val_dataloader)))
    real_A = imgs['A'].to(device)
    real_B = imgs['B'].to(device)
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2) #在长度的维度上拼接
    torchvision.utils.save_image(img_sample, f'images/{file_name}', nrow=5, normalize=True)


# ----------Training----------
prev_time = time.time()
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)

        #Adversial ground truths
        vaild = torch.ones((real_A.size(0), *patch), requires_grad=False, device=device)
        fake = torch.zeros((real_A.size(0), *patch), requires_grad=False, device=device)

        #=====================
        #Train Generators
        #=====================

        optimizer_G.zero_grad()
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, vaild)
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        #total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # =====================
        # Train Discriminator
        # =====================

        optimizer_D.zero_grad()

        #real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, vaild)

        #fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        #total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

    #print log
    time_used = time.time() - prev_time
    prev_time = time.time()

    print(
        f"[Epoch {epoch}/{num_epochs}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}] [Time: {time_used}]"
    )

    sample_images(f"{epoch}.png")

    if save_model:
        torch.save(generator.state_dict(), 'saved_models/generator.pth')
        torch.save(discriminator.state_dict(), 'saved_models/discriminator.pth')












