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
from UNet.UNet import Unet
from discriminator import Discriminator, weights_init_normal
from utils import *
from datasets import ImageDataset

#===============================================================================
# 经过实践发现，这个训练是非常慢的。
# 可以尝试一下换一个unet
# cyclegan的训练通常需要小的学习率，大概在0.0002左右，而且训练时间也比较长，通常需要几天的时间。
# 我推荐使用batch size为1，这样可以有更多次数的backward。
# 我实验感觉是当batch size为1时，生成效果最好。
#===============================================================================


if not os.path.exists('images'):
    os.makedirs('images')

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# 定义超参数
epoch = 0
num_epochs = 200
decay_epoch = 100
batch_size = 1 #cyclegan通常使用1作为batch_size，这样训练单张图像可以更好的保持细节和特征，而且增加了backward的次数。
learning_rate = 0.0002
img_height = 256
img_width = 256
channels = 3
lambda_cyc = 10.0
lambda_id = 5.0
dataset_name = 'apple2orange'
save_model = False

#calculate output of image discriminator (PatchGAN)
patch = (1, img_height // 2 ** 4, img_width // 2 ** 4) # 256 / 2**4 = 16, 对应了discriminator的4次下采样

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss() #用于计算cycle loss
criterion_identity = torch.nn.L1Loss() #用于计算identity loss

input_shape = (channels, img_height, img_width) #pytorch中图像格式为BCHW

# Initialize generator and discriminator
G_AB = Unet(input_channels=channels, output_channels=channels)
G_BA = Unet(input_channels=channels, output_channels=channels)
D_A = Discriminator(in_channels=3) #因为用的是现成的判别器，所以需要修改一下输入通道数，1.5 * 2 == 3
D_B = Discriminator(in_channels=3)

G_AB.to(device)
G_BA.to(device)
D_A.to(device)
D_B.to(device)

# G_AB.apply(weights_init_normal)
# G_BA.apply(weights_init_normal)
# D_A.apply(weights_init_normal)
# D_B.apply(weights_init_normal)

#Optimizer
optimizer_G = torch.optim.Adam( #itertools.chain()可以将多个迭代器合并成一个迭代器
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=learning_rate, betas=(0.5, 0.999)
) #可以使用一个优化器同时优化两个模型
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(num_epochs, epoch, decay_epoch).step
)

lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(num_epochs, epoch, decay_epoch).step
)

lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(num_epochs, epoch, decay_epoch).step
)

#Buffer of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Configure dataloaders
transforms_ = [
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)), #随机裁剪
    transforms.RandomHorizontalFlip(), #随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dateloader = DataLoader(
    ImageDataset(f'datasets/{dataset_name}', transforms_=transforms_, unaligned=True),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)
#Test dataloader
val_dataloader = DataLoader(
    ImageDataset(f'datasets/{dataset_name}', transforms_=transforms_, mode='test'),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)

def sample_images(file_name):
    img = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = img['A'].to(device)
    fake_B = G_AB(real_A)
    real_B = img['B'].to(device)
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True) #将多张图片拼接在一起
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, f'images/{file_name}', normalize=False)


# Training
prev_time = time.time()
start_epoch = epoch
for epoch in range(start_epoch, num_epochs):
    for i, batch in enumerate(dateloader):

        #set model input
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)

        #Adversarial ground truths
        valid = torch.ones((real_B.size(0), *patch), device=device, requires_grad=False)
        fake = torch.zeros((real_B.size(0), *patch), device=device, requires_grad=False)

        #=====================
        #train Generators
        #=====================
        G_AB.train()
        G_BA.train()
        optimizer_G.zero_grad()
        #Identity loss
        # 生成器G_BA用来生成A风格的图片，那么把real_A输入到G_BA中，应该仍然得到real_A
        # 只有这样才能证明G具有生成y风格的能力。因此，我们需要计算G_BA(real_A)和real_A之间的L1损失
        # 如果不加该loss，那么生成器可能会自主的修改图像的色调，使得整体的颜色产生变化。
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        #GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        #Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        #Total loss
        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()


        #=====================
        #Train DiscriminatorA
        #=====================
        optimizer_D_A.zero_grad()
        loss_real = criterion_GAN(D_A(real_A), valid)
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        #=====================
        #Train DiscriminatorB
        #=====================
        optimizer_D_B.zero_grad()
        loss_real = criterion_GAN(D_B(real_B), valid)
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        #total Discriminator loss
        loss_D = (loss_D_A + loss_D_B) / 2

    #Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    print(
        f"[Epoch {epoch}/{num_epochs}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}] [Time: {time.time() - prev_time}]"
    )
    prev_time = time.time()

    # if epoch % 5 == 0:
    sample_images(f"epoch{epoch}.png")



















