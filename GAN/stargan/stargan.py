import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch

#利用现成的生成器和判别器
from UNet.UNet import Unet
from model import Discriminator, weights_init_normal, GeneratorResNet
from datasets import *

#========================================================================================================
# StarGAN不需要 identity loss，因为它的多域转换机制依赖于标签信息来控制转换，而不是像 CycleGAN 那样依赖于循环一致性。
# StarGAN 通过标签控制转换，这使得生成器已经具备了在不同域之间进行正确转换的能力，因此不需要 identity loss 来辅助。
#========================================================================================================

if not os.path.exists('images'):
    os.makedirs('images')

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# 定义超参数
start_epoch = 0
num_epochs = 200
decay_epoch = 100
batch_size = 16 #cyclegan通常使用1作为batch_size，这样训练单张图像可以更好的保持细节和特征，而且增加了backward的次数。
learning_rate = 0.0002
img_height = 128
img_width = 128
channels = 3
save_model = False
selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
n_critic = 5
residual_blocks = 6
dataset_name = "img_align_celeba"


c_dim = len(selected_attrs)
img_shape = (channels, img_height, img_width)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss functions
criterion_cycle = torch.nn.L1Loss() #用于计算cycle loss

def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

# Loss weights
lambda_cls = 1
lambda_rec = 10 # Reconstruction loss weight 重建损失权重
lambda_gp = 10 # Gradient penalty weight 梯度惩罚权重

#========================================================================================================
#StarGAN 采用了 WGAN-GP 的梯度惩罚技术。
#在训练判别器时，StarGAN 在随机插值的图像上计算梯度惩罚，这有助于确保判别器满足 1-Lipschitz 条件，稳定训练过程。
#========================================================================================================


# Initialize generator and discriminator
generator = GeneratorResNet(img_shape=img_shape,res_blocks=residual_blocks ,c_dim=c_dim)
discriminator = Discriminator(img_shape=img_shape, c_dim=c_dim)

generator.to(device)
discriminator.to(device)

# Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Configure data loader
train_transforms = [
    transforms.Resize(int(1.12*img_height), Image.BICUBIC),
    transforms.RandomCrop(img_height),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    CelebADataset(
        f"../../data/{dataset_name}", attributes=selected_attrs, transform_=train_transforms, mode="train"
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

val_transforms = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

val_dataloader = DataLoader(
    CelebADataset(
        f"../../data/{dataset_name}", attributes=selected_attrs, transform_=val_transforms, mode="test"
    ),
    batch_size=10,
    shuffle=False,
    num_workers=1,
)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """
    计算梯度惩罚 用于WGAN-GP
    :param D:
    :param real_samples:
    :param fake_samples:
    :return: gradient_penalty
    """
    #随机插值，between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True) #这里利用了广播机制
    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates, requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() #计算梯度惩罚
    return gradient_penalty

label_changes = [
    ((0, 1), (1, 0), (2, 0)),  # Set to black hair
    ((0, 0), (1, 1), (2, 0)),  # Set to blond hair
    ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
    ((3, -1),),  # Flip gender
    ((4, -1),),  # Age flip
]

def sample_images(file_name):
    val_imgs, val_labels = next(iter(val_dataloader))
    val_imgs = val_imgs.to(device)
    val_labels = val_labels.to(device)
    img_samples = None
    for i in range(10):
        img, label = val_imgs[i], val_labels[i]
        # Repeat for each attribute
        imgs = imgs.repeat(c_dim, 1, 1, 1)
        labels = label.repeat(c_dim, 1)
        #make changes to the labels
        for sample_i, changes in enumerate(label_changes):
            for col, val in changes:
                labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val

        gen_imgs = generator(imgs, labels)
        gen_imgs = torch.cat([x for x in gen_imgs], -1)
        img_sample = torch.cat((img.data, gen_imgs.data), -1)
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples.view(1, *img_samples.shape), f'images/{file_name}', normalize=True)



# Training
saved_samples = []
start_time = time.time()
for epoch in range(start_epoch, num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        sampled_c = torch.randint(0, 2, (imgs.size(0), c_dim), device=device).float()
        fake_imgs = generator(imgs, sampled_c)

        #=========================
        #Train Discriminator
        #=========================

        optimizer_D.zero_grad()

        # Real images
        real_validity, pred_cls = discriminator(imgs)
        # Fake images
        fake_validity, _ = discriminator(fake_imgs.detach())
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data)
        # Adversarial loss
        loss_D_adv = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        # Classification loss
        loss_D_cls = criterion_cls(pred_cls, labels)
        # Total loss
        loss_D = loss_D_adv + lambda_cls * loss_D_cls

        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        #Every n_critic iterations, train the generator
        if i % n_critic == 0:

            #=========================
            #Train Generator
            #=========================

            # Translate and reconstruct image
            gen_imgs = generator(imgs, sampled_c)
            recov_imgs = generator(gen_imgs, labels)
            # Discriminator evaluates translated image
            fake_validity, pred_cls = discriminator(gen_imgs)
            # Adversarial loss
            loss_G_adv = -torch.mean(fake_validity)
            # Classification loss
            loss_G_cls = criterion_cls(pred_cls, sampled_c)
            # Reconstruction loss
            loss_G_rec = criterion_cycle(recov_imgs, imgs)
            # Total loss
            loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec

            loss_G.backward()
            optimizer_G.step()

            # Print log
            print(
                f"\r[Epoch {epoch}/{num_epochs}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}] [time: {time.time()-start_time:.2f}]",
            )
            start_time = time.time()
            sample_images(f"epoch{epoch}.png")





























