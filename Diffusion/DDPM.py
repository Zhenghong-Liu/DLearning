"""
Math2Latex DDPM.py
2024年06月12日
by Zhenghong

实现DDPM模型，生成Mnist图像。
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#使用Unet作为去噪神经网络
from Unet import Unet

def extract(a, t, x_shape):
    print(f"a.device: {a.device}, t.device: {t.device}")
    t.to(a.device)
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *(1,) * (len(x_shape) - 1))

class DDPM(nn.Module):
    def __init__(self,loss_type, beta_schedule, T, device, **kwargs):
        super(DDPM, self).__init__()
        self.model = Unet()
        self.T = T
        self.device = torch.device(device)
        self.IMG_SIZE = kwargs["IMG_SIZE"]

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.002, T, dtype=torch.float32)

        alphas = 1.0 - betas

        alpha_cumprod = torch.cumprod(alphas, dim = 0)
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]], dim = 0)

        # 构建regisiter buffer,存储静态变量
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)

        #前向过程，用于从x0一步计算加噪图像 q(xt|x0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alpha_cumprod))

        #反向过程
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        #反向过程，通过xt预测x0
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod - 1.0))

        #反向过程，通过x0，xt预测xt-1  q(xt-1|xt,x0)
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod))

        self.loss_fn = F.l1_loss

    # ================训练过程======================================================
    def q_sample(self, x_start, t, noise):
        '''
        x_start : [batch_size, 1, 28, 28]
        t : [batch_size, 1]
        noise : [batch_size, 1, 28, 28]
        '''
        device = self.device
        sqrt_alpha_cumprod = extract(self.sqrt_alphas_cumprod, t, x_start.shape).to(device)
        sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape).to(device)

        return x_start * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod

    def forward_diffusion_sample(self, x_start, t):
        '''
        x_start : [batch_size, 1, 28, 28]
        '''
        device = self.device
        noise = torch.randn_like(x_start, device=device, requires_grad=False)  # 生成最原始的噪声。
        noisy_img = self.q_sample(x_start, t, noise)

        return noisy_img, noise

    def loss_process(self,noise_pred, noise):
        self.loss = self.loss_fn(noise_pred, noise)
        print("noise_pred.shape: ", noise_pred.shape, "noise.shape: ")
        self.loss.backward()

    #训练过程
    def forward(self, x_start):
        print("Training..., x_start.shape: ", x_start.shape)
        device = self.device
        t = torch.randint(0, self.T, (x_start.shape[0],), device=device).long()
        noisy_img, noise = self.forward_diffusion_sample(x_start, t) #采样过程, 从噪声中采样，生成图像
        noise_pred = self.model(noisy_img, t)
        self.loss_process(noise_pred, noise)

    # ================采样过程======================================================

    def p_sample(self, noise, t):
        #计算均值
        device = self.device
        sqrt_recip_alphas = extract(self.sqrt_recip_alphas, t, noise.shape).to(device)
        betas = extract(self.betas, t, noise.shape).to(device)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, noise.shape).to(device)
        predic_noise = self.model(noise, t)
        model_mean = sqrt_recip_alphas * (noise - betas * predic_noise / sqrt_one_minus_alphas_cumprod)

        model_variance = extract(self.posterior_variance, t, noise.shape).to(device)

        if t == 0:
            return model_mean
        else:
            epsilons = torch.randn_like(noise, device=device)
            return model_mean + torch.sqrt(model_variance) * epsilons


    def p_sample_loop(self, IMG_SIZE, step_size):
        device = self.device
        noise = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)

        for i in reversed(range(0, self.T)):
            t = torch.full((1,), i, device=device).long()
            noise = self.p_sample(noise, t)
            noise = noise.clamp(-1, 1)

            if i % step_size == 0:
                plt.subplot(1, 10, i // step_size + 1)
                # print(f'img shape : {noise.shape}')
                self.show_img(noise.detach().cpu()[0])
        plt.savefig('sample.png')
        plt.show()



    #sample过程
    def sample_plot_image(self):
        IMG_SIZE = self.IMG_SIZE
        device = self.device

        plt.figure(figsize=(15, 15))
        plt.axis('off')
        num_images = 10
        stepsize = int(T / num_images)

        self.p_sample_loop(IMG_SIZE, stepsize)

    def show_img(self, image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        plt.imshow(reverse_transforms(image))


if __name__ == '__main__':
    epochs = 100
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device : {device}")
    T = 100
    IMG_SIZE = 64
    model = DDPM("l1", "linear", T, device, IMG_SIZE = 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    #=================加载cifar10数据集=================
    BATCH_SIZE = 64
    # image in range [-1, 1]
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            model(batch[0].to(device)) #batch[0]是图像, batch[1]是label
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {model.loss.item()} ")
                model.sample_plot_image()