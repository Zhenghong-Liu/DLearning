"""
Math2Latex Unet.py
2024年06月12日
by Zhenghong Liu
"""


import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinosoidaPosEmb(nn.Module): #用于对时间t编码
    def __init__(self, dim):
        super(SinosoidaPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device)* -emb)
        # emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = x[:, None] * emb[None, :] #这里的None是为了增加维度，使得两个张量能够相乘
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()
        self.time_mlp = nn.Sequential(
            SinosoidaPosEmb(64),
            nn.Linear(64, 64),
            nn.Mish(),
        )
        self.down1 = DownSample(3, 64)
        self.down2 = DownSample(64, 128)

        self.conv1 = ConvBlock(128, 256)
        self.conv2 = ConvBlock(256, 256)

        self.up1 = UpSample(256, 128)
        self.up2 = UpSample(128, 64)

        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 3)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]

        _x1, x = self.down1(x)
        x = x + time_emb
        _x2, x = self.down2(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.up1(x, _x2)
        x = self.up2(x, _x1)

        x = self.conv3(x)
        x = self.conv4(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, self.pool(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, x_down):
        x = self.up(x)
        x = torch.cat([x, x_down], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.mish = nn.Mish()

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))

if __name__ == '__main__':
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 100, (2,))
    print(f't shape: {t.shape}')
    model = Unet()
    y = model(x, t)
    print(y.shape)


