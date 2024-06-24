"""
Math2Latex bnn_minist.py
2024年06月14日
by Zhenghong Liu

使用mine_bnn中的bnn网络，来实现对minist数据集的分类。

本模型采用两阶段的训练
1：首先训练一个变分自编码器VAE，学习一个有图片到隐空间的映射
2：之后拿出VAE中的编码器部分，并锁定其参数
3：添加bayes层bayesLayer，学习分类。
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


from mine_bnn import BayesLayer







if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = 5
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)