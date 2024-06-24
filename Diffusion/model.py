"""
Math2Latex model.py
2024年06月12日
by Zhenghong Liu
"""

import torch
import math
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np



# Load the dataset and make data loader

BATCH_SIZE = 64
IMG_SIZE = 64

# image in range [-1, 1]
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
     ])

trainset = torchvision.datasets.CIFAR10(root='../data', train = True, download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

testset = torchvision.datasets.CIFAR10(root='../data', train = False, download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, drop_last = True)

def show_img(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    plt.imshow(reverse_transforms(image))

first_batch = next(iter(train_loader))[0]
plt.figure(figsize=(15,15))
plt.axis('off')

for idx in range(0, 10):
    plt.subplot(1, 10, idx + 1)
    img = show_img(first_batch[idx])

plt.show()



#==========================================




