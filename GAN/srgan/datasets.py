import glob #用于查找符合特定规则的文件路径名
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms

#normalizatin parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        h_height, h_width = hr_shape

        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose([
            transforms.Resize((h_height // 4, h_width // 4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.hr_transform = transforms.Compose([
            transforms.Resize((h_height, h_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.files = sorted(glob.glob(root + "/*.*")) #返回所有匹配的文件路径列表

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)