import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CelebADataset(Dataset):
    def __init__(self, root, transform_=None, mode="train", attributes=None):
        self.transform = transforms.Compose(transform_)

        self.selected_attrs = attributes
        self.files = sorted(glob.glob(f"{root}/*.jpg"))
        self.files = self.files[:-2000] if mode == "train" else self.files[-2000:] #分割数据集
        self.label_path = glob.glob(f"{root}/*.txt")[0]
        self.annotations = self.get_annotations()

    def get_annotations(self):
        """
        获取标签
        :return: 返回一个字典，key为图片名，value为标签
        """
        annotations = {}
        lines = [line.rstrip() for line in open(self.label_path, "r")] #读取标签文件, rsrtip()去掉字符串末尾的空格
        self.label_names = lines[1].split()
        for _, line in enumerate(lines[2:]):
            filename, *values = line.split()
            labels = []
            for attr in self.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == "1")) #将标签转换为0或1,获取one-hot编码
            annotations[filename] = labels #labels就是一个one-hot编码，以列表的形式存储
        return annotations

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split("/")[-1]
        img = self.transform(Image.open(filepath))
        label = self.annotations[filename]
        label = torch.FloatTensor(label)

        return img, label

    def __len__(self):
        return len(self.files)