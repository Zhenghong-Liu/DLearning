import os
from pathlib import Path #Path是一个类，用于处理文件路径
from torch.utils.data import Dataset
from PIL import Image

class FFHQ(Dataset):
    def __init__(self, transform, path = "../../data/ffhq", file_prefix = "", file_postfix = ".png"):
        self.transform = transform
        self.path = Path(path)
        self.file_prefix = file_prefix
        self.file_postfix = file_postfix
        self.length = len(os.listdir(path))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if (index >= 9000): index += 2000 #这里是为了跳过一些损坏的图片
        padded_index = str(index).rjust(5, "0") #rjust() 方法返回一个原字符串右对齐,并使用空格填充至指定长度的新字符串
        file_path = self.file_prefix + padded_index + self.file_postfix
        file_path = Path.joinpath(self.path, file_path)
        image = Image.open(file_path)
        image = self.transform(image)
        return image

