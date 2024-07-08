import random
import time
import datetime
import sys

import torch
import numpy as np
from torchvision.utils import save_image


#学习率更新策略
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch


    #每个epoch调用一次,返回学习率的缩放因子
    #相当于线性下降,从decay_start_epoch开始下降
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


#定义一个ReplayBuffer类，用于存储生成器的生成结果
class ReplayBuffer:
    def __init__(self, max_len = 50):
        assert (max_len > 0), "Length of buffer must be greater than 0"
        self.max_len = max_len
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0) #在第0维增加一个维度,相当于在第0维增加一个batch_size=1，用于之后的cat操作
            if len(self.data) < self.max_len:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_len - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)
