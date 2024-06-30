"""
Math2Latex GPTHelper.py
2024年06月30日
by Zhenghong Liu
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SineDataset(Dataset):
    def __init__(self):
        x, y = self._preprocess()
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        #将x归一化到0-1之间
        self.x = (self.x - self.x.min()) / (self.x.max() - self.x.min())

    def _preprocess(self, n = 300):
        np.random.seed(32)
        x = np.linspace(0, 1 * 2 * np.pi, n)
        y1 = 3 * np.sin(x)
        y1 = np.concatenate((np.zeros(60), y1 + np.random.normal(0, 0.15 * np.abs(y1), n), np.zeros(60)))
        x = np.concatenate((np.linspace(-3, 0, 60), np.linspace(0, 3 * 2 * np.pi, n),
                            np.linspace(3 * 2 * np.pi, 3 * 2 * np.pi + 3, 60)))
        y2 = 0.1 * x + 1
        y = y1 + y2
        return x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

