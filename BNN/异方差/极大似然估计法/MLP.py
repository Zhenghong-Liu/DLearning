import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, 20)
        self.hidden2 = nn.Linear(20, 50)
        self.hidden3 = nn.Linear(50, 20)
        self.mean = nn.Linear(20, 1)
        self.log_sigma = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        mean = self.mean(x)
        log_sigma = self.log_sigma(x)
        return mean, log_sigma


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, mean, log_sigma, y):
        sigma = torch.exp(log_sigma)
        loss = (y - mean) ** 2 / (2 * sigma ** 2) + log_sigma
        return loss.mean()
