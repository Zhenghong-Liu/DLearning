"""
Math2Latex train_cifar10.py
2024年07月04日
by Zhenghong Liu
"""

from typing import Dict, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from minddpm import DDPM
from TEmbdUnet import Unet



def train_cifar10(
        n_epoch:int = 100, device:str = "cuda", load_pth: Optional[str] = None
)->None:

    ddpm = DDPM(eps_model=Unet(3, 3), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth))

    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(root="../../../data", train=True, transform=tf, download=True)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-5)


    for i in range(n_epoch):
        print(f"Epoch {i}")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for img, _ in pbar:
            optimizer.zero_grad()
            img = img.to(device)
            loss = ddpm(img)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"Loss: {loss_ema:.4f}")
            optimizer.step()

        if i % 10 == 0:
            ddpm.eval()
            with torch.no_grad():
                xh = ddpm.sample(8, (3, 32, 32), device)
                xset = torch.cat([img[:8], xh], 0) #上面8个是真实的图像, 下面8个是生成的图像
                grid = make_grid(xset, nrow=4, normalize=True, value_range=(-1, 1))
                save_image(grid, f"./contents/ddpm_sample_cifar{i}.png")

                # Save model
                # torch.save(ddpm.state_dict(), f"./contents/ddpm_cifar.pth")


if __name__ == '__main__':
    train_cifar10()