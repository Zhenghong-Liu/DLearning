"""
Math2Latex ddim.py
2024年07月04日
by Zhenghong Liu
"""

"""
DDPM存在的问题：
1.加噪过程和去噪过程太长，导致生成图像很慢。
DDPM本身推导出来了跳步加噪的过程。
所以DDIM的目的是实现跳步去噪的过程，这样就可以加速生成图像。
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from minddpm import DDPM


# https://arxiv.org/abs/2010.02502
class DDIM(DDPM):
    def __init__(
            self,
            eps_model: nn.Module,
            betas: Tuple[float, float],
            eta: float,
            n_T: int,
            criterion: nn.Module = nn.MSELoss(),
    ):
        super(DDIM, self).__init__(eps_model, betas, n_T, criterion)
        self.eta = eta

    # modified from https://github.com/ermongroup/ddim/blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/functions/denoising.py#L10-L32
    def sample(self, n_sample:int, size, device) -> torch.tensor:
        x_i = torch.randn(n_sample, *size, device=device) #这里是生成初始的x_i ~ N(0, 1)

        for i in range(self.n_T, 1, -1):
            z = torch.randn(n_sample, *size, device=device)  if i > 2 else 0
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T, device=device).repeat(n_sample, 1)
            )

            x0_t = (x_i - eps * (1 - self.alphabar_t[i]).sqrt()) / self.alphabar_t[i].sqrt()
            c1 = self.eta * ((1 - self.alphabar_t[i] / self.alphabar_t[i - 1]) * (1 - self.alphabar_t[i - 1]) / (
                    1 - self.alphabar_t[i])).sqrt()
            c2 = ((1 - self.alphabar_t[i - 1]) - c1 ** 2).sqrt()
            x_i = self.alphabar_t[i - 1].sqrt() * x0_t + c1 * z + c2 * eps

        return x_i