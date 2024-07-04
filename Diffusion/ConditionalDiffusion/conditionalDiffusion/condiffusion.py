"""
Math2Latex minddpm.py
2024年07月04日
by Zhenghong Liu
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ddpm_schedules(beta1:float, beta2:float, T:int) -> Dict[str, torch.tensor]:
    """
    在训练过程中后获取参数
    """
    assert beta1 < beta2 < 1.0, "beta1 must be less than beta2, and beta2 must be less than 1.0"

    beta_t = (beta2 - beta1) * torch.arange(0, T+1, dtype=torch.float32) / T + beta1 #这里torch.arange(0, T+1)是从0到T，共T+1个数
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1.0 - beta_t

    alphabar_t = torch.cumprod(alpha_t, dim=0) #这里是对alpha_t进行累乘

    sqrt_alphabar_t = torch.sqrt(alphabar_t)
    sqrt_1m_alphabar_t = torch.sqrt(1.0 - alphabar_t)

    #用于Sample过程
    oneover_sqrt_alpha_t = 1.0 / torch.sqrt(alpha_t)
    omabt_over_sqrt_1m_alphabar_t = (1.0 - alpha_t) / sqrt_1m_alphabar_t

    return {
        "alpha_t": alpha_t,
        "oneover_sqrt_alpha_t": oneover_sqrt_alpha_t,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrt_alphabar_t": sqrt_alphabar_t,
        "sqrt_1m_alphabar_t": sqrt_1m_alphabar_t,
        "omabt_over_sqrt_1m_alphabar_t": omabt_over_sqrt_1m_alphabar_t
    }


class DDPM(nn.Module):

    def __init__(
            self,
            eps_model: nn.Module,
            betas: Tuple[float, float],
            n_T: int,
            criterion: nn.Module = nn.MSELoss(),
            drop_prob = 0.1
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        #register_buffer是将变量注册到buffer中，这样在保存模型的时候，这些变量不会被保存
        #register_buffer允许我们按名称自由访问这些张量。它有助于设备放置。
        for k, v in ddpm_schedules(*betas, n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion
        self.drop_prob = drop_prob

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T + 1, (x.size(0),), device=x.device)
        eps = torch.randn_like(x)

        x_t = (
            self.sqrt_alphabar_t[_ts, None, None, None] * x
            + self.sqrt_1m_alphabar_t[_ts, None, None, None] * eps
        )

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(x.device) #drop_prob 是结果为1的概率

        eps_pred = self.eps_model(x_t, c, _ts / self.n_T, context_mask)

        return self.criterion(eps, eps_pred)

    def sample(self, n_sample, size, device, guide_w = 0.0, n_classes = 10):

        x_i = torch.randn(n_sample, *size, device=device)
        c_i = torch.arange(0, n_classes).to(device)
        c_i = c_i.repeat(n_sample // c_i.shape[0])

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        #double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.0

        x_i_store = [] #用于存储生成图片的过程

        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T], device=device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            #double the batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size, device=device) if i > 1 else 0

            eps = self.eps_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]

            x_i = (
                self.oneover_sqrt_alpha_t[i] * (x_i - self.omabt_over_sqrt_1m_alphabar_t[i] * eps)
                + self.sqrt_beta_t[i] * z
            )
            if i % 20 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store