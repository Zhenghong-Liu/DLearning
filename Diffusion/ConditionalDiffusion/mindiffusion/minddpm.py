"""
Math2Latex minddpm.py
2024年07月04日
by Zhenghong Liu
"""

from typing import Dict, Tuple

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
    alpha_t = 1.0 - sqrt_beta_t

    alphabar_t = torch.cumprod(alpha_t, dim=0) #这里是对alpha_t进行累乘

    sqrt_alphabar_t = torch.sqrt(alphabar_t)
    sqrt_1m_alphabar_t = torch.sqrt(1.0 - alphabar_t)

    #用于Sample过程
    oneover_sqrt_alpha_t = 1.0 / torch.sqrt(alpha_t)
    omabt_over_sqrt_1m_alphabar_t = (1.0 - alphabar_t) / sqrt_1m_alphabar_t

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
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        #register_buffer是将变量注册到buffer中，这样在保存模型的时候，这些变量不会被保存
        #register_buffer允许我们按名称自由访问这些张量。它有助于设备放置。
        for k, v in ddpm_schedules(*betas, n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x:torch.tensor) -> torch.tensor:
        """
        这部分就是DDPM文章中的左边部分，Training部分
        进行正向扩散x_t，并尝试使用eps_model从x_t猜测epsilon值。
        """
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],), device=x.device) # （x.shape[0],）确定生成batch_size个数,输出结果是一个一维张量

        #t ~ U[1, n_T]
        eps = torch.randn_like(x)

        x_t = (
            self.sqrt_alphabar_t[_ts, None, None, None] * x
            + self.sqrt_1m_alphabar_t[_ts, None, None, None] * eps
        ) #根据x0计算xt，sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps

        #我们应该根据这个x_t来预测epsilon，然后计算loss
        eps_pred = self.eps_model(x_t, _ts / self.n_T) #这里的_ts / self.n_T是为了将t归一化到[0, 1]之间

        return self.criterion(eps, eps_pred)

    def sample(self, n_sample:int, size, device) -> torch.tensor:
        """
        这部分就是DDPM文章中的右边部分，Sampling部分
        """
        x_i = torch.randn(n_sample, *size, device=device) #x_t ~ N(0, 1)

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size, device=device)  if i > 1 else 0 #z ~ N(0, 1) 最后一次不需要加噪声
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T, device=device).repeat(n_sample, 1)
            )

            x_i= (
                self.oneover_sqrt_alpha_t[i] * (x_i - self.omabt_over_sqrt_1m_alphabar_t[i] * eps)
                + self.sqrt_beta_t[i] * z
            )

        return x_i


