"""
Math2Latex diffusion.py
2024年06月11日
by Zhenghong Liu
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, pred, targ, weighted=1.0):
        '''
        pred, targ : [batch_size, action_dim]
        '''
        loss = self._loss(pred, targ)
        WeightedLoss = (loss * weighted).mean()
        return WeightedLoss

class L1Loss(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class L2Loss(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {"L1": L1Loss, "L2": L2Loss}

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *(1,) * (len(x_shape) - 1))

class SinosoidaPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinosoidaPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device)* -emb)
        # emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = x[:, None] * emb[None, :] #这里的None是为了增加维度，使得两个张量能够相乘
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

#去噪神经网络，可以是MLP，ResNet，Unet等
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim=16): #扩散模型中，每一步的时间是需要编码的，因此需要t_dim
        super(MLP, self).__init__()

        self.t_dim = t_dim
        self.a_dim = action_dim
        self.device = device

        self.time_mlp = nn.Sequential(
            SinosoidaPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(), #再diffusion模型中，通常使用Mish激活函数
            nn.Linear(t_dim * 2, t_dim)
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )
        self.final_layer = nn.Linear(hidden_dim, action_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, time, state): #这里的x是action
        t_emb = self.time_mlp(time)
        x = torch.cat([x, state, t_emb], dim=1)
        x = self.mid_layer(x)
        x = self.final_layer(x)
        return x

class Diffusion(nn.Module):
    def __init__(self, loss_type, beta_schedule='linear', clip_denoised=True, predict_epsilon=True, **kwargs):
        super(Diffusion, self).__init__()
        self.state_dim = kwargs['obs_dim']
        self.action_dim = kwargs['act_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.T = kwargs['T']
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.device = torch.device(kwargs['device'])
        self.model = MLP(self.state_dim, self.action_dim, self.hidden_dim, self.device)

        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.002, self.T, dtype=torch.float32)

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, 0) #累乘, 例如[1,2,3]的累乘结果是[1,2,6]
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]], dim=0)

        #构建regisiter buffer,存储静态变量
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('alpha_cumprod_prev', alpha_cumprod_prev)

        #前向过程
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alpha_cumprod))

        #反向过程
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        #反向过程 - 用于估计x0
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alpha_cumprod - 1.0))

        #反向过程
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod))

        #Loss function
        self.loss_fn = Losses[loss_type]()

    def q_posterior(self, x_start, x, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
        return posterior_mean, posterior_variance, posterior_log_variance


    def predict_start_from_noise(self, x, t, pred_noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise)

    def p_mean_variance(self, x, t, state):
        pred_noise = self.model(x, t, state)
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        x_recon.clamp_(-1, 1) #限制x_recon的范围在[-1, 1]之间,clamp_是inplace操作
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, state):
        batch, *_, device = x.shape, x.device
        model_mean, model_log_variance = self.p_mean_variance(x, t, state)
        noise = torch.randn_like(x)

        nonzero_mask = (1 - (t == 0).float()).reshape(batch, *(1,) * (len(x.shape) - 1))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, state, shape, *args, **kwargs):
        '''
        state : [batch_size, state_dim]
        shape : [batch_size, action_dim]
        '''
        device = self.device
        batch_size = state.shape[0]
        x = torch.randn(shape, device=device, requires_grad=False) #生成最原始的噪声

        #进行反向的过程
        for i in reversed(range(0, self.T)):
            t = torch.full([batch_size, 1], i, device=device)
            x = self.p_sample(x, t, state)
        return x

    def sample(self, state, *args, **kwargs):
        '''
        state : [batch_size, state_dim]
        '''
        #采样过程
        batch_size = state.shape[0]
        shape = [batch_size, self.action_dim] #初始化噪声
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp(-1.0, 1.0) #限制动作范围

    # =========================== 训练过程 ===========================

    def q_sample(self, x_start, t, noise):
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        x_recon = self.model(x_noisy, t, state)

        loss = self.loss_fn(x_recon, x_noisy, weights)
        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = x.shape[0]
        t = torch.randint(0, self.T, [batch_size], device=self.device).long()
        return self.p_losses(x, state, t, weights)

    #采样过程
    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

if __name__ == '__main__':
    device = 'cpu'
    x = torch.randn(256, 2).to(device)
    state = torch.randn(256, 11).to(device)
    model = Diffusion(
        loss_type='L2',
        obs_dim=11,
        act_dim=2,
        hidden_dim=256,
        T=100,
        device=device
    )
    action = model(state)

    loss = model.loss(x, state)

    print(f'action : {action}; loss : {loss.item()}')


