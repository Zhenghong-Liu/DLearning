"""
马尔可夫链蒙特卡罗（MCMC）算法是一种用于从复杂概率分布中采样的方法，特别是在这些分布难以直接采样的情况下。

从目标概率分布中采样得到的样本有很多用途，主要体现在以下几个方面：
1 估计期望值和不确定性, 2 构建置信区间

我学习到的内容是MCMC一般可以用来计算后验。也就是它可以在后验分布上采样，从而我们可以估计期望和不确定性。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

import seaborn as sns
import pandas as pd

# 生成带噪声的模拟数据
np.random.seed(42)
true_a = 2.0
true_b = 1.0
x = np.linspace(0, 10, 100)
y = true_a * x + true_b + np.random.normal(0, 10, x.shape) # y = ax + b + noise

# 定义先验分布（假设先验分布为均值为0，方差为10的正态分布）
def log_prior_prob(a, b):
    return norm.logpdf(a, loc=0, scale=10) + norm.logpdf(b, loc=0, scale=10)

# 定义似然函数
def log_likelihood(data_x, data_y, a, b):
    y_model = a * data_x + b
    return np.sum(norm.logpdf(data_y, loc=y_model, scale=1)) #norm.pdf(x, loc, scale) 计算以y_model为均值，1为方差时，出现data_y的概率

# 定义后验分布（先验分布 * 似然函数）
#这里没有PD，是因为在Metropolis-Hastings算法中，PD会被消掉
def log_posterior_prob(a, b, data_x, data_y):
    return log_prior_prob(a, b) + log_likelihood(data_x, data_y, a, b)

# Metropolis-Hastings算法
T = 4_000
a_samples = np.zeros(T)
b_samples = np.zeros(T)
a_samples[0] = 0  # 初始值
b_samples[0] = 0  # 初始值
sigma_a = 0.5  # 采样的标准差
sigma_b = 0.5  # 采样的标准差

for t in range(1, T):
    a_new = norm.rvs(loc=a_samples[t-1], scale=sigma_a)
    b_new = norm.rvs(loc=b_samples[t-1], scale=sigma_b)
    alpha = min(1, log_posterior_prob(a_new, b_new, x, y) - log_posterior_prob(a_samples[t-1], b_samples[t-1], x, y))
    u = random.uniform(0, 1)
    if u < alpha:
        a_samples[t] = a_new
        b_samples[t] = b_new
    else:
        a_samples[t] = a_samples[t-1]
        b_samples[t] = b_samples[t-1]

# 绘制后验分布
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(a_samples[1000:], bins=50, density=True, alpha=0.7)
plt.xlabel('Parameter a')
plt.ylabel('Density')
plt.title('Posterior distribution of parameter a')

plt.subplot(1, 2, 2)
plt.hist(b_samples[1000:], bins=50, density=True, alpha=0.7)
plt.xlabel('Parameter b')
plt.ylabel('Density')
plt.title('Posterior distribution of parameter b')

plt.tight_layout()
plt.show()

# 绘制专业的后验分布
# 去掉初始的1000个样本，以减少采样初期的影响
burn_in = 1000
a_samples = a_samples[burn_in:]
b_samples = b_samples[burn_in:]

# 使用seaborn绘制后验分布
df = pd.DataFrame({'a': a_samples, 'b': b_samples})
sns.pairplot(df, kind='scatter', diag_kind='hist').fig.set_size_inches(6, 6)
plt.suptitle('Posterior distributions of parameters a and b', y=1.02) # y=1.02是为了让标题不与图重叠
plt.show()


# 绘制拟合的图像
plt.figure(figsize=(12, 6))
plt.scatter(x, y, label='Data')
mean_a = np.mean(a_samples)
mean_b = np.mean(b_samples)
std_a = np.std(a_samples)
std_b = np.std(b_samples)

plt.plot(x, mean_a * x + mean_b, color='red', label='Fitted line')
plt.fill_between(x, (mean_a - std_a) * x + (mean_b - std_b), (mean_a + std_a) * x + (mean_b + std_b), color='red', alpha=0.3, label='Uncertainty')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Fitted line with uncertainty')
plt.show()


