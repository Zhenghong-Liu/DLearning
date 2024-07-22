# MCMC 马尔可夫链蒙特卡洛算法
# 学习的代码地址： https://github.com/tech-meow/MCMC/blob/main/MCMC.ipynb
import math

# ======================================================
# 接受拒绝采样 （Rejection Sampling）
# ======================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random

mean = 1
standard_deviation = 2

# 生成正态分布的样本 (目标分布)
x_values = np.arange(-2, 2, 0.1)
y_values = norm(mean, standard_deviation)

# plt.plot(x_values, y_values.pdf(x_values))
# plt.show()

# 可以使用使用一个均匀分布把这个分布包起来
# 对于这个高斯分布，我们是可以确定他的峰值的
print(f"该高斯分布的峰值为：{ 1 / (standard_deviation * math.sqrt(2 * math.pi)) }")

# 所以我们用一个峰值为0.2的均匀分布包裹这个高斯分布
plt.plot(x_values, y_values.pdf(x_values))
plt.plot(x_values, 0.2 * np.ones_like(x_values), '--', label='k*q(z)')
plt.legend()
plt.show()


# 接受拒绝采样
def accept_reject():
    while True:
        u = random.uniform(0, 0.2)
        q = random.uniform(-2, 2)
        x = y_values.pdf(q)
        if u <= x:
            return q

# 采样
samples = [accept_reject() for _ in range(100_000)]

# 绘制采样结果，以及各种分布的对比
plt.plot(x_values, y_values.pdf(x_values), label='p(z)')
normed_value = y_values.cdf(2.0) - y_values.cdf(-2.0)
plt.plot(x_values, y_values.pdf(x_values) / normed_value, label='normed pdf')

plt.plot(x_values, 0.2 * np.ones_like(x_values), '--', label='k*q(z)')
plt.hist(samples, bins=20, density=True, label='sampling')
plt.legend()
plt.show()




# ======================================================
# 马尔可夫链   平稳分布
# ======================================================

# 例如： 一个简单的马尔可夫链
# 1. 有三个状态：单身，恋爱，结婚
# 2. 单身有0.2的概率保持单身，0.8的概率恋爱，0的概率结婚
# 3. 恋爱有0.2的概率变为单身，0.6的概率保持恋爱，0.2的概率结婚
# 4. 结婚有0.1的概率变为单身，0的概率变为恋爱状态，0.9的概率保持结婚

# 状态转移矩阵
transfer_matrix = np.array([[0.2, 0.8, 0.0],[0.2, 0.6, 0.2],[0.1, 0.0, 0.9]], dtype=np.float32)
dist = np.array([1.0, 0.0, 0.0], dtype=np.float32) #初始状态

def get_long_time_distribution(transfer_matrix, dist, steps):
    # 马尔可夫链的平稳分布
    single = []  # 记录单身的概率
    love = []  # 记录恋爱的概率
    married = []  # 记录结婚的概率

    for _ in range(steps):
        dist = np.dot(dist, transfer_matrix)
        single.append(dist[0])
        love.append(dist[1])
        married.append(dist[2])

    print(f"平稳分布：{dist}")

    x = np.arange(steps)
    plt.plot(x, single, label='single')
    plt.plot(x, love, label='love')
    plt.plot(x, married, label='married')
    plt.legend()
    plt.show()


get_long_time_distribution(transfer_matrix, dist, 30)
# 换一个初始状态，看看平稳分布
dist = np.array([0.4, 0.3, 0.3], dtype=np.float32)
get_long_time_distribution(transfer_matrix, dist, 30)


# ======================================================
# 马尔可夫链蒙特卡洛算法 （MCMC）
# 1. 输入任意给定的马尔可夫链状态转移矩阵Q（通常就是一个某一点附近采样的分布，比如高斯分布等）
# 2. 给定一个初始状态x0   for t = 0, ..., T
# 3. 从条件分布Q(x|xt)中采样得到x*
# 4. 从[0, 1]均匀分布中得到u
# 5. 如果 u <= pi(x*)Q(x*|xt)  则接受x*，否则继续保留xt

# 注意，通常为了保证输出结果比较好，一般从某个样本n开始才真正使用
# 前面的样本会被丢弃，因为早期的样本是由收敛的状态分布产生的
# ======================================================

# ======================================================
# MCMC算法2： Metropolis-Hastings算法
# 1. 输入任意给定的马尔可夫链状态转移矩阵Q（通常就是一个某一点附近采样的分布，比如高斯分布等）
# 2. 给定一个初始状态x0   for t = 0, ..., T
# 3. 从条件分布Q(x|xt)中采样得到x*
# 4. 从[0, 1]均匀分布中得到u
# 5. 如果 u <= min( (pi(x*)Q(x*|xt) / (pi(xt)Q(xt|x*))) , 1 ), 则输出x*，否则输出xt
# ======================================================

# MCMC算法1

def norm_dist_prob(x, mean = 1, std = 2):
    return norm(mean, std).pdf(x)

n_1 = 1000

T = 50_000
pi = [0 for _ in range(T)] # 保存采样结果
t = 2
sigma = 1
while t < T -1 :
    t = t + 1
    p_new = norm.rvs(loc=pi[t-1], scale=sigma, size=1, random_state=None) # 在上一步的范围随机采样，这里sigma范围是1
    alpha = norm_dist_prob(p_new) * norm(p_new, sigma).pdf(pi[t-1])
    u = random.uniform(0, 1)
    if u < alpha:
        pi[t] = p_new[0]
    else:
        pi[t] = pi[t-1]

plt.scatter(pi[1000:], norm_dist_prob(pi[1000:]), label='Target Distribution')
num_bins = 50
plt.hist(pi[1000:], num_bins, density=True, label='MCMC Sampling Distribution', alpha=0.7)
plt.legend()
plt.show()


# MCMC算法2： Metropolis-Hastings算法
T = 50_000
pi = [0 for _ in range(T)] # 保存采样结果
t = 0
sigma = 1

while t < T - 1:
    t = t + 1
    p_new = norm.rvs(loc=pi[t-1], scale=sigma, size=1, random_state=None) # 在上一步的范围随机采样，这里sigma范围是1
    alpha = min(1, norm_dist_prob(p_new[0]) / norm_dist_prob(pi[t-1]))
    u = random.uniform(0, 1)
    if u < alpha:
        pi[t] = p_new[0]
    else:
        pi[t] = pi[t-1]

plt.scatter(pi[1000:], norm_dist_prob(pi[1000:]), label='Target Distribution')
num_bins = 50
plt.hist(pi[1000:], num_bins, density=True, label='MCMC Sampling Distribution', alpha=0.7)
plt.legend()
plt.show()















