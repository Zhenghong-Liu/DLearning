import numpy as np
import scipy
import matplotlib.pyplot as plt

# 投掷硬币，观测得到正面的数量n_top, 反面的数量n_bottom, 以及总次数n
n = 10000
n_top = scipy.stats.binom.rvs(n, 0.5) # 二项分布, 从二项分布中随机抽取样本
n_bottom = n - n_top
print(f"一共投掷了{n}次硬币，其中{n_top}次是正面，{n_bottom}次是反面")

# 极大似然估计=============================================================
x = np.linspace(0, 1, 1001)
def likelihood(x):
    return scipy.stats.binom.pmf(n_top, n, x)
y = likelihood(x)

# 输出结果================================================================
print(f"Maximum likelihood estimate: {x[np.argmax(y)]}")
plt.plot(x, y)
# 在最大值的位置画一条竖线
plt.axvline(x[np.argmax(y)], color="red", linestyle="--")
plt.title("max Likehood")
plt.xlabel("Positive probability")
plt.ylabel("Likelihood")
plt.show()

# 结果发现：最大似然估计的概率和 n_top / n 极为相近。这其中应该是存在联系的。


