"""
Math2Latex EM.py
2024年07月01日
by Zhenghong Liu
"""

'''
实现EM算法，估计抛硬币问题
有两个质量不均匀的硬币A和B,硬币A正面朝上的概率为PA，硬币B正面朝上的概率为PB
现在有一个硬币序列，每次以概率PP选择硬币A，以概率1-PP选择硬币B
在选定一个硬币后，连续抛掷10次，记录正面和反面的序列。如

A:正反反正正正正正反正
B:正正反正正正正正反正
A:正反反正正正正正反正
A:正反反正正正正正反正
B:正正反正正正正正反正

现在我在不知道该序列中，每行记录的是哪个硬币的情况下，如何估计PA,PB,PP
'''

'''
使用EM算法，首先根据PA=0.3,PB=0.6,PP=0.4生成一个硬币序列
然后使用EM算法估计PA,PB,PP

参考博客：https://zhuanlan.zhihu.com/p/439081824
'''

import numpy as np
import random

#设置随机数种子
random.seed(32)
def generate_data(PA, PB, PP, N):
    '''
    生成数据
    :param PA: 硬币A正面朝上的概率
    :param PB: 硬币B正面朝上的概率
    :param PP: 选择硬币A的概率
    :param N: 生成的序列长度
    :return: 硬币序列
    '''
    data = []
    for _ in range(N):
        if random.random() < PP:
            p = PA
        else:
            p = PB
        tmp = []
        for j in range(10):
            tmp.append('H' if random.random() < p else 'T')
        data.append(tmp[:])
    return data

def countH(lst):
    count = 0
    for c in lst:
        count += c == 'H'
    return count


def EM(data, PA, PB, PP, max_iter=100):
    '''
    EM算法
    :param data: 数据
    :param PA: 硬币A正面朝上的概率
    :param PB: 硬币B正面朝上的概率
    :param PP: 选择硬币A的概率
    :param max_iter: 最大迭代次数
    :return: 估计的PA,PB,PP
    '''
    for _ in range(max_iter):
        #E步
        PAH = 0
        PAT = 0
        PBH = 0
        PBT = 0
        isA = 0
        for record in memo:
            likeA = pow(PA, record) * pow(1-PA, 10-record)
            likeB = pow(PB, record) * pow(1-PB, 10-record)

            useA = likeA / (likeA + likeB)
            useB = 1 - useA

            PAH += useA * record
            PAT += useA * (10 - record)
            PBH += useB * record
            PBT += useB * (10 - record)
            isA += likeA > likeB

        #M步
        PA = PAH / (PAH + PAT)
        PB = PBH / (PBH + PBT)
        PP = isA / len(memo)
    return PA, PB, PP



if __name__ == '__main__':
    PA = 0.3
    PB = 0.6
    PP = 0.4
    data = generate_data(PA, PB, PP, 1000)
    memo = [countH(lst) for lst in data]
    totalH = sum(memo)

    print("尝试初值为0.2，0.7， 0.3==================")
    PA_, PB_, PP_ = EM(data, 0.2, 0.7, 0.3, max_iter=10)
    print('real PA:{}, PB:{}, PP:{}'.format(PA, PB, PP))
    print('estimated PA:{}, PB:{}, PP:{}'.format(PA_, PB_, PP_))

    print("尝试初值为0.5，0.5， 0.5==================")
    PA_, PB_, PP_ = EM(data, 0.5, 0.5, 0.5, max_iter=100)
    print('real PA:{}, PB:{}, PP:{}'.format(PA, PB, PP))
    print('estimated PA:{}, PB:{}, PP:{}'.format(PA_, PB_, PP_))

    print("尝试初值为0.7，0.2， 0.9==================")
    PA_, PB_, PP_ = EM(data, 0.7, 0.2, 0.9, max_iter=100)
    print('real PA:{}, PB:{}, PP:{}'.format(PA, PB, PP))
    print('estimated PA:{}, PB:{}, PP:{}'.format(PA_, PB_, PP_))

