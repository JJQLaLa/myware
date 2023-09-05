# -*- coding: utf-8 -*-
# Time : 2023/8/29 8:40
# Author : chen
# Software: PyCharm
# File : k-means-底层实现.py
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.cluster import MiniBatchKMeans,k_means
X = np.array([
    [1, 2],
    [2, 3],
    [5, 6],
    [7, 8]]
)
# 定义k个质心
C = np.array([[1.0, 2.0], [2.0, 2.0]])

# 展示处理前的样本点和质心
plt.scatter(X[:, 0], X[:, 1], marker='.', s=15, c='b')
plt.scatter(C[:, 0], C[:, 1], marker='*', c='r')
plt.show()

while True:
    # 计算每个样本点到质心的距离
    B = []
    C_ = copy.deepcopy(C)
    for c in C:
        dist = np.sqrt(((X - c) ** 2).sum(axis=1))  # 欧式距离
        B.append(dist)

    # 求归属，分类，判断每个样本点，距离哪个质心更近
    min_dix = np.argmin(B, axis=0)

    # 更新质心
    for i in range(len(C)):
        C[i] = np.mean(X[min_dix == i], axis=0)
    # 如果计算得出的新中心点与原中心点一样（质心不再移动），那么结束，否则重新进行第二步过程
    if (C ==C_).all():
        break
    # 聚类后 质心和样本点的位置
    plt.scatter(X[:, 0], X[:, 1], marker='.', s=15, c='b')
    plt.scatter(C[:, 0], C[:, 1], marker='*', c='r')
    plt.show()