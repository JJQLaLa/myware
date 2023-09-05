# -*- coding: utf-8 -*-
# Time : 2023/8/29 10:49
# Author : chen
# Software: PyCharm
# File : k-means-底层实现.py
#1导包
import numpy as np
import matplotlib.pyplot as plt
import copy
#生成样本点
X = np.array([
    [1,2],
    [2,3],
    [5,6],
    [7,8]
])
#生成k个质心
C = np.array([[1.0,2.0],[2.0,2.0]])
#展示处理前的样本点和质心
plt.scatter(X[:,0],X[:,1],s=15,c='b')
plt.scatter(C[:,0],C[:,1],c='r')
plt.show()
while True:
    #计算样本点到质心的距离
    B = []
    C_ = copy.deepcopy(C)
    for c in C:
        #shape[0]  shape[1]
        #欧式距离代码实现
        dist = np.sqrt(((X - c)**2).sum(axis=1))
        B.append(dist)
    #求归属 分类 判断每个样本点距离哪个质心较近
    min_dix = np.argmax(B,axis=0)
    #更新质心
    for i in range(len(C)):
        C[i] = np.mean(X[min_dix==i],axis=0)
        #如果计算的样本得出的中心点跟质心距离是一样的（质心不在移动） 那么就over 相反  接着进行
        # 第二步过程
    if (C==C_).all():
        break

    plt.scatter(X[:, 0], X[:, 1], s=15, c='b')
    plt.scatter(C[:, 0], C[:, 1], c='r')
    plt.show()



















