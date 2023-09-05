# -*- coding: utf-8 -*-
# Time : 2023/8/31 15:13
# Author : chen
# Software: PyCharm
# File : pca底层实现.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
data = load_iris()
x = data.data
y = data.target
p = PCA(2)
p.fit(x)
new_x = p.transform(x)
color = ['red','blue','black']
for p in [0,1,2]:
    plt.scatter(new_x[y==p,0],
                new_x[y==p,1],
                alpha=1.0,
                c=color[p],
                label=data.target_names[p]
                )
plt.show()





#1.数据集表示的是m行n列  每一行代表的是数据样本
#2.均值缩放
# 协方差公式
def m_x(x):
    z_m = np.nanmean(x,axis=0)
    return x - z_m
def pca(x,n):
    new_x = m_x(x)
    cov = 1 / n * np.dot(new_x.T,new_x)
    u,s,v = np.linalg.svd(cov)
    #两列  获取一列
    cc = np.dot(new_x,u)
    return cc[:,0]
def mm():
    x = np.array([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2, 1.6],
        [1, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
    ])
    print(x.shape)
    rss = pca(x,1)
    print(rss.shape)
if __name__ == '__main__':
    mm()












