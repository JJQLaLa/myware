# -*- coding: utf-8 -*-
# Time : 2023/8/31 14:02
# Author : chen
# Software: PyCharm
# File : 复习.py
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
# 使用k-means调库实现以下试题
# 1.加载数据集luckynome.txt
x = np.loadtxt('luckynome.txt',delimiter=',')
    # 2.构建模型
k_list1 = np.arange(2,9)
j_list = []
for k in k_list1:
    kk = KMeans(n_clusters=k)
    # 3.训练模型
    kk.fit(x)
    # 4.获取代价值并添加到列表里
    j = kk.inertia_#代价值
    j_list.append(j)
# 5.绘制肘部曲线(代价图)得到最优的曲线
plt.plot(k_list1,j_list)
plt.show()
# 6.获取聚类后的y矩阵
kk1 = KMeans(3)
kk1.fit(x)
y = kk1.labels_
print('聚类后的y矩阵',y)





