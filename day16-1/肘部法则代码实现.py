# -*- coding: utf-8 -*-
# Time : 2023/8/29 15:57
# Author : chen
# Software: PyCharm
# File : 肘部法则代码实现.py
import numpy as np
import matplotlib.pyplot as plt
X = np.loadtxt('../day17-1/luckynome.txt', delimiter=',')
from sklearn.cluster import KMeans
k_list = [2,3,4,5,6,7,8]
j_list = []
for k in k_list:
    #调用k-means方法
    kk = KMeans(n_clusters=k)
    kk.fit(X)
    #获取代价值
    j = kk.inertia_#代价值
    j_list.append(j)
#绘制肘部法则图/代价图
plt.plot(k_list,j_list)
plt.show()
kk1  = KMeans(3)
kk1.fit(X)
y = kk1.labels_
print(y)






