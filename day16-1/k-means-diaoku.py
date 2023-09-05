# -*- coding: utf-8 -*-
# Time : 2023/8/29 11:44
# Author : chen
# Software: PyCharm
# File : k-means-diaoku.py
import numpy as np
from sklearn.cluster import KMeans#cluster聚类
from sklearn.cluster import MiniBatchKMeans#cluster聚类
x = np.loadtxt('../day17-1/luckynome.txt', delimiter=',')
k = KMeans(n_clusters=2)
k.fit(x)
y = k.labels_
print(y)
#聚类后的矩阵









