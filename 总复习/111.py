# -*- coding: utf-8 -*-
# Time : 2023/9/5 8:55
# Author : chen
# Software: PyCharm
# File : 111.py

from math import sqrt

import numpy as np

T = [[3, 104, 98],
    [2, 100, 93],
    [1, 81, 95],
    [101, 10, 16],
    [99, 5, 8],
    [98, 2, 7]]
x = [18, 90]
K = 5

dis = []
# 计算距离
for i in T:
    d = sqrt((i[0] - x[0]) ** 2 + (i[1] - x[1]) ** 2)
    dis.append(d)
# 排序
dis.sort()
print(dis)

y_pre = np.mean(dis[:K])
print(y_pre)
'''58.26267081096732'''