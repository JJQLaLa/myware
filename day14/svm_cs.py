# -*- coding: utf-8 -*-
# Time : 2023/8/28 9:27
# Author : chen
# Software: PyCharm
# File : svm_cs.py
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# datas = np.loadtxt('apple.txt',delimiter=',')
# X = datas[:,:-1]
# y = datas[:,-1:]
# s = SVC()
# s.fit(X,y)
# plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), zorder=10, cmap=plt.cm.Paired, edgecolor='k', s=20)
# plt.show()
# zorder: z方向上排列顺序，数值越大，在上方显示 # paired两个色彩相近输出(paired)
# 圈出测试数据plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors=‘none’, zorder=10, edgecolor=‘k’)