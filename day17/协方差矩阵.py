# -*- coding: utf-8 -*-
# Time : 2023/8/30 9:50
# Author : chen
# Software: PyCharm
# File : 协方差矩阵.py
# 1.数据集X,特征个数n=2,样本个数m=3
import numpy as np
X= np.array([[0.0,2.0],[1.0,1.0],[2.0,0.0]])#X.shape(2,3),
# m=3, n=2(1)
# 均值缩放
X = X - X.mean()
# (2) 转置得到X_
X_ = X.T
# (3) 求解协方差矩阵A=1.0/n*np.dot(X_,X_.T)
n = X_.shape[0]
A =1.0/n*np.dot(X_,X_.T)
print('A=',A)
print('np.cov()=', np.cov(X_))   #或者调用np.cov()
#求特征值和特征向量
cov_mat = np.cov(X.T)
eig_vals,eig_vecs = np.linalg.eig(cov_mat)
print('特征值',eig_vals)
print('特征向量',eig_vecs)
