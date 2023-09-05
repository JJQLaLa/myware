# -*- coding: utf-8 -*-
# Time : 2023/8/31 11:38
# Author : chen
# Software: PyCharm
# File : 数据降维.py

import numpy as np
import matplotlib.pyplot as plt
# 对数据中心化（零均值化）处理
def Centralization_function(data):
    '''
    :param data: data为{x1,x2...xn}的多维向量，设维度为m维的话，考可以将数据集写出m行n列的矩阵A(m*n)
    :return: 矩阵data-data中相应向量的平均数
    '''
    zero_mean_matrix = np.nanmean(data, axis=0)
    return data - zero_mean_matrix

def pca_svd(data, k):
    new_data = Centralization_function(data)
    Cov_mat = 1 / k * np.dot(new_data.T, new_data)
    U, s, V = np.linalg.svd(Cov_mat)
    pc = np.dot(new_data, U)
    return pc[:, 0]

def test():
    x = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])
    print(x.shape)
    result_eig = pca_svd(x, 1)
    print(result_eig.shape)
    plt.scatter(x[:,0],x[:,1],c='r')
    plt.show()

if __name__ == '__main__':
    test()