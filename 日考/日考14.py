# -*- coding: utf-8 -*-
# Time : 2023/8/28 17:11
# Author : chen
# Software: PyCharm
# File : 日考14.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
def main():
    # 1.完成数据集的加载和初始化（8分）
    datas = np.loadtxt('bread.txt',delimiter=',')
    x = datas[:,:2]
    y = datas[:,-1:]
    # 2.将数据集洗牌(6分)，合理分割成训练集和测试集(6分)
    m = len(x)
    np.random.seed(0)
    np_perm = np.random.permutation(m)
    x = x[np_perm]
    y = y[np_perm]
    train_x,test_x,train_y,test_y = train_test_split(x,y)
    # 3.正确调用SVM库函数并训练模型(30分)
    s = SVC()
    s.fit(train_x,train_y)
    # 4.分别求出训练集和测试集的准确率(20分)
    print('训练集的精度=',s.score(train_x,train_y))
    print('测试集的精度=',s.score(test_x,test_y))
    # 5.画出整个样本数据并画出分界线(30分)
    plt.scatter(test_x[:,0],test_x[:,1],c=test_y.flatten(),marker='x')
    plt.show()
if __name__ == '__main__':
    main()