# -*- coding: utf-8 -*-
# Time : 2023/8/28 14:32
# Author : chen
# Software: PyCharm
# File : 乳腺癌分类.py

# 乳腺癌诊断分类
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
cancers = load_breast_cancer()  # 下载乳腺癌数据集
X = cancers.data  # 获取特征值
Y = cancers.target  # 获取标签

# 训练集占80%，测试集占20%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
np.unique(Y)  # 查看label都由哪些分类  (unique去除数组中的重复数字，并进行排序之后输出)
plt.scatter(X[:, 0], X[:, 1], c=Y)  # 任选某2个特征 与 良性恶性的关系
X = X[Y<2,:2] # 只取y<2的类别，也就是0 1 并且只取前两个特征
Y = Y[Y<2] # 只取y<2的类别
plt.show() #显示图像
plt.scatter(X[Y==0,0],X[Y==0,1],color='red')
plt.scatter(X[Y==1,0],X[Y==1,1],color='blue')
plt.show()