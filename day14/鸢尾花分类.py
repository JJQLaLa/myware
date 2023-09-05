# -*- coding: utf-8 -*-
# Time : 2023/8/28 14:52
# Author : chen
# Software: PyCharm
# File : 鸢尾花分类.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#加载鸢尾花数据集
iris = datasets.load_iris()
#获取特征和标签
X = iris.data
Y = iris.target

X = X[Y<2,:2] # 只取y<2的类别，也就是0 1 并且只取前两个特征
Y = Y[Y<2] # 只取y<2的类别

# 分别画出类别 0 和 1 的点
plt.scatter(X[Y==0,0],X[Y==0,1],color='red')
plt.scatter(X[Y==1,0],X[Y==1,1],color='blue')
plt.show()
# 对鸢尾花数据集进行分类（模型自己选择）
# 对于构建好的模型进行训练
# 请预测并输入
# 最后展示分类后的样本散点图类别