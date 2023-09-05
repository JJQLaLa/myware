# -*- coding: utf-8 -*-
# Time : 2023/8/28 15:25
# Author : chen
# Software: PyCharm
# File : 练习.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
#1.加载鸢尾花数据集
data_iris = load_iris()
#2.获取特征和标签
x = data_iris.data
y = data_iris.target
 # 3.只取y<2的类别，也就是0 1 并且只取前两个特征 （注：鸢尾花数据集有三个类别 只获取前两个类别 0和1）
x = x[y<2,:2]
# 4.只取y<2的类别
y = y[y<2]
# 5.分别画出类别 0 和 1 的点
plt.scatter(x[y==0,0],x[y==0,1],c='black')
plt.scatter(x[y==1,0],x[y==1,1],c='blue')
plt.show()
# 6.对鸢尾花数据集进行分类（模型自己选择）
s = SVC()
#7. 对于构建好的模型进行训练
s.fit(x,y)
# 8.请预测并输入
s_p = s.predict(x)
print(s_p)
print(len(s_p))
# 9.最后展示分类后的样本散点图类别
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()