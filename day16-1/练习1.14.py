# -*- coding: utf-8 -*-
# Time : 2023/8/29 16:10
# Author : chen
# Software: PyCharm
# File : 练习1.14.py
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report#分类报告
# 三、Kmeans+svm调库
# 1、加载数据集data.txt，保存到x中
x = np.loadtxt('data.txt',delimiter=',')
x = x[:,:2]
# 2、创建kmeans方法
# 3、设置k为4
k = KMeans(4)
# 4、训练数据
k.fit(x)
# 5、将获得标签设置为y
y = k.labels_
# 6、创建svm模型
# 7、松弛因子惩罚系数为0.3
# 8、核函数为高斯核函数
s = SVC(
    C=0.3,
    kernel='rbf'
)
# 9、传入x,y进行训练
s.fit(x,y)
# 10、输出模型精度
print(s.score(x, y))
# 11、输出分类报告
print(classification_report(y,s.predict(x)))



