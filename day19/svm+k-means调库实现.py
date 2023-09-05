# -*- coding: utf-8 -*-
# Time : 2023/9/4 10:13
# Author : chen
# Software: PyCharm
# File : svm+k-means调库实现.py
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# 三、Kmeans+svm调库
# 1、加载数据集data.txt，保存到x中
x = np.loadtxt('data1.txt',delimiter=',')
x = x[:,:-1]
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
print('模型精度',s.score(x,y))
# 11、输出分类报告
print('分类报告',confusion_matrix(y,s.predict(x)))
# 12、输出混淆矩阵
print('混淆矩阵',classification_report(y,s.predict(x)))