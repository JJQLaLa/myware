# -*- coding: utf-8 -*-
# Time : 2023/8/26 9:05
# Author : chen
# Software: PyCharm
# File : sv_ss.py
import matplotlib.pyplot as plt
from sklearn.svm import SVC#svm模型
from sklearn.datasets import load_breast_cancer#乳腺癌数据集
from sklearn.model_selection import train_test_split
datas = load_breast_cancer()
x = datas.data
y = datas.target
train_x,test_x,train_y,test_y = train_test_split(x,y)
s = SVC(
    kernel='rbf'#线性不可分
)
#线性核  ‘linear’
#sigmoid核  'sigmoid'
#多项式核#   'poly'
#高斯核  'rbf'
s.fit(train_x,train_y)
test_x1 = s.predict(test_x)
print(test_x1)
plt.scatter(test_x[:,0],test_x[:,1],c=test_y.flatten(),marker='*')
plt.show()













