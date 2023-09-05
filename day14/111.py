# -*- coding: utf-8 -*-
# Time : 2023/8/28 10:41
# Author : chen
# Software: PyCharm
# File : 111.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# 1.完成数据集的加载apple.txt
aa = np.loadtxt('../周考三/apple.txt', delimiter=',')
# 2.获取特征和标签
x = aa[:,:-1]
y = aa[:,-1:]
# 3.进行特征缩放
a_s = StandardScaler()
x = a_s.fit_transform(x)
# 4.对其添加常数项
a_p = PolynomialFeatures()
x = a_p.fit_transform(x)
# 5.将数据集洗牌
m = len(x)
jj = np.random.permutation(m)
x = x[jj]
y = y[jj]
# 6.合理分割成训练集和测试集
d=int(0.7*m)
# 计算数据集的分割位置
x_train,x_test=np.split(x,(d,),axis=0)   # 等同于 X_train1 = X[:d],X_test = X[d:]
y_train,y_test=np.split(y,(d,),axis=0)    # 等同于 y_train = y[:d],y_test = y[d:]
# 7.正确调用SVM库函数（惩罚系数0.6，使用高斯核函数）
svm = SVC(
    C=0.6,
    kernel='rbf'
)
# 8.训练模型
svm.fit(x_train,y_train)
# 9.进行模型预测predict()
s_pred = svm.predict(x_test)
print(s_pred)
# 10.求出训练集准确率
s_score = svm.score(x_train,y_train)
print(s_score)
# 11.求出测试集的准确率
s_score1 = svm.score(x_test,y_test)
print(s_score1)
# 12.输出混淆矩阵#输入都是测试集的
print(confusion_matrix(y_test,s_pred))
# 13.输出分类报告
print(classification_report(y_test,s_pred))
# 14.输出支持向量的索引
print('支持向量的索引',svm.support_)
# 15.输出支持向量的列表
print('支持向量的列表',svm.support_vectors_)
# 16.输出支持向量所属每个类别的个数
print('支持向量所属每个类别的个数',svm.n_support_)
# 17.绘制整个样本散点图
import matplotlib.pyplot as plt
plt.scatter(x_test[:,1],x_test[:,2],c='r',marker='+',s=20)
plt.show()



