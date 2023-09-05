# -*- coding: utf-8 -*-
# Time : 2023/8/28 17:22
# Author : chen
# Software: PyCharm
# File : 练习3.py
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# 1、数据集加载
# a)正确添加数据集d2.txt
aa = np.loadtxt('d2.txt',delimiter=',')
x  = aa[:,:-1]
y  = aa[:,-1:]
# 2、数据预处理：
# a)特征缩放
x = StandardScaler().fit_transform(x)
# b)添加常数项
x= PolynomialFeatures().fit_transform(x)
# 3、数据选择：
#  a)将数据分为训练集和测试集两部分
# b)训练集75%
#  c)测试集25%
train_x,test_x,train_y,test_y = train_test_split(x,y)
#  4、创建支持向量机模型
#  b)惩罚系数：15
#  c)gamma：0.5
s = SVC(
    C=15,
    gamma=0.5
)
#  5、训练模型
s.fit(train_x,train_y)
#  a)输出测试集准确率
print(s.score(test_x,test_y))
#  b)输出测试集的混淆矩阵
print(confusion_matrix(test_y,s.predict(test_x)))
#  c)输出测试集的分类报告
print(classification_report(test_y,s.predict(test_x)))
#  7、输出模型重要属性：
# 输出模型的支持向量
print('模型的支持向量',s.support_)
# a)支持向量的下标
print(s.support_vectors_)
#  8、输出模型重要属性：
# 输出每种类别支持向量的个数
print(s.n_support_)