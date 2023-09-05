# -*- coding: utf-8 -*-
# Time : 2023/8/28 15:38
# Author : chen
# Software: PyCharm
# File : 神将网络调库实现.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits#手写数字识别
#导入sklearn.neural_network库
from sklearn.neural_network import MLPClassifier

#导入统计相关库     混淆矩阵 分类报告
from sklearn.metrics import confusion_matrix,classification_report

#设置正常显示汉字、负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#导入手写数字识别数据集
#手写数字识别数据集imgX: 共包含5000个数据，分为10类，每个数据包含400属性
#标签集labely: 共包含50 00个数据，每个数据为数字1-10
#独立属性: 数据包含400个独立属性, 代表每个手写数字图像的点阵数据
data_digits  = load_digits()
# X = np.loadtxt(r'E:\PC_Project\machine learning1\day10\imagesData.txt',delimiter=',')
# y = np.loadtxt(r'E:\PC_Project\machine learning1\day10\labelsData.txt',delimiter=',')
X = data_digits.data
y = data_digits.target
#数据预处理
def preprocess(X):
    #进行0-1缩放特征
    X_min = np.min(X)
    X_max = np.max(X)
    X = (X-X_min)/(X_max-X_min)
    return X

X = preprocess(X)

#洗牌
m = len(X)
np.random.seed(3)  #设置随机种子
order = np.random.permutation(m)
X = X[order]
y = y[order]

#数据切分
a = int(0.7*m)
train_X,test_X = np.split(X,[a])
train_y,test_y = np.split(y,[a])

#alpha正则化参数   学习率
# 隐藏层共2层，单元数分别为400,100；正则化参数0.1；最大迭代次数为300
mlp = MLPClassifier(hidden_layer_sizes=(400,100),alpha=0.1,max_iter=300)

# -使用训练集完成模型的训练，并计算和输出训练集和测试集的准确率
mlp.fit(train_X,train_y.ravel())   #train_y要转成一维数组格式  .ravel()

print('训练集准确率=',mlp.score(train_X,train_y))
print('测试集准确率=',mlp.score(test_X,test_y))

# -分别计算并输出训练集和测试集的混淆矩阵和分类报告
#注意传入的第一个参数为真实值 第二个参数为预测值
print('训练集混淆矩阵\n',confusion_matrix(train_y,mlp.predict(train_X)))
print('测试集混淆矩阵\n',confusion_matrix(test_y,mlp.predict(test_X)))
print('训练集分类报告\n',classification_report(train_y,mlp.predict(train_X),))
print('测试集分类报告\n',classification_report(test_y,mlp.predict(test_X)))
