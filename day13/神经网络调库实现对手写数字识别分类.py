# -*- coding: utf-8 -*-
# Time : 2023/8/28 15:51
# Author : chen
# Software: PyCharm
# File : 神经网络调库实现对手写数字识别分类.py
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.neural_network import MLPClassifier#神经网络分类模型
from sklearn.neural_network import MLPRegressor#神经网络回归模型
from sklearn.metrics import confusion_matrix#混淆矩阵
from sklearn.metrics import classification_report#分类报告
from sklearn.datasets import load_digits#手写数字识别数据集
from sklearn.model_selection import train_test_split#数据集分割库
from sklearn.metrics import accuracy_score
#导入统计相关库     混淆矩阵 分类报告
#导入手写数字识别数据集
datas_digits = load_digits()
x = datas_digits.data
y = datas_digits.target
#手写数字识别数据集imgX: 共包含5000个数据，分为10类，每个数据包含400属性
#标签集labely: 共包含50 00个数据，每个数据为数字1-10
#独立属性: 数据包含400个独立属性, 代表每个手写数字图像的点阵数据
#数据预处理
#洗牌
m = len(x)
np.random.seed(3)#随机数种子
np_perm = np.random.permutation(m)
x = x[np_perm]
y = y[np_perm]
#数据切分
train_x,test_x,train_y,test_y= train_test_split(x,y,test_size=0.2)
#alpha正则化参数<==>学习率
# 隐藏层共2层，单元数分别为400,100；正则化参数0.1；最大迭代次数为300
mlp = MLPClassifier(hidden_layer_sizes=(400,100),alpha=0.1,max_iter=300)
# -使用训练集完成模型的训练，并计算和输出训练集和测试集的准确率
mlp.fit(train_x,train_y.ravel())#.ravel()将数据转换为1维
print('score--训练集准确率=',mlp.score(train_x,train_y))
print('acc_score--训练集准确率=',accuracy_score(train_y,mlp.predict(train_x)))
print('score--测试集准确率=',mlp.score(test_x,test_y))
print('acc_score--测试集准确率=',accuracy_score(test_y,mlp.predict(test_x)))
# -分别计算并输出训练集和测试集的混淆矩阵和分类报告
print('训练集混淆矩阵\n',confusion_matrix(train_y,mlp.predict(train_x)))
print('训练集分类报告\n',classification_report(train_y,mlp.predict(train_x)))
print('测试集混淆矩阵\n',confusion_matrix(test_y,mlp.predict(test_x)))
print('测试集分类报告\n',classification_report(test_y,mlp.predict(test_x)))