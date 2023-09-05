# -*- coding: utf-8 -*-
# Time : 2023/8/25 14:03
# Author : chen
# Software: PyCharm
# File : mlp_01.py
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve,auc
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import OneHotEncoder#独热处理-编码处理
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix#混淆矩阵
from sklearn.metrics import classification_report#分类报告
# 一、神经网络+独热
# 使用ex2data1.txt数据集完成神经网络训练
# 1、加载ex2data1.txt数据集
datas = np.loadtxt('ex2data1.txt',delimiter=',')
# 2、获取特征矩阵与标签矩阵
x = datas[:,:-1]
y = datas[:,-1:]
# 3、创建线性模型函数
def H(x,theta):#theta竖着放==#列向量
    return np.dot(x,theta)
# 4、创建激活函数/sigmoid函数/预测函数/逻辑模型函数
def S(h):
    return 1.0/(1.0+np.exp(-h))
# 5、代价函数
def J(a3,y):#h  s  a3
    return -np.mean(y*np.log(a3) +(1-y) * np.log(1-a3))
# 6、创建正向传播函数#模型
def FP(x,theta1,theta2):
    a1 = x
    h2 = H(a1,theta1)
    a2 = S(h2)
    h3 = H(a2, theta2)
    a3 = S(h3)
    return a1,a2,a3
# 7、创建反向传播函数
def BP(a1,a2,a3,y,theta2):
    #样本
    m = len(y)
    #输出层误差
    delet3 = a3 - y
    #隐藏层误差
    delet2 = np.dot(delet3,theta2.T) * a2 * (1-a2)
    #隐藏层的梯度值
    deletheta2 = 1.0/m * np.dot(a2.T,delet3)
    deletheta1 = 1.0/m * np.dot(a1.T,delet2)
    return deletheta1,deletheta2
# 8、创建梯度下降函数
# 9、隐藏神经元8个
def gredDesc(x,y,lr=0.01,h=8):
    am,an = x.shape
    bm,bn = y.shape
    theta1 = np.ones((an,h))
    theta2 = np.ones((h,bn))
    j_list= []
    for g in range(1000):
        a1, a2, a3 = FP(x,theta1,theta2)
        j = J(a3,y)
        j_list.append(j)
        deletheta1, deletheta2 = BP(a1,a2,a3,y,theta2)
        theta1 = theta1 - deletheta1 * lr
        theta2 = theta2 - deletheta2 * lr
    return theta1,theta2,j_list
# 10、创建精度函数/acc_score()
def acc_score(a3,y):#a3==>预测值  y==>真实值
    eq = np.equal(a3,y)
    acc_mean = np.mean(eq)
    return acc_mean
# 11、特征洗牌
np_x = np.random.permutation(len(x))
x = x[np_x]
y = y[np_x]
# 12、缩放
def ff_cc(x):
    a = np.mean(x)
    b = np.std(x)
    x = (x - a) / b
    return x
x = ff_cc(x)
# 13、数据拼接
x = np.c_[np.ones(len(x)),x]
y = np.c_[y]
# 15、标签独热化
y = OneHotEncoder(sparse=False).fit_transform(y)
# 14、切分训练集、测试集
train_x,test_x,train_y,test_y = train_test_split(x,y)
# 16、训练模型
# 17、调整超参数
theta1,theta2,j_list = gredDesc(train_x,train_y,lr=0.01)
# 18、画出代价函数图
plt.plot(j_list)
plt.show()
# 19、输出测试集预测值，预测值不能为概率，必须为整数
#1调模型函数==>fp()
a1,a2,a3 = FP(test_x,theta1,theta2)
test_where = np.where(a3>0.5,1,0)
print(f'预测值{test_where[:,1]}')
# 20、输出测试集精度==>acc_score()
test_score = acc_score(test_where,test_y)
print(f'测试集精度{test_score}')
#21输出训练集精度==>acc_score()
b1,b2,b3 = FP(train_x,theta1,theta2)
trian_where = np.where(b3>0.5,1,0)
train_score = acc_score(trian_where,train_y)
print(f'训练集精度{train_score}')
#22输出混淆矩阵=误差矩阵
print(f'混淆矩阵，{confusion_matrix(np.argmax(test_y,axis=1),np.argmax(test_where,axis=1))}')
#23输出分类报告==>分类评估指标
print(f'分类报告，{classification_report(np.argmax(test_y,axis=1),np.argmax(test_where,axis=1))}')
fpr,tpr,thresholds = roc_curve(test_y,test_where)
roc_auc = auc(fpr,tpr)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()