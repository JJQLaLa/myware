# -*- coding: utf-8 -*-
# Time : 2023/8/28 16:33
# Author : chen
# Software: PyCharm
# File : 练习1.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# 1.完成数据集的加载smt.txt
datas = np.loadtxt('smt.txt',delimiter=',')
# 2.对数据集进行初始化
x = datas[:,:2]
y = datas[:,-1:]
# 3.将数据集洗牌
m = len(x)
np.random.seed(1)
np_perm = np.random.permutation(m)
x = x[np_perm]
y = y[np_perm]
# 4.合理分割成训练集和测试集
train_x,test_x,train_y,test_y = train_test_split(x,y)
# 5.合理选取神经网络模型结构
# 6.并对神经网络权重theta进行初始化
# 7.实现激活函数
def H(x,theta):
    return np.dot(x,theta)
def S(h):
    return 1.0/(1.0+np.exp(-h))
def FP(x,theta1,theta2):
    a1 = x
    h2 = H(a1,theta1)
    a2 = S(h2)
    h3 = H(a2,theta2)
    a3 = S(h3)
    return a1,a2,a3
def BP(a1,a2,a3,y,theta2,):
    #输出层的误差
    dele3 = a3 - y
    #隐藏层的误差
    dele2 = np.dot(dele3,theta2.T) * a2*(1-a2)
    #隐藏层的梯度
    deletheta2 = 1.0/m * np.dot(a2.T,dele3)
    #隐输入层的梯度
    deletheta1 = 1.0/m * np.dot(a1.T,dele2)
    return deletheta1,deletheta2
def J(a3,y):
    return -np.mean(y*np.log(a3) + (1-y)* np.log(1-a3))
# 8.实现梯度下降并记录代价函数
# -实现正向传播算法
# -实现反向传播算法
# -记录代价函数
def gredDesc(x,y,lr=0.01,hidden=8):
    am,an = x.shape
    bm,bn = y.shape
    theta1 = np.ones((an,hidden))
    theta2 = np.ones((hidden,bn))#此时bn=1
    j_list = []
    for i in range(1000):
        #计算模型值==>前向传播
        a1,a2,a3 = FP(x,theta1,theta2)
        #计算代价值
        j = J(a3,y)
        j_list.append(j)
        #计算梯度值
        deletheta1, deletheta2 = BP(a1,a2,a3,y,theta2)
        theta1 = theta1 - deletheta1 * lr
        theta2 = theta2 - deletheta2 * lr
    return theta1,theta2,j_list
def acc_score(a3,y):
    eq = np.equal(a3,y)
    mean = np.mean(eq)
    return mean
if __name__ == '__main__':
    # 9.完成模型的训练
    theta1, theta2, j_list = gredDesc(train_x,train_y,lr=0.01)
    # 10.并计算在训练集上的准确率
    a1,a2,a3 = FP(train_x,theta1,theta2)
    train_where = np.where(a3>0.5,1,0)
    train_score = acc_score(train_where,train_y)
    print('训练集上的准确率=',train_score)
    # 11.画出代价函数曲线
    plt.plot(j_list)
    plt.show()
    # # 12.在测试集上完成了预测
    b1,b2,b3 = FP(test_x,theta1,theta2)
    test_where = np.where(b3>0.5,1,0)
    print('预测值',test_where)
    print(test_where.shape)
    print(test_y.shape)
    # # 13.计算在测试集上的准确率
    test_score = acc_score(test_where,test_y)
    print('测试集上的准确率=',test_score)
    # # 14.输出模型值
    print('theta1',theta1)
    print('theta2',theta2)
    c1,c2,c3 = FP(test_x,theta1,theta2)
    test_where1 = np.where(c3>0.5,1,0)
    print('此时模型值为=',test_where1)
    # 15.输出代价值
    print('代价值',j_list[-1])
