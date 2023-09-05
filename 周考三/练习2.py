# -*- coding: utf-8 -*-
# Time : 2023/8/28 17:10
# Author : chen
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# File : 练习2.py
def main():
    # 1.完成数据集的加载apple.txt
    datas = np.loadtxt('apple.txt',delimiter=',')
    # 2.获取特征和标签
    x = datas[:,:2]
    y = datas[:,-1:]
    # 3.进行特征缩放
    x = StandardScaler().fit_transform(x)
    # 4.对其添加常数项
    x = PolynomialFeatures().fit_transform(x)
    # 5.将数据集洗牌
    m = len(x)
    np.random.seed(0)
    np_perm = np.random.permutation(m)
    x = x[np_perm]
    y = y[np_perm]
    # 6.合理分割成训练集和测试集
    train_x,test_x,train_y,test_y = train_test_split(x,y)
    # 7.正确调用SVM库函数（惩罚系数0.6，使用高斯核函数）
    s = SVC(
        C=0.6,
        kernel='rbf'
    )
    # 8.训练模型
    s.fit(train_x,train_y)
    # 9.进行模型预测
    test_p = s.predict(test_x)
    print(test_p)
    # 10.求出训练集准确率
    print(s.score(train_x,train_y))
    # 11.求出测试集的准确率
    print(s.score(test_x,test_y))
    # 12.输出混淆矩阵
    print(confusion_matrix(test_y,s.predict(test_x)))
    # 13.输出分类报告
    print(classification_report(test_y,s.predict(test_x)))
    # 14.输出支持向量的索引
    print(s.support_)
    # 15.输出支持向量的列表
    print(s.support_vectors_)
    # 16.输出支持向量所属每个类别的个数
    print(s.n_support_)
    # 17.绘制整个样本散点图
    plt.scatter(test_x[:,1],test_x[:,2],c='r',marker='+')
    plt.show()
if __name__ == '__main__':
    main()