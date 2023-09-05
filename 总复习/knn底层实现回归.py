# -*- coding: utf-8 -*-
# Time : 2023/9/5 8:52
# Author : chen
# Software: PyCharm
# File : knn底层实现回归.py
# 思想
# KNN（K-最近邻算法）的底层实现回归包括以下步骤：
#
# 1.  计算样本之间的距离：对于训练集中的每个样本，计算其与待预测样本之间的距离。常用的距离度量方法包括欧氏距离、曼哈顿距离等。
#
# 2.  选择最近的K个样本：根据计算得到的距离，选择距离最近的K个样本作为最近邻样本。
#
# 3.  对最近的K个样本进行加权平均：对于回归问题，通常使用加权平均的方法预测待预测样本的输出。加权平均可以使用距离的倒数作为权重，即距离越近的样本权重越大。
#
# 4.  预测输出：根据加权平均的结果，得到待预测样本的输出值。
#
# 需要注意的是，KNN算法在进行回归时对于分类问题有一些不同之处。
# 在分类问题中，KNN算法通常使用投票的方法确定类别，而在回归问题中
# KNN算法将最近的K个样本的输出作为预测输出。
# 此外，KNN算法中的K值也需要根据具体问题进行选择和调优。
import numpy as np


class KNNRegression:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            distances = self.calculate_distance(X_test[i])
            top_k_indices = np.argsort(distances)[:self.k]
            top_k_labels = self.y_train[top_k_indices]
            y_pred.append(np.mean(top_k_labels))
        return np.array(y_pred)

    def calculate_distance(self, x):
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

# 创建示例数据
X_train = np.array([[1, 2], [2, 3], [1, 1]])
y_train = np.array([2, 3, 1])
X_test = np.array([[1, 3], [2, 2]])

# 创建KNNRegression对象，并指定k值
knn = KNNRegression(k=2)

# 训练数据
knn.fit(X_train, y_train)

# 预测数据
y_pred = knn.predict(X_test)

print(y_pred)  # 输出预测结果
