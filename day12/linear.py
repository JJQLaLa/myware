# -*- coding: utf-8 -*-
# Time : 2023/8/25 11:46
# Author : chen
# Software: PyCharm
# File : linear.py
# 1.导入相关的库
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split

# 2.加载数据
diabetes = load_diabetes()

# 3.打印输出数据的键
print(diabetes.keys())

# 4.取出data和target中的数据作为X和Y
X = diabetes.data
Y = diabetes.target

# 5.打印输出X，Y数据的维度，并打印前5个数据
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("First 5 X values:")
print(X[:5])
print("First 5 Y values:")
print(Y[:5])

# 6.仅取出X数据中的一个特征做训练
X = X[:, np.newaxis, 2]

# 7.切分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# 8.定义线性回归类
class LinearRegression():
    def __init__(self, learning_rate=0.001, num_iterations=10000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    # 9.定义代价函数
    def cost_func(self, X, Y, w, b):
        m = len(Y)
        cost = (1 / (2 * m)) * np.sum((np.dot(X, w) + b - Y) ** 2)
        return cost

    # 10.定义训练函数
    def train(self, X, Y):
        m, n = X.shape
        self.w = np.zeros((n, 1))
        self.b = 0

        for i in range(self.num_iterations):
            prediction = np.dot(X, self.w) + self.b
            error = prediction - Y

            dw = (1 / m) * np.dot(X.T, error)
            db = (1 / m) * np.sum(error)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b -self.learning_rate * db

            # 13.每500次打印并输出代价
            if i % 500 == 0:
                cost = self.cost_func(X, Y, self.w, self.b)
                print("Cost after", i, "iterations:", cost)

    # 11.定义预测函数
    def predict(self, X):
        prediction = np.dot(X, self.w) + self.b
        return prediction


# 12.实例化线性回归类
lr = LinearRegression()

# 14.训练模型
lr.train(X_train, Y_train)

# 15.使用训练好的权重和偏置在测试集上进行预测
predictions = lr.predict(X_test)
print("Predictions on test set:")
print(predictions)
