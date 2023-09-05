# -*- coding: utf-8 -*-
# Time : 2023/9/4 9:24
# Author : chen
# Software: PyCharm
# File : knn1.py
import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, X1, X2):
        return np.sqrt(np.sum((X1 - X2) ** 2))

    def predict(self, X_test):
        y_pred = []

        for x in X_test:
            distances = []

            for i, x_train in enumerate(self.X_train):
                distance = self.euclidean_distance(x, x_train)
                distances.append((distance, self.y_train[i]))

            distances.sort(key=lambda x: x[0])

            k_nearest_neighbors = distances[:self.k]
            labels = [neighbor[1] for neighbor in k_nearest_neighbors]

            y_pred.append(max(set(labels), key=labels.count))

        return y_pred
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 7], [7, 9], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([[5, 5], [2, 2], [9, 8]])

knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(predictions)  # 输出：[1, 0, 1]
