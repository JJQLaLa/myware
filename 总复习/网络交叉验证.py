# -*- coding: utf-8 -*-
# Time : 2023/9/5 8:37
# Author : chen
# Software: PyCharm
# File : 网络交叉验证.py
import KNN
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

model = LogisticRegression(solver='liblinear', multi_class='ovr')
scores = cross_val_score(model, X, y, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
