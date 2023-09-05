# -*- coding: utf-8 -*-
# Time : 2023/9/5 8:47
# Author : chen
# Software: PyCharm
# File : 鸢尾花数据交叉验证.py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 创建支持向量机模型
model = SVC()

# 创建网络交叉验证分割器
cross_val = StratifiedKFold(n_splits=5)

# 进行网络交叉验证
scores = []
for train_index, test_index in cross_val.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练模型
    model.fit(X_train, y_train)

    # 评估模型
    score = model.score(X_test, y_test)
    scores.append(score)

# 输出每次交叉验证的得分
print("交叉验证得分：", scores)
print("平均得分：", np.mean(scores))
