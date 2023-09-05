# -*- coding: utf-8 -*-
# Time : 2023/8/28 15:19
# Author : chen
# Software: PyCharm
# File : svm相关题型代码实现.py
import numpy as np
from sklearn.svm import SVC

# 设置随机种子
np.random.seed(1)

# 生成两个shape=[20,2]的随机数组
data1 = np.random.randn(20, 2)
data2 = np.random.randn(20, 2)

# 广播加3，减3的操作
data1 = data1 + 3
data2 = data2 - 3

# 拼接生成shape=[40,2]的X数据
X = np.concatenate((data1, data2), axis=0)
# 生成对应的标签
y = np.concatenate((np.ones(20), -np.ones(20)), axis=0)
# 创建SVC模型
model = SVC(kernel='linear')

# 训练模型
model.fit(X, y)

# 打印支持向量的数量
print("支持向量的数量：", model.support_vectors_.shape[0])

# 打印预测结果
print("预测结果：", model.predict(X))
