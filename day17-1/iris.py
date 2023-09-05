# -*- coding: utf-8 -*-
# Time : 2023/8/31 16:08
# Author : chen
# Software: PyCharm
# File : iris.py
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
# Pca调库实现
# 1.加载乳腺癌数据集
data = load_breast_cancer()
# 2.获取特征和标签
x = data.data
y = data.target
# 3.调用pca库函数训练模型
p = PCA(2)
p.fit(x)
new_x = p.transform(x)
print(new_x)
# 4.输出特征值的方差
print('特征值的方差',p.explained_variance_)
# 5.输出特征值的比例
print('特征值的比例',p.explained_variance_ratio_)
# 6.绘制癌症数据分布
plt.scatter(new_x[:,0],new_x[:,1],c=y,marker='+',s=50)
plt.show()
