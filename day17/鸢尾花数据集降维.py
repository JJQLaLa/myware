# -*- coding: utf-8 -*-
# Time : 2023/8/31 11:49
# Author : chen
# Software: PyCharm
# File : 鸢尾花数据集降维.py
# 基于PCA实现对鸢尾花四维数据进行降维处理
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
Y = iris.target  # 数据集标签 ['setosa', 'versicolor', 'virginica']，山鸢尾、变色鸢尾、维吉尼亚鸢尾
X = iris.data  # 数据集特征 四维，花瓣的长度、宽度，花萼的长度、宽度
pca = PCA(n_components=2)
pca = pca.fit(X)
X_dr = pca.transform(X)
# 对三种鸢尾花分别绘图
colors = ['red', 'black', 'orange']
# iris.target_names
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_dr[Y == i, 0],
                X_dr[Y == i, 1],
                alpha=1,
                c=colors[i],
                label=iris.target_names[i])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()
