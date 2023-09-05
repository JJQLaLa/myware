# -*- coding: utf-8 -*-
# Time : 2023/9/4 9:53
# Author : chen
# Software: PyCharm
# File : k-means-pca.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
#k-means做聚类
#1.用西瓜数据集做聚类
x = np.loadtxt('xigua.txt',delimiter=',')
#2.显示画图中文以及负号
plt.rcParams['font.sans-serif'] = 'SimHei'#显示中文
# plt.rcParams['axes.unicode_minus'] = False
#3.画出肘部法则图
k_list = np.arange(2,9)
innr_list = []
for k in k_list:
    kk = KMeans(n_clusters=k)
    kk.fit(x)
    inner = kk.inertia_
    innr_list.append(inner)
plt.plot(k_list,innr_list)
plt.title('肘部曲线')
plt.show()
#4.选择最优的k值创建kmeans模型拟合数据
ks = KMeans(3)
ks.fit(x)
#5.画出分类后的样本点  并根据标签上色分类
lables = ks.labels_
print(lables)
plt.scatter(x[:,0],x[:,1],c=lables.flatten())
plt.show()
#6.画出聚类中心
centers = ks.cluster_centers_#聚类中心
plt.scatter(centers[:,0],centers[:,1],c='r',marker='x')
plt.title('质心')
plt.show()


# Pca调库实现
# 1.加载乳腺癌数据集
datas = load_breast_cancer()
# 2.获取特征和标签
x = datas.data#;print(x)
y = datas.target
# 3.调用pca库函数训练模型
p = PCA(2)
new_x = p.fit_transform(x)
print(new_x)
# 4.输出特征值的方差
print('特征值的方差',p.explained_variance_)
# 5.输出特征值的比例
print('特征值的比例',p.explained_variance_ratio_)
# 6.绘制癌症数据分布图
plt.scatter(new_x[:,0],new_x[:,1],c=y,marker='+')
plt.show()
