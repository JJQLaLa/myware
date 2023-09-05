# -*- coding: utf-8 -*-
# Time : 2023/9/1 15:37
# Author : chen
# Software: PyCharm
# File : 111.py
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import graphviz

wine = datasets.load_wine()
# print(wine) 字典形式
# print(wine.data)
# print(wine.data.shape) #(178, 13)一共有13个特征
# print(wine.target)

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
# test_size=0.3——>0.3是训练接，0.7是测试集
# print(x_train)
print(x_train.shape)  # (124, 13)
print(y_train.shape)  # (124,)

clf = tree.DecisionTreeClassifier(criterion="entropy")
#实例化
clf = clf.fit(x_train, y_train)  #训练模型
score = clf.score(x_test, y_test)  # 返回预测的精确度accuracy
print(score)  # 0.7962962962962963

feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']
#将特征值改为中文

dot_data = tree.export_graphviz(clf,
                                out_file='tree.dot',
                                feature_names=feature_name,
                                class_names=['琴酒', '雪莉', '贝尔摩德'],
                                filled=True,  #填充颜色
                                rounded=True  #边框略圆
                                )
with open('tree.dot', encoding='utf-8') as f:
    dot_grapth = f.read()
graph = graphviz.Source(dot_grapth.replace("helvetica", "MicrosoftYaHei"))
#为了显示中文，所以只能改变dot文件里的字体

graph.render(r'D:\wine')
graph.view()
