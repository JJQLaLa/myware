# -*- coding: utf-8 -*-
# Time : 2023/9/1 16:31
# Author : chen
# Software: PyCharm
# File : 决策树001.py
import graphviz
from sklearn import tree
from sklearn.datasets import load_wine#红酒数据集
from sklearn.tree import DecisionTreeClassifier#分类模型
data = load_wine()
x = data.data
y = data.target
d = DecisionTreeClassifier(
    random_state=0,
    # criterion='gini',#悉尼系数
    criterion='entropy'#信息增益
)##悉尼系数
d.fit(x,y)
# feature_names = ['酒精'、'苹果酸'、灰、灰分的碱度、镁、总酚、黄酮类化合物、非黄烷类酚类、原花色素、颜色强度、色调、稀释葡萄酒的OD280/OD315、脯氨]
ress = tree.export_graphviz(
    d,
    out_file=None,
    feature_names = data.feature_names,
    filled=True,
    rounded=True,
    class_names = ['0','1','2']
)
graphviz = graphviz.Source(ress)
graphviz.render('data-pdf')
graphviz.view()#视图


f_d = d.feature_importances_
arr = [*zip(f_d,d.feature_importances_)]
print(arr)












