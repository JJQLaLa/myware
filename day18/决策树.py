from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 训练决策树模型
clf = DecisionTreeClassifier(random_state=0,criterion="entropy")
clf.fit(X, y)
# 绘制决策树
dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    filled=True,
    rounded=True,
    class_names=["0","1","2"],
    feature_names=iris.feature_names)
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")  # 保存决策树到文件iris_decision_tree.pdf
graph.view()  # 在窗口中显示决策树
