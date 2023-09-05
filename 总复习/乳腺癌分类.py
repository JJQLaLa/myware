# -*- coding: utf-8 -*-
# Time : 2023/9/4 15:35
# Author : chen
# Software: PyCharm
# File : 乳腺癌分类.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

### 导入数据，删除有缺失值的数据
# column_names=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data=pd.read_csv('breast-cancer.csv')
# data=data.replace(to_replace='?',value=np.nan)
# data=data.dropna(how='any')
# print(data.shape)
print(data.head(2))
# ### 分割数据集为训练集和测试集
# X_train,X_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
# # print(y_train.value_counts())
# # # print(y_test.value_counts())
#
# ### 标准化数据
# ss=StandardScaler()
# X_train=ss.fit_transform(X_train)
# X_test=ss.fit_transform(X_test)
#
# ### 用两种方法训练参数并进行预测
# lr=LogisticRegression()
# sgdc=SGDClassifier()
# lr.fit(X_train,y_train)
# lr_y_predict=lr.predict(X_test)
# # print(lr_y_predict)
# sgdc.fit(X_train,y_train)
# sgdc_y_predict=sgdc.predict(X_test)
# # print(lr_y_predict)
#
# ### 两种方法的性能评估
# print('Accuracy of LR Classifier:',lr.score(X_test,y_test))
# print(classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))
#
# print('Accuarcy of SGD Classifier:',sgdc.score(X_test,y_test))
# print(classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant']))
