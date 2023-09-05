# -*- coding: utf-8 -*-
# Time : 2023/8/29 9:32
# Author : chen
# Software: PyCharm
# File : 深拷贝和浅拷贝.py
import copy
a = [1,2,3,4,['a','n']]
b = a#进行赋值
c = copy.copy(a)#浅拷贝
d = copy.deepcopy(a)#深拷贝
a.append(5)
a[4].append('c')
print('a=',a)
print('b=',b)
print('c=',c)
print('d=',d)
