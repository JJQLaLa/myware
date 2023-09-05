# -*- coding: utf-8 -*-
# Time : 2023/9/5 11:18
# Author : chen
# Software: PyCharm
# File : sql_login.py
import pymysql
conn = pymysql.Connection(
    host='localhost',
    port=3306,
    user='root',
    password='root'
)
cursor = conn.cursor()
conn.select_db('test')
cursor.execute("create table test_pymysql(id int);")
conn.close()