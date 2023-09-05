# -*- coding: utf-8 -*-
# Time : 2023/9/5 10:28
# Author : chen
# Software: PyCharm
# File : ui_oiuyt.py
# 用户注册 用户信息单独文件保存
def user_register():
    user_id = input('输入账户id：')
    user_pwd = input('输入用户密码：')
    user_name = input('输入用户昵称：')
    user = {'u_id': user_id, 'u_pwd': user_pwd, 'u_name': user_name}
    user_path = "F:/机器一项目文档/2109A/ui_login/user_path44" + user_id # 新建文件夹保存信息
    file_user = open(user_path, 'w')
    file_user.write(str(user))
    file_user.close()
user_register()
