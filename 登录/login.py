import

# 登录页面的URL
login_url = "http://example.com/login"

# 登录所需的用户名和密码
username = "your_username"
password = "your_password"

# 创建一个会话
session = requests.Session()

# 发送GET请求，获取登录页面的内容，用于提取登录所需的表单数据
response = session.get(login_url)
login_page_content = response.text

# 提取登录所需的表单数据，例如表单的字段名为"username"和"password"
# 可以使用BeautifulSoup或正则表达式等方法提取
# 假设表单的字段名为"username"和"password"
form_data = {
    "username": username,
    "password": password
}

# 发送POST请求，模拟登录
response = session.post(login_url, data=form_data)

# 检查登录是否成功
if response.status_code == 200:
    print("登录成功！")
else:
    print("登录失败！")