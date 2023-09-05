import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#线性模型函数
def H(x,theta):
    return np.dot(x,theta)
#逻辑模型函数
def S(h):
    return 1.0/(1.0+np.exp(-h))
#神经网络正向传播/前向传播/神经网络模型函数
def FP(x,theta1,theta2):
    #输入层
    a1 = x
    #隐藏层
    a2 = S(H(a1,theta1))
    #输出层
    a3 = S(H(a2,theta2))
    #返回得到的三层值=即预测值
    return a1,a2,a3
#神经网络代价函数
def J(a3,y):
    j = -np.mean(y*np.log(a3) +(1-y) * np.log(1-a3))
    return j
#反向传播函数
def BP(a1,a2,a3,y,theta2):
    m = len(y)
    #输出层的误差
    delt3 = a3 - y
    #隐藏层的误差
    delt2 = np.dot(delt3,theta2.T) * a2*(1-a2)
    #隐藏层的梯度值
    deletheta2 = 1.0/m * np.dot(a2.T,delt3)
    #输入层的梯度值
    deletheta1 = 1.0 /m * np.dot(a1.T, delt2)
    return deletheta1,deletheta2

#梯度下降函数
#x==>特征 y==>标签 lr==>学习率 h==>指定隐藏层的神经元数量
#items_number==>梯度下降最大迭代次数
def G(x,y,lr=0.01,h=10,items_number=1000):
    #初始化theta
    xm,xn = x.shape
    ym,yn = y.shape
    theta1 = np.ones((xn,h))
    theta2 = np.ones((h,yn))
    j_list =  []
    for i in range(items_number):
        #计算模型值
        a1, a2, a3 = FP(x,theta1,theta2)
        #计算代价值
        j = J(a3,y)
        j_list.append(j)
        #计算梯度值
        deletheta1, deletheta2 = BP(a1,a2,a3,y,theta2)
        theta1 = theta1 - deletheta1 * lr
        theta2 = theta2 - deletheta2 * lr
    return theta1,theta2,j_list

def main():
    #加载数据集
    path = '../day12/data1.txt'
    datas = np.loadtxt(f'{path}',delimiter=',')
    #获取标签和特征
    x = datas[:,:2]
    y = datas[:,-1:]
    #数据集分割处理
    train_x,test_x,train_y,test_y= train_test_split(x,y)
    #模型训练
    theta1, theta2, j_list = G(train_x,train_y)
    plt.plot(j_list)
    plt.show()
if __name__ == '__main__':
    main()












