#逻辑模型
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def S(x,theta):#逻辑模型函数
    return 1.0/(1.0+np.exp(-np.dot(x,theta)))


def J(s,y):#代价函数
    return -np.mean(y*np.log(s) + (1-y) * np.log(1-s))


def G(x,y,lr=0.01):#梯度下降函数
    m = len(y)
    theta = np.ones((x.shape[1],1))
    j_list = []
    for i in range(1000):
        s = S(x,theta)
        j = J(s,y)
        j_list.append(j)
        deltheta = 1.0/m * np.dot(x.T,s-y)
        theta = theta - deltheta * lr
    return theta,j_list

def acc_score(s,y):#准确率函数
    ew = np.where(s>0.5,1,0)
    eq = np.equal(ew,y)
    acc_mean = np.mean(eq)
    return acc_mean

def f_c(x):#特征缩放
    a = np.mean(x)
    b = np.std(x)
    x = (x - a) / b
    return x
def main():
    #加载数据集
    datas = np.loadtxt('apple (1).txt',delimiter=',')
    #获取特征和标签
    x = datas[:,:2]
    y = datas[:,-1:]
    #对x进行特征缩放 在逻辑里面不能对y进行特征缩放
    x = f_c(x)
    #数据拼接
    x = np.c_[(np.ones(len(x)), x)]
    #数据洗牌
    np_m = np.random.permutation(len(x))
    x = x[np_m]
    y = y[np_m]
    #数据集切分
    train_x, test_x, train_y, test_y = train_test_split(x, y)
    #模型训练
    theta, j_list = G(train_x,train_y,lr=0.1)
    #打印预测值
    test_pred = S(test_x,theta)
    ew = np.where(test_pred > 0.5, 1, 0)
    print(ew)
    #打印测试集的精度
    print(acc_score(ew, test_y))
    #打印训练集的精度
    train_pred = S(train_x, theta)
    et = np.where(train_pred > 0.5, 1, 0)
    print(acc_score(et, train_y))
    #绘制样本散点图
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thersholds = roc_curve(test_y, test_pred)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # plt.scatter(test_x[:, 1], test_x[:,2], c='r', marker='*')
    # plt.show()
if __name__ == '__main__':
    main()


































