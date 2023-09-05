# PCA算法（Principal Component Analysis）是一种线性降维的算法，
# 用于将高维数据降低到低维空间。其步骤如下：

# 1. 将数据集表示为m行n列的矩阵X，其中每一行代表一个数据样本。
# 2. 对数据集的每一个特征（每一行）减去各自特征的平均值，即进行零均值化处理，
# 得到归一化后的矩阵B。
# 3. 求协方差矩阵C，即C=1/n * B * B.T，其中n是数据样本数量。
# 4. 对协方差矩阵C进行特征值分解，得到特征值和特征向量。
# 5. 将特征值从大到小排序，并选择其中最大的k个特征值对应的特征向量作为行向量，
# 得到特征向量矩阵P。
# 6. 将原始数据集X与特征向量矩阵P相乘，得到降维后的结果Y=PX。
#
# 在Python中，可以使用numpy库实现PCA算法。具体的代码实现如下：

# ```python
import numpy as np

# 对数据中心化（零均值化）处理
def Centralization_function(data):
    '''
    :param data: data为{x1,x2...xn}的多维向量，设维度为m维的话，考可以将数据集写出m行n列的矩阵A(m*n)
    :return: 矩阵data-data中相应向量的平均数
    '''
    zero_mean_matrix = np.nanmean(data, axis=0)
    return data - zero_mean_matrix

def pca_svd(data, k):
    new_data = Centralization_function(data)
    Cov_mat = 1 / k * np.dot(new_data.T, new_data)
    U, s, V = np.linalg.svd(Cov_mat)
    pc = np.dot(new_data, U)
    print(pc)
    return pc[:, 0]

def test():
    data = np.array([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2, 1.6],
        [1, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
    ])
    result_eig = pca_svd(data, 1)
    print(result_eig)

if __name__ == '__main__':
    test()

# 以上是对PCA算法的简要介绍和Python实现代码。其中，
# test函数中的data代表输入的数据集，k代表降维后的维度。运行test函数后，
# 将输出降维后的结果