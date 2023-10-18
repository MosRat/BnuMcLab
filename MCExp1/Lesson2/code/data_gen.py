# import numpy as np
# from sklearn.datasets import make_gaussian_quantiles

from utils import *

mean1 = np.array([
    0, 0
])
mean2 = np.array([
    4, 5
])
cov = np.diag([2, 4])
cov_diff1 = np.array([[4, 1],
                      [1, 3]])
cov_diff2 = np.array([[5, 2],
                      [2, 6]])

if __name__ == '__main__':
    # 生成六组数据，数据已经保存在pyi文件中，无需运行多次
    # 第一组和第二组均值为mean1，mean2，协方差均为对角矩阵cov，它们构成第一个训练集
    # 第三组和第四组均值为mean1，mean2，协方差均为非对角矩阵cov_diff1，它们构成第二个训练集
    # 第五组和第六组均值为mean1，mean2，协方差为非对角矩阵cov_diff1和cov_diff2，它们构成第三个训练集

    class1_data = np.random.multivariate_normal(mean1, cov, 3000)
    class2_data = np.random.multivariate_normal(mean2, cov, 3000)

    class3_data = np.random.multivariate_normal(mean1, cov_diff1, 3000)
    class4_data = np.random.multivariate_normal(mean2, cov_diff1, 3000)

    class5_data = np.random.multivariate_normal(mean1, cov_diff1, 3000)
    class6_data = np.random.multivariate_normal(mean2, cov_diff2, 3000)

    np.save('c1', class1_data)
    np.save('c2', class2_data)
    np.save('c3', class3_data)
    np.save('c4', class4_data)
    np.save('c5', class5_data)
    np.save('c6', class6_data)
