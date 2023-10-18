import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from utils import *
from data_gen import *

data = np.load('c1.npy')

# u = np.array([1, 1])  # 均值
# o = np.array([[1, 0], [0, 1]])  # 协方差矩阵
params_mean = [mean1, mean2, mean1, mean2, mean1, mean2]
params_cov = [cov, cov, cov_diff1, cov_diff1, cov_diff1, cov_diff2]


def plot_scatter(ax, data, label, edge=None):
    """
    绘制散点图及分类边界
    :param ax: 绘图轴
    :param data: 数据
    :param label: 标签
    :param edge: 边界
    :return:
    """
    ax.scatter(data[:, 0], data[:, 1], c=label, cmap='coolwarm')
    ax.plot(edge[:, 0], edge[:, 1], c="#FA7752")
    return ax


def make_gussin_data(u, o):
    """
    按照期望和协方差生成数据
    :param u: 期望
    :param o: 协方差
    :return:
    """
    num = 400
    l = np.linspace(-8, 8, num)
    X, Y = np.meshgrid(l, l)

    pos = np.concatenate((np.expand_dims(X, axis=2), np.expand_dims(Y, axis=2)), axis=2)  # 定义坐标点

    a = np.dot((pos - u), np.linalg.inv(o))  # o的逆矩阵
    b = np.expand_dims(pos - u, axis=3)
    # Z = np.dot(a.reshape(200*200,2),(pos-u).reshape(200*200,2).T)
    Z = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        Z[i] = np.array([np.dot(a[i, j], b[i, j]) for j in range(num)]).reshape(-1)  # 计算指数部分

    Z = np.exp(Z * (-0.5)) / (2 * np.pi * math.sqrt(np.linalg.det(o)))
    return X, Y, Z


def plot_3d_edge(ax, edge):
    """
    绘制决策面
    :param ax: 绘图轴
    :param edge: 边界数据
    :return:
    """
    x, y = edge[:, 0], edge[:, 1]
    # x = np.linspace(-8, 8, 40)
    z = np.linspace(-0.01, 0.03, 40)
    Z, X = np.meshgrid(z, x)
    Z, Y = np.meshgrid(z, y)
    # Z = np.ones_like(X)
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.6, cmap='Oranges')


def plot_dis_3d(ax, X, Y, Z):
    """
    绘制三维概率密度函数
    :param ax: 绘图轴
    :param X: X
    :param Y: Y
    :param Z: Z
    :return:
    """
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.6, cmap=cm.coolwarm)
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    cset = ax.contour(X, Y, Z, 20, zdir='z', offset=0, cmap=cm.coolwarm)  # 绘制xy面投影
    cset = ax.contour(X, Y, Z, zdir='x', offset=-2, cmap=mpl.cm.winter)  # 绘制zy面投影
    # cset = ax.contour(X, Y, Z, zdir='y', offset=8, cmap=mpl.cm.winter)  # 绘制zx面投影

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_3d_main():
    fig = plt.figure(dpi=200, figsize=(10, 15))  # 绘制图像

    for i in range(1, 4):
        ax = fig.add_subplot(int(f'31{i}'), projection='3d')
        X, Y, Z = make_gussin_data(params_mean[2 * i - 2], params_cov[2 * i - 2])
        plot_dis_3d(ax, X, Y, Z)
        X, Y, Z = make_gussin_data(params_mean[2 * i - 1], params_cov[2 * i - 1])
        ax.set_title(f'C{i} distribute')
        plot_dis_3d(ax, X, Y, Z)
    plt.show()


def plot_2d_main():
    fig = plt.figure(dpi=150, figsize=(12, 16))  # 绘制图像

    for i in range(1, 4):
        plot_2d(fig, i)

    plt.show()


def plot_2d(fig, i):
    # ax = fig.add_subplot(int(f'31{i}'))
    ax = plt.gca()
    x, y = np.meshgrid(np.linspace(-6, 10, 100), np.linspace(-6, 10, 100))
    pos = np.dstack((x, y))

    # 计算二元正态分布的概率密度
    rv1 = multivariate_normal(params_mean[2 * i - 2], params_cov[i * 2 - 2])
    rv2 = multivariate_normal(params_mean[i * 2 - 1], params_cov[i * 2 - 1])
    pdf1 = rv1.pdf(pos)
    pdf2 = rv2.pdf(pos)
    ax.set_xlim([-6, 10])
    ax.set_xticks(np.arange(-6, 10))
    ax.set_ylim([-6, 10])
    ax.set_yticks(np.arange(-6, 10))
    ax.contourf(x, y, pdf1 + pdf2, cmap='coolwarm')
    # ax.contourf(x, y, pdf2, cmap='cool')
    # ax.set_colorbar(label='$p$')
    ax.grid(True)
    plt.colorbar(ax=ax, mappable=cm.ScalarMappable(cmap='coolwarm'))
    ax.set_title(f'C{2 * i - 1} vs C{2 * i}')
    return ax


def plot_matrix(data):
    """
    绘制混淆矩阵
    :param data: 数据
    :return:
    """
    plt.figure(dpi=200)
    sns.heatmap(data, cmap="coolwarm", annot=True, fmt=".3f", xticklabels=['C1_True', 'C2_True'],
                yticklabels=['C1_Predict', 'C2_predict'])
    plt.title("Confusion Matrix")

    plt.show()


if __name__ == '__main__':
    # 绘制三个数据集的概率密度曲面
    plot_3d_main()
    # 绘制三个数据集的等概率密度图
    # plot_2d_main()





    # fig = plt.figure(dpi=160, figsize=(8, 8))  # 绘制图像
    # ax = fig.add_subplot(int(f'111'), projection='3d')
    # x = np.linspace(-8, 8, 80).reshape(-1, 1)
    # y = -0.8 * x + 4
    # plot_3d_edge(ax, np.concatenate([x, y], axis=1))
    #
    # X, Y, Z = make_gussin_data(params_mean[0], params_cov[0])
    # plot_dis_3d(ax, X, Y, Z)
    #
    # X, Y, Z = make_gussin_data(params_mean[1], params_cov[1])
    # plot_dis_3d(ax, X, Y, Z)
    #
    # # ax.set_title(f'C{i} distribute')
    # # plot_dis_3d(ax, X, Y, Z)
    # plt.show()
    # fig = plt.figure(dpi=100)
    # plot_2d(fig,1)
    # plot_2d(fig,2)
    # plot_2d(fig,3)
    # plt.show()
    # plot_3d_main()
# plot_dis_3d(*m
# pake_gussin_data(u, o))
# plot_2d_main()
# plt.show()
# plot_matrix([[1, 0], [0, 1]])
