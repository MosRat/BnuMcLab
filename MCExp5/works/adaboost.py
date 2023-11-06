# from math import log
import numpy as np

np.set_printoptions(precision=3)

x = np.arange(10)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1.])
w = np.full((10,), 0.1)
alphas = []
his = []


def alpha(err):
    """
    计算损失（即权重）alpha
    :param err: 原始损失
    :return:
    """
    return 0.5 * np.log((1 - err) / err)


def train():
    """
    训练模型
    :return:
    """
    global x, y, w, alphas
    for e in range(10):  # 训练10个弱分类器
        div = min(((i, j, 1 - sum(np.r_[y[:i] == j, y[i:] == -j] * w)) for i in range(10) for j in [1, -1]),
                  key=lambda x: x[-1])  # 以分类准确度为损失函数、训练弱分类器x>v
        his.append((div[0], div[1]))  # 记录弱分类器
        print(f'第{e + 1}个分类器：x<{div[0]} y = {div[1]} err={div[2]}')
        a = alpha(div[2])  # 记录弱分类器的alpha损失（权重）
        alphas.append(a)
        mask = np.array([np.exp(-a if i else a) for i in np.r_[y[:div[0]] == div[1], y[div[0]:] == -div[1]]])  # 更新样本权重
        w *= mask
        w /= w.sum()  # 概率归一化
        print(f'D{e}:{w}')


def G(x):
    """
    最终获得的强分类器
    :param x:
    :return:
    """
    # 根据弱分类器加权组合后的符号来预测
    return 1 if sum(np.array([j if x < i else -j for i, j in his]) * np.array(alphas)) > 0 else -1


if __name__ == '__main__':
    train()
    print('y    ', *y.astype(np.int_), sep='\t')
    print('y_hat', *(G(i) for i in range(10)), sep='\t')
    print('正确率：', sum(y.astype(np.int_)[i] == G(i) for i in range(10)) / 10)
