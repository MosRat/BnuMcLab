import numpy as np

from utils import *
from sklearn.model_selection import train_test_split


def load_data():
    """
    载入数据
    :return:
    """
    return np.load('c1.npy'), np.load('c2.npy'), np.load('c3.npy'), np.load('c4.npy'), np.load('c5.npy'), np.load(
        'c6.npy')


def make_label(cl1, cl2):
    """
    为两组数据生成标签，并将它们合并后打乱
    :param cl1: 第一组高斯分布
    :param cl2: 第二组高斯分布
    :return:
    """
    data = np.concatenate([c1, c2], axis=0)
    labels = np.concatenate([np.full((cl1.shape[0], 1), 0), np.full((cl2.shape[0], 1), 1)], axis=0)
    data_set = np.hstack((data, labels))
    np.random.shuffle(data_set)
    return data_set


def save_dataset():
    """
    保存生产的数据集
    :return:
    """
    data_set_1 = make_label(c1, c2)
    data_set_2 = make_label(c3, c4)
    data_set_3 = make_label(c5, c6)
    np.save("data_set_1", data_set_1)
    np.save("data_set_2", data_set_2)
    np.save("data_set_3", data_set_3)


def load_dataset():
    """
    载入数据
    :return:
    """
    return (
        np.load("data_set_1.npy"),
        np.load("data_set_2.npy"),
        np.load("data_set_3.npy"),
    )


if __name__ == '__main__':
    c1, c2, c3, c4, c5, c6 = load_data()
    # save_dataset()
    # def make_split()
    # print(make_label(c1, c2)[:, -1])
    # print(X_test.shape)
