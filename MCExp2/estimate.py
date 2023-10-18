from typing import Tuple

import matplotlib.pyplot as plt
from scipy.stats import norm

import dataset
from utils import *


class Estimator:

    def __init__(self, **kwargs):
        """
        估计器基类
        :param kwargs:
        """
        self.x = None
        self._distance = None
        self._data = None

    def __repr__(self):
        return f"<Estimator at {id(self)}>"

    def _make_dis_matrix(self):
        """
        计算分布矩阵
        :return:
        """
        self._distance = np.broadcast_to(self._data, (len(self._data), len(self._data)))
        # print(self._distance)

    def _cal_distance(self):
        """
        计算测试点到数据集的距离
        :return:
        """
        self._distance = np.abs(self._data - self.x)

    def fit(self, *args: np.ndarray | Tuple[np.ndarray], **kwargs):
        """
        拟合数据集
        :param args: 一维数组或者多个一维数组
        :param kwargs:
        :return:
        """
        if args:
            self._data = args[0]
        if kwargs:
            self._data = kwargs.get('data')
        self.n = len(self._data)

    def predict(self, x: np.ndarray | float) -> float | np.ndarray:
        raise NotImplementedError


class KNeighborEstimator(Estimator):
    def __init__(self, k=6, **kwargs):
        """
        K近邻预测器
        :param k:
        :param kwargs:
        """
        self.k = k
        super().__init__(**kwargs)

    def predict(self, x: np.ndarray | float) -> float | np.ndarray:
        """
        预测一个点或者一组点的概率密度
        :param x:
        :return:
        """
        if isinstance(x, float):
            self.x = np.array([x]).reshape(-1, 1)
        elif isinstance(x, np.ndarray):
            self.x = x.reshape(-1, 1)
        else:
            raise ValueError("Not allow type")
        self._cal_distance()
        self.order_dis = np.argsort(self._distance, axis=1)
        self.b_data = np.broadcast_to(self._data, self.order_dis.shape)
        self.edge = self.b_data[range(len(self.x)), self.order_dis[:, min(self.k - 1, self.n - 1)]]
        self.h = self._distance[range(len(self.x)), self.order_dis[:, min(self.k - 1, self.n - 1)]]
        self.inner = self.b_data[:, self.order_dis[:, :self.k]]
        self.p = (self.k / self.n / (2 * self.h)).reshape(-1)
        if len(self.p) == 1:
            return float(self.p)
        return self.p


class Kernel:
    def __init__(self, h: float, **kwargs):
        """
        核函数基类
        :param h:
        :param kwargs:
        """
        self.h = h

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class RectKernel(Kernel):
    """
    矩形核
    """

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        x /= self.h
        return np.sum((x < 0.5), axis=1, keepdims=True).astype(np.int_) / self.h


class GaussianKernel(Kernel):
    """
    高斯核
    """

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        x /= self.h
        return np.sum(norm.pdf(x), axis=1, keepdims=True) / self.h


class KernelDensityEstimator(Estimator):
    def __init__(self, kernel: Kernel, **kwargs):
        """
        核密度估计
        :param kernel:
        :param kwargs:
        """
        self.p = None
        self.kernel = kernel
        super().__init__(**kwargs)

    def predict(self, x: np.ndarray | float) -> float | np.ndarray:
        """
        预测一个点或者一组点的概率密度
        :param x:
        :return:
        """
        if isinstance(x, float):
            self.x = np.array([x]).reshape(-1, 1)
        elif isinstance(x, np.ndarray):
            self.x = x.reshape(-1, 1)
        else:
            raise ValueError("Not allow type")
        self._cal_distance()
        self.p = self.kernel(self._distance) / self.n
        if len(self.p) == 1:
            return float(self.p)
        return self.p


if __name__ == '__main__':
    # d = np.array([1, 2, 3])
    # print(np.broadcast_to(d, (len(d), len(d))))
    from plot import plot_dis_2d

    k = range(2, 10)

    #
    # plt.show()

    # raise ex

    # raise e
    # print(e)
    # print(np.arange(1, 10).reshape(3, 3)[:,[[0, 1], [2, 1], [0, 2]]])
