from numbers import Number
from typing import Iterable

from utils import *
from plot import plot_dis_2d
from estimate import *


def plot_kernel_dis(data_, debug=False):
    """
    绘制核概率密度函数
    :param data_: 数据，从data1,2,3,4中取
    :param debug: 打印估计器各个内部参量形状
    :return:
    """
    params = [0.25, 1, 4]
    kernels = [RectKernel, GaussianKernel]
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for i, kernel in enumerate(kernels):
        for j, param in enumerate(params):
            e = KernelDensityEstimator(kernel(param))
            try:
                e.fit(data_)
                x = np.linspace(-8, 8, 1200)
                pre = e.predict(x)
                axs[i, j].plot(x, pre, label='pre')
                plot_dis_2d(axs[i, j], data_)
                axs[i, j].set_title(f'k={kernel.__name__},h = {param}')
                axs[i, j].legend()
            except Exception as ex:
                raise ex
            finally:
                if debug:
                    for k, v in e.__dict__.items():
                        if isinstance(v, np.ndarray):
                            print(f'{k} :{v.shape}')
    plt.show()


def plot_KNeighbor_dis(data_: np.ndarray, ks: Iterable[Number], debug=False):
    """
    绘制K近邻概率密度估计图
    :param data_: 数据，从data1,2,3,4中取
    :param ks: k的取值范围，取六个值
    :param debug: 打印估计器各个内部参量形状
    :return:
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.reshape(-1)
    for i, k in enumerate(ks):
        e = KNeighborEstimator(k=k)
        try:
            e.fit(data_)
            x = np.linspace(-8, 8, 1600)
            pre = e.predict(x)
            axs[i].plot(x, pre, label='pre')
            plot_dis_2d(axs[i], data_)
            axs[i].set_title(f'k={k}')
            axs[i].legend()
        except Exception as ex:
            raise ex
        finally:
            if debug:
                for key, v in e.__dict__.items():
                    if isinstance(v, np.ndarray):
                        print(f'{key} :{v.shape}')
    plt.show()


if __name__ == '__main__':
    # data_ 取值dataset.datai,i=1,2,3,4 分别对应16、256、1000、2000点的估计
    ks = [2**i for i in range(6, 12)]
    plot_KNeighbor_dis(dataset.data4, ks)
    plot_kernel_dis(dataset.data4)
