import itertools
from collections.abc import Iterable
from plot import *
from utils import *
from dataset import *


class Fishier:
    def __init__(self):
        """
        LDA分类器
        """
        self.W = None
        self.Sb = None
        self.Sw = None
        self.Si = None
        self.mu = None
        self.norm = None
        self.m = None
        self.nums = None
        self.types = None
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        在指定数据集上拟合
        :param X: 样本特征
        :param y: 样本标签
        :return: None
        """
        self.X = X
        self.y = y
        # 类数量和每类的样本数
        self.types, self.nums = np.unique(y, return_counts=True)
        # 每类均值向量
        self.m = np.array([X[y == idx].mean(axis=0) for idx in self.types])
        # 总体均值向量
        self.mu = X.mean(axis=0)
        # 样本减去自己类的均值
        self.norm = [X[y == idx] - self.m[i] for i, idx in enumerate(self.types)]
        # 每类的类内离散度矩阵
        self.Si = [sum(j.reshape(-1, 1) * j for j in i) for i in self.norm]
        # 所有类的类内离散度矩阵求和
        self.Sw = sum(self.Si)
        # 多分类的类间散度矩阵
        self.Sb = sum(m * ((mi - self.mu).reshape(-1, 1) * (mi - self.mu)) for mi, m in zip(self.m, self.nums))
        # 求解降维方向W，即矩阵Sw^-1@Sb的前n个特征值
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(self.Sw).dot(self.Sb))
        sorted_indices = np.argsort(eig_vals)
        # 此处n取2，即将为2维
        self.W = eig_vecs[:, sorted_indices[:-3:-1]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测（降维）数据
        :param X: 高维样本特征
        :return: 降维后样本
        """
        return X @ self.W

    def _debug(self):
        """
        调试用，打印实例中各个数组形状
        :return:
        """
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                print(f'{k}:{v.shape}')
            else:
                if isinstance(v, Iterable):
                    print(f'{k}:{[i.shape for i in v]}')
                else:
                    print(f'{k}:{v}')


def main_dp():
    try:
        fig, axs = plt.subplots(3, 2, dpi=300, figsize=(12, 18))
        # axs = axs.reshape(-1)
        for i, tp in enumerate(itertools.combinations(range(3), 2)):
            print(tp)
            X, y = make_data(tp)
            b = LDA()
            pd = b.fit_transform(X, y)
            a = Fishier()
            a.fit(X, y)
            predict_data = a.predict(X)
            plot_depos(axs[i][0], pd, y)
            plot_depos(axs[i][1], predict_data, y)
            p = Perceptron()
            p.fit(predict_data.reshape(-1, 1), y)
            print(p.coef_, p.intercept_)
            print(p.score(predict_data.reshape(-1, 1), y))
        plt.show()
        # print(p.predict(predict_data.reshape(-1, 1)) == y)
        # plot_depos(predict_data, y)
        # print()
    except Exception as e:
        a._debug()
        raise e


def main():
    plt.figure(dpi=300)
    try:
        b = LDA()
        pd = b.fit_transform(X, y)
        a = Fishier()
        a.fit(X, y)
        predict_data = a.predict(X)
        plot_depos(plt.gca(), predict_data, y)
        # P1、P2用作降维前后对比
        # p1 = Perceptron(fit_intercept=True, max_iter=1000, shuffle=True)
        # p2 = Perceptron(fit_intercept=True, max_iter=1000, shuffle=True)
        p1 = SVC(kernel='linear')
        p2 = SVC(kernel='linear')
        p1.fit(predict_data, y)
        p2.fit(X, y)
        # 使用非线性分类器时无法使用以下语句绘制决策面，需要注释掉
        plot_edge(plt.gca(), p1.coef_, p1.intercept_)
        print(f'降维后(2维)得分:{p1.score(predict_data, y)}')
        print(f'原始数据(4维)得分:{p2.score(X, y)}')
    except Exception as e:
        a._debug()
        raise e
    finally:
        plt.show()


def test():
    try:
        a = Fishier()
        a.fit(X, y)
        predict_data = a.predict(X)
        plt.scatter(predict_data[:, 0], predict_data[:, 1], c=y)
        plt.show()
        b = LDA()
        pd = b.fit_transform(X, y)
        plt.scatter(pd[:, 0], pd[:, 1], c=y)
        plt.show()

        # print(predict_data)
    except Exception as e:
        a._debug()
        raise e


if __name__ == '__main__':
    main()
