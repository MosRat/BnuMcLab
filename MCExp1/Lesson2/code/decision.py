from typing import List
from plot import *

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from utils import *
from data_deal import *
from scipy.stats import multivariate_normal


def _norm_pdf(mu: np.ndarray, sigma: np.ndarray):
    """
    根据指定的期望和方差，返回多元高斯分布概率密度函数
    :param mu: 期望
    :param sigma: 协方差矩阵，必须半正定
    :return:  概率密度函数
    """

    def _pdf(value):
        a = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
        e = -0.5 * ((value - mu).T @ np.linalg.inv(sigma) @ (value - mu))
        return a * np.exp(e)

    # 与下面的语句等价
    # _pdf = multivariate_normal(mu, sigma).pdf

    return _pdf


class BayesDecision:
    def __init__(self, **kwargs):
        """
        贝叶斯决策器类
        :param kwargs:
        """
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_mean_c1 = self.X_mean_c2 = self.X_cov_c1 = self.X_cov_c2 = None
        self.pdf_c1 = self.pdf_c1 = lambda x: x
        self.last_predict = None

    def __repr__(self):
        return f'<BayesDecision Instance at {hex(id(self))}> \n\tprior:{self.prior}\n\tClass1 \n\t\tmu={self.X_mean_c1} \n\t\tsigma={self.X_cov_c1} \n\tClass2 \n\t\tmu={self.X_mean_c2} \n\t\tsigma={self.X_cov_c2} \n'

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def _make_train_test(self, dataset, rate=0.3333):
        """
        将数据集分为测试集和训练集，并分离标签和特征
        :param dataset: 数据集，本次实验中为6000*3的数组
        :param rate: 测试集占比
        :return: None
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset[:, :2], dataset[:, -1],
                                                                                random_state=42,
                                                                                shuffle=False, test_size=rate)

    def _cal_prior(self):
        """
        计算先验概率，用训练集的每类样本占比估计，由于标签选用整数0，1，故使用对训练集标签求均值实现

        :return: None
        """
        c1 = self.y_train.mean()
        self.prior = np.array([c1, 1 - c1])

    def _cal_likelihood_params(self):
        """

        使用最大似然估计对每个类概率密度进行参数估计，假设每个类符合高斯分布，使用训练集均值与协方差估计其期望和方差参数，进而得到概率密度函数

        :return: None
        """
        self.X_mean_c1 = self.X_train[self.y_train == 0].mean(axis=0)
        self.X_mean_c2 = self.X_train[self.y_train == 1].mean(axis=0)
        self.X_cov_c1 = (self.X_train[self.y_train == 0] - self.X_mean_c1).T @ (
                self.X_train[self.y_train == 0] - self.X_mean_c1) / len(self.X_train[self.y_train == 0])
        self.X_cov_c2 = (self.X_train[self.y_train == 1] - self.X_mean_c2).T @ (
                self.X_train[self.y_train == 1] - self.X_mean_c2) / len(self.X_train[self.y_train == 1])
        self.pdf_c1 = _norm_pdf(self.X_mean_c1, self.X_cov_c1)
        self.pdf_c2 = _norm_pdf(self.X_mean_c2, self.X_cov_c2)

    def _cal_likelihood_density(self, value: np.ndarray):
        """
        计算指定样本在每一类上的概率（似然概率），因为只需要比大小，故使用概率密度代替
        :param value: 样本特征向量,本次实验为 1*2数组
        :return: None
        """
        return np.array([np.log(self.pdf_c1(value)), np.log(self.pdf_c2(value))])

    def fit(self, *args, **kwargs) -> None:
        """
        在指定数据集拟合，计算每个类的先验概率密度和类概率密度函数并保存
        :param args: 数据集，可以是需要分割的完整数据集或者仅有训练集，后者标签与数据分开传入
        :param kwargs: 数据集分割参数
        :return: None
        """
        assert isinstance(args[0], np.ndarray)
        if len(args) == 1:
            self._make_train_test(args[0], **kwargs)
        elif len(args) == 2:
            self.X_train = args[0]
            self.y_train = args[1].reshape(-1, 1)
        self._cal_prior()
        self._cal_likelihood_params()

    def predict(self, data=None) -> np.ndarray:
        raise NotImplementedError

    def score(self, data=None, labels: np.ndarray = None) -> float:
        """
        评价分类器得分
        :param data: 测试样本集，如果没有则使用实例保存的测试数据
        :param labels: 测试标签 ，如果没有则使用实例保存的测试数据
        :return: 测试正确率，即预测标签等于标注的样本占比
        """
        if data is None:
            data = self.X_test
            labels = self.y_test
        predict = self.predict(data)
        return (predict == labels.reshape(-1, 1)).mean()

    def confuse_matrix(self, data=None, labels: np.ndarray = None) -> List:
        """
        计算混淆矩阵，验证模型的灵敏度和特异性
        :param data: 试样本集，如果没有则使用实例保存的测试数据
        :param labels: 测试标签 ，如果没有则使用实例保存的测试数据
        :return: 混淆矩阵
        """
        if self.last_predict is None:
            self.last_predict = self.predict(data)
        if labels is None:
            labels = self.y_test.reshape(-1, 1)
        return [[((self.last_predict == 0) & (labels == 0)).mean(),
                 ((self.last_predict == 0) & (labels == 1)).mean()],
                [((self.last_predict == 1) & (labels == 0)).mean(),
                 ((self.last_predict == 1) & (labels == 1)).mean()]
                ]

    def get_dec_edge(self, xrange: np.ndarray, yrange: np.ndarray) -> np.ndarray:
        """
        计算决策面，遍历指定范围，找出在两类中先验与似然之积近似相等的样本点
        :param xrange: x计算范围，由np.linspace生成
        :param yrange: y计算范围，由np.linspace生成
        :return: 处于决策面上的(x,y)，按 n*2 数组排列
        """
        result = []
        for x in xrange:
            for y in yrange:
                p1, p2 = self._cal_likelihood_density(np.array([x, y]))
                if np.isclose(p1, p2, rtol=1e-3):
                    print(p1, p2)
                    result.append((x, y))
        # print(result)
        return np.array(result)

    def f1_score(self, data=None, labels: np.ndarray = None) -> float:
        """
        计算模型的 f1 score
        :param data: 测试样本集，如果没有则使用实例保存的测试数据
        :param labels: 测试标签 ，如果没有则使用实例保存的测试数据
        :return: f1分数
        """
        (TP, FP),(FN, TN) = self.confuse_matrix(data, labels)
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        return 2 * (P * R) / (P + R)


class BayesMinErrorDecision(BayesDecision):
    """
    最小错误率贝叶斯决策
    """

    def predict(self, data=None) -> np.ndarray:
        """
        对指定测试集进行预测，将样本按照特征分类，具体实现方式为计算样本在各类的先验概率和似然概率（密度）的乘积，然后比较大小，选择大作为决策结果
        :param data:  测试集，如果没有则使用实例保存的测试数据
        :return: None
        """

        if data is None:
            data = self.X_test
        result = np.zeros(data.shape[0])

        for idx, s in enumerate(data):
            p = self._cal_likelihood_density(s) * self.prior
            result[idx] = np.argmax(p)
        self.last_predict = result.reshape(-1, 1)
        return self.last_predict


class BayesMinRiskDecision(BayesDecision):
    """
    最小风险贝叶斯决策
    """

    def __init__(self, decision_table: np.ndarray = None, **kwargs):
        super().__init__(**kwargs)
        if decision_table is None:
            decision_table = np.array([[0, 1], [1, 0]])
        assert decision_table.shape == (2, 2)
        self.decision_table = decision_table

    def predict(self, data=None) -> np.ndarray:
        """
        对指定测试集进行预测，将样本按照特征分类，具体实现方式为计算样本在各类的先验概率和似然概率（密度）的乘积，然后按决策表指定风险参数进行加权，最后根据计算结果大小进行分类
        :param data:  测试集，如果没有则使用实例保存的测试数据
        :return: None
        """

        if data is None:
            data = self.X_test
        result = np.zeros(data.shape[0])

        for idx, s in enumerate(data):
            p = self._cal_likelihood_density(s) * self.prior
            p = np.array([p @ self.decision_table[0], p @ self.decision_table[1]])
            result[idx] = np.argmin(p)
        self.last_predict = result.reshape(-1, 1)
        return self.last_predict


if __name__ == '__main__':
    d1, d2, d3 = load_dataset()

    handle = BayesMinErrorDecision()
    handle.fit(d3)
    print(handle)
    print(handle.predict())
    print(handle.score())
    print(handle.confuse_matrix())
    fig = plt.figure(dpi=300, figsize=(8, 8))
    plot_scatter(plt.gca(), handle.X_train, handle.y_train,
                 handle.get_dec_edge(np.linspace(-6, 8, 500), np.linspace(-6, 8, 500)))
    plt.show()

    print()
# print(np.linalg.norm(np.array([3, 4])))
