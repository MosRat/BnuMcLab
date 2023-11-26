import numpy as np
from dataset import X, y


def dis_euclidean(e, matrix):
    return np.sum((matrix - e) ** 2, axis=1)


def dis_std_euclidean(e, matrix):
    std = np.std(matrix, axis=0)
    if np.allclose(std, 0):
        return np.sum((matrix - e) ** 2, axis=1)
    matrix = np.nan_to_num((matrix - np.mean(matrix, axis=0)) / std)
    return np.sum((matrix - e) ** 2, axis=1)


def dis_manhattan(e, matrix):
    return np.sum(np.abs(matrix - e), axis=1)


def dis_chebyshev(e, matrix):
    return np.max(np.abs(matrix - e), axis=1)


class Kmeans:
    def __init__(self, k=10, max_iter=100, dis_func=dis_euclidean, random_state=42):
        self.centers = None
        self.k = k
        self.max_iter = max_iter
        self.random = np.random.RandomState(random_state)
        self.dis = dis_func

    def fit(self, X: np.ndarray):
        size = X.shape[0]
        init_index = self.random.randint(0, size, (self.k,))  # 随机初始化类中心
        centers = X[init_index, :]
        for _ in range(self.max_iter):  # 进行迭代
            m = np.c_[tuple(self.dis(c, X) for c in centers)]  # 计算各样本点到类中心的距离
            c = np.argmin(m, axis=1)  # 找出到对应样本距离最小的中心点
            centers_new = np.array([np.mean(X[c == i], axis=0) for i in range(self.k)])  # 根据各样本新的分类确定新的中心点
            if np.allclose(centers_new, centers):  # 如果中心点稳定那么退出迭代
                break
            else:
                centers = centers_new
        self.centers = centers_new
        self.inertia_ = sum(np.min(np.c_[tuple(dis_euclidean(c, X) for c in self.centers)], axis=1))  # 计算均方误差之和
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmin([self.dis(c, X) for c in self.centers], axis=1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans as skl

    # from sklearn.cluster import

    sns.set_theme(palette='cool')

    fig, axes = plt.subplots(2, 2, dpi=150, figsize=(12, 12))
    for i, ax in enumerate(axes.reshape(-1), 1):
        his1 = []
        his2 = []
        his3 = []
        his4 = []
        his5 = []
        for k in range(2, 22):
            p1 = Kmeans(k, max_iter=100 * i)
            p2 = skl(n_clusters=k, n_init='auto', max_iter=100 * i)
            # p3 = Kmeans(k, max_iter=10 * i, dis_func=dis_std_euclidean)
            # p4 = Kmeans(k, max_iter=10 * i, dis_func=dis_manhattan)
            # p5 = Kmeans(k, max_iter=10 * i, dis_func=dis_chebyshev)

            p1.fit(X)
            p2.fit(X)
            # p3.fit(X)
            # p4.fit(X)
            # p5.fit(X)
            his1.append(p1.inertia_)
            his2.append(p2.inertia_)
            # his3.append(p3.inertia_)
            # his4.append(p4.inertia_)
            # his5.append(p5.inertia_)
        ax.plot(range(2, 22), his1, label='kmeans')
        ax.plot(range(2, 22), his2, label='sklearn(kmeans++)')
        # ax.plot(range(2, 22, 2), his3, label='std_euclidean')
        # ax.plot(range(2, 22, 2), his4, label='manhattan')
        # ax.plot(range(2, 22, 2), his5, label='chebyshev')
        ax.set_xticks(range(2, 22))
        ax.legend()
        ax.set_title(f'max_iter={100 * i}')
    plt.show()
