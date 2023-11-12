import time

from utils import np, plt, sns, mpl, pd, MDS, Isomap, TSNE
from pca import PCA
from dataset import XY, Z, mnist_xy, mnist_z
from plot import plot_3d

if __name__ == '__main__':
    # sns.set_style('darkgrid')

    # 瑞士卷：测试不同降维算法
    # dps = [PCA(n_composition=2), MDS(n_jobs=8, normalized_stress='auto'), Isomap(n_neighbors=5, n_jobs=8),
    #        TSNE(angle=0.1, n_iter=2000, n_jobs=8)]
    # fig, axes = plt.subplots(2, 2, dpi=300, figsize=(16, 16))
    # for p, ax in zip(dps, axes.reshape(-1)):
    #     # p = TSNE(n_components=2)
    #     t = time.perf_counter()
    #     xy = p.fit_transform(XY)
    #     ax.scatter(xy[:, 0], xy[:, 1], marker='.', s=1, c=Z, cmap='rainbow')
    #     ax.set(
    #         title=f'{p.__class__.__name__}({time.perf_counter() - t:.2f}s)',
    #         xticks=np.linspace(-20, 20, 10),
    #         yticks=np.linspace(-20, 20, 10),
    #     )
    #     ax.set_xticks(np.linspace(-20, 20, 10),color='w')
    #     # ax.set_yticks(np.linspace(-20, 20, 10),color='w')
    #     ax.yaxis.set_major_locator(plt.NullLocator())
    #     ax.xaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(hspace=0.5)
    # plt.show()

    # MNIST：测试不同算法
    dps = [PCA(n_composition=2), MDS(n_jobs=8, normalized_stress='auto'), Isomap(n_neighbors=20, n_jobs=8), TSNE()]
    fig, axes = plt.subplots(1, 1, dpi=300, figsize=(8, 8))
    for p, ax in zip(dps, axes.reshape(-1)):
        t = time.perf_counter()
        xy = p.fit_transform(mnist_xy)
        for point, label in zip(xy, mnist_z):
            ax.text(*point, label, color=plt.cm.rainbow(label / 10), fontdict={'weight': 'bold', 'size': 4})
        ax.set(
            title=f'{p.__class__.__name__}({time.perf_counter() - t:.2f}s)',
            xlim=(np.min(xy[:, 0]), np.max(xy[:, 0])),
            ylim=(np.min(xy[:, 1]), np.max(xy[:, 1])),
        )
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(hspace=0.5)
    plt.show()

    # 瑞士卷：测试TSNE不同迭代步数
    # dps = [TSNE(n_jobs=8, n_iter=50+i * 500) for i in range(1, 7)]
    # fig, axes = plt.subplots(2, 3, dpi=300, figsize=(16, 12))
    # for p, ax in zip(dps, axes.reshape(-1)):
    #     # p = TSNE(n_components=2)
    #     xy = p.fit_transform(XY)
    #     ax.scatter(xy[:, 0], xy[:, 1], marker='.', s=1, c=Z, cmap='rainbow')
    #     ax.set(
    #         title=str(p.__class__.__name__) + str(p.n_iter),
    #         xticks=np.linspace(-20, 20, 10),
    #         yticks=np.linspace(-20, 20, 10),
    #     )
    #     # ax.set_xticks(np.linspace(-20, 20, 10),color='w')
    #     # ax.set_yticks(np.linspace(-20, 20, 10),color='w')
    #     ax.yaxis.set_major_locator(plt.NullLocator())
    #     ax.xaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(hspace=0.5)
    # plt.show()



    # MNIST：测试TSNE不同迭代步数
    # dps = [TSNE(n_jobs=8, n_iter=50+i * 500) for i in range(1, 7)]
    # fig, axes = plt.subplots(2, 3, dpi=300, figsize=(16, 12))
    # for p, ax in zip(dps, axes.reshape(-1)):
    #     # p = TSNE(n_components=2)
    #     xy = p.fit_transform(mnist_xy)
    #     ax.scatter(xy[:, 0], xy[:, 1], marker='.', s=1, c=mnist_z, cmap='rainbow')
    #     ax.set(
    #         title=str(p.__class__.__name__) + str(p.n_iter),
    #         xticks=np.linspace(-20, 20, 10),
    #         yticks=np.linspace(-20, 20, 10),
    #     )
    #     # ax.set_xticks(np.linspace(-20, 20, 10),color='w')
    #     # ax.set_yticks(np.linspace(-20, 20, 10),color='w')
    #     ax.yaxis.set_major_locator(plt.NullLocator())
    #     ax.xaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(hspace=0.5)
    # plt.show()
