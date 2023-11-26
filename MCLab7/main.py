from kmean import Kmeans, dis_chebyshev, dis_euclidean, dis_std_euclidean, dis_manhattan
from dataset import X, y

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans as skl

sns.set_theme(palette='cool')

fig, axes = plt.subplots(2, 2, dpi=150, figsize=(12, 12))
for i, ax in enumerate(axes.reshape(-1), 1):
    his1 = []
    his2 = []
    his3 = []
    his4 = []
    his5 = []
    for k in range(2, 22, 2):
        p1 = Kmeans(k, max_iter=10 * i)
        p2 = skl(n_clusters=k, n_init='auto', max_iter=10 * i)
        p3 = Kmeans(k, max_iter=10 * i, dis_func=dis_std_euclidean)
        p4 = Kmeans(k, max_iter=10 * i, dis_func=dis_manhattan)
        p5 = Kmeans(k, max_iter=10 * i, dis_func=dis_chebyshev)

        p1.fit(X)
        p2.fit(X)
        p3.fit(X)
        p4.fit(X)
        p5.fit(X)
        his1.append(p1.inertia_)
        his2.append(p2.inertia_)
        his3.append(p3.inertia_)
        his4.append(p4.inertia_)
        his5.append(p5.inertia_)
    ax.plot(range(2, 22, 2), his1, label='kmeans')
    ax.plot(range(2, 22, 2), his2, label='sklearn')
    ax.plot(range(2, 22, 2), his3, label='std_euclidean')
    ax.plot(range(2, 22, 2), his4, label='manhattan')
    ax.plot(range(2, 22, 2), his5, label='chebyshev')
    ax.set_xticks(range(2, 22, 2))
    ax.legend()
    ax.set_title(f'max_iter={10 * i}')
plt.show()
