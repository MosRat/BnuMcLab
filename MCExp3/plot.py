import matplotlib.pyplot as plt

import dataset
from utils import *
from dataset import *

sns.set_style('dark')
df['target'] = df['target'].map(dict(zip(range(3), labels)))
sns.set_palette("cool")


def plot_pairs():
    sns.pairplot(df, hue='target', palette="Pastel1")
    # plt.title("Distribution of IrisDataset")
    plt.show()


def plot_corr():
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.drop(['target'], axis=1).corr(), linewidth=1.5, fmt=".4f", annot=True, cmap="RdBu_r", vmin=-1.5,
                vmax=1.5)
    plt.show()


def plot_depos(ax, predict_data, y):
    # fig, ax = plt.subplots(figsize=(8, 8),dpi=300)
    scatter = ax.scatter(predict_data[:, 0], predict_data[:, 1], c=y, cmap=sns.color_palette("bwr", as_cmap=True))
    legend1 = ax.legend(*scatter.legend_elements(), title="Iris")
    ax.add_artist(legend1)


def plot_edge(ax, ceof, intercept):
    x = np.linspace(-3, 2, 100)
    ys = []
    for i, (w, b) in enumerate(zip(ceof, intercept)):
        y = -(w[0] * x + b) / w[1]
        ys.append(y)
        ax.plot(x, y, color=sns.color_palette("bwr")[i * 2], linewidth=6)
        ax.fill_between(x, -3 if i!=0 else 0, y, facecolor=sns.color_palette("bwr")[i * 2], alpha=0.3)
        ax.set_xlim(-3, 2)
        ax.set_ylim(-2.6, -1)


# plot_corr()
if __name__ == '__main__':
    plot_pairs()
    # print([1, 2, 3, 4][:-2:-1])
