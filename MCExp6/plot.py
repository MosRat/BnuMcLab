from utils import np, plt, sns, mpl
from dataset import XY, Z


def plot_3d(x, y, z, c, s, ax=None, show=False,
            xticks=np.linspace(-15, 15, 6),
            yticks=np.linspace(-15, 15, 6), ):
    if ax is None:
        fig = plt.figure(dpi=150, figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=c, s=s,
               cmap='rainbow')
    ax.set(
        xticks=xticks,
        yticks=yticks,
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
    )
    if show:
        plt.show()
    return ax


def plot_2d(x, y, c, ax=None,
            xticks=np.linspace(-15, 15, 6),
            yticks=np.linspace(-1, 1, 6), show=False):
    if ax is None:
        fig = plt.figure(dpi=150, figsize=(16, 16))
        ax = fig.add_subplot(111)
    ax.scatter(x, y, c=c,
               cmap='rainbow')
    ax.set(
        # xticks=xticks,
        # yticks=yticks,
        xlabel='X',
        ylabel='Y',
        # zlabel='Z',
    )
    if show:
        plt.show()
    return ax


if __name__ == '__main__':
    plot_3d(XY[:, 0], XY[:, 1], XY[:, 2] * 4, c=Z, s=2.5, )
    # plot_2d(np.linspace(-15, 15, 5000), np.random.rand(5000), c=np.linspace(-1, 1, 5000))
    plt.show()
