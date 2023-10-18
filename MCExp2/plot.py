import matplotlib.pyplot as plt

from utils import *
from dataset import *
import seaborn as sns

large = 22;
med = 16;
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'figure.dpi': 150,
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
# plt.style.use('seaborn-whitegrid')
sns.set_style("darkgrid")


def plot_dis_2d(ax, data_: np.ndarray):
    """
    绘制单个概率密度图
    :param ax:
    :param data_:
    :return:
    """
    data_ = pd.Series(data_)
    ax.hist(data_,
            bins=20,
            density=True,
            histtype='barstacked',
            color='#B1CDE3')
    data_.plot(ax=ax, kind='kde', color='#FFB5AF', label="real")
    # plt.show()


def plot_dis_data():
    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1)
        plot_dis_2d(ax, vars().get(f'data{i + 1}'))
        ax.set_title(f"data{i + 1}")
    plt.show()


if __name__ == '__main__':
    plot_dis_data()
