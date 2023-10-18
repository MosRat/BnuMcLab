import matplotlib.pyplot as plt

from utils import *

sns.set_style('whitegrid')
sns.set_palette('Pastel1')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率


def plot_single(data):
    try:
        a = data['gamma']
        sns.relplot(data=data,
                    x='lg(c)',
                    y='score',
                    col='gamma',
                    kind='line',
                    style='gamma',
                    hue='gamma',
                    palette='Pastel1',
                    markers=True,
                    # size='score',
                    col_wrap=2
                    )
    except KeyError:
        sns.relplot(data=data,
                    x='lg(c)',
                    y='score',
                    kind='line',
                    palette='Pastel1',
                    markers=True,
                    # size='score',
                    legend='full'
                    )
    # plt.legend()

    # plt.show()
