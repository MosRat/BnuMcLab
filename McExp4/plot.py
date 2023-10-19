import pandas as pd

from utils import *

sns.set_style('whitegrid')
sns.set_palette('Pastel1')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率


def plot_single(data: pd.DataFrame) -> None:
    try:
        a = data['gamma']
        sns.relplot(data=data,
                    x='lg(c)',
                    y='score',
                    col='gamma',
                    kind='line',
                    style='gamma',
                    hue='type',
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
                    hue='type',
                    # size='score',
                    legend='full'
                    )


if __name__ == '__main__':
    print(sns.load_dataset("flights")
          .pivot(index="year", columns="month", values="passengers")
          )
