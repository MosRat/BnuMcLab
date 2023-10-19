from typing import Dict, List, Tuple

from utils import *
from plot import plot_single
from data import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import functools

train_data, valid_data, test_data = load_dataset()

linear_params = {
    'C': [10 ** i for i in range(-5, 3)]
}
gaussian_params = {
    'C': [10 ** i for i in np.linspace(-3, 2, 10)],  # 范围内过采样便于画图
    'gamma': [10 ** i for i in range(-4, 0)]
}

linear_grid = GridSearchCV(
    SVC(kernel='linear'),
    param_grid=linear_params,
    n_jobs=6,
    cv=9,
)
gaussian_grid = GridSearchCV(
    SVC(),
    param_grid=gaussian_params,
    n_jobs=6,
    cv=9,
)


def k_split(i: int,
            k: int,
            X: np.ndarray,
            y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    K折分离数据集
    :param i:选取第几折作为验证集
    :param k:K折
    :param X:特征数据
    :param y:标签数据
    :return:训练集特征X、训练集特征y、测试集特征X，测试集特征y
    """
    length = len(X)
    return (np.concatenate([X[:i * (length // k)], X[(i + 1) * (length // k):]], axis=0),
            np.concatenate([y[:i * (length // k)], y[(i + 1) * (length // k):]], axis=0),
            X[i * (length // k):(i + 1) * (length // k)],
            y[i * (length // k):(i + 1) * (length // k)])


def grid_search(estimator: ClassifierMixin.__class__ | RegressorMixin.__class__,
                params: Dict[str, List],
                cv: int = 10,
                n_workers: int = 6,
                **kwargs) -> pd.DataFrame:
    """
    网格搜索函数，在指定决策器上，按照给定范围寻找最佳参数，采用K折交叉验证取平均计算某参数条件下决策器性能。

    :param estimator: 决策器类，不要实例化
    :param params: 待决策参数取值范围，将作为实例化决策器的参数
    :param cv: K交叉验证的K
    :param n_workers: 运算进程数
    :param kwargs: 决策器的共享参数，例如svc的kernel
    :return: dataframe，每一列为一各参数，最后一列是评价器得分，每一行是一种参数组合
    """
    grids = [{v: t[i] for i, v in enumerate(params)} for t in itertools.product(*(params.values()))]
    procs = {}
    his = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for param in grids:
            procs[pool.submit(single_fit, estimator, kwargs, param, cv)] = param
        for fu in tqdm(as_completed(procs), total=len(procs), desc=f'{n_workers} 进程训练：'):
            e, score = fu.result()
            procs[fu].update({'score': score})
            his.append((
                procs[fu], e
            ))
    optim_param, optim_estimator = max(his, key=lambda x: x[0]['score'])
    print(f'best param:{optim_param},best model: {optim_estimator}')
    df = pd.DataFrame([i[0] for i in his])
    # df['valid'] = True
    df['type'] = 'valid'
    df1 = df.copy()
    # df1['valid'] = False
    df1['type'] = 'test'
    df1['score'] = [i[1].score(test_data[:, 1:], test_data[:, 0]) for i in his]
    df = pd.concat([df, df1], axis=0)
    df['lg(c)'] = np.log10(df['C'])
    try:
        return df.sort_values(by=['lg(c)', 'gamma'])
    except KeyError:
        return df.sort_values(by=['lg(c)'])


def single_fit(estimator, kwargs, param, cv):
    """
    在指定决策器上拟合一次，用于多进程搜索,参数同搜索函数

    :param estimator:
    :param kwargs:
    :param param:
    :param cv:
    :return:
    """
    s = estimator(**kwargs, **param)
    his = []
    for i in range(cv):
        X, y, vX, vy = k_split(i, cv, train_data[:, 1:], train_data[:, 0])
        s.fit(X, y)
        his.append(s.score(vX, vy))
    return s, sum(his) / len(his)


if __name__ == '__main__':
    df = grid_search(SVC, linear_params, kernel='linear')
    plot_single(df)
    plt.show()

    df = grid_search(SVC, gaussian_params, )
    plot_single(df)
    plt.show()

    # 以下是使用sklearn.gridCv进行搜索,因为windows多进程的限制，不能和以上代码一起运行
    # linear_grid.fit(train_data[:, 1:], train_data[:, 0])
    # print(linear_grid.best_params_, linear_grid.best_score_)
    # gaussian_grid.fit(train_data[:, 1:], train_data[:, 0])
    # print(gaussian_grid.best_params_, gaussian_grid.best_score_)
