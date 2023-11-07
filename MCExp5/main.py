from utils import *
from dataset import get_feature_name
from estimator import random_tree, ada_boost, gain_tree, gain_rate_tree
from plot import plot_tree, plot_line, X_train, y_train, X_test, y_test

# X_train, y_train, X_test, y_test = load_data()
# 定义决策器模型
features = get_feature_name()
ps = [gain_tree(i) for i in range(1, 8)] + [gain_rate_tree(i) for i in range(1, 8)]
ts = [random_tree(i * 10) for i in range(1, 9)]
ads = [ada_boost(i * 25) for i in range(1, 9, 2)]


def fit_and_score(p: ClassifierMixin) -> Tuple[ClassifierMixin, float]:
    """
    测试函数，训练并评价模型
    """
    p.fit(X_train, y_train)
    s = p.score(X_test, y_test)
    # print(s)
    return p, s


if __name__ == '__main__':
    # 测试决策树
    his = {'gain': [], 'gainRate': []}
    with multiprocess_as_completed(6, fit_and_score, ps,
                                   names=[f'gain_{i}' for i in range(1, 8)] + [f'gainRate_{i}' for i in
                                                                               range(1, 8)], ) as (it, names):
        for i in it:
            p, s = i.result()
            name = names[i]
            his[name.split('_')[0]].append(s)
            print(f'{name}: {s}')
    # 需要配置graphic才能绘制决策树的图
    # plot_tree(p, 'tree')
    print(p)
    plot_line(his.values(), his.keys(), 'decision_tree', 'max_depth', 'score')

    # 测试随机森林
    # his = {'train': [], 'test': []}
    # with multiprocess_as_completed(6, fit_and_score, ts, names=range(1, 9), ) as (it, names):
    #     for i in it:
    #         p, s = i.result()
    #         his['train'].append(p.score(X_train, y_train))
    #         his['test'].append(s)
    #         name = names[i]
    #         print(f'{name}: {s}')
    # # plot_tree(p.estimator_, 'forest')
    # plot_line(his.values(), his.keys(), 'random_forest', 'n_estimator($10^1$)', 'score')

    # 测试AdaBoost 每个迭代分类器的错误率
    # his = {f'n_estimator={i * 25}': [] for i in range(1, 9, 2)}
    # with multiprocess_as_completed(6, fit_and_score, ads,
    #                                names=[f'n_estimator={i * 25}' for i in range(1, 9, 2)], ) as (
    #         it, names):
    #     for i in it:
    #         p, s = i.result()
    #         name = names[i]
    #         his[name] = np.sort(p.estimator_errors_)[::-1]
    #         print(f'{name}: {s} ')
    # # print(p.estimator_.fit(X_train, y_train), p.estimator_.score(X_test, y_test))
    # # plot_tree(p.estimator_, 'adaboost')
    # plot_line(his.values(), his.keys(), 'AdaBoost', 'estimator_iter', 'error', marks=[None] * 4)

    # 测试AdaBoost不同分类器数目下的错误率
    # his = []
    # ads = [ada_boost(i * 5) for i in range(1, 20)]
    # with multiprocess_as_completed(6, fit_and_score, ads,
    #                                names=range(1, 20), ) as (
    #         it, names):
    #     for i in it:
    #         p, s = i.result()
    #         name = names[i]
    #         his.append(s)
    #         print(f'{name}: {s} ')
    # # print(p.estimator_.fit(X_train, y_train), p.estimator_.score(X_test, y_test))
    # # plot_tree(p.estimator_, 'adaboost')
    # plot_line([his], ['score'], 'AdaBoost', 'n_estimator($10^1$)', 'score')

    # 测试AdaBoost不同分类器（决策树、感知机、线性核支持向量机、SGD随机梯度下降）下的错误率
    # his = {'tree': [], 'mlp': [], 'SVC': [], 'SGD': []}
    # ads = ([ada_boost(i * 5) for i in range(1, 20)]
    #        + [ada_boost_mlp(i * 5) for i in range(1, 20)]
    #        + [ada_boost_svc(i * 5) for i in range(1, 20)]
    #        + [ada_boost_sgd(i * 5) for i in range(1, 20)]
    #        )
    # with multiprocess_as_completed(6, fit_and_score, ads,
    #                                names=[f'ada_boost_tree_{i * 5}' for i in range(1, 20)]
    #                                      + [f'ada_boost_mlp_{i * 5}' for i in range(1, 20)]
    #                                      + [f'ada_boost_SVC_{i * 5}' for i in range(1, 20)]
    #                                      + [f'ada_boost_SGD_{i * 5}' for i in range(1, 20)], ) as (
    #
    #         it, names):
    #     for i in it:
    #         p, s = i.result()
    #         name = names[i]
    #         his[name.split('_')[-2]].append(s)
    #         print(f'{name}: {s} ')
    # # print(p.estimator_.fit(X_train, y_train), p.estimator_.score(X_test, y_test))
    # # plot_tree(p.estimator_, 'adaboost')
    # plot_line(his.values(), his.keys(), 'AdaBoost', 'n_estimator($10^1$)', 'score')
