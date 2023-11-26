import numpy as np
import pandas as pd
import matplotlib

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from dtreeviz import model

np.set_printoptions(precision=3)
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']

table = pd.read_excel('data.xls', skiprows=1).drop(['序号'], axis=1)
cats = set(table.columns[:-1])
data = table.values
data = np.concatenate([LabelEncoder().fit_transform(i).reshape(-1, 1) for i in data.T], axis=1)
vals = {i: list(pd.unique(table[i])) for i in table.columns}
D = len(data)
target = '是否给予贷款'


@np.vectorize
def entropy(p):
    """
    计算熵
    :param p: 概率
    :return: 熵
    """
    if np.isclose(p, 0):
        return 0.
    return -p * np.log2(p)


def ent(name, table, D):
    """
    计算某一特征分割下的熵
    :param name: 特征
    :param table: 数据集
    :param D: 数据集大小
    :return:
    """
    Di = np.array([sum(table[name] == i) for i in vals[name]]) # 这个是每个特征取值样本的数目
    matrix = np.array([
        [sum((table[target] == i) & (table[name] == j)) for i in vals[target]]
        for j in vals[name]
    ]) / Di.reshape(-1, 1) # 这个mxn矩阵，每列是每个分类取值样本，每行是每个该特征取值
    matrix = np.nan_to_num(matrix) # 处理0
    return Di / D * entropy(matrix).sum(axis=1) # 概率转化为熵并求和


def Gain(name, table):
    """
    计算信息增益
    :param name: 特征名称
    :param table: 数据集
    :return: 信息增益
    """
    D = len(table)
    ent_D = entropy(np.array([sum(table[target] == i) / D for i in vals[target]])).sum()
    ent_DA = ent(name, table, D).sum()
    print(name,ent_DA)


    # 以下代码计算A划分数据集D的熵，用来实现C4.5算法
    # ent_A = entropy(np.array([sum(table[name] == i) for i in vals[name]]) / D).sum()
    # print(f"ent_D:{ent_D}")

    return ent_D - ent_DA


def plot_tree(e, name, X_train, y_train):
    """
    绘制决策树图
    """
    viz_model = model(e,
                      X_train=X_train,
                      y_train=y_train,
                      feature_names=table.columns[:-1],
                      target_name='safe',
                      class_names=['no', 'yes'],

                      )
    v = viz_model.view()
    v.show()
    v.save(f"{name}.svg")


print('计算信息增益：', *((i, round(Gain(i, table), 3)) for i in cats))
div = max(((i, Gain(i, table)) for i in cats), key=lambda x: x[1])
print(f"选择{div[0]}")
cats.remove(div[0])

for c in vals[div[0]]:
    t = table.drop([div[0]], axis=1)[table[div[0]] == c]
    d = max(((i, Gain(i, t)) for i in cats), key=lambda x: x[1])
    print(f'计算信息增益 {div[0]}:{c}', *((i, round(Gain(i, t), 3)) for i in cats))
    print(f"选择{d[0]}")

    for subc in vals[d[0]]:
        y = dict(((t[t[d[0]] == subc])[target]).value_counts())
        print(f'{div[0]}:{c}->{d[0]}:{subc} ==> {y}')

tree = DecisionTreeClassifier(criterion="entropy", max_depth=2)

tree.fit(data[:, :-1], data[:, -1])
# print(tree)
# pprint.pprint(pt(tree))
# print(help(type(tree.tree_)))
# print(tree.feature_importances_)
# plot_tree(tree, '1', data[:, :-1], data[:, -1])
# print(tree.predict(data[:, :-1]))
# print(tree.score(data[:, :-1], data[:, -1]))
