from utils import *
from typing import Literal


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


def std(iter_, axis=0):
    """
    将数量转化为比例（概率）
    """
    arr = np.array(list(iter_))
    s = arr.sum(axis=axis, keepdims=True)
    res = np.nan_to_num(arr / s)
    return res


class _Tree:
    def __init__(self, depth: int, max_depth=2, criterion: Literal['gain', 'gain_rate'] = 'gain', X: np.ndarray = None,
                 y: np.ndarray = None):
        """
        决策树内部节点
        """
        self.subtrees = None
        self.depth = depth
        self.feature = -1
        self.classes = dict(zip(*np.unique(y, return_counts=True)))
        c = self.classes
        if not c:
            return

        ce = entropy(std(c.values()))
        if criterion == 'gain':
            self.feature = max((i for i in range(X.shape[1])), key=lambda i: ce.sum() - (entropy(std(
                [[sum(y[X[:, i] == 0] == j) for j in self.classes], [sum(y[X[:, i] == 1] == j) for j in self.classes]],
                axis=1)).sum(axis=1) * std([sum(X[:, i] == 0), sum(X[:, i] == 1)])).sum()
                               )

        else:
            self.feature = max((i for i in range(X.shape[1])), key=lambda i: (ce.sum() - (entropy(std(
                [[sum(y[X[:, i] == 0] == j) for j in self.classes], [sum(y[X[:, i] == 1] == j) for j in self.classes]],
                axis=1)).sum(axis=1) * std([sum(X[:, i] == 0), sum(X[:, i] == 1)])).sum()) / entropy(
                std([sum(X[:, i] == 0), sum(X[:, i] == 1)])).sum()
                               )
        if len(self.classes) != 1 and depth < max_depth:
            self.subtrees = [
                _Tree(depth + 1, max_depth, criterion, X[X[:, self.feature] == i], y[X[:, self.feature] == i])
                for i in range(2) if len(y[X[:, self.feature] == i])
            ]

    def forward(self, X):
        # print(self.feature, self.classes)
        if len(self.classes) == 1:
            return list(self.classes.keys())[0]
        if self.subtrees is not None:
            if len(self.subtrees) == 1:
                return self.subtrees[0].forward(X)
            if self.subtrees[int(X[self.feature])].feature != -1:
                return self.subtrees[int(X[self.feature])].forward(X)
            else:
                return max((i for i in self.subtrees[int(1 - X[self.feature])].classes),
                           key=lambda x: self.subtrees[int(1 - X[self.feature])].classes[x])
        return max((i for i in self.classes), key=lambda x: self.classes[x])

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def __repr__(self):
        sep = "\n " + "  " * self.depth
        if self.subtrees:
            return f'Tree<div={self.feature} classes={self.classes}>({sep}{sep.join((i.__repr__() for i in self.subtrees))}{sep})'
        return f'Tree<div={self.feature} classes={self.classes}>()' + f'=> {list(self.classes.keys())[0]} ' if len(
            self.classes) == 1 else ''


class DecisionTree:
    def __init__(self, max_depth: int = 2, criterion: Literal['gain', 'gain_rate'] = 'gain', random_state=42):
        """
        决策树模型
        """
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self._tree = None
        self.classes = None
        self._features = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._tree = _Tree(1, self.max_depth, self.criterion, X, y)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) == 2:
            return np.array([self._tree(i) for i in data])
        else:
            return self._tree(data)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # print(self.predict(X))
        return (self.predict(X) == y).sum() / len(y)

    def __repr__(self):
        return f'DecisionTree(max_depth={self.max_depth}, criterion = {self.criterion}, random_state={self.random_state})\nTrees:{self._tree}'


def gini_tree(x): return DecisionTreeClassifier(criterion='gini', max_depth=x, random_state=42)


def entropy_tree(x): return DecisionTreeClassifier(criterion='entropy', max_depth=x, random_state=42)


def gain_tree(x): return DecisionTree(criterion='gain', max_depth=x, random_state=42)


def gain_rate_tree(x): return DecisionTree(criterion='gain_rate', max_depth=x, random_state=42)


def random_tree(x): return RandomForestClassifier(n_estimators=x, random_state=42)


def ada_boost(x, e=DecisionTreeClassifier(max_depth=2)): return AdaBoostClassifier(e,
                                                                                   n_estimators=x,
                                                                                   random_state=42)


def ada_boost_mlp(x, e=Perceptron(max_iter=10, alpha=1e-7, eta0=10, l1_ratio=1e-5)): return AdaBoostClassifier(e,
                                                                                                               n_estimators=x,
                                                                                                               random_state=42,
                                                                                                               algorithm='SAMME')


def ada_boost_svc(x, e=SVC(kernel='linear')): return AdaBoostClassifier(e,
                                                                        n_estimators=x,
                                                                        random_state=42,
                                                                        algorithm='SAMME')


def ada_boost_sgd(x, e=SGDClassifier(learning_rate='constant', eta0=1e-5)): return AdaBoostClassifier(e,
                                                                                                      n_estimators=x,
                                                                                                      random_state=42,
                                                                                                      algorithm='SAMME')


if __name__ == '__main__':
    from dataset import load_data

    X_train, y_train, X_test, y_test = load_data()
    t = DecisionTree(max_depth=3)
    t.fit(X_train, y_train)
    print(t.score(X_test, y_test))
    print(t)
