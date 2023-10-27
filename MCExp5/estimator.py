from utils import *


def gini_tree(x): return DecisionTreeClassifier(criterion='gini', max_depth=x, random_state=42)


def entropy_tree(x): return DecisionTreeClassifier(criterion='entropy', max_depth=x, random_state=42)


def random_tree(x): return RandomForestClassifier(n_estimators=x, random_state=42)


def ada_boost(x, e=DecisionTreeClassifier(max_depth=2)): return AdaBoostClassifier(e,
                                                                                   n_estimators=x,
                                                                                   random_state=42)


def ada_boost_mlp(x, e=Perceptron(max_iter=10, alpha=1e-7, eta0=10,l1_ratio=1e-5)): return AdaBoostClassifier(e,
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

# gini_tree.set_params()
