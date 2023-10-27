import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import *
from dataset import get_feature_name, load_data, get_classes_name

X_train, y_train, X_test, y_test = load_data()
vec_func = np.vectorize(lambda x: 0 if x == 'p' else 1)
y_train = vec_func(y_train)
y_test = vec_func(y_test)

sns.set_style('darkgrid')
sns.set_palette('Pastel1')
# print(__name__, y_train, y_train.shape)


def classification_tree_to_dot(tree, feature_names, class_names, categorical_dict):
    """ 把分类树转化成dot data

    参数
        tree: DecisionTreeClassifier的输出
        feature_names: vnames, 除去目标变量所有变量的名字
        class_names: 目标变量所有的分类
        categorical_dict: 储存所有名称及分类的字典

    输出
        graphvic_str: the dot data
    """
    tree_ = tree.tree_

    # store colors that distinguish discrete chunks of data
    if len(class_names) <= 10:
        # get the colorblind friendly colors
        color_palette = adjust_colors(None)['classes'][len(class_names)]
    else:
        color_palette = sns.color_palette("coolwarm", len(class_names)).as_hex()

    feature_name = [
        feature_names[i] if i != tt.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # initialize the dot data string
    graphvic_str = 'digraph Tree {node [shape=oval, penwidth=0.1, width=1, fontname=helvetica] ; edge [fontname=helvetica] ;'

    # print(graphvic_str)

    def recurse(node, depth, categorical_dict):
        # store the categorical_dict information of each side
        categorical_dict_L = categorical_dict.copy()
        categorical_dict_R = categorical_dict.copy()
        # non local statement of graphvic_str
        nonlocal graphvic_str
        # variable is not dummy by default
        is_dummy = False
        # get the threshold
        threshold = tree_.threshold[node]

        # get the feature name
        name = feature_name[node]
        # judge whether a feature is dummy or not by the indicator "_isDummy_"
        if "_isDummy_" in str(name) and name.split('_isDummy_')[0] in list(categorical_dict.keys()):
            is_dummy = True
            # if the feature is dummy, the threshold is the value following name
            name, threshold = name.split('_isDummy_')[0], name.split('_isDummy_')[1]

        # get the data distribution of current node
        value = tree_.value[node][0]
        # get the total amount
        n_samples = tree_.n_node_samples[node]
        # calculate the weight
        weights = [i / sum(value) for i in value]
        # get the largest class
        class_name = class_names[np.argmax(value)]

        # pair the color and weight
        fillcolor_str = ""
        for i, j in enumerate(color_palette):
            fillcolor_str += j + ";" + str(weights[i]) + ":"
        fillcolor_str = '"' + fillcolor_str[:-1] + '"'

        if tree_.feature[node] != tt.TREE_UNDEFINED:
            # if the node is not a leaf
            graphvic_str += ('{} [style=wedged, label=<{}<br/>{}>, fillcolor =' + fillcolor_str + '] ;').format(node,
                                                                                                                n_samples,
                                                                                                                name)
            # print(('{} [style=wedged, label=<{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,name))
            if is_dummy:
                # if the feature is dummy and if its total categories > 5
                categorical_dict_L[name] = [str(i) for i in categorical_dict_L[name] if i != threshold]
                categorical_dict_R[name] = [str(threshold)]
                if len(categorical_dict[name]) > 5:
                    # only show one category on edge
                    threshold_left = "not " + threshold
                    threshold_right = threshold
                else:
                    # if total categories <= 5, list all the categories on edge
                    threshold_left = ", ".join(categorical_dict_L[name])
                    threshold_right = threshold
            else:
                # if the feature is not dummy, then it is numerical
                threshold_left = "<=" + str(round(threshold, 3))
                threshold_right = ">" + str(round(threshold, 3))
            graphvic_str += ('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="{}"] ;').format(node,
                                                                                                     tree_.children_left[
                                                                                                         node],
                                                                                                     threshold_left)
            graphvic_str += ('{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="{}"] ;').format(node,
                                                                                                      tree_.children_right[
                                                                                                          node],
                                                                                                      threshold_right)
            # print(('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="{}"] ;').format(node,tree_.children_left[node],threshold_left))
            # print(('{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="{}"] ;').format(node,tree_.children_right[node],threshold_right))

            recurse(tree_.children_left[node], depth + 1, categorical_dict_L)
            recurse(tree_.children_right[node], depth + 1, categorical_dict_R)
        else:
            # the node is a leaf
            graphvic_str += (
                    '{} [shape=box, style=striped, label=<{}<br/>{}>, fillcolor =' + fillcolor_str + '] ;').format(
                node, n_samples, class_name)
            # print(('{} [shape=box, style=striped, label=<{}<br/>{}>, fillcolor ='+fillcolor_str+'] ;').format(node,n_samples,class_name))

    recurse(0, 1, categorical_dict)
    return graphvic_str + "}"


def plot_tree(e, name):
    """
    绘制决策树图
    """
    viz_model = model(e,
                      X_train=X_train,
                      y_train=y_train,
                      feature_names=get_feature_name(),
                      target_name='safe',
                      class_names=['p', 'e']
                      )
    v = viz_model.view()
    v.show()
    v.save(f"./imgs/{name}.svg")


def plot_line(data, label, title, x_label, y_label, marks=None):
    """
    绘制折线图
    """
    plt.figure(dpi=300)
    if marks is None:
        marks = ['x', 'v', 'o', '.']
    for i, l, m in zip(data, label, marks):
        plt.plot(i, label=l, marker=m, linewidth=3)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
