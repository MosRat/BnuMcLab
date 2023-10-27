import json

import numpy as np

from utils import *


def prepare_data():
    vec = DictVectorizer()
    data = pd.read_csv('蘑菇分类数据集.csv')
    data_array = np.hstack([data['class'].values.reshape(-1, 1),
                            vec.fit_transform(data.drop(['class', 'odor', 'stalk-color-below-ring'], axis=1).to_dict(
                                'records')).toarray()]
                           )
    # json.dump(list(vec.get_feature_names_out()), open('1.json', 'w', encoding='utf8'))
    np.random.shuffle(data_array)
    np.save('data.npy', data_array)
    exit(0)


def load_data():
    data = np.load('data.npy', allow_pickle=True)
    return data[:500, 1:], data[:500, 0], data[-3000:, 1:], data[-3000:, 0]


def get_classes_name():
    return ['e', 'p']


def get_feature_name():
    return [
        "cap-color=b",
        "cap-color=c",
        "cap-color=e",
        "cap-color=g",
        "cap-color=n",
        "cap-color=p",
        "cap-color=r",
        "cap-color=u",
        "cap-color=w",
        "cap-color=y",
        "cap-shape=b",
        "cap-shape=c",
        "cap-shape=f",
        "cap-shape=k",
        "cap-shape=s",
        "cap-shape=x",
        "cap-surface=f",
        "cap-surface=g",
        "cap-surface=s",
        "cap-surface=y",
        "gill-color=b",
        "gill-color=e",
        "gill-color=g",
        "gill-color=h",
        "gill-color=k",
        "gill-color=n",
        "gill-color=o",
        "gill-color=p",
        "gill-color=r",
        "gill-color=u",
        "gill-color=w",
        "gill-color=y",
        "habitat=d",
        "habitat=g",
        "habitat=l",
        "habitat=m",
        "habitat=p",
        "habitat=u",
        "habitat=w",
        # "odor=a",
        # "odor=c",
        # "odor=f",
        # "odor=l",
        # "odor=m",
        # "odor=n",
        # "odor=p",
        # "odor=s",
        # "odor=y",
        "population=a",
        "population=c",
        "population=n",
        "population=s",
        "population=v",
        "population=y",
        "ring-number=n",
        "ring-number=o",
        "ring-number=t",
        "ring-type=e",
        "ring-type=f",
        "ring-type=l",
        "ring-type=n",
        "ring-type=p",
        "spore-print-color=b",
        "spore-print-color=h",
        "spore-print-color=k",
        "spore-print-color=n",
        "spore-print-color=o",
        "spore-print-color=r",
        "spore-print-color=u",
        "spore-print-color=w",
        "spore-print-color=y",
        "stalk-color-above-ring=b",
        "stalk-color-above-ring=c",
        "stalk-color-above-ring=e",
        "stalk-color-above-ring=g",
        "stalk-color-above-ring=n",
        "stalk-color-above-ring=o",
        "stalk-color-above-ring=p",
        "stalk-color-above-ring=w",
        "stalk-color-above-ring=y",
        # "stalk-color-below-ring=b",
        # "stalk-color-below-ring=c",
        # "stalk-color-below-ring=e",
        # "stalk-color-below-ring=g",
        # "stalk-color-below-ring=n",
        # "stalk-color-below-ring=o",
        # "stalk-color-below-ring=p",
        # "stalk-color-below-ring=w",
        # "stalk-color-below-ring=y",
        "stalk-root=?",
        "stalk-root=b",
        "stalk-root=c",
        "stalk-root=e",
        "stalk-root=r",
        "stalk-surface-above-ring=f",
        "stalk-surface-above-ring=k",
        "stalk-surface-above-ring=s",
        "stalk-surface-above-ring=y",
        "stalk-surface-below-ring=f",
        "stalk-surface-below-ring=k",
        "stalk-surface-below-ring=s",
        "stalk-surface-below-ring=y",
        "veil-color=n",
        "veil-color=o",
        "veil-color=w",
        "veil-color=y"
    ]


if __name__ == '__main__':
    # pass
    prepare_data()
    data = np.load('data.npy', allow_pickle=True)
    print(data.shape)
    # print(data)
    # p = DecisionTreeClassifier(max_depth=1, random_state=42)
    # p = RandomForestClassifier(random_state=42)
    p = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42)
    p.fit(data[:500, 1:], data[:500, 0])
    print(p.score(data[-1000:, 1:], data[-1000:, 0]))
    print(np.mean(p.predict(data[-1000:, 1:]) == data[-1000:, 0]))
