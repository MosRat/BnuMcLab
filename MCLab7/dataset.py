import numpy as np
import pandas as pd
from pathlib import Path


def deal_raw_data():
    import cv2

    dataset = Path('./dataset/selected_mnist_data')
    ls = []
    for c in dataset.iterdir():
        for i in c.iterdir():
            img = cv2.imread(str(i), cv2.IMREAD_GRAYSCALE)
            # print(img.shape, img.dtype)
            img = img.reshape(-1)
            emp = np.append(img, int(c.name))
            ls.append(emp)
    data = np.array(ls)
    np.random.shuffle(data)
    np.save(Path("dataset/dataset.npy"), data)
    # print(data[:, -1])


def load_dataset():
    data = np.load(Path('dataset/dataset.npy'))
    return data[:, :-1] / 255, data[:, -1]


X, y = load_dataset()
print(X.shape, y.shape)
