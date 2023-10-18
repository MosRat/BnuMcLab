import numpy as np
import pandas as pd

from utils import *


def get_dataset():
    # test_csv = pd.read_csv('test.csv').values
    train_csv = pd.read_csv('train.csv', index_col=None).values
    train_data = train_csv[:800]
    valid_data = train_csv[800:900]
    test_data = train_csv[900:1000]

    print(test_data.shape)
    print(train_data.shape)
    print(valid_data.shape)
    np.save('train', train_data)
    np.save('valid', valid_data)
    np.save('test', test_data)


def normalize(x: np.ndarray):
    x = x.astype('float')
    x[:, 1:] = x[:, 1:] / 255
    return x


def load_dataset():
    return map(normalize, (np.load('train.npy'), np.load('valid.npy'), np.load('test.npy')))


if __name__ == '__main__':
    # get_dataset()
    train_data, valid_data, test_data = load_dataset()
    print(train_data[3])
