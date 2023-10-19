import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sklearn.svm import SVC
from sklearn.base import RegressorMixin,ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm

if __name__ == '__main__':
    print(type(SVC))
    # import torch
    # print(torch.cuda.is_available())