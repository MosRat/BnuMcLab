from utils import *


def deal_raw_data():
    """
    处理原始数据
    :return:
    """
    r16 = pd.read_excel('raw_data/sampled_data_16.xlsx')
    r256 = pd.read_excel('raw_data/sampled_data_256.xlsx')
    r1000 = pd.read_excel('raw_data/sampled_data_1000.xlsx')
    r2000 = pd.read_excel('raw_data/sampled_data_2000.xlsx')
    np.save('data/data_16', r16['Sample'].values)
    np.save('data/data_256', r256['Sample'].values)
    np.save('data/data_1000', r1000['Sample'].values)
    np.save('data/data_2000', r2000['Sample'].values)


data1, data2, data3, data4 = np.load('data/data_16.npy'), np.load('data/data_256.npy'), np.load(
    'data/data_1000.npy'), np.load('data/data_2000.npy')
