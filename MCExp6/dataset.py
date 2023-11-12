from utils import *

XY, Z = make_swiss_roll(n_samples=1500, noise=0.25, random_state=42)
mnist = pd.read_csv('train.csv').values
mnist_xy, mnist_z = mnist[::42, 1:], mnist[::42, 0]
