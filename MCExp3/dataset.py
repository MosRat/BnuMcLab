from utils import Iris

iris_data = Iris()
df = Iris(as_frame=True).frame
labels = iris_data.target_names
X = iris_data.data
y = iris_data.target


def make_data(label=None):
    if label is None:
        label = [0, 1]
    return X[(y == label[0]) | (y == label[1])], y[(y == label[0]) | (y == label[1])]
