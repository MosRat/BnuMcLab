from sklearn.datasets import make_classification, make_circles, make_moons
from utils import *
from decision import *
from plot import *

# data = make_moons(random_state=42, n_samples=400, noise=1e-1)
# data = make_circles(random_state=42)
data = make_classification(random_state=42, n_samples=250, n_features=2, n_redundant=0, n_informative=2)
plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1])
data = np.concatenate([data[0], data[1].reshape(-1, 1)], axis=1)

b = BayesMinErrorDecision()
b.fit(data)
edge = b.get_dec_edge(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
plt.gca().plot(edge[:, 0], edge[:, 1], c="#FA7752")
plt.show()

print(b.score())

# print(data)
