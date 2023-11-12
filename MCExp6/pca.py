from utils import np, pd, plt
from dataset import XY, Z


class PCA:
    def __init__(self, n_composition=2):
        self.n_composition = n_composition
        self.dp_matrix = None

    def fit(self, X: np.ndarray):
        X_std = X - X.mean(axis=0, keepdims=True)
        cov = np.cov(X_std.T)
        val, vec = np.linalg.eig(cov)
        sort_val = np.sort(val)[::-1]
        sort_vec = vec[np.argsort(val)[::-1]]
        self.dp_matrix = sort_vec[:, :self.n_composition]

    def transform(self, X: np.ndarray):
        X_std = X - X.mean(axis=0, keepdims=True)
        return (X_std @ self.dp_matrix).astype('double')

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

        # print(X.mean(axis=0, keepdims=True).shape)
        # vec, vel = np.linalg.eig(X_std)


if __name__ == '__main__':
    p = PCA()
    xy = p.fit_transform(XY)
    plt.scatter(xy[:, 0], xy[:, 1], c=Z)
    plt.show()
