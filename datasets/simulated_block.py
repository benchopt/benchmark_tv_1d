from benchopt import BaseDataset
from benchopt import safe_import_context
from numpy import transpose

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import random as sprandom
    from scipy.sparse.linalg import LinearOperator


class Dataset(BaseDataset):

    name = "Simulated blocked"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # np.convolve(A, blocked signal) + noise ~ N(mu, sigma)
    parameters = {
        'sigma': [0.1],
        'mu': [0],
        'n_features': [500],
        'n_samples': [400],
        'type_A': ['conv'],
        # 'type_A': ['identity', 'random', 'conv'],
        'num_block': [10]}

    def __init__(self, mu=0, sigma=0.3, n_features=10, n_samples=5,
                 type_A='identity', random_state=27, num_block=1):
        # Store the parameters of the dataset
        self.mu = mu
        self.sigma = sigma
        self.n_features, self.n_samples = n_features, n_samples
        self.type_A = type_A
        self.num_block = num_block
        self.random_state = random_state

    def set_A(self, rng):
        len_A = self.n_features - self.n_samples + 1
        # if len_A == 1 and self.type_A == 'identity':
        #     A = np.array([1])
        # else:
        #     A = rng.randn(len_A)
        if self.type_A == 'identity':
            A = LinearOperator(
                dtype=np.float64,
                matvec=lambda x: x,
                matmat=lambda X: X,
                shape=(self.n_samples, self.n_features),
            )
        elif self.type_A == 'conv':
            filt = rng.randn(len_A)
            A = LinearOperator(
                dtype=np.float64,
                matvec=lambda x: np.convolve(filt, x, mode='full'),
                matmat=lambda X: np.array(
                    [np.convolve(filt, x, mode='full') for x in X.T]
                ).T,
                shape=(self.n_samples, self.n_features),
                rmatvec=lambda x: np.correlate(filt, x, mode='valid'),
                rmatmat=lambda X: np.array(
                    [np.correlate(filt, x, mode='valid') for x in X.T]
                ).T
            )
        return A

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        w = np.cumsum(rng.randn(self.n_features) * sprandom(1, self.n_features,
                      density=self.num_block/self.n_features,
                      random_state=rng, data_rvs=np.ones).A[0])
        A = self.set_A(rng)
        y = A @ w + rng.normal(self.mu, self.sigma, self.n_samples)
        data = dict(A=A, y=y)
        return data
