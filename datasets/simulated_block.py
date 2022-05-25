from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import random as sprandom


class Dataset(BaseDataset):

    name = "Simulated blocked"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # blocked signal + noise ~ N(mu, sigma)
    parameters = {
        'sigma': [0.1],
        'mu': [0],
        'n_samples': [500],
        'n_features': [400],
        'type_A': ['identity', 'random_square', 'random_nonsquare'],
        'num_block': [10]}

    def __init__(self, mu=0, sigma=0.3, n_samples=10, n_features=5,
                 type_A='identity', random_state=27, num_block=1):
        # Store the parameters of the dataset
        self.mu = mu
        self.sigma = sigma
        self.n_samples, self.n_features = n_samples, n_features
        self.type_A = type_A
        self.num_block = num_block
        self.random_state = random_state

    def set_A(self, rng):
        if self.type_A == 'random_square':
            A = rng.randn(self.n_samples, self.n_samples)
        elif self.type_A == 'random_nonsquare':
            A = rng.randn(self.n_samples, self.n_features)
        else:
            A = np.identity(self.n_samples)
        return A

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        A = self.set_A(rng)
        w = np.cumsum(rng.randn(A.shape[1]) * sprandom(1, A.shape[1],
                      density=self.num_block/A.shape[1],
                      random_state=rng, data_rvs=np.ones).A[0])
        y = A @ w + rng.normal(self.mu, self.sigma, A.shape[0])
        data = dict(A=A, y=y)

        return data
