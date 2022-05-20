from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import random as sprandom
    from scipy import stats


class Dataset(BaseDataset):

    name = "Simulated blocked"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # blocked signal + noise ~ N(mu, sigma)
    parameters = {
        'sigma': [0.1],
        'mu': [0],
        'n_features': [50],
        'n_samples': [40],
        'type_A': ['identity', 'random_square', 'random_nonsquare'],
        'num_block': [3]}

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
        if self.type_A == 'random_square':
            A = rng.randn(self.n_samples, self.n_samples)
        elif self.type_A == 'random_nonsquare':
            A = rng.randn(self.n_features, self.n_samples)
        else:
            A = np.identity(self.n_samples)
        return A

    def get_data(self):
        rng = np.random.RandomState(47)
        rvs = stats.norm(loc=0, scale=1).rvs
        w = np.cumsum(sprandom(1, self.n_samples,
                      density=self.num_block/self.n_samples,
                      random_state=rng, data_rvs=rvs).A[0])
        A = self.set_A(rng)
        y = A @ w + rng.normal(self.mu, self.sigma, A.shape[0])
        data = dict(A=A, y=y)

        return data
