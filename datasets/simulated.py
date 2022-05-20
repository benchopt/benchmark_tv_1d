from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated sinusoidal"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # cos + noise ~ N(mu, sigma)
    parameters = {
        'sigma': [0.1],
        'mu': [0],
        'n_features': [500],
        'n_samples': [400],
        'type_A': ['identity', 'random_square', 'random_nonsquare']}

    def __init__(self, mu=0, sigma=0.3, n_features=10, n_samples=5,
                 type_A='identity', random_state=27):
        # Store the parameters of the dataset
        self.mu = mu
        self.sigma = sigma
        self.n_features, self.n_samples = n_features, n_samples
        self.type_A = type_A
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
        t = np.arange(self.n_samples)
        rng = np.random.RandomState(47)
        w = np.cos(np.pi*t/self.n_samples*10)
        A = self.set_A(rng)
        y = A @ w + rng.normal(self.mu, self.sigma, A.shape[0])
        data = dict(A=A, y=y)

        return data
