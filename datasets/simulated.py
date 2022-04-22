import numpy as np
from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # cos + bruit ~ N(mu, sigma)
    parameters = {
        'sigma': [0.1],
        'mu': [0],
        'K': [500],
        'type_A': ['identity', 'diagonal', 'triangular', 'random']}

    def __init__(self, mu=0, sigma=0.3, K=10,
                 type_A='identity', random_state=27):
        # Store the parameters of the dataset
        self.mu = mu
        self.sigma = sigma
        self.K = K
        self.type_A = type_A
        self.random_state = random_state

    def set_A(self, rng):
        if self.type_A == 'diagonal':
            A = np.diag(rng.random(self.K))
        elif self.type_A == 'triangular':
            A = np.triu(rng.randn(self.K, self.K))
        elif self.type_A == 'random':
            A = rng.randn(self.K, self.K)
        else:
            A = np.identity(self.K)
        return A

    def get_data(self):
        t = np.arange(self.K)
        rng = np.random.RandomState(47)
        y = np.cos(np.pi*t/self.K*10) + rng.normal(self.mu, self.sigma, self.K)
        data = dict(A=self.set_A(rng), y=y)

        return y.shape[0], data
