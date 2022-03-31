import numpy as np
from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # cos + bruit ~ N(mu, sigma)
    parameters = {
        'sigma': np.linspace(0.1, 0.5, 3).round(1),
        'mu': [0],
        'K': [20]}

    def __init__(self, mu=0, sigma=0.3, K=10, random_state=27):
        # Store the parameters of the dataset
        self.mu = mu
        self.sigma = sigma
        self.K = K
        self.random_state = random_state

    def get_data(self):
        t = np.arange(self.K)
        rng = np.random.RandomState(47)
        y = np.cos(np.pi*t/self.K*10) + rng.normal(self.mu, self.sigma, self.K)
        # different matrix A that we can choose
        # A = np.eye(self.K, dtype=int)  # identity
        A = np.diag(rng.random(self.K))  # diagonal
        # A = np.triu(rng.rand(self.K, self.K))  #  triangulaire
        # A = rng.rand(self.K, self.K)  # totally random
        # A = A / A.sum(axis=1)  # nomalized by row
        data = dict(A=A, y=y)

        return y.shape[0], data
