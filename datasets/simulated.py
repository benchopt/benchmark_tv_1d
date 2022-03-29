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
        'T': [1000]}

    def __init__(self, mu=0, sigma=0.3, T=10, random_state=27):
        # Store the parameters of the dataset
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.random_state = random_state

    def get_data(self):
        t = np.arange(self.T)
        rng = np.random.RandomState(47)
        y = np.cos(np.pi*t/self.T*10) + rng.normal(self.mu, self.sigma, self.T)
        reg_max = np.max(abs(y - np.mean(y)))
        data = dict(reg_max=reg_max, y=y)

        return y.shape[0], data
