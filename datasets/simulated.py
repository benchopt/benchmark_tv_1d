from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated sinusoidal"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # np.convolve(A, cos) + noise ~ N(mu, sigma)
    parameters = {
        'sigma': [0.1],
        'mu': [0],
        'n_features': [500],
        'n_samples': [400],
        'type_A': ['identity', 'random']}

    def __init__(self, mu=0, sigma=0.3, n_features=10, n_samples=5,
                 type_A='identity', random_state=27):
        # Store the parameters of the dataset
        self.mu = mu
        self.sigma = sigma
        self.n_features, self.n_samples = n_features, n_samples
        self.type_A = type_A
        self.random_state = random_state

    def set_A(self, rng):
        len_A = self.n_features - self.n_samples + 1
        if len_A == 1 and self.type_A == 'identity':
            A = np.array([1])
        else:
            A = rng.randn(len_A)
        return A

    def get_data(self):
        t = np.arange(self.n_samples)
        rng = np.random.RandomState(self.random_state)
        w = np.cos(np.pi*t/self.n_samples*10)
        A = self.set_A(rng)
        y = np.convolve(w, A) + rng.normal(self.mu, self.sigma,
                                           self.n_features)
        data = dict(A=A, y=y)

        return data
