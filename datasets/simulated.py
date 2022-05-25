from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import rand as sprand


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples': [400],
        'n_features': [500],
        'n_block': [10],
        'mu': [0],
        'sigma': [0.1],
        'type_A': ['identity', 'random'],
        'type_x': ['block', 'sin'],
        'random_state': [27]
    }

    def __init__(self, n_samples=10, n_features=10, n_block=2,
                 mu=0, sigma=0.3, type_A='identity', type_x='block',
                 random_state=27):
        # Store the parameters of the dataset
        self.n_samples, self.n_features = n_samples, n_features
        self.n_block = n_block
        self.mu, self.sigma = mu, sigma
        self.type_A, self.type_x = type_A, type_x
        self.random_state = random_state

    def get_A(self, rng):
        if self.type_A == 'random':
            A = rng.randn(self.n_samples, self.n_features)
        else:
            assert self.n_samples == self.n_features
            A = np.identity(self.n_samples)
        return A

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        if self.type_x == 'sin':
            # A * cos + noise ~ N(mu, sigma)
            t = np.arange(self.n_features)
            x = np.cos(np.pi*t/self.n_features * self.n_blocks)
        else:
            # A * blocked signal + noise ~ N(mu, sigma)
            z = sprand(
                1, self.n_features, density=self.n_block/self.n_features,
                random_state=rng
            ).toarray()[0]
            x = np.cumsum(rng.randn(self.n_features) * z)
        A = self.get_A(rng)
        y = A @ x + rng.normal(self.mu, self.sigma, self.n_samples)
        data = dict(A=A, y=y, x=x)

        return data
