from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import rand as sprand
    from scipy.sparse.linalg import LinearOperator


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples': [400],
        'n_features': [250],
        'n_blocks': [10],
        'loc, scale': [(0, 0.1)],
        'type_A': ['identity', 'random', 'conv'],
        'type_x': ['block', 'sin'],
        'type_n': ['gaussian', 'laplace'],
        'random_state': [27]
    }

    # This makes sure that for each solver, we have one simulated dataset that
    # will be compatible in the test_solver.
    test_parameters = {
        'type_A': ['random', 'conv'],
        'n_samples, n_features': [(10, 5)],
        'type_n': ['gaussian']
    }

    def __init__(self, n_samples=5, n_features=5, n_blocks=1,
                 loc=0, scale=0.01, type_A='identity', type_x='block',
                 type_n='gaussian', random_state=27):
        # Store the parameters of the dataset
        self.n_samples, self.n_features = n_samples, n_features
        self.n_blocks = n_blocks
        self.loc, self.scale = loc, scale
        self.type_A, self.type_x, self.type_n = type_A, type_x, type_n
        self.random_state = random_state

    def get_A(self, rng):
        if self.type_A == 'identity':
            A = LinearOperator(
                dtype=np.float64,
                matvec=lambda x: x,
                matmat=lambda X: X,
                rmatvec=lambda x: x,
                rmatmat=lambda X: X,
                shape=(self.n_samples, self.n_features),
            )
        elif self.type_A == 'random':
            A = rng.randn(self.n_samples, self.n_features)
        elif self.type_A == 'conv':
            len_A = self.n_samples - self.n_features + 1
            filt = rng.randn(len_A)
            A = LinearOperator(
                dtype=np.float64,
                matvec=lambda x: np.convolve(x, filt, mode='full'),
                matmat=lambda X: np.array(
                    [np.convolve(x, filt, mode='full') for x in X.T]
                ).T,
                rmatvec=lambda x: np.correlate(x, filt, mode='valid'),
                rmatmat=lambda X: np.array(
                    [np.correlate(x, filt, mode='valid') for x in X.T]
                ).T,
                shape=(self.n_samples, self.n_features)
            )
        return A

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        if self.type_x == 'sin':
            t = np.arange(self.n_features)
            x = np.cos(np.pi*t/self.n_features * self.n_blocks)
        elif self.type_x == 'block':
            z = sprand(
                1, self.n_features, density=self.n_blocks/self.n_features,
                random_state=rng
            ).toarray()[0]
            x = np.cumsum(rng.randn(self.n_features) * z)
        if self.type_n == 'gaussian':
            # noise ~ N(loc, scale)
            n = rng.normal(self.loc, self.scale, self.n_samples)
        elif self.type_n == 'laplace':
            # noise ~ L(loc, scale)
            n = rng.laplace(self.loc, self.scale, self.n_samples)
        A = self.get_A(rng)
        y = A @ x + n
        data = dict(A=A, y=y, x=x)

        return data
