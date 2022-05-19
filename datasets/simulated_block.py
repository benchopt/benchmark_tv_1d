from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import random
    from scipy.sparse import random as sprandom
    from scipy import stats


class Dataset(BaseDataset):

    name = "Simulated blocked"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # cos + bruit ~ N(mu, sigma)
    parameters = {
        'sigma': [0.1],
        'mu': [0],
        'K': [500],
        'len_output': [400],
        'type_A': ['identity', 'random_square', 'random_nonsquare'],
        'num_block': [10]}

    def __init__(self, mu=0, sigma=0.3, K=10, len_output=5,
                 type_A='identity', random_state=27, num_block=1):
        # Store the parameters of the dataset
        self.mu = mu
        self.sigma = sigma
        self.K, self.len_output = K, len_output
        self.type_A = type_A
        self.num_block = num_block
        self.random_state = random_state

    def set_A(self, rng):
        if self.type_A == 'random_square':
            A = rng.randn(self.K, self.K)
        elif self.type_A == 'random_nonsquare':
            A = rng.randn(self.K, self.len_output)
        else:
            A = np.identity(self.K)
        return A

    def get_data(self):
        rng = np.random.RandomState(47)
        random.seed(47)
        slice_list = sorted(random.sample(range(self.K),
                                          self.num_block - 1),
                            reverse=False)
        slice_list.insert(0, 0)
        slice_list.append(self.K)
        slice_schedule = [np.ones(self.K).tolist()[slice_list[i]:
                                                   slice_list[i + 1]]
                          for i in range(self.num_block)]

        rvs = stats.norm(loc=0, scale=2).rvs
        slice_value = sprandom(1, self.num_block, density=0.5,
                               random_state=rng, data_rvs=rvs).A[0]
        y = np.array([])
        for i in range(self.num_block):
            y = np.concatenate(
                (y, np.array(slice_schedule[i]) * slice_value[i]))

        y = np.array(y) + rng.normal(self.mu, self.sigma, self.K)
        data = dict(A=self.set_A(rng), y=y)

        return data
