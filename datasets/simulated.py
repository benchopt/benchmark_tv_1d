import numpy as np
from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # cos + bruit ~ N(mu, sigma)
    parameters = {
        'sigma' : [0.1, 0.5],
	    'mu' : [0]}

    def __init__(self, mu=0, sigma=0.3, random_state=27):
        # Store the parameters of the dataset
        self.mu = mu
        self.sigma = sigma
        self.random_state = random_state

    def get_data(self):
        t = np.arange(1000)
        T = 1000
    
        rng = np.random.RandomState(47)
        y = np.cos(np.pi*t/T) + rng.normal(self.mu, self.sigma, len(t))
        
        data = dict(y=y)
        
        return y.shape[0], data
