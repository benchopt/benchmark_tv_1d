import numpy as np


def test_inverse(D, D_inv):
    n = D.shape[0]
    return np.allclose(D @ D_inv, np.identity(n))
