import numpy as np
from scipy.sparse import spdiags
from solvers.DPGD import get_inverse_D


def test_inverse_D():
    rng = np.random.RandomState(42)
    p = rng.randint(2, 100)
    data = np.array([-np.ones(p), np.ones(p)])
    diags = np.array([0, 1])
    D = spdiags(data, diags, p-1, p)
    D_inv = get_inverse_D(p)
    np.testing.assert_almost_equal(D @ D_inv, np.identity(p-1))
