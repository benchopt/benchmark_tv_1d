import pytest
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import check_grad as check_grad

from benchopt.utils.safe_import import set_benchmark

# this means this test has to be run from the root
set_benchmark('./')

try:
    from solvers.ADMM import loss
    from solvers.ADMM import jac_loss
except Exception:
    raise


@pytest.mark.parametrize('random_state', [0, 27, 42, 66])
def test_grad(random_state):
    n = 10
    p = 10
    rng = np.random.RandomState(random_state)
    A = rng.randn(n, p)
    A_op = LinearOperator(
        dtype=np.float64,
        matvec=lambda x: A @ x,
        matmat=lambda X: A @ X,
        rmatvec=lambda x: A.T @ x,
        rmatmat=lambda X: A.T @ X,
        shape=(n, p)
    )
    y = rng.normal(0, 1, n)
    delta = 0.9

    z = rng.randn(p - 1)
    mu = rng.randn(p - 1)
    gamma = 1.9

    def func(u):
        return loss(y, A, u, delta, z, mu, gamma)

    def jac(u):
        return jac_loss(y, A, u, delta, z, mu, gamma)

    def func_op(u):
        return loss(y, A_op, u, delta, z, mu, gamma)

    def jac_op(u):
        return jac_loss(y, A_op, u, delta, z, mu, gamma)

    np.testing.assert_almost_equal(0, check_grad(func, jac,
                                                 x0=rng.randn(p)),
                                   decimal=5)
    np.testing.assert_almost_equal(0, check_grad(func_op, jac_op,
                                                 x0=rng.randn(p)),
                                   decimal=5)
