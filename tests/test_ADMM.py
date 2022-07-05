import pytest
import numpy as np
from scipy.sparse.linalg import LinearOperator
from solvers.ADMM import sum_grad_L_huber
from solvers.ADMM import jac_sum_grad_L_huber
from scipy.optimize import check_grad as check_grad


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
    u_tmp = (np.diff(mu, append=0, prepend=0)
             - gamma * np.diff(z, append=0, prepend=0))
    gDtD = LinearOperator(shape=(p, p),
                          matvec=lambda x: - gamma * np.diff(
                          np.diff(x), append=0, prepend=0))

    def func(u):
        return sum_grad_L_huber(y, A, u, delta, gDtD, u_tmp)

    def jac(u):
        return jac_sum_grad_L_huber(y, A, u, delta, gDtD, u_tmp)

    def func_op(u):
        return sum_grad_L_huber(y, A_op, u, delta, gDtD, u_tmp)

    def jac_op(u):
        return jac_sum_grad_L_huber(y, A_op, u, delta, gDtD, u_tmp)

    np.testing.assert_almost_equal(0,
                                   check_grad(func, jac, x0=np.zeros(p)),
                                   decimal=6)
    np.testing.assert_almost_equal(0,
                                   check_grad(func_op, jac_op, x0=np.zeros(p)),
                                   decimal=6)
