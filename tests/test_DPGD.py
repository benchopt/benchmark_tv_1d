import pytest
import numpy as np

from scipy.sparse.linalg import LinearOperator
from solvers.DPGD import Solver as DPGDSolver


@pytest.mark.parametrize('random_state', [0, 27, 42, 66])
def test_LO_solve_dual_eq(random_state):
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

    reg = 1.
    c = 1.
    delta = 0.9
    data_fit = 'quad'
    max_iter = 50

    global n_iter

    def cb(u):
        global n_iter
        n_iter += 1
        if n_iter <= max_iter:
            return True

    solver = DPGDSolver()
    solver.alpha = 1.
    solver.use_acceleration = True
    solver.set_objective(A, reg, y, c, delta, data_fit)
    n_iter = 0
    solver.run(cb)
    u_1 = solver.u
    assert n_iter > max_iter
    solver.set_objective(A_op, reg, y, c, delta, data_fit)
    n_iter = 0
    solver.run(cb)
    u_2 = solver.u
    assert n_iter > max_iter

    np.testing.assert_almost_equal(u_1, u_2)
