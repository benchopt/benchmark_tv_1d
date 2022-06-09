import numpy as np
from scipy.sparse import rand as sprand
from scipy.sparse.linalg import LinearOperator
from solvers.DPGD import Solver as DPGDSolver


def test_LO_solve_dual_eq():
    n = 40
    p = 25
    n_blocks = 10
    rng = np.random.RandomState(42)
    len_A = n - p + 1
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
        shape=(n, p)
    )
    # y = A * blocked signal + noise ~ N(0, 0.1)
    z = sprand(1, p, density=n_blocks/p,
               random_state=rng
               ).toarray()[0]
    x = np.cumsum(rng.randn(p) * z)
    y = A @ x + rng.normal(0, 0.1, n)

    reg = 1.
    c = 1.
    delta = 0.9
    data_fit = 'quad'
    global n_iter

    solver = DPGDSolver()
    solver.alpha = 1.
    solver.use_acceleration = True
    solver.set_objective(A, reg, y, c, delta, data_fit)
    n_iter = 0
    solver.run(cb)
    u_1 = solver.u
    solver.set_objective(A @ np.identity(p), reg, y, c, delta, data_fit)
    n_iter = 0
    solver.run(cb)
    u_2 = solver.u

    np.testing.assert_almost_equal(u_1, u_2)


def cb(u):
    global n_iter
    n_iter += 1
    if n_iter > 5:
        return True
