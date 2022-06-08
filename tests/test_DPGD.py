import numpy as np
from scipy.sparse import rand as sprand
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from solvers.DPGD import get_inverse_D


def get_u_DPGD(alpha, use_acceleration, reg, c, y, A, n_iter):
    n, p = A.shape
    if isinstance(A, np.ndarray):
        DA_inv = np.diff(np.linalg.pinv(A), axis=0)
        DA_invDA_invt = DA_inv @ DA_inv.T
        DA_invy = DA_inv @ y
        AtA_inv = np.linalg.pinv(A.T @ A)
    else:
        D_inv = get_inverse_D(p)
        DA_inv = np.linalg.pinv(A @ D_inv)
        AtA = LinearOperator(
            dtype=np.float64,
            matvec=lambda x: A.T @ A @ x,
            shape=(p, p)
        )
    Aty = A.T @ y
    tol_cg = 1e-12
    # alpha / rho
    stepsize = alpha / (np.linalg.norm(DA_inv, ord=2)**2)
    # initialisation
    u = c * np.ones(p)
    v = np.zeros(p - 1)
    v_tmp = np.zeros(p)
    v_old = v.copy()
    v_acc = v.copy()

    t_new = 1
    for _ in range(n_iter):
        if use_acceleration:
            t_old = t_new
            t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            v_old[:] = v
            v[:] = v_acc
        if isinstance(A, np.ndarray):
            v = np.clip(v - stepsize * (DA_invDA_invt @ v - DA_invy),
                        -reg, reg)
        else:
            v_tmp, _ = cg(AtA, Aty + np.diff(v, append=0, prepend=0),
                          x0=v_tmp, tol=tol_cg)
            v = np.clip(v + stepsize * np.diff(v_tmp),
                        -reg, reg)

        if use_acceleration:
            v_acc[:] = v + (t_old - 1.) / t_new * (v - v_old)

        if isinstance(A, np.ndarray):
            u = AtA_inv @ (Aty + np.diff(v, append=0, prepend=0))
        else:
            u, _ = cg(AtA, Aty + np.diff(v, append=0, prepend=0),
                      x0=u, tol=tol_cg)
    return u


def test_LO_solve_dual_eq():
    n = 400
    p = 250
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

    alpha = 1.
    use_acceleration = True
    reg = 1.
    c = 1.
    u_1 = get_u_DPGD(alpha, use_acceleration, reg, c, y, A, 3)
    u_2 = get_u_DPGD(alpha, use_acceleration, reg, c, y, A @ np.identity(p), 3)

    np.testing.assert_almost_equal(u_1, u_2, decimal=8)
