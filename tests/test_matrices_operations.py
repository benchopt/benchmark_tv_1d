import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import rand as sprand
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from solvers.DPGD import get_inverse_D


def test_inverse_D():
    rng = np.random.RandomState(42)
    p = rng.randint(2, 100)
    data = np.array([-np.ones(p), np.ones(p)])
    diags = np.array([0, 1])
    D = spdiags(data, diags, p-1, p)
    D_inv = get_inverse_D(p)
    np.testing.assert_almost_equal(D @ D_inv, np.identity(p-1))


def test_LO_solve_dual_eq():
    def dual(y, A):
        p = A.shape[1]
        v = np.zeros(p - 1)
        v_tmp = np.zeros(p)
        tol_cg = 1e-15
        alpha = 1.
        reg = 1.

        DA_inv_1 = np.diff(np.linalg.pinv(A @ np.identity(p)), axis=0)
        DA_invDA_invt = DA_inv_1 @ DA_inv_1.T
        DA_invy = DA_inv_1 @ y
        stepsize_1 = alpha / (np.linalg.norm(DA_inv_1, ord=2)**2)
        v_1 = np.clip(v - stepsize_1 * (DA_invDA_invt @ v - DA_invy),
                      -reg, reg)

        D_inv_2 = get_inverse_D(p)
        DA_inv_2 = np.linalg.pinv(A @ D_inv_2)
        AtA = LinearOperator(
            dtype=np.float64,
            matvec=lambda x: A.T @ A @ x,
            shape=(p, p)
        )
        Aty = A.T @ y
        stepsize_2 = alpha / (np.linalg.norm(DA_inv_2, ord=2)**2)
        v_2_tmp, _ = cg(AtA, Aty + np.diff(v, append=0, prepend=0),
                        x0=v_tmp, tol=tol_cg)
        v_2 = np.clip(v + stepsize_2 * np.diff(v_2_tmp),
                      -reg, reg)

        np.testing.assert_almost_equal(v_1, v_2, decimal=5)

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
    dual(y, A)
