from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg
    from scipy.optimize import minimize
    from benchmark_utils.shared import st, huber, grad_huber


def loss(y, A, u, delta, z, mu, gamma):
    R = y - A @ u
    return huber(R, delta) + gamma / 2 * np.linalg.norm(
        np.diff(u) - z + mu / gamma, ord=2) ** 2


def jac_loss(y, A, u, delta, z, mu, gamma):
    R = y - A @ u
    return - A.T @ grad_huber(R, delta) - gamma * np.diff(
        np.diff(u) - z + mu / gamma, append=0, prepend=0)


class Solver(BaseSolver):
    """Alternating direction method for analysis formulation."""
    name = 'ADMM analysis'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'gamma': [25.],
                  'update_pen': [False]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        p = self.A.shape[1]
        self.u = u = self.c * np.ones(p)
        z = np.zeros(p - 1)
        mu = np.zeros(p - 1)
        gamma = self.gamma
        tol_cg = 1e-12
        Aty = self.A.T @ self.y
        if isinstance(self.A, np.ndarray):
            data = np.array([-np.ones(p), np.ones(p)])
            diags = np.array([0, 1])
            D = spdiags(data, diags, p-1, p)
            AtA_gDtD_inv = np.linalg.pinv(self.A.T @ self.A + gamma * D.T @ D)
        else:
            # D @ x = np.diff(x)
            # D.T @ x = -np.diff(x, append=0, prepend=0)
            AtA_gDtD = LinearOperator(shape=(p, p),
                                      matvec=lambda x: self.A.T @ (
                self.A @ x) - gamma * np.diff(
                np.diff(x), append=0, prepend=0))

        while callback():
            z_old = z
            if self.data_fit == 'quad':
                u_tmp = (Aty + np.diff(mu, append=0, prepend=0)
                         - gamma * np.diff(z, append=0, prepend=0))
                if isinstance(self.A, np.ndarray):
                    u = np.ravel(AtA_gDtD_inv @ u_tmp)
                else:
                    u, _ = cg(AtA_gDtD, u_tmp, x0=u, tol=tol_cg)
            else:
                def func(u):
                    return loss(self.y, self.A, u,
                                self.delta, z, mu, gamma)

                def jac(u):
                    return jac_loss(self.y, self.A, u,
                                    self.delta, z, mu, gamma)

                u = minimize(func, x0=u, jac=jac,
                             method='BFGS', tol=1e-15).x

            z = st(np.diff(u) + mu / gamma, self.reg / gamma)
            mu += gamma * (np.diff(u) - z)

            if self.update_pen:
                r = np.linalg.norm(np.diff(u) - z, ord=2)
                s = np.linalg.norm(
                    gamma * np.diff(z - z_old, append=0, prepend=0), ord=2)
                if r > 10 * s:
                    gamma *= 2
                if s > 10 * r:
                    gamma /= 2
            self.u = u

    def get_result(self):
        return dict(u=self.u)
