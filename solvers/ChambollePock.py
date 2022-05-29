from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg


class Solver(BaseSolver):
    """A first-order primal-dual algorithm for synthesis formulation."""
    name = 'Chambolle-Pock analysis'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'sigma': [0.5],
                  'theta': [1.]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        n, p = self.A.shape
        data = np.array([-np.ones(p), np.ones(p)])
        diags = np.array([0, 1])
        D = spdiags(data, diags, p-1, p).toarray()
        sigma = self.sigma
        tau = 1. / np.linalg.norm(self.A, ord=2)**2
        I_tauAtA = LinearOperator(
            dtype=np.float64,
            matvec=lambda x: np.identity(p) @ x + tau * self.A.T @ self.A @ x,
            shape=(p, p),
        )
        K = LinearOperator(
            dtype=np.float64,
            matvec=lambda x: np.r_[D @ x, self.A @ x],
            rmatvec=lambda x: np.c_[D.T @ x[:p-1],
                                    self.A @ x[p-1:]],
            shape=(n + p - 1, p),
        )

        u = self.c * np.ones(p)
        v = np.zeros(p - 1)
        w = np.r_[v, self.A @ u]
        u_bar = u

        while callback(u):
            u_old = u
            if self.data_fit == 'quad':
                v_tmp = v + sigma * np.diff(u_bar)
                v = v_tmp - sigma * self.st(
                    v_tmp / sigma,
                    self.reg / sigma)
                u_tmp = u - tau * (-np.diff(v, append=0, prepend=0))
                u, _ = cg(I_tauAtA, u_tmp + tau * self.A.T @ self.y)
            else:
                x = w + tau * K @ u_bar
                w[:p - 1] = x[:p - 1] - tau * \
                    self.st(x[:p - 1] / tau,
                            self.reg / tau)
                R = tau * self.y - x[p - 1:]
                w[p - 1:] = x[p - 1:] - \
                    np.where(abs(R) < self.delta * (tau + 1),
                             tau *
                             (self.y + x[p - 1:]) / (tau + 1),
                             x[p - 1:] + self.delta * np.sign(R))
                u -= tau * K.T @ w
            u_bar = u + self.theta * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w
