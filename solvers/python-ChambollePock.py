from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags


class Solver(BaseSolver):
    """A first-order primal-dual algorithm for synthesis formulation."""
    name = 'Chambolle-Pock analysis'

    stopping_criterion = SufficientProgressCriterion(
        patience=50, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'theta': [1.]}

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
        D = spdiags(data, diags, p-1, p)
        sigma = 0.5
        tau = 1. / np.linalg.norm(self.A, ord=2)**2
        I_tauAtA_inv = np.linalg.pinv(
            np.eye(n, p, k=0) + tau * self.A.T @ self.A)
        K = np.r_[D.toarray(), self.A]

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
                u = I_tauAtA_inv @ (u - tau * (-np.diff(v, append=0,
                                    prepend=0)) + tau * self.A.T @ self.y)
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
