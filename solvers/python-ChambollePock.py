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
        patience=20, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'sigma': [1.],
                  'theta': [1.]}

    def set_objective(self, A, reg_scaled, y, c, delta, data_fit):
        self.reg_scaled = reg_scaled
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        p = self.A.shape[1]
        data = np.array([-np.ones(p), np.ones(p)])
        diags = np.array([0, 1])
        D = spdiags(data, diags, p-1, p)
        tau = 1. / (np.linalg.norm(self.A, ord=2)**2)
        I_tauAtA_inv = np.linalg.pinv(np.identity(
            p) + tau * self.A.T @ self.A)
        tauAty = tau * self.A.T @ self.y
        K = np.r_[D.toarray(), self.A]

        u = self.c * np.ones(p)
        v = np.zeros(p - 1)
        w = np.r_[v, self.A @ u]
        u_bar = u

        while callback(u):
            u_old = u
            if self.data_fit == 'quad':
                v = v + self.sigma * np.diff(u_bar) - self.sigma * self.st(
                    v / self.sigma + np.diff(u_bar),
                    self.reg_scaled / self.sigma)
                u_tmp = u - tau * (-np.diff(v, append=0, prepend=0))
                u = I_tauAtA_inv @ (tauAty + u_tmp)
            else:
                x = w + self.sigma * K @ u_bar
                w[:p - 1] = x[:p - 1] - self.sigma * \
                    self.st(x[:p - 1] / self.sigma,
                            self.reg_scaled / self.sigma)
                R = self.sigma * self.y - x[p - 1:]
                w[p - 1:] = x[p - 1:] - \
                    np.where(abs(R) < self.delta * (self.sigma + 1),
                             self.sigma *
                             (self.y + x[p - 1:]) / (self.sigma + 1),
                             x[p - 1:] + self.delta * np.sign(R))
                u -= tau * K.T @ w
            u_bar = u + self.theta * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w
