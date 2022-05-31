from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Primal-Dual Splitting Method for analysis formulation."""
    name = 'CondatVu analysis'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'ratio': [1.],
                  'eta': [1.]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        n, p = self.A.shape
        # Block preconditioning (2x2)
        LD = 2.0  # Lipschitz constant associated to D (only for 1d!!)
        LA = np.linalg.norm(self.A @ np.identity(p), ord=2)
        sigma_v = 1.0 / (self.ratio * LD)
        sigma_w = 1.0 / (self.ratio * LA)
        tau = 1 / (LA ** 2 / 2 + sigma_v * LD ** 2)
        eta = self.eta
        # initialisation
        u = self.c * np.ones(p)
        v = np.zeros(p - 1)
        w = self.A @ u

        while callback(u):
            if self.data_fit == 'quad':
                u_tmp = (u - tau * self.A.T @ (self.A @ u - self.y)
                         - tau * (-np.diff(v, append=0, prepend=0))
                         )
            else:
                u_tmp = (u
                         - tau * (- np.diff(v, append=0, prepend=0)
                                  + self.A.T @ w)
                         )
                w_tmp = w + sigma_w * self.A @ (2 * u_tmp - u)
                R_tmp = sigma_w * self.y - w_tmp
                w_tmp = (w_tmp
                         - np.where(abs(R_tmp) < (self.delta * (sigma_w + 1)),
                                    sigma_w * ((self.y + w_tmp)
                                               / (sigma_w + 1)),
                                    w_tmp + (self.delta * np.sign(R_tmp)))
                         )
                w = eta * w_tmp + (1 - eta) * w
            v_tmp = (v + sigma_v * np.diff(2 * u_tmp - u)
                     - sigma_v * self.st(v / sigma_v +
                                         np.diff(2 * u_tmp - u),
                                         self.reg / sigma_v)
                     )
            u = eta * u_tmp + (1 - eta) * u
            v = eta * v_tmp + (1 - eta) * v
        self.u = u

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w
