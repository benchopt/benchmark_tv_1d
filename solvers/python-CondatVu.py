from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import norm as spnorm


class Solver(BaseSolver):
    """Primal-Dual Splitting Method for synthesis and analysis formulation."""
    name = 'CondatVu'

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'eta': [0.5, 1],
                  'swap': [False]}

    def set_objective(self, A, reg, y):
        self.reg = reg
        self.A, self.y = A, y

    def run(self, callback):
        len_y = len(self.y)
        data = np.array([np.ones(len_y), -np.ones(len_y)])
        diags = np.array([0, 1])
        D = spdiags(data, diags, len_y-1, len_y)
        u = np.zeros(len_y)  # initialisation
        z = np.zeros(len_y - 1)

        sigma = 0.5
        eta = self.eta
        tau = 1 / (np.linalg.norm(self.A.T @ self.A, ord=2) /
                   2 + sigma * spnorm(D)**2)

        while callback(u):
            if self.swap:
                z_tmp = z + sigma * D @ u - sigma * \
                    self.st(z / sigma + D @ u, self.reg / sigma)
                u_tmp = u - tau * \
                    self.A.T @ (self.A @ u - self.y) - \
                    tau * D.T @ (2 * z_tmp - z)
            else:
                u_tmp = u - tau * \
                    self.A.T @ (self.A @ u - self.y) - tau * D.T @ z
                z_tmp = z + sigma * D @ (2 * u_tmp - u) - sigma * self.st(
                    z / sigma + D @ (2 * u_tmp - u), self.reg / sigma)
            u = eta * u_tmp + (1 - eta)*u
            z = eta * z_tmp + (1 - eta)*z
        self.u = u

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w -= np.sign(w) * abs(np.clip(w, -mu, mu))
        return w
