from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """A first-order primal-dual algorithm for synthesis formulation."""
    name = 'ChambollePock analysis'

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'sigma': [1.],
                  'theta': [1.]}

    def skip(self, A, reg, y, c, delta, data_fit):
        if data_fit == 'huber':
            return True, "solver does not work with huber loss"
        return False, None

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        len_y = len(self.y)
        tau = 1. / (np.linalg.norm(self.A, ord=2)**2)
        I_tauAtA_inv = np.linalg.pinv(np.identity(
            len_y) + tau * self.A.T @ self.A)
        tauAty = tau * self.A.T @ self.y
        u = self.c * np.ones(len_y)
        v = np.zeros(len_y - 1)
        u_bar = u

        while callback(u):
            u_old = u
            v = np.clip(v + self.sigma * np.diff(u_bar), -self.reg, self.reg)
            u_tmp = u - tau * (-np.diff(v, append=0, prepend=0))
            u = I_tauAtA_inv @ (tauAty + u_tmp)
            u_bar = u + self.theta * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u
