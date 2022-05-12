from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags


class Solver(BaseSolver):
    """A first-order primal-dual algorithm for synthesis formulation."""
    name = 'ChambollePock (higher dualization) analysis'

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )


    # any parameter defined here is accessible as a class attribute
    parameters = {'ratio': [10.],
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
        # Block preconditioning (2x2)
        L = 2.0 # Lipschitz constant associated to D (only for 1d!!)
        Lh = np.linalg.norm(self.A, ord=2)**2
        tau = self.ratio / (L + np.sqrt(Lh))
        sigma_v = 1. / (self.ratio * L)
        sigma_w = 1. / (self.ratio * np.sqrt(Lh))
        # Init variables
        n, p = self.A.shape
        len_y = len(self.y)
        u = self.c * np.ones(p)
        v = np.zeros(p - 1) # we consider non-cyclic finite difference
        w = np.zeros(p)
        u_bar = u

        while callback(u):
            u_old = u
            v = np.clip(v + sigma_v * np.diff(u_bar), -self.reg, self.reg)
            w = (w + sigma_w * (self.A @ u_bar - self.y)) / (1. + sigma_w)
            # grad.T = -div, hence + sign
            u = u + tau * np.diff(v, prepend=0, append=0) - tau * self.A.T @ w
            u_bar = u + self.theta * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u
