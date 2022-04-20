from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Proximal gradient descent for synthesis formulation."""
    name = 'ISTA'

    stopping_strategy = "callback"
    # any parameter defined here is accessible as a class attribute

    def set_objective(self, A, reg, y):
        self.reg = reg
        self.A, self.y = A, y

    def run(self, callback):
        len_y = len(self.y)
        L = np.tri(len_y)
        AL = self.A @ L
        stepsize = 1 / (np.linalg.norm(AL, ord=2)**2)  # 1/ rho
        z = np.zeros(len_y)  # initialisation
        while callback(L @ z):
            z = self.st(z - stepsize * AL.T @ (AL @ z - self.y),
                        self.reg * stepsize)
        self.u = L @ z

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w0 = w[0]
        w -= np.sign(w) * abs(np.clip(w, -mu, mu))
        w[0] = w0
        return w
