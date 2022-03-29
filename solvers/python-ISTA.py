import numpy as np
from benchopt import BaseSolver


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'ISTA'

    # any parameter defined here is accessible as a class attribute

    def set_objective(self, reg_max, y, reg):
        self.reg = reg
        self.reg_max, self.y = reg_max, y

    def run(self, n_iter):
        len_y = len(self.y)
        L = np.tri(len_y)
        A = np.eye(len_y, dtype=int)
        AL = A @ L
        stepsize = 1 / (np.linalg.norm(AL, ord=2)**2)  # 1/ rho
        reg_tot = self.reg*self.reg_max
        z = np.zeros(len_y)  # initialisation
        for _ in range(n_iter):
            z = self.st(z + stepsize * AL.T @ (self.y - AL @ z),
                        reg_tot * stepsize)
        self.x = L.dot(z)

    def get_result(self):
        return self.x

    def st(self, w, mu):
        w0 = w[0]
        w -= np.sign(w) * abs(np.clip(w, -mu, mu))
        w[0] = w0
        return w
