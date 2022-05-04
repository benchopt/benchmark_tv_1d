from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Fixed point with block updates."""
    name = 'FP'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1.9]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        len_y = len(self.y)
        L = np.tri(len_y)
        AL = self.A @ L
        # alpha / rho
        stepsize = self.alpha / (len_y * np.max((AL**2).sum(axis=1)))
        # initialisation
        z = np.zeros(len_y)
        z[0] = self.c
        mu = np.zeros((len_y, len_y))
        nu = np.zeros(len_y)
        while callback(np.cumsum(z)):
            mu = z - stepsize * (len_y * (AL @ z - self.y) * AL.T).T
            nu = np.mean(mu, axis=0)
            z = self.st(nu, stepsize * self.reg)
        self.u = np.cumsum(z)

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w0 = w[0]
        w -= np.sign(w) * abs(np.clip(w, -mu, mu))
        w[0] = w0
        return w

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - A.T @ R
        else:
            return - A.T @ self.grad_huber(R, self.delta)

    def grad_huber(self, R, delta):
        return np.where(np.abs(R) < delta, R, np.sign(R) * delta)
