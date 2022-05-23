from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Proximal gradient descent for synthesis formulation."""
    name = 'Primal PGD synthesis (ISTA)'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1.9],
                  'use_acceleration': [False, True]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.n_samples = y.shape[0] - A.shape[0] + 1
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        L = np.tri(self.n_samples)
        # alpha / rho
        stepsize = self.alpha / (np.linalg.norm(self.A, ord=2))**2
        # initialisation
        z = np.zeros(self.n_samples)
        z[0] = self.c
        z_old = z.copy()
        z_acc = z.copy()

        t_new = 1
        while callback(np.cumsum(z)):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                z_old[:] = z
                z[:] = z_acc
            z = self.st(z - stepsize * self.grad_z(self.A, L, z),
                        self.reg * stepsize)
            if self.use_acceleration:
                z_acc[:] = z + (t_old - 1.) / t_new * (z - z_old)
        self.u = np.cumsum(z)

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w0 = w[0]
        w -= np.clip(w, -mu, mu)
        w[0] = w0
        return w

    def grad_z(self, A, L, z):
        R = self.y - np.convolve(L @ z, A)
        if self.data_fit == 'quad':
            return - L.T @ np.correlate(R, A, mode="valid")
        else:
            return - L.T @ np.correlate(self.grad_huber(R, self.delta),
                                        A, mode="valid")

    def grad_huber(self, R, delta):
        return np.where(np.abs(R) < delta, R, np.sign(R) * delta)
