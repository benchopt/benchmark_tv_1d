from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Fixed point with block updates for synthesis formulation."""
    name = 'FP synthesis'

    stopping_criterion = SufficientProgressCriterion(
        patience=50, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1.9]}

    def skip(self, A, reg, y, c, delta, data_fit):
        if data_fit == 'huber':
            return True, "solver does not work with huber loss"
        return False, None

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.n_samples = y.shape[0] - A.shape[0] + 1
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        n = self.y.shape[0]
        L = np.tri(self.n_samples)

        # initialisation
        z = np.zeros(self.n_samples)
        z[0] = 1

        # alpha / rho
        stepsize = self.alpha / \
            (n * 10 * np.max(np.convolve(L @ z, self.A)**2))

        z[0] = self.c
        mu = np.zeros((self.n_samples, self.n_samples))
        nu = np.zeros(self.n_samples)
        while callback(np.cumsum(z)):
            mu = z - stepsize * (n * self.grad_z(self.A, L, z)).T
            nu = np.mean(mu, axis=0)
            z = self.st(nu, stepsize * self.reg)
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
