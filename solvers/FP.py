from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Fixed point with block updates for synthesis formulation."""
    name = 'FP synthesis'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1.9]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        n, p = self.A.shape
        L = np.tri(p)
        AL = self.A @ L
        # alpha / rho
        stepsize = self.alpha / (n * np.max((AL**2).sum(axis=1)))
        # initialisation
        z = np.zeros(p)
        z[0] = self.c
        mu = np.zeros((p, p))
        nu = np.zeros(p)
        while callback(np.cumsum(z)):
            mu = z - stepsize * (n * self.grad(AL, z) * AL.T).T
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

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - R
        else:
            return - self.grad_huber(R, self.delta)

    def grad_huber(self, R, delta):
        return np.where(np.abs(R) < delta, R, np.sign(R) * delta)
