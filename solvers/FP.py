from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    prox_z = import_ctx.import_from('shared', 'prox_z')
    grad_huber = import_ctx.import_from('shared', 'grad_huber')


class Solver(BaseSolver):
    """Fixed point with block updates for synthesis formulation."""
    name = 'FP synthesis'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1.9],
                  'use_acceleration': [False, True]}

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

        z_old = z.copy()
        z_acc = z.copy()

        t_new = 1
        while callback(np.cumsum(z)):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                z_old[:] = z
                z[:] = z_acc
            mu = z - stepsize * (n * self.grad(AL, z) * AL.T).T
            nu = np.mean(mu, axis=0)
            z = prox_z(nu, stepsize * self.reg)
            if self.use_acceleration:
                z_acc[:] = z + (t_old - 1.) / t_new * (z - z_old)
        self.u = np.cumsum(z)

    def get_result(self):
        return self.u

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - R
        else:
            return - grad_huber(R, self.delta)
