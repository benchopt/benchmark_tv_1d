from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import prox_tv as ptv


class Solver(BaseSolver):
    """Proximal gradient descent for analysis formulation."""
    name = 'Primal PGD analysis'

    install_cmd = 'conda'
    requirements = ['pip:prox-tv']

    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1.],
                  'use_acceleration': [False, True]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.n_samples = y.shape[0] - A.shape[0] + 1
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        # alpha / rho
        stepsize = self.alpha / \
            np.linalg.norm(np.convolve(
                np.ones(self.n_samples), self.A), ord=2)**2
        # initialisation
        u = self.c * np.ones(self.n_samples)
        u_acc = u.copy()
        u_old = u.copy()

        t_new = 1
        while callback(u):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                u_old[:] = u
                u[:] = u_acc
            u = ptv.tv1_1d(
                u - stepsize * self.grad(self.A, u),
                self.reg * stepsize, method='condat')
            if self.use_acceleration:
                u_acc[:] = u + (t_old - 1.) / t_new * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u

    def grad(self, A, u):
        R = self.y - np.convolve(u, A)
        if self.data_fit == 'quad':
            return - np.correlate(R, A, mode="valid")
        else:
            return - np.correlate(self.grad_huber(R, self.delta),
                                  A, mode="valid")

    def grad_huber(self, R, delta):
        return np.where(np.abs(R) < delta, R, np.sign(R) * delta)
