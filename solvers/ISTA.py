from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.shared import prox_z
    from benchmark_utils.shared import grad_huber


class Solver(BaseSolver):
    """Proximal gradient descent for synthesis formulation."""
    name = 'Primal PGD synthesis (ISTA)'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1., 1.9],
                  'use_acceleration': [False, True]}

    def skip(self, A, reg, y, c, delta, data_fit):
        if self.use_acceleration and self.alpha > 1:
            return True, "FISTA only converge when stepsize <= 1/L"
        return False, None

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        p = self.A.shape[1]
        L = np.tri(p)
        AL = self.A @ L
        # alpha / rho
        stepsize = self.alpha / (np.linalg.norm(AL, ord=2)**2)
        # initialisation
        z = np.zeros(p)
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
            z = prox_z(z - stepsize * self.grad(AL, z),
                       self.reg * stepsize)
            if self.use_acceleration:
                z_acc[:] = z + (t_old - 1.) / t_new * (z - z_old)
        self.u = np.cumsum(z)

    def get_result(self):
        return self.u

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - A.T @ R
        else:
            return - A.T @ grad_huber(R, self.delta)
