from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import prox_tv as ptv
    from benchmark_utils.shared import grad_huber
    from benchmark_utils.shared import get_l2norm


class Solver(BaseSolver):
    """Proximal gradient descent for analysis formulation."""
    name = 'Primal PGD analysis'

    install_cmd = 'conda'

    # We need blas devel to get the include file for BLAS/LAPACK operations
    requirements = ["blas-devel", 'pip:prox-tv']

    stopping_criterion = SufficientProgressCriterion(
        patience=3, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'alpha': [1.],
        'use_acceleration': [False, True]
    }

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        p = self.A.shape[1]
        # alpha / rho
        stepsize = self.alpha / get_l2norm(self.A)**2
        # initialisation
        self.u = self.c * np.ones(p)
        u_acc = self.u.copy()
        u_old = self.u.copy()

        t_new = 1
        while callback():
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                u_old[:] = self.u
                self.u[:] = u_acc
            self.u = ptv.tv1_1d(
                self.u - stepsize * self.grad(self.A, self.u),
                self.reg * stepsize, method='condat')
            if self.use_acceleration:
                u_acc[:] = self.u + (t_old - 1.) / t_new * (self.u - u_old)

    def get_result(self):
        return dict(u=self.u)

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - A.T @ R
        else:
            return - A.T @ grad_huber(R, self.delta)
