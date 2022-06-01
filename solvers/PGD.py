from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import prox_tv as ptv


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
    parameters = {'alpha': [1.],
                  'use_acceleration': [False, True]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        p = self.A.shape[1]
        # alpha / rho
        stepsize = self.alpha / self.get_l2norm(self.A)**2
        # initialisation
        u = self.c * np.ones(p)
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
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - A.T @ R
        else:
            return - A.T @ self.grad_huber(R, self.delta)

    def grad_huber(self, R, delta):
        return np.where(np.abs(R) < delta, R, np.sign(R) * delta)

    def get_l2norm(self, A, n_iter=100):
        if isinstance(A, np.ndarray):
            return np.linalg.norm(A, ord=2)
        else:
            AtA = A.T @ A
            x = np.random.randn(AtA.shape[0])
            for _ in range(n_iter):
                x = AtA @ x
                x /= np.linalg.norm(x)
            return np.sqrt(np.linalg.norm(AtA @ x))
