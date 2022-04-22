from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import prox_tv as ptv


class Solver(BaseSolver):
    """Proximal gradient descent for analysis formulation."""
    name = 'PGD'

    install_cmd = 'conda'
    requirements = ['pip:prox-tv']

    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute
    parameters = {'use_acceleration': [False, True]}

    def set_objective(self, A, reg, y):
        self.reg = reg
        self.A, self.y = A, y

    def run(self, callback):
        len_y = len(self.y)
        stepsize = 1 / (np.linalg.norm(self.A, ord=2)**2)  # 1/ rho

        # initialisation
        S = np.sum(self.A, axis=1)
        c = (S @ self.y)/(S @ S)
        u = c * np.ones(len_y)
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
                u - stepsize * self.A.T @ (self.A @ u - self.y),
                self.reg * stepsize, method='condat')
            if self.use_acceleration:
                u_acc[:] = u + (t_old - 1.) / t_new * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u
