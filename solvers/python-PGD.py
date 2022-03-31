import numpy as np
from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import prox_tv as ptv


class Solver(BaseSolver):
    """analytique"""
    name = 'PGD'

    install_cmd = 'conda'
    requirements = ['pip:prox-tv']

    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute

    def set_objective(self, A, reg, reg_max, y):
        self.reg, self.reg_max = reg, reg_max
        self.A, self.y = A, y

    def run(self, callback):
        reg_tot = self.reg*self.reg_max
        stepsize = 1 / (np.linalg.norm(self.A, ord=2)**2)  # 1/ rho
        u = np.zeros(len(self.y))  # initialisation
        while callback(u):
            u = ptv.tv1_1d(
                u - stepsize * self.A.T @ (self.A @ u - self.y),
                reg_tot * stepsize, method='condat')
        self.u = u

    def get_result(self):
        return self.u
