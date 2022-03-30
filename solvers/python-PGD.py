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

    # any parameter defined here is accessible as a class attribute

    def set_objective(self, A, reg, reg_max, y):
        self.reg, self.reg_max = reg, reg_max
        self.A, self.y = A, y

    def run(self, n_iter):
        reg_tot = self.reg*self.reg_max
        stepsize = 1 / (np.linalg.norm(self.A, ord=2)**2)  # 1/ rho
        x = np.zeros(len(self.y))  # initialisation
        for _ in range(n_iter):
            x = ptv.tv1_1d(
                x + stepsize * self.A.T @ (self.y - self.A @ x), 
                reg_tot * stepsize, method='condat')
        self.x = x

    def get_result(self):
        return self.x
