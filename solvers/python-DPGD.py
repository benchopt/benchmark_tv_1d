import numpy as np
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion


class Solver(BaseSolver):
    """analysis and synthesis : primal and dual"""
    name = 'DPGD'

    stopping_criterion = SufficientProgressCriterion(
        patience=50, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute

    def set_objective(self, A, reg, y):
        self.reg = reg
        self.A, self.y = A, y

    def run(self, callback):
        len_y = len(self.y)
        D = (np.eye(len_y, k=1) - np.identity(len_y))[:-1]
        DA_inv = D @ np.linalg.pinv(self.A)
        v = np.zeros(len_y - 1)
        u = np.zeros(len_y)
        stepsize = 1 / (np.linalg.norm(DA_inv, ord=2)**2)  # 1/ rho

        while callback(u):
            v -= stepsize * (DA_inv @ DA_inv.T @ v - DA_inv @ self.y)
            v = np.clip(v, -self.reg, self.reg)
            u = np.linalg.pinv(self.A.T @ self.A) @ (self.A.T @ self.y - D.T @ v)
        self.u = u

    def get_result(self):
        return self.u
