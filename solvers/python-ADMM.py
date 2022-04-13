import numpy as np
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion


class Solver(BaseSolver):
    """analysis and synthesis : primal and dual"""
    name = 'ADMM'

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'gamma': [1.5],
                  'update_penalization': [False, True]}

    def set_objective(self, A, reg, y):
        self.reg = reg
        self.A, self.y = A, y

    def run(self, callback):
        len_y = len(self.y)
        D = (np.eye(len_y, k=1) - np.identity(len_y))[:-1]
        u = np.zeros(len_y)
        z = np.zeros(len_y - 1)
        mu = np.zeros(len_y - 1)
        gamma = self.gamma

        while callback(u):
            z_old = z
            A_tmp = np.linalg.pinv(self.A.T @ self.A + gamma * D.T @ D)
            u = A_tmp @ (self.A.T @ self.y - D.T @ mu + gamma * D.T @ z)
            z = self.st(D @ u + mu / gamma, self.reg / gamma)
            mu += gamma*(D @ u - z)

            if self.update_penalization:
                r = np.linalg.norm(D @ u - z, ord=2)
                s = np.linalg.norm(gamma * D.T @ (z - z_old), ord=2)
                if r > 10 * s:
                    gamma *= 2
                if s > 10 * r:
                    gamma /= 2
        self.u = u

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w -= np.sign(w) * abs(np.clip(w, -mu, mu))
        return w
