from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags


class Solver(BaseSolver):
    """Alternating direction method for synthesis and analysis formulation."""
    name = 'ADMM analysis and synthesis'

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'gamma': [1.5, 1.9],
                  'update_pen': [False]}

    def skip(self, A, reg, y, c, delta, data_fit):
        if data_fit == 'huber':
            return True, "solver does not work with huber loss"
        return False, None

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        len_y = len(self.y)
        data = np.array([np.ones(len_y), -np.ones(len_y)])
        diags = np.array([0, 1])
        D = spdiags(data, diags, len_y-1, len_y)
        u = self.c * np.ones(len_y)
        z = np.zeros(len_y - 1)
        mu = np.zeros(len_y - 1)
        gamma = self.gamma
        AtA = self.A.T @ self.A
        DtD = D.T @ D
        Aty = self.A.T @ self.y
        A_tmp = np.linalg.pinv(AtA + gamma * DtD)

        while callback(u):
            z_old = z
            u = np.ravel(A_tmp @ (Aty - D.T @ mu + gamma * D.T @ z))
            z = self.st(D @ u + mu / gamma, self.reg / gamma)
            mu += gamma * (D @ u - z)

            if self.update_pen:
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
        w -= np.clip(w, -mu, mu)
        return w
