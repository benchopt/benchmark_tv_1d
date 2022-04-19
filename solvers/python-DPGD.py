from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags


class Solver(BaseSolver):
    """Dual Projected gradient descent for synthesis formulation."""
    name = 'DPGD'

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute

    def set_objective(self, A, reg, y):
        self.reg = reg
        self.A, self.y = A, y

    def run(self, callback):
        len_y = len(self.y)
        data = np.array([np.ones(len_y), -np.ones(len_y)])
        diags = np.array([0, 1])
        D = spdiags(data, diags, len_y-1, len_y).toarray()
        DA_inv = D @ np.linalg.pinv(self.A)
        v = np.zeros(len_y - 1)
        u = np.zeros(len_y)
        stepsize = 1 / (np.linalg.norm(DA_inv, ord=2)**2)  # 1/ rho
        DA_invDA_invt = DA_inv @ DA_inv.T
        DA_invy = DA_inv @ self.y
        AtA_inv = np.linalg.pinv(self.A.T @ self.A)
        Aty = self.A.T @ self.y

        while callback(u):
            v -= stepsize * (DA_invDA_invt @ v - DA_invy)
            v = np.clip(v, -self.reg, self.reg)
            u = AtA_inv @ (Aty - D.T @ v)
        self.u = u

    def get_result(self):
        return self.u
