from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg


class Solver(BaseSolver):
    """Dual Projected gradient descent for analysis formulation."""
    name = 'Dual PGD analysis'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1.],
                  'use_acceleration': [False, True]}

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
        n, p = self.A.shape
        data = np.array([-np.ones(p), np.ones(p)])
        diags = np.array([0, 1])
        D = spdiags(data, diags, p-1, p).toarray()
        DA_inv = np.linalg.pinv(self.A @ np.linalg.pinv(D))
        AtA = LinearOperator(
            dtype=np.float64,
            matvec=lambda x: self.A.T @ self.A @ x,
            shape=(p, p)
        )
        Aty = self.A.T @ self.y
        # alpha / rho
        stepsize = self.alpha / (np.linalg.norm(DA_inv, ord=2)**2)
        # initialisation
        u = self.c * np.ones(p)
        v = np.zeros(p - 1)
        v_old = v.copy()
        v_acc = v.copy()

        t_new = 1
        while callback(u):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                v_old[:] = v
                v[:] = v_acc
            v_tmp, _ = cg(AtA, Aty + np.diff(v, append=0, prepend=0))
            v = np.clip(v + stepsize * np.diff(v_tmp),
                        -self.reg, self.reg)
            if self.use_acceleration:
                v_acc[:] = v + (t_old - 1.) / t_new * (v - v_old)
            u, _ = cg(AtA, Aty + np.diff(v, append=0, prepend=0))
        self.u = u

    def get_result(self):
        return self.u
