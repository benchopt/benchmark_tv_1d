from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg


def get_inverse_D(p):
    return - np.arange(1, p, 1) / p + np.tri(p)[:, 1:]


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
        if max(y.shape) > 1e4:
            return True, "solver has to do a too large densification"
        return False, None

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        n, p = self.A.shape
        if isinstance(self.A, np.ndarray):
            DA_inv = np.diff(np.linalg.pinv(self.A), axis=0)
            DA_invDA_invt = DA_inv @ DA_inv.T
            DA_invy = DA_inv @ self.y
            AtA_inv = np.linalg.pinv(self.A.T @ self.A)
        else:
            DA_inv = np.diff(np.linalg.pinv(self.A @ np.identity(p)), axis=0)
            AtA = LinearOperator(
                dtype=np.float64,
                matvec=lambda x: self.A.T @ self.A @ x,
                shape=(p, p)
            )
        Aty = self.A.T @ self.y
        tol_cg = 1e-12
        # alpha / rho
        stepsize = self.alpha / (np.linalg.norm(DA_inv, ord=2)**2)
        # initialisation
        self.u = self.c * np.ones(p)
        v = np.zeros(p - 1)
        v_tmp = np.zeros(p)
        v_old = v.copy()
        v_acc = v.copy()

        t_new = 1
        while callback():
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                v_old[:] = v
                v[:] = v_acc
            if isinstance(self.A, np.ndarray):
                v = np.clip(v - stepsize * (DA_invDA_invt @ v - DA_invy),
                            -self.reg, self.reg)
            else:
                v_tmp, _ = cg(AtA, Aty + np.diff(v, append=0, prepend=0),
                              x0=v_tmp, tol=tol_cg)
                v = np.clip(v + stepsize * np.diff(v_tmp),
                            -self.reg, self.reg)

            if self.use_acceleration:
                v_acc[:] = v + (t_old - 1.) / t_new * (v - v_old)

            if isinstance(self.A, np.ndarray):
                self.u = AtA_inv @ (Aty + np.diff(v, append=0, prepend=0))
            else:
                self.u, _ = cg(AtA, Aty + np.diff(v, append=0, prepend=0),
                          x0=self.u, tol=tol_cg)

    def get_result(self):
        return dict(u=self.u)
