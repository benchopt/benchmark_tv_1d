from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg


class Solver(BaseSolver):
    """Dual Projected gradient descent for analysis formulation."""
    name = 'Dual PGD analysis'

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1.],
                  'use_acceleration': [True]}

    def skip(self, A, reg, y, c, delta, data_fit):
        if data_fit == 'huber':
            return True, "solver does not work with huber loss"
        return False, None

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.n_samples = y.shape[0] - A.shape[0] + 1
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        data = np.array([-np.ones(self.n_samples), np.ones(self.n_samples)])
        diags = np.array([0, 1])
        D = spdiags(data, diags, self.n_samples-1, self.n_samples).toarray()
        D_inv = np.linalg.pinv(D)
        AD_inv = np.array([np.convolve(D_inv_col, self.A)
                          for D_inv_col in D_inv.T]).T
        DA_inv = np.linalg.pinv(AD_inv)
        # alpha / rho
        stepsize = self.alpha / (np.linalg.norm(DA_inv, ord=2)**2)
        Aty = np.correlate(self.y, self.A, mode="valid")
        AtA = LinearOperator(shape=(self.n_samples, self.n_samples),
                             matvec=lambda x: np.correlate(
            np.convolve(x, self.A), self.A, mode="valid"))
        # initialisation
        u = self.c * np.ones(self.n_samples)
        v = np.zeros(self.n_samples - 1)
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
