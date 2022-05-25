import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    import numpy as np
    from celer import Lasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    """Coordinate descent for synthesis formulation as lasso problem."""
    name = 'Celer synthesis'

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy='iteration'
    )
    install_cmd = 'conda'
    requirements = ['pip:celer']
    references = [
        'M. Massias, A. Gramfort and J. Salmon, ICML, '
        '"Celer: a Fast Solver for the Lasso with Dual Extrapolation", '
        'vol. 80, pp. 3321-3330 (2018)'
    ]

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

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.lasso = Lasso(
            alpha=self.reg / self.A.shape[0], max_iter=1,
            max_epochs=100000,
            tol=1e-12, prune=True, fit_intercept=False,
            warm_start=False, positive=False, verbose=False,
        )
        self.run(2)

    def run(self, n_iter):
        n, p = self.A.shape
        L = np.tri(p)[:, 1:]
        S = np.sum(self.A, axis=1)
        AL = self.A @ L
        A_op = self.A @ np.ones((p, p)) @ (self.A.T) / (S @ S)
        y_new = self.y - A_op @ self.y
        AL_new = AL - A_op @ AL
        self.lasso.max_iter = n_iter
        self.lasso.fit(AL_new, y_new)
        z = self.lasso.coef_.flatten()
        c = S @ (self.y - AL @ z) / (S @ S)
        self.u = np.r_[0, np.cumsum(z)] + c

    @staticmethod
    def get_next(previous):
        """Linear growth for n_iter."""
        return previous + 1

    def get_result(self):
        return self.u
