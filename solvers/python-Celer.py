import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    import numpy as np
    from celer import Lasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'Celer'

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

    def set_objective(self, A, reg, y):
        self.reg = reg
        self.A, self.y = A, y

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.lasso = Lasso(
            alpha=self.reg / self.y.shape[0], max_iter=1, max_epochs=100000,
            tol=1e-12, prune=True,
            warm_start=False, positive=False, verbose=False,
        )

    def run(self, n_iter):
        len_y = self.y.shape[0]
        L = np.tri(len_y)
        AL = self.A @ L
        y_new = self.y - np.mean(self.y)
        AL_new = AL - np.mean(AL, axis=0, keepdims=True)
        self.lasso.max_iter = n_iter
        self.lasso.fit(AL_new, y_new)
        z = self.lasso.coef_.flatten()
        c = np.mean(self.y - AL @ z)
        self.u = L @ z + np.linalg.pinv(self.A) @ (np.ones(len_y) * c)

    @staticmethod
    def get_next(previous):
        "Linear growth for n_iter."
        return previous + 1

    def get_result(self):
        return self.u
