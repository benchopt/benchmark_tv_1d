import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    import numpy as np
    from skglm import GeneralizedLinearEstimator
    from skglm.datafits import Huber
    from skglm.penalties import WeightedL1
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    """Coordinate descent for synthesis formulation as lasso problem."""
    name = 'Celer-Huber'

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy='iteration'
    )

    install_cmd = 'conda'
    requirements = ['pip:skglm']
    references = [
        'M. Massias, A. Gramfort and J. Salmon, ICML, '
        '"Celer: a Fast Solver for the Lasso with Dual Extrapolation", '
        'vol. 80, pp. 3321-3330 (2018)'
    ]

    def skip(self, A, reg, y, c, delta, data_fit):
        if data_fit == 'quad':
            return True, "solver does not work with quadratic loss"
        return False, None

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        weights = np.ones(self.y.shape[0])
        weights[0] = 0
        self.clf = GeneralizedLinearEstimator(
            Huber(self.delta),
            WeightedL1(self.reg / self.y.shape[0], weights),
            is_classif=False,
            max_iter=1, max_epochs=100000,
            tol=1e-12, fit_intercept=False,
            warm_start=False, verbose=False,
        )

    def run(self, n_iter):
        len_y = self.y.shape[0]
        L = np.tri(len_y)
        AL = self.A @ L
        self.clf.max_iter = n_iter
        self.clf.fit(AL, self.y)
        z = self.clf.coef_.flatten()
        self.u = np.cumsum(z)

    @staticmethod
    def get_next(previous):
        """Linear growth for n_iter."""
        return previous + 1

    def get_result(self):
        return self.u
