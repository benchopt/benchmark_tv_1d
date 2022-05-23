import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    import numpy as np
    from skglm import GeneralizedLinearEstimator
    from skglm.datafits import Quadratic
    from skglm.datafits import Huber
    from skglm.penalties import WeightedL1
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    """Coordinate descent for synthesis formulation."""
    name = 'skglm synthesis'

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy='iteration'
    )

    install_cmd = 'conda'
    requirements = ["pip:git+https://github.com/EnLAI111/skglm@Huber_datafit"]
    references = [
        'M. Massias, P. Bannier, Q. Klopfenstein and Q. Bertrand.'
        '"skglm: Fast and modular Generalized Linear Models'
        'with support for models missing in scikit-learn."'
    ]

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        weights = np.ones(self.A.shape[1])
        weights[0] = 0

        if data_fit == 'quad':
            self.clf = GeneralizedLinearEstimator(
                Quadratic(),
                WeightedL1(self.reg / self.A.shape[0], weights),
                is_classif=False,
                max_iter=1, max_epochs=100000,
                tol=1e-12, fit_intercept=False,
                warm_start=False, verbose=False,
            )
        else:
            self.clf = GeneralizedLinearEstimator(
                Huber(self.delta),
                WeightedL1(self.reg / self.A.shape[0], weights),
                is_classif=False,
                max_iter=1, max_epochs=100000,
                tol=1e-12, fit_intercept=False,
                warm_start=False, verbose=False,
            )
        self.run(2)

    def run(self, n_iter):
        p = self.A.shape[1]
        L = np.tri(p)
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
