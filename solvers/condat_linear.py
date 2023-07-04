from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.tv_numba import linearized_taut_string


class Solver(BaseSolver):
    """Solve the TV minimization problem using a majorization-minimization algorithm.

    Condat, L. (2013). A direct algorithm for 1D total variation denoising.
    https://github.com/albarji/proxTV/blob/master/src/TVL1opt.cpp
    """

    name = "Linearized Taut String"
    stopping_criterion = SingleRunCriterion()
    requirements = ["pip:numba"]

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.data_fit = data_fit
        # Delta is ignored, only used for huber function.
        self.run(5)

    def skip(self, **objective_dict):
        if objective_dict["data_fit"] != "quad":
            return True, "TVMM solver only useable for quadratic data_fit"
        return False, None

    def run(self, n_iter):
        self.u = np.zeros_like(self.y)
        linearized_taut_string(self.y, self.reg, self.u)

    def get_result(self):
        return self.u
