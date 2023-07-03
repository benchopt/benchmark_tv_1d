from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import numba as nb
    from benchmark_utils.tv_numba import tv_mm

class Solver(BaseSolver):
    """Solve the TV minimization problem using a majorization-minimization algorithm."""

    name = 'tv_mm'
    stopping_strategy = 'iteration'
    parameters = {"tol": [1e-3, 1e-4, 1e-5, 1e-6]}
    requirements = ['pip:numba']

    def set_objective(self, A, reg, y, c , delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.data_fit = data_fit
        # Delta is ignored, only used for huber function.
        self.run(5)

    def skip(self, **objective_dict):
        if objective_dict['data_fit'] != 'quad':
            return True, "TVMM solver only useable for quadratic data_fit"
        return False, None

    def run(self, n_iter):
        self.u = tv_mm(self.y, self.reg, n_iter,self.tol)

    def get_result(self):
        return self.u
