from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    from benchmark_utils.tv_numba import tv_mm, jit_module



class Solver(BaseSolver):
    """TV Denoising with Majoration-Minimisation.

    I. Selesnick, “Total Variation Denoising (an MM algorithm)”.
    https://eeweb.engineering.nyu.edu/iselesni/lecture_notes/TVDmm/
    """

    name = "tv_mm"
    stopping_strategy = "iteration"
    requirements = ["pip:numba"]

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.data_fit = data_fit
        # Delta is ignored, only used for huber function.
        self.run(5)
        jit_module()

    def skip(self, **objective_dict):
        if objective_dict["data_fit"] != "quad":
            return True, "TVMM solver only useable for quadratic data_fit"
        A = objective_dict["A"]
        if getattr(A,"_op_type", None) != "identity":
            return True, "TVMM solver only useable for identity matrix"
        return False, None


    def run(self, n_iter):
        self.u = tv_mm(self.y, self.reg, n_iter, 0.0)

    def get_result(self):
        return self.u
