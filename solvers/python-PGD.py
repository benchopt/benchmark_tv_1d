from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import prox_tv as ptv


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'PGD'

    install_cmd = 'conda'
    requirements = ['pip:prox-tv']

    # any parameter defined here is accessible as a class attribute

    def set_objective(self, reg_max, y, reg):
        self.reg = reg
        self.reg_max, self.y = reg_max, y

    def run(self, n_iter):
        reg_tot = self.reg*self.reg_max
        x = ptv.tv1_1d(self.y, reg_tot)
        self.x = x

    def get_result(self):
        return self.x
