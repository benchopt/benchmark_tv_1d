import prox_tv as ptv
from benchopt import BaseSolver


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'PGD'

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
