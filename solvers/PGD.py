from functools import partial
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import prox_tv as ptv
    from benchmark_utils.shared import grad_huber
    from benchmark_utils.shared import get_l2norm
    from benchmark_utils.tv_numba import (
        tv_mm, gtv_mm_tol2, jit_module, prox_condat,
    )


class Solver(BaseSolver):
    """Proximal gradient descent for analysis formulation."""
    name = 'Primal PGD analysis'

    install_cmd = 'conda'

    # We need blas devel to get the include file for BLAS/LAPACK operations
    requirements = ['blas-devel', 'pip:prox-tv', 'numba']

    stopping_criterion = SufficientProgressCriterion(
        patience=3, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'alpha': [1.],
        'use_acceleration': [False, True],
        'prox_op': ["condat_C", "tv_mm", "condat_numba", "gtv_mm3", "gtv_mm5"]
    }

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

        if self.prox_op != "condat_C":
            # We jit here to avoid using non jitted function definitions
            jit_module()
        self.prox_f = None
        if self.prox_op == "condat_C":
            self.prox_f = partial(ptv.tv1_1d, method='condat')
        elif self.prox_op == "tv_mm":
            self.prox_f = partial(tv_mm, max_iter=1000, tol=1e-6)
        elif self.prox_op == "condat_numba":
            self.prox_f = prox_condat
        elif "gtv_mm" in self.prox_op:
            K = int(self.prox_op[-1])
            self.prox_f = partial(gtv_mm_tol2, max_iter=1000, tol=1e-15, K=K)

    def run(self, callback):
        p = self.A.shape[1]
        # alpha / rho
        stepsize = self.alpha / get_l2norm(self.A)**2
        # initialisation
        u = self.c * np.ones(p)
        u_acc = u.copy()
        u_old = u.copy()

        t_new = 1
        while callback(u):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                u_old[:] = u
                u[:] = u_acc
            u = self.prox_f(
                u - stepsize * self.grad(self.A, u),
                self.reg * stepsize,
            )
            if self.use_acceleration:
                u_acc[:] = u + (t_old - 1.) / t_new * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - A.T @ R
        else:
            return - A.T @ grad_huber(R, self.delta)
