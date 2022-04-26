from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import prox_tv as ptv


class Solver(BaseSolver):
    """Proximal gradient descent for analysis formulation."""
    name = 'PGD'

    install_cmd = 'conda'
    requirements = ['pip:prox-tv']

    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute
    parameters = {'use_acceleration': [False, True]}

    def set_objective(self, A, reg, y, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        len_y = len(self.y)
        stepsize = 1 / (np.linalg.norm(self.A, ord=2)**2)  # 1/ rho

        # initialisation
        S = np.sum(self.A, axis=1)
        if self.data_fit == 'quad':
            c = (S @ self.y)/(S @ S)
        else:
            c = self.c_huber(S, self.delta, 100)
        u = c * np.ones(len_y)
        u_acc = u.copy()
        u_old = u.copy()

        t_new = 1
        while callback(u):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                u_old[:] = u
                u[:] = u_acc
            u = ptv.tv1_1d(
                u - stepsize * self.grad(self.A, u),
                self.reg * stepsize, method='condat')
            if self.use_acceleration:
                u_acc[:] = u + (t_old - 1.) / t_new * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u

    def to_dict(self):
        return dict(A=self.A, reg=self.reg, y=self.y)

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - A.T @ R
        else:
            return self.grad_huber(A, R, self.delta)

    def grad_huber(self, A, R, delta):
        return - A.T @ (np.where(np.abs(R) < delta, R, np.sign(R) * delta))

    def c_huber(self, S, delta, niter):
        list_c = np.linspace(min(self.y), max(self.y), niter)
        diff = []
        for c in list_c:
            R = self.y - S * c
            diff.append(abs((np.where(np.abs(R) < delta, self.y - c,
                                      np.sign(R) * delta)).sum()))
        index = diff.index(min(diff))
        return list_c[index]
