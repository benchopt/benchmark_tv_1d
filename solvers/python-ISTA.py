from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Proximal gradient descent for synthesis formulation."""
    name = 'ISTA'

    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    parameters = {'alpha': [1., 1.5, 1.9]}

    def set_objective(self, A, reg, y, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        len_y = len(self.y)
        L = np.tri(len_y)
        AL = self.A @ L
        stepsize = self.alpha / (np.linalg.norm(AL, ord=2)**2)  # alpha / rho
        # initialisation
        S = np.sum(self.A, axis=1)
        if self.data_fit == 'quad':
            c = (S @ self.y)/(S @ S)
        else:
            c = self.c_huber(S, self.delta, 100)
        z = np.zeros(len_y)
        z[0] = c
        while callback(np.cumsum(z)):
            z = self.st(z - stepsize * self.grad(AL, z),
                        self.reg * stepsize)
        self.u = np.cumsum(z)

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w0 = w[0]
        w -= np.sign(w) * abs(np.clip(w, -mu, mu))
        w[0] = w0
        return w

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
