from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import optimize


class Objective(BaseObjective):
    name = "TV1D"

    parameters = {'reg': [0.1],
                  'delta': [0.9],
                  'data_fit': ['quad', 'huber']
                  }

    def set_data(self, A, y):
        self.A = A
        self.y = y
        self.n_samples = y.shape[0] - A.shape[0] + 1
        S = np.convolve(np.ones(self.n_samples), A)
        self.c = self.get_c(S, self.delta)
        self.reg_scaled = self.reg*self.get_reg_max(self.c)

    def compute(self, u):
        R = self.y - np.convolve(u, self.A)
        if self.data_fit == 'quad':
            return .5 * R @ R + self.reg_scaled*(abs(np.diff(u)).sum())
        else:
            return self.huber(R, self.delta) \
                + self.reg_scaled*(abs(np.diff(u)).sum())

    def get_one_solution(self):
        return np.zeros(self.n_samples)

    def to_dict(self):
        return dict(A=self.A, reg=self.reg_scaled, y=self.y, c=self.c,
                    delta=self.delta, data_fit=self.data_fit)

    def huber(self, R, delta):
        norm_1 = np.abs(R)
        loss = np.where(norm_1 < delta,
                        0.5 * norm_1**2,
                        delta * norm_1 - 0.5 * delta**2)
        return np.sum(loss)

    def get_c(self, S, delta):
        if self.data_fit == 'quad':
            return (S @ self.y)/(S @ S)
        else:
            return self.c_huber(S, delta)

    def c_huber(self, S, delta):
        def f(c):
            R = self.y - S * c
            return abs(S @ self.grad_huber(R, delta))
        yS = self.y / S
        return optimize.golden(f, brack=(min(yS), max(yS)))

    def get_reg_max(self, c):
        L = np.tri(self.n_samples)
        z = np.zeros(self.n_samples)
        z[0] = c
        return np.max(abs(self.grad_z(self.A, L, z)))

    def grad_z(self, A, L, z):
        R = self.y - np.convolve(L @ z, A)
        if self.data_fit == 'quad':
            return - L.T @ np.correlate(R, A, mode="valid")
        else:
            return - L.T @ np.correlate(self.grad_huber(R, self.delta),
                                        A, mode="valid")

    def grad_huber(self, R, delta):
        return np.where(np.abs(R) < delta, R, np.sign(R) * delta)
