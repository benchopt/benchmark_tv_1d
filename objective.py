from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import optimize


class Objective(BaseObjective):
    name = "TV1D"

    parameters = {'reg': [0.5],
                  'delta': [0.9],
                  'data_fit': ['quad', 'huber']
                  }

    def set_data(self, A, y, x):
        self.A, self.y, self.x = A, y, x
        S = self.A @ np.ones(self.A.shape[1])
        self.c = self.get_c(S, self.delta)
        self.reg_scaled = self.reg*self.get_reg_max(self.c)

    def compute(self, u):
        R = self.y - self.A @ u
        reg_TV = abs(np.diff(u)).sum()
        if self.data_fit == 'quad':
            loss = .5 * R @ R + self.reg_scaled * reg_TV
        else:
            loss = self.huber(R, self.delta) + self.reg_scaled * reg_TV

        norm_x = np.linalg.norm(u - self.x)

        return dict(value=loss, norm_x=norm_x)

    def get_one_solution(self):
        return np.zeros(self.A.shape[1])

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
            return abs((S * self.grad_huber(R, delta)).sum())
        yS = self.y / S
        return optimize.golden(f, brack=(min(yS), max(yS)))

    def get_reg_max(self, c):
        L = np.tri(self.A.shape[1])
        AL = self.A @ L
        z = np.zeros(self.A.shape[1])
        z[0] = c
        return np.max(abs(self.grad(AL, z)))

    def grad(self, A, u):
        R = self.y - A @ u
        if self.data_fit == 'quad':
            return - A.T @ R
        else:
            return - A.T @ self.grad_huber(R, self.delta)

    def grad_huber(self, R, delta):
        return np.where(np.abs(R) < delta, R, np.sign(R) * delta)
