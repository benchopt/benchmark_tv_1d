import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Ordinary Least Squares"

    parameters = {'reg': [0.5],
                  'delta': [0.9],
                  'data_fit': ['quad', 'huber']
                  # 'reg': np.linspace(0.1, 0.9, 3).round(1)
                  }

    def __init__(self, reg=0.5, delta=100, data_fit='quad'):
        self.reg = reg  # 0<reg<1
        self.delta = delta
        self.data_fit = data_fit

    def set_data(self, A, y):
        self.A = A
        self.y = y
        S = np.sum(self.A, axis=1)
        L = np.tri(y.shape[0])
        AL = self.A @ L
        if self.data_fit == 'quad':
            c = (S @ self.y)/(S @ S)
            reg_max = np.max(abs(AL.T @ (S * c - self.y)))
        else:
            c = self.c_huber(S, self.delta, 100)
            reg_max = self.reg_huber(AL, S, self.delta, c)
        self.reg = self.reg*reg_max

    def compute(self, u):
        R = self.y - self.A @ u
        if self.data_fit == 'quad':
            return .5 * R @ R + self.reg*(abs(np.diff(u)).sum())
        else:
            return self.huber(R, self.delta) + self.reg*(abs(np.diff(u)).sum())

    def to_dict(self):
        return dict(A=self.A, reg=self.reg, y=self.y, delta=self.delta,
                    data_fit=self.data_fit)

    def huber(self, R, delta):
        norm_1 = np.abs(R)
        loss = np.where(norm_1 < delta,
                        0.5 * norm_1**2,
                        delta * norm_1 - 0.5 * delta**2)
        return np.sum(loss)

    def c_huber(self, S, delta, niter):
        list_c = np.linspace(min(self.y), max(self.y), niter)
        diff = []
        for c in list_c:
            R = self.y - S * c
            diff.append(abs((np.where(np.abs(R) < delta, self.y - c,
                                      np.sign(R) * delta)).sum()))
        index = diff.index(min(diff))
        return list_c[index]

    def reg_huber(self, AL, S, delta, c):
        R = self.y - S * c
        return np.max(abs(AL.T @ np.where(np.abs(R) < delta, R,
                                          np.sign(R) * delta)))
