import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Ordinary Least Squares"

    def __init__(self, reg=0.5):
        self.reg = reg  # 0<reg<1

    def set_data(self, A, y):
        self.A = A
        self.y = y

    def compute(self, u):
        R = self.y - self.A @ u
        return .5 * R @ R + self.reg*self.reg_max*(abs(np.diff(u)).sum())

    def to_dict(self):
        S = np.sum(self.A, axis=1)
        c = S.dot(self.y)/(S.dot(S))*np.eye(len(self.y), dtype=int)
        reg_max = np.max(abs((self.A).T.dot(self.A.dot(c) - self.y)))
        self.reg_max = reg_max
        return dict(A=self.A, reg=self.reg, reg_max=self.reg_max, y=self.y)
