import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Ordinary Least Squares"

    def __init__(self, reg=0.1):
        self.reg = reg  # 0<reg<1

    def set_data(self, reg_max, y):
        self.reg_max = reg_max
        self.y = y

    def compute(self, x):
        A = np.eye(len(self.y), dtype=int)
        R = self.y - A @ x
        # S = np.sum(A, axis=1)
        # c = S.dot(self.y)/(S.dot(S))*np.eye(len(self.y), dtype=int)
        # reg_max = np.max(abs(np.transpose(A).dot(A.dot(c) - self.y)))
        return .5 * R @ R + self.reg*self.reg_max*(abs(np.diff(x)).sum())

    def to_dict(self):
        return dict(reg=self.reg, reg_max=self.reg_max, y=self.y)
