import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Ordinary Least Squares"

    parameters = {'reg': np.linspace(0.1, 0.9, 3).round(1)}

    def __init__(self, reg=0.5):
        self.reg = reg  # 0<reg<1

    def set_data(self, A, y):
        self.A = A
        self.y = y
        S = np.sum(self.A, axis=1)
        c = (S @ self.y)/(S @ S)
        L = np.tri(y.shape[0])
        reg_max = np.max(abs(L.T @ (self.A.T @ (S * c - self.y))))
        self.reg = self.reg*reg_max

    def compute(self, u):
        R = self.y - self.A @ u
        return .5 * R @ R + self.reg*(abs(np.diff(u)).sum())

    def to_dict(self):
        return dict(A=self.A, reg=self.reg, y=self.y)
