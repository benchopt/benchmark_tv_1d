import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Ordinary Least Squares"

    def __init__(self, reg=0.5):
        self.reg = reg  # 0<reg<1

    def set_data(self, reg_max, y):
        self.reg_max = reg_max
        self.y = y

    def compute(self, x):
        A = np.eye(len(self.y), dtype=int)
        R = self.y - A @ x
        return .5 * R @ R + self.reg*self.reg_max*(abs(np.diff(x)).sum())

    def to_dict(self):
        return dict(reg=self.reg, reg_max=self.reg_max, y=self.y)
