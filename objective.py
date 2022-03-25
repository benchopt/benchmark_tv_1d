import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Ordinary Least Squares"


    def __init__(self, reg = 100):
        self.reg = reg

    def set_data(self, y):
        self.y = y

    def compute(self, x):
        A = np.eye(len(self.y), dtype=int)
        R = self.y - A @ x
        return .5 * R @ R + self.reg*(abs(np.diff(x)).sum())

    def to_dict(self):
        return dict(reg=self.reg, y=self.y)
