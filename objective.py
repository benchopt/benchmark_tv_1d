import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Ordinary Least Squares"

    parameters = {'fit_intercept': [False]}

    def __init__(self, fit_intercept=False):
        self.fit_intercept = fit_intercept

    def set_data(self, reg, y):
        self.reg, self.y = reg, y

    def compute(self, x):
        A = np.eye(len(self.y), dtype=int)
        R = self.y - A @ x
        return .5 * R @ R + self.reg*(abs(np.diff(x)).sum())

    def to_dict(self):
        return dict(reg=self.reg, y=self.y)
