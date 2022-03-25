import numpy as np
from benchopt import BaseSolver


class Solver(BaseSolver):
    """Gradient descent solver, optionally accelerated."""
    name = 'ISTA'

    # any parameter defined here is accessible as a class attribute

    def set_objective(self, reg, y):
        self.reg = reg
        self.y = y
    
    def run(self, n_iter):
        L = np.tri(len(self.y))
        A = np.eye(len(self.y), dtype=int)
        AL = A @ L
        rho = np.linalg.norm(AL, ord=2)**2
    	
        z = np.concatenate(([self.y[0]], np.diff(self.y)))
        for _ in range(n_iter):
            z = self.st(z + 1/rho * AL.T @ (self.y - AL @ z), self.reg / rho)
        self.x = L.dot(z)

    def get_result(self):
        return self.x

    def st(self, w, mu):
        w0 = w[0]
        w -= np.sign(w) * abs(np.clip(w, -mu, mu))
        w[0] = w0
        return w
 
