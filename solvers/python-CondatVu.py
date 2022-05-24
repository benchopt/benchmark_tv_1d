from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags


class Solver(BaseSolver):
    """Primal-Dual Splitting Method for analysis formulation."""
    name = 'CondatVu analysis'

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'eta': [1.]}

    def skip(self, A, reg, y, c, delta, data_fit):
        if data_fit == 'huber':
            return True, "solver does not work with huber loss"
        return False, None

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.n_samples = y.shape[0] - A.shape[0] + 1
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        data = np.array([-np.ones(self.n_samples), np.ones(self.n_samples)])
        diags = np.array([0, 1])
        D = spdiags(data, diags, self.n_samples-1, self.n_samples).toarray()

        # initialisation
        u = self.c * np.ones(self.n_samples)
        v = np.zeros(self.n_samples - 1)

        sigma = 0.5
        eta = self.eta

        tau = 1 / (10 * np.linalg.norm(self.A**2, ord=2) /
                   2 + sigma * np.linalg.norm(D, ord=2) ** 2)

        while callback(u):
            u_tmp = u - tau * np.correlate(np.convolve(u, self.A) - self.y,
                                           self.A, mode="valid") - \
                    tau * (-np.diff(v, append=0, prepend=0))
            v_tmp = v + sigma * np.diff(2 * u_tmp - u) - \
                sigma * self.st(v / sigma +
                                np.diff(2 * u_tmp - u),
                                self.reg / sigma)
            u = eta * u_tmp + (1 - eta)*u
            v = eta * v_tmp + (1 - eta)*v
        self.u = u

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w

    def _prox_huber(self, u, mu):
        return np.where(
            np.abs(u) <= self.delta * (mu + 1.0),
            u / (mu + 1.0),
            u - self.delta * mu * np.sign(u),
        )
