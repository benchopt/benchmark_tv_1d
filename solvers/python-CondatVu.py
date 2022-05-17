from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import norm as spnorm


class Solver(BaseSolver):
    """Primal-Dual Splitting Method for synthesis and analysis formulation."""
    name = 'CondatVu analysis and synthesis'

    stopping_criterion = SufficientProgressCriterion(
        patience=40, strategy='callback'
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'eta': [0.5, 1]}

    def set_objective(self, A, reg_scaled, y, c, delta, data_fit):
        self.reg_scaled = reg_scaled
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        len_y = len(self.y)
        data = np.array([-np.ones(len_y), np.ones(len_y)])
        diags = np.array([0, 1])
        D = spdiags(data, diags, len_y-1, len_y)
        K = np.r_[D.toarray(), self.A]

        # initialisation
        u = self.c * np.ones(len_y)
        v = np.zeros(len_y - 1)
        w = np.r_[v, self.A @ u]
        w_tmp = w

        sigma = 0.5
        eta = self.eta

        if self.data_fit == 'quad':
            tau = 1 / (np.linalg.norm(self.A.T @ self.A, ord=2) /
                       2 + sigma * spnorm(D)**2)
        else:
            tau = 1 / (sigma * np.linalg.norm(K, ord=2)**2)

        while callback(u):
            if self.data_fit == 'quad':
                u_tmp = u - tau * self.A.T @ (self.A @ u - self.y) - \
                    tau * (-np.diff(v, append=0, prepend=0))
                v_tmp = v + sigma * np.diff(2 * u_tmp - u) - \
                    sigma * self.st(v / sigma +
                                    np.diff(2 * u_tmp - u),
                                    self.reg_scaled / sigma)
                u = eta * u_tmp + (1 - eta)*u
                v = eta * v_tmp + (1 - eta)*v
            else:
                u_tmp = u - tau * K.T @ w

                x_tmp = w + sigma * K @ (2 * u_tmp - u)
                w_tmp[:len_y - 1] = x_tmp[:len_y - 1] - \
                    sigma * self.st(x_tmp[:len_y - 1] /
                                    sigma, self.reg_scaled / sigma)
                R_tmp = sigma * self.y - x_tmp[len_y - 1:]
                w_tmp[len_y - 1:] = x_tmp[len_y - 1:] - \
                    np.where(abs(R_tmp) < self.delta * (sigma + 1),
                             sigma *
                             (self.y + x_tmp[len_y - 1:]) / (sigma + 1),
                             x_tmp[len_y - 1:] + self.delta * np.sign(R_tmp))
                u = eta * u_tmp + (1 - eta)*u
                w = eta * w_tmp + (1 - eta)*w
        self.u = u

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w
