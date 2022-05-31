from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientDescentCriterion
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import LinearOperator


class Solver(BaseSolver):
    """Primal-Dual Splitting Method for analysis formulation."""
    name = 'CondatVu analysis'

    stopping_criterion = SufficientDescentCriterion(
        patience=3, strategy="callback"
    )

    # any parameter defined here is accessible as a class attribute
    parameters = {'eta': [1.]}

    def set_objective(self, A, reg, y, c, delta, data_fit):
        self.reg = reg
        self.A, self.y = A, y
        self.c = c
        self.delta = delta
        self.data_fit = data_fit

    def run(self, callback):
        n, p = self.A.shape
        data = np.array([-np.ones(p), np.ones(p)])
        diags = np.array([0, 1])
        D = spdiags(data, diags, p-1, p).toarray()
        K = LinearOperator(
            dtype=np.float64,
            matvec=lambda x: np.r_[np.diff(x), self.A @ x],
            matmat=lambda X: np.r_[np.diff(X, axis=0), self.A @ X],
            rmatvec=lambda x: - np.diff(x[:p-1], append=0, prepend=0)
            + self.A.T @ x[p-1:],
            shape=(n + p - 1, p),
        )
        norm_AtA = np.linalg.norm(self.A.T @ self.A @ np.identity(p), ord=2)
        norm_K = np.linalg.norm(K @ np.eye(p, n + p - 1), ord=2)

        # initialisation
        u = self.c * np.ones(p)
        v = np.zeros(p - 1)
        w = np.r_[v, self.A @ u]
        w_tmp = w

        sigma = 0.5
        eta = self.eta

        if self.data_fit == 'quad':
            tau = 1 / (norm_AtA / 2 + sigma * np.linalg.norm(D, ord=2) ** 2)
        else:
            tau = 1 / (sigma * norm_K ** 2)

        while callback(u):
            if self.data_fit == 'quad':
                u_tmp = (u - tau * self.A.T @ (self.A @ u - self.y)
                         - tau * (-np.diff(v, append=0, prepend=0))
                         )
                v_tmp = (v + sigma * np.diff(2 * u_tmp - u)
                         - sigma * self.st(v / sigma +
                                           np.diff(2 * u_tmp - u),
                                           self.reg / sigma)
                         )
                u = eta * u_tmp + (1 - eta)*u
                v = eta * v_tmp + (1 - eta)*v
            else:
                u_tmp = u - tau * K.T @ w

                x_tmp = w + sigma * K @ (2 * u_tmp - u)
                x_tmp_1 = x_tmp[:p - 1]
                w_tmp[:p - 1] = x_tmp_1 - sigma * self.st(x_tmp_1 / sigma,
                                                          self.reg / sigma)
                x_tmp_2 = x_tmp[p - 1:]
                R_tmp = sigma * self.y - x_tmp[p - 1:]
                w_tmp[p - 1:] = (x_tmp_2
                                 - np.where(abs(R_tmp) < (self.delta
                                                          * (sigma + 1)),
                                            sigma * ((self.y + x_tmp_2)
                                                     / (sigma + 1)),
                                            x_tmp_2 + (self.delta
                                                       * np.sign(R_tmp)))
                                 )
                u = eta * u_tmp + (1 - eta)*u
                w = eta * w_tmp + (1 - eta)*w
        self.u = u

    def get_result(self):
        return self.u

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w
