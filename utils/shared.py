import numpy as np


def get_l2norm(A, n_iter=100):
    if isinstance(A, np.ndarray):
        return np.linalg.norm(A, ord=2)
    else:
        x = np.random.randn(A.shape[1])
        for _ in range(n_iter):
            x = A.T @ (A @ x)
            x /= np.linalg.norm(x)
        return np.sqrt(np.linalg.norm(A.T @ (A @ x)))
