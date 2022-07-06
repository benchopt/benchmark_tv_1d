import numpy as np


def get_l2norm(A, n_iter=100):
    if isinstance(A, np.ndarray):
        return np.linalg.norm(A, ord=2)
    else:
        # multiplication for the smaller size of matrice
        if A.shape[0] < A.shape[1]:
            A = A.T
        AtA = A.T @ A
        x = np.random.randn(A.shape[1])
        for _ in range(n_iter):
            x = AtA @ x
            x /= np.linalg.norm(x)
        return np.sqrt(np.linalg.norm(AtA @ x))


def st(w, mu):
    w -= np.clip(w, -mu, mu)
    return w


def prox_z(w, mu):
    w0 = w[0]
    w = st(w, mu)
    w[0] = w0
    return w


def huber(R, delta):
    norm_1 = np.abs(R)
    loss = np.where(norm_1 < delta,
                    0.5 * norm_1**2,
                    delta * norm_1 - 0.5 * delta**2)
    return np.sum(loss)


def grad_huber(R, delta):
    return np.where(np.abs(R) < delta, R,
                    np.sign(R) * delta)
