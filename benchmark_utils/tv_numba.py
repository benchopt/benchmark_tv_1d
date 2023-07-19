r"""
Proximity Operator for Total Variation in 1D.

This solve the problem:
.. math::
    \min_x \frac{1}{2} \|x - y\|_2^2 + \lambda \|x\|_1

Three implemtation are availables:
- linearizedTautString: Algorihtm by Condat in O(n)
- TV MM
- Group TV MM
"""
import numpy as np

import numba as nb

nbr1d64 = nb.types.Array(nb.types.float64, 1, "A", readonly=True)
nbr1d32 = nb.types.Array(nb.types.float32, 1, "A", readonly=True)

nbr2d64 = nb.types.Array(nb.types.float64, 2, "A", readonly=True)
nbr2d32 = nb.types.Array(nb.types.float32, 2, "A", readonly=True)

nb2d64 = nb.types.Array(nb.types.float64, 2, "A")
nb2d32 = nb.types.Array(nb.types.float32, 2, "A")
nb1d64 = nb.types.Array(nb.types.float64, 1, "A")
nb1d32 = nb.types.Array(nb.types.float32, 1, "A")

nbr1d64c = nb.types.Array(nb.types.complex128, 1, "A", readonly=True)
nbr1d32c = nb.types.Array(nb.types.complex64, 1, "A", readonly=True)

nbr2d64c = nb.types.Array(nb.types.complex128, 2, "A", readonly=True)
nbr2d32c = nb.types.Array(nb.types.complex64, 2, "A", readonly=True)

nb2d64c = nb.types.Array(nb.types.complex128, 2, "A")
nb2d32c = nb.types.Array(nb.types.complex64, 2, "A")
nb1d64c = nb.types.Array(nb.types.complex128, 1, "A")
nb1d32c = nb.types.Array(nb.types.complex64, 1, "A")


def linearized_taut_string(y, lmbd, x):
    """Linearized Taut String algorithm.

    Follows the algorithm described in [1]_, analoguous to the algorithm [2]_.

    Parameters
    ----------
    y: np.ndarray
        Input data.
    lambda: float
        Regularization parameter.
    x: np.ndarray
        Return value

    Returns
    -------
    np.ndarray
        Output data.

    References
    ----------
    .. [1] https://github.com/albarji/proxTV/blob/master/src/TVL1opt.cpp
    .. [2] Condat, L. (2013) A direct algorithm for 1D total variation
    denoising.
    """

    i = 0
    # Relative height of majorant and minorant sploes at the current points
    # w.r.t the tube center.
    mn_height = mx_height = 0
    # minorant and majorant of slope
    mn, mx = y[0] - lmbd, y[0] + lmbd
    # last breakpoint of taut string
    last_break = -1
    # Last breakpoint of majorant and minorant
    mn_break = mx_break = 0
    N = len(y)
    while i < N:
        while i < N - 1:
            mn_height += mn - y[i]
            if lmbd < mn_height:
                i = mn_break + 1
                x[last_break + 1: mn_break + 1] = mn
                last_break = mn_break
                mn = y[i]
                mx = 2 * lmbd + mn
                mx_height = lmbd
                mn_height = -lmbd
                mn_break = mx_break = i
                i += 1
                continue
            mx_height += mx - y[i]
            if -lmbd > mx_height:
                i = mx_break + 1
                x[last_break + 1: mx_break + 1] = mx
                last_break = mx_break
                mx = y[i]
                mn = mx - 2 * lmbd
                mn_height = lmbd
                mx_height = -lmbd
                mn_break = mx_break = i
                i += 1
                continue
            if mx_height > lmbd:
                mx += (lmbd - mx_height) / (i - last_break)
                mx_height = lmbd
                mx_break = i
            if mn_height <= -lmbd:
                mn += (-lmbd - mn_height) / (i - last_break)
                mn_height = -lmbd
                mn_break = i
            i += 1
        mn_height += mn - y[i]
        if mn_height > 0:
            i = mn_break + 1
            x[last_break + 1: mn_break + 1] = mn
            last_break = mn_break
            mn = y[i]
            mx = 2 * lmbd + mn
            mx_height = mn_height = -lmbd
            mn_break = mx_break = i
            continue
        mx_height += mx - y[i]
        if mx_height < 0:
            i = mx_break + 1
            x[last_break + 1: mx_break + 1] = mx
            last_break = mx_break
            mx = y[i]
            mn = mx - 2 * lmbd
            mn_height = mx_height = lmbd
            mn_break = mx_break = i
            continue
        if mn_height <= 0:
            mn += (-mn_height) / (i - last_break)
        i += 1
    x[last_break + 1:] = mn
    return x


linearized_taut_string.jitter = nb.njit(
    [
        nb1d32(nbr1d32, nb.types.float32, nb1d32),
        nb1d64(nbr1d64, nb.types.float64, nb1d64),
    ]
)


def prox_condat(y, lmbd):
    x = np.zeros_like(y)
    linearized_taut_string(y, lmbd, x)
    return x


#######################################
#   ProxTV 1D using an MM algorithm   #
#######################################


def fast_cost(y, x, r, lmbd):
    return 0.5 * np.sqrt(np.sum(np.abs(y - x) ** 2)) + lmbd * np.sum(r)


fast_cost.jitter = nb.njit(
    [
        nb.types.float32(nbr1d32, nb1d32, nb1d32, nb.types.float32),
        nb.types.float64(nbr1d64, nb1d64, nb1d64, nb.types.float64),
    ],
    fastmath=True,
)


def difft(x):
    r"""Apply the matrix D^T to x.

    Parameters
    ----------
    x: np.ndarray
        Input data.

    Returns
    -------
    np.ndarray
        Output data.

    Notes
    -----
    This function is equivalent to:
    ..math ::
        y = \begin{bmatrix}
        -1 & 1  & 0  & \dots & 0 \\
        0 & \ddots & \ddots &  \ddots &  \vdots \\
        \vdots & \ddots & \ddots &  \ddots &  0 \\
        0 & \dots & 0 & -1 & 1 \\
        \end{bmatrix}^T x
    """
    y = np.zeros(len(x) + 1, dtype=x.dtype)
    y[0] = -x[0]
    y[1:-1] = x[:-1] - x[1:]
    y[-1] = x[-1]
    return y


difft.jitter = nb.njit(
    [
        nb1d32(nb1d32),
        nb1d64(nb1d64),
    ],
    fastmath=True,
)


def TDMA(a, b, c, d, x):
    r"""
    Solve a tridiagonal system of equations.

    Parameters
    ----------
    a: np.ndarray
        Lower diagonal.
    b: np.ndarray
        Diagonal.
    c: np.ndarray
        Upper diagonal.
    d: np.ndarray
        Right hand side.
    x: np.ndarray
        Solution.

    Notes
    -----
    This function is equivalent to:
    ..math ::
        \begin{bmatrix}
        b_0 & c_0 & 0 & \dots & 0 \\
        a_1 & b_1 & c_1 & \dots & 0 \\
        0 & a_2 & b_2 & \dots & 0 \\
        \vdots & \ddots & \ddots & \ddots & \vdots \\
        0 & \dots & 0 & a_{n-1} & b_{n-1} \\
        \end{bmatrix}
        \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\   x_{n-1}\end{bmatrix} =
        \begin{bmatrix} d_0 \\ d_1 \\ \vdots \\ d_{n-1} \\\end{bmatrix}
    See [1]_ for more details.
    This function is based on the implementation in [2]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    .. [2] https://stackoverflow.com/questions/8733015
    """

    n = len(d)
    w = np.zeros(n - 1, a.dtype)
    g = np.zeros(n, a.dtype)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    x[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        x[i - 1] = g[i - 1] - w[i - 1] * x[i]


TDMA.jitter = nb.njit(
    [
        "void(f8[:],f8[:],f8[:],f8[:],f8[:])",
        "void(f4[:],f4[:],f4[:],f4[:],f4[:])",
    ],
    fastmath=True,
)


def tv_mm(y, lmbd, max_iter=100, tol=1e-3):
    """ProxTV 1D using an MM algorithm.

    I. Selesnick, “Total Variation Denoising (an MM algorithm)”.
    https://eeweb.engineering.nyu.edu/iselesni/lecture_notes/TVDmm/
    """
    N = len(y)
    ddt_up = -np.ones(N - 1, dtype=y.dtype)
    ddt_diag = 2 * np.ones(N - 1, dtype=y.dtype)
    ddt_down = -np.ones(N - 1, dtype=y.dtype)
    ddt_up[-1] = 0
    ddt_down[0] = 0

    x = np.copy(y)
    tmp = np.zeros(N - 1, dtype=y.dtype)
    diffy = y[1:] - y[:-1]
    cost_prev = 1e40
    for i in range(max_iter):
        tmp = np.abs(x[1:] - x[:-1])
        if np.all(tmp == 0):
            break
        # cost =  0.5 * np.sqrt(np.sum(np.abs(y - x)**2)) + lmbd * np.sum(tmp)
        cost = fast_cost(y, x, tmp, lmbd)
        f_diag = (tmp / lmbd) + ddt_diag
        TDMA(ddt_down, f_diag, ddt_up, diffy, tmp)
        x = y - difft(tmp)
        if (cost_prev - cost) / cost_prev <= tol:
            break
        else:
            cost_prev = cost
    return x


tv_mm.jitter = nb.njit(
    [
        nb1d32(nbr1d32, nb.types.float32, nb.types.int16, nb.types.float32),
        nb1d64(nbr1d64, nb.types.float64, nb.types.int16, nb.types.float64),
    ]
)


def running_sum_valid(arr, K):
    """Compute the running sum of an array.

    This is equivalent to `np.convolve(a, np.ones(K), mode='same')`,
    and is slighly faster for our usecase.
    This reimplementation is necessary to enable numba to properly compile
    the gtv function.
    """
    N = len(arr)
    ret = np.empty(N, arr.dtype)
    i = 0
    r = K // 2
    running_buf = np.zeros(K + 1, arr.dtype)
    running_buf[: K // 2] = arr[: K // 2]
    s = np.sum(running_buf)
    for i in range(N):
        if i + K // 2 >= N:
            running_buf[r] = 0
        else:
            running_buf[r] = arr[i + K // 2]
        s += running_buf[r]
        r = (r + 1) % (K + 1)
        s -= running_buf[r]
        ret[i] = s
    return ret


running_sum_valid.jitter = nb.njit(
    [nb1d32(nb1d32, nb.types.int16), nb1d64(nb1d64, nb.types.int16)],
    fastmath=True,
    error_model="numpy",
)


def gtv_mm_tol2(y, lmbd, K=1, max_iter=100, tol=1e-3):
    """Group Total Variation denoising with Majoration-Minimization algorithm.

    Parameters
    ----------
    y : np.ndarray
        Input data.
    lmbd : float
        Regularization parameter.
    K : int
        Size of the group.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for the stopping criterion.

    Returns
    -------
    np.ndarray
        Output data.

    References
    ----------
    .. [1]  I. W. Selesnick and P.-Y. Chen, “Total variation denoising with
      overlapping group sparsity” in 2013 IEEE International Conference on
      Acoustics, Speech and Signal Processing, Vancouver, BC, Canada,
      May 2013, pp. 5696–5700. doi: 10.1109/ICASSP.2013.6638755.
    """
    N = len(y)
    ddt_up = -np.ones(N - 1, dtype=y.dtype)
    ddt_diag = 2 * np.ones(N - 1, dtype=y.dtype)
    ddt_down = -np.ones(N - 1, dtype=y.dtype)
    f_diag = np.empty_like(y)
    x = y.copy()
    tmp = np.zeros(N - 1, dtype=y.dtype)
    diffy = y[1:] - y[:-1]
    cost_prev = 1e40
    for i in range(max_iter):
        tmp = (x[1:] - x[:-1]) ** 2
        if np.min(tmp) < 1e-10:
            break
        tmp = running_sum_valid(tmp, K)
        tmp = np.sqrt(tmp)
        # cost =  0.5 * np.sqrt(np.sum(np.abs(y - x)**2)) + lmbd * np.sum(tmp)
        cost = fast_cost(y, x, tmp, lmbd)
        tmp = 1 / tmp
        tmp = running_sum_valid(tmp, K)
        f_diag = 1 / (tmp * lmbd) + ddt_diag
        TDMA(ddt_down, f_diag, ddt_up, diffy, tmp)
        x = y - difft(tmp)
        if (cost_prev - cost) <= tol * cost_prev:
            break
        else:
            cost_prev = cost
    return x


gtv_mm_tol2.jitter = nb.njit(
    [
        nb.types.Array(nb.types.float64, 1, "A")(
            nbr1d64,
            nb.types.float64,
            nb.types.int16,
            nb.types.int16,
            nb.types.float64,
        ),
        nb.types.Array(nb.types.float32, 1, "A")(
            nbr1d32,
            nb.types.float32,
            nb.types.int16,
            nb.types.int16,
            nb.types.float32,
        ),
    ],
    error_model="numpy",
)

JITTED = False


def jit_module():
    """Jit all functions in this module."""
    global JITTED
    if JITTED:
        return
    JITTED = True
    for name, func in globals().items():
        if hasattr(func, "jitter"):
            globals()[name] = func.jitter(func)
