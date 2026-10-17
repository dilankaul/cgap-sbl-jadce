import numpy as np
from scipy.sparse import csr_matrix

from cgap_sbl_jadce.algorithms.cgap_sbl import _run_cgap_sbl, CGAPSBLResult


def normalized_connectivity(C, N):
    if np.iscomplexobj(C):
        raise ValueError("GL C must contain real non-negative weights")
    C = csr_matrix(C, dtype=float, copy=True)
    C.sum_duplicates()
    if C.shape != (N, N):
        raise ValueError("C must have shape (N, N)")
    difference = C - C.T
    if (not np.all(np.isfinite(C.data)) or np.any(C.data < 0)
            or difference.nnz and np.max(np.abs(difference.data)) > 1e-10):
        raise ValueError("GL C must be finite, symmetric and non-negative")
    xi = np.asarray(C.sum(axis=1)).ravel()
    if not np.all(np.isfinite(xi)) or np.any(xi <= 0):
        raise ValueError("GL connectivity must be finite and strictly positive")
    return C, 1.0 / np.sqrt(xi)


def gl_penalty(C, inverse_sqrt_xi, gamma, alpha):
    # Retain C's diagonal and freeze gamma at the previous iteration.
    return float(alpha) * np.maximum(gamma - inverse_sqrt_xi * (C @ (inverse_sqrt_xi * gamma)), 0.0)


def cgap_sbl_gl(Theta, Y, sigma2, C, alpha, **kwargs) -> CGAPSBLResult:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("GL alpha must be finite and non-negative")
    C, inverse_sqrt_xi = normalized_connectivity(C, Theta.shape[1])
    return _run_cgap_sbl(Theta, Y, sigma2,
        lambda gamma: gl_penalty(C, inverse_sqrt_xi, gamma, alpha), **kwargs)
