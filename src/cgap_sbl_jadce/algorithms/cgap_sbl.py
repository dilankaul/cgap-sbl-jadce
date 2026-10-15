from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from cgap_sbl_jadce.algorithms.posterior import posterior_mmv
from cgap_sbl_jadce.metrics import nmse
from cgap_sbl_jadce.models.spatial_dependence import anc_penalty


@dataclass(frozen=True)
class CGAPSBLResult:
    gamma: np.ndarray
    mu: np.ndarray
    iterations: int
    converged: bool
    gamma_history: np.ndarray | None = None
    lambda_history: np.ndarray | None = None
    nmse_history: np.ndarray | None = None


def stable_gamma_update(eta: np.ndarray, lambda_: np.ndarray, M: int) -> np.ndarray:
    """gamma = 2 eta / (1 + sqrt(1 + 4 lambda eta / M))."""
    eta = np.asarray(eta, dtype=float)
    lambda_ = np.asarray(lambda_, dtype=float)
    if eta.shape != lambda_.shape:
        raise ValueError("eta and lambda must have the same shape")
    if M <= 0 or np.any(eta < 0) or np.any(lambda_ < 0):
        raise ValueError("M must be positive; eta and lambda must be non-negative")
    if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(lambda_)):
        raise ValueError("eta and lambda must be finite")
    return 2.0 * eta / (1.0 + np.sqrt(1.0 + 4.0 * (lambda_ / M) * eta))


def relative_gamma_change(gamma_new: np.ndarray, gamma: np.ndarray) -> float:
    change = float(np.linalg.norm(gamma_new - gamma))
    scale = float(np.linalg.norm(gamma))
    return change / scale if scale > 0 else (0.0 if change == 0 else float("inf"))


def _run_cgap_sbl(
    Theta: np.ndarray, Y: np.ndarray, sigma2: float,
    penalty: Callable[[np.ndarray], np.ndarray],
    gamma_init: float = 0.1, K: int = 500, delta: float = 1e-4,
    keep_history: bool = False, Z_true: np.ndarray | None = None,
) -> CGAPSBLResult:
    if gamma_init <= 0 or K < 1 or delta <= 0:
        raise ValueError("gamma_init, K, and delta must be positive")
    N, M = Theta.shape[1], Y.shape[1]
    gamma = np.full(N, float(gamma_init))
    gamma_history, lambda_history, nmse_history = [], [], []
    converged = False
    for k in range(1, K + 1):
        posterior = posterior_mmv(Theta, Y, sigma2, gamma)
        eta = posterior.Sigma_diag + np.sum(np.abs(posterior.mu) ** 2, axis=1) / M
        lambda_ = penalty(gamma)
        gamma_new = stable_gamma_update(eta, lambda_, M)
        if keep_history:
            gamma_history.append(gamma_new.copy())
            lambda_history.append(lambda_.copy())
            if Z_true is not None:
                nmse_history.append(nmse(Z_true, posterior.mu))
        change = relative_gamma_change(gamma_new, gamma)
        gamma = gamma_new
        if change < delta:
            converged = True
            break
    posterior = posterior_mmv(Theta, Y, sigma2, gamma)
    return CGAPSBLResult(
        gamma, posterior.mu, k, converged,
        np.asarray(gamma_history) if keep_history else None,
        np.asarray(lambda_history) if keep_history else None,
        np.asarray(nmse_history) if keep_history and Z_true is not None else None,
    )


def cgap_sbl(
    Theta: np.ndarray, Y: np.ndarray, sigma2: float, Omega: np.ndarray,
    gamma_init: float = 0.1, K: int = 500, delta: float = 1e-4,
    keep_history: bool = False, Z_true: np.ndarray | None = None,
) -> CGAPSBLResult:
    """lambda = positive_part(Omega @ gamma)."""
    Omega = np.asarray(Omega, dtype=float)
    N = Theta.shape[1]
    if Omega.shape != (N, N) or not np.all(np.isfinite(Omega)) or not np.allclose(Omega, Omega.T):
        raise ValueError("Omega must be a finite symmetric (N, N) matrix")
    return _run_cgap_sbl(
        Theta, Y, sigma2, lambda gamma: np.maximum(Omega @ gamma, 0.0),
        gamma_init, K, delta, keep_history, Z_true,
    )


def cgap_sbl_anc(
    Theta: np.ndarray, Y: np.ndarray, sigma2: float, C: np.ndarray,
    alpha: float, beta: float, **kwargs,
) -> CGAPSBLResult:
    C = csr_matrix(C, dtype=float)
    if C.shape != (Theta.shape[1], Theta.shape[1]):
        raise ValueError("C must have shape (N, N)")
    difference = C - C.T
    if not np.all(np.isfinite(C.data)) or (difference.nnz and np.max(np.abs(difference.data)) > 1e-10):
        raise ValueError("C must be finite and symmetric")
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha < 0 or beta < 0:
        raise ValueError("ANC alpha and beta must be non-negative")
    return _run_cgap_sbl(
        Theta, Y, sigma2, lambda gamma: anc_penalty(C, gamma, alpha, beta), **kwargs,
    )
