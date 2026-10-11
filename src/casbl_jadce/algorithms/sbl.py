from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from casbl_jadce.algorithms.posterior import posterior_mmv
from casbl_jadce.algorithms.cgap_sbl import relative_gamma_change
from casbl_jadce.metrics import nmse


@dataclass(frozen=True)
class SBLResult:
    gamma: np.ndarray
    mu: np.ndarray
    iterations: int
    converged: bool
    gamma_history: np.ndarray | None = None
    nmse_history: np.ndarray | None = None


def sbl(
    Theta: np.ndarray,
    Y: np.ndarray,
    sigma2: float,
    gamma_init: float = 0.1,
    K: int = 500,
    delta: float = 1e-4,
    keep_history: bool = False,
    Z_true: np.ndarray | None = None,
) -> SBLResult:
    if gamma_init <= 0 or K < 1 or delta <= 0:
        raise ValueError("gamma_init, K, and delta must be positive")
    N = Theta.shape[1]
    M = Y.shape[1]
    gamma = np.full(N, float(gamma_init), dtype=float)
    gamma_hist: list[np.ndarray] = []
    nmse_hist: list[float] = []
    converged = False

    for k in range(1, K + 1):
        posterior = posterior_mmv(Theta, Y, sigma2, gamma)
        eta = posterior.Sigma_diag + np.sum(np.abs(posterior.mu) ** 2, axis=1) / float(M)
        gamma_new = np.maximum(eta.real, 0.0)
        if keep_history:
            gamma_hist.append(gamma_new.copy())
            if Z_true is not None:
                nmse_hist.append(nmse(Z_true, posterior.mu))
        change = relative_gamma_change(gamma_new, gamma)
        gamma = gamma_new
        if change < delta:
            converged = True
            break

    posterior = posterior_mmv(Theta, Y, sigma2, gamma)
    return SBLResult(
        gamma=gamma,
        mu=posterior.mu,
        iterations=k,
        converged=converged,
        gamma_history=np.asarray(gamma_hist) if keep_history else None,
        nmse_history=np.asarray(nmse_hist) if (keep_history and Z_true is not None) else None,
    )
