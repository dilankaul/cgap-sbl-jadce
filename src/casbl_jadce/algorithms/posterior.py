from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


@dataclass(frozen=True)
class Posterior:
    mu: np.ndarray
    Sigma_diag: np.ndarray


def posterior_mmv(Theta: np.ndarray, Y: np.ndarray, sigma2: float, gamma: np.ndarray) -> Posterior:
    """Posterior for complex MMV SBL using the L x L covariance Pi.

    Pi = Theta diag(gamma) Theta^H + sigma2 I.
    Returns mu and diag(Sigma) without explicitly inverting Pi or forming Sigma.
    """
    Theta = np.asarray(Theta, dtype=np.complex128)
    Y = np.asarray(Y, dtype=np.complex128)
    gamma = np.asarray(gamma, dtype=float)
    L, N = Theta.shape
    if Y.ndim != 2 or Y.shape[0] != L:
        raise ValueError("Y must have shape (L, M).")
    if gamma.shape != (N,):
        raise ValueError("gamma must have shape (N,).")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")

    Theta_gamma = Theta * gamma[None, :]
    Pi = Theta_gamma @ Theta.conj().T
    Pi.flat[:: L + 1] += float(sigma2)

    factor = cho_factor(Pi, lower=True, overwrite_a=False, check_finite=False)
    Pi_inv_Y = cho_solve(factor, Y, check_finite=False)

    mu = gamma[:, None] * (Theta.conj().T @ Pi_inv_Y)

    # If Pi=Lc Lc^H, then theta_i^H Pi^{-1} theta_i = ||Lc^{-1} theta_i||_2^2.
    # Only one triangular solve is needed for all N pilot columns, instead of
    # computing the full Pi^{-1}Theta with two triangular solves.
    Lc = factor[0]
    whitened_Theta = solve_triangular(Lc, Theta, lower=True, check_finite=False)
    diag_gram = np.sum(np.abs(whitened_Theta) ** 2, axis=0)
    Sigma_diag = gamma - gamma * gamma * diag_gram
    # Negative values here should only be roundoff-level numerical errors.
    Sigma_diag = np.maximum(Sigma_diag, 0.0)
    return Posterior(mu=mu, Sigma_diag=Sigma_diag)


def posterior_mmv_full_reference(Theta: np.ndarray, Y: np.ndarray, sigma2: float, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slow full-matrix reference implementation used only by tests."""
    Theta = np.asarray(Theta, dtype=np.complex128)
    Y = np.asarray(Y, dtype=np.complex128)
    gamma = np.asarray(gamma, dtype=float)
    L = Theta.shape[0]
    Gamma = np.diag(gamma)
    Pi = Theta @ Gamma @ Theta.conj().T + sigma2 * np.eye(L)
    Pi_inv = np.linalg.inv(Pi)
    Sigma = Gamma - Gamma @ Theta.conj().T @ Pi_inv @ Theta @ Gamma
    mu = Gamma @ Theta.conj().T @ Pi_inv @ Y
    return mu, Sigma
