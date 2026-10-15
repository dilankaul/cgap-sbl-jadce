from __future__ import annotations
import numpy as np


def pairwise_distances(loc: np.ndarray) -> np.ndarray:
    loc = np.asarray(loc, dtype=float)
    if loc.ndim != 2 or loc.shape[1] != 2:
        raise ValueError("loc must have shape (N, 2).")
    delta = loc[:, None, :] - loc[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def build_C(loc: np.ndarray, rho: float = 3.78, U: float = 15.0) -> np.ndarray:
    """C_ij = max((exp(-u_ij/rho)-exp(-U/rho))/(1-exp(-U/rho)), 0)."""
    loc = np.asarray(loc, dtype=float)
    N = loc.shape[0]
    if rho <= 0:
        return np.eye(N, dtype=float)
    d = pairwise_distances(loc)
    denom = 1.0 - np.exp(-U / rho)
    C = (np.exp(-d / rho) - np.exp(-U / rho)) / denom
    C = np.where(d <= U, C, 0.0)
    return np.maximum(C, 0.0)


def build_Omega_ANC(C: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square.")
    if alpha < 0 or beta < 0:
        raise ValueError("ANC alpha and beta must be non-negative")
    return float(alpha) * (float(beta) * np.ones_like(C) - C)


def anc_penalty(C: np.ndarray, gamma: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """lambda = alpha * positive_part(beta * sum(gamma) - C @ gamma)."""
    gamma = np.asarray(gamma, dtype=float)
    return float(alpha) * np.maximum(float(beta) * np.sum(gamma) - C @ gamma, 0.0)
