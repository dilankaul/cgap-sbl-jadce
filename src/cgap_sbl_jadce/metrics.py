from __future__ import annotations
import numpy as np


def support_from_gamma(gamma: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(gamma) >= float(tau)


def support_from_rows(Z_hat: np.ndarray, B_q: int) -> np.ndarray:
    energy = np.sum(np.abs(Z_hat) ** 2, axis=1)
    support = np.zeros(len(energy), dtype=bool)
    if B_q == 0:
        return support
    if not (0 < B_q <= len(energy)):
        raise ValueError("B_q must satisfy 0 <= B_q <= number of rows.")
    support[np.argpartition(energy, -B_q)[-B_q:]] = True
    return support


def oracle_sparsity_budget(S_q: int, L: int, N: int) -> int:
    """Feasible oracle support budget for OMP/CoSaMP.

    The greedy baselines receive the realized activity count whenever it is
    identifiable within the pilot dimension. If the natural event model
    produces more active MTDs than measurements, cap the solver budget at L
    (and at N defensively): B_q = min(S_q, L, N).
    """
    S_q = int(S_q); L = int(L); N = int(N)
    if S_q < 0:
        raise ValueError("S_q must be non-negative.")
    if L <= 0 or N <= 0:
        raise ValueError("L and N must be positive.")
    return min(S_q, L, N)


def detection_counts(a_true: np.ndarray, a_hat: np.ndarray) -> tuple[int, int, int, int]:
    a_true = np.asarray(a_true, dtype=bool); a_hat = np.asarray(a_hat, dtype=bool)
    tp = int(np.sum(a_true & a_hat)); fp = int(np.sum(~a_true & a_hat))
    fn = int(np.sum(a_true & ~a_hat)); tn = int(np.sum(~a_true & ~a_hat))
    return tp, fp, fn, tn


def precision_recall_f1(a_true: np.ndarray, a_hat: np.ndarray) -> tuple[float, float, float]:
    tp, fp, fn, _ = detection_counts(a_true, a_hat)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def nmse(Z_true: np.ndarray, Z_hat: np.ndarray) -> float:
    """Normalized channel-estimation MSE.

    NMSE is undefined when a realization contains no active channel energy.
    Return NaN in that case so aggregation can exclude it instead of being
    poisoned by an artificial infinity. Detection metrics are still evaluated.
    """
    denom = float(np.linalg.norm(Z_true) ** 2)
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(Z_true - Z_hat) ** 2 / denom)
