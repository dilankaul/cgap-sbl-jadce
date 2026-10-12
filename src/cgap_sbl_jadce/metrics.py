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


def nmse(Z_true: np.ndarray, Z_hat: np.ndarray, a_true: np.ndarray | None = None) -> float:
    """Channel NMSE on true active rows, excluding inactive-row estimates.

    The true activity mask is used only for scoring. If omitted (e.g. for
    convergence histories), support is inferred from nonzero true channel rows.
    Zero true channel energy returns NaN and is excluded from mean NMSE.
    """
    Z_true = np.asarray(Z_true)
    Z_hat = np.asarray(Z_hat)
    if Z_true.ndim != 2 or Z_true.shape != Z_hat.shape:
        raise ValueError("True and estimated channels must have matching (N, M) shapes.")
    active = np.any(Z_true != 0, axis=1) if a_true is None else np.asarray(a_true, dtype=bool)
    if active.shape != (Z_true.shape[0],):
        raise ValueError("True activity must have shape (N,).")
    truth = Z_true[active]
    estimate = Z_hat[active]
    denom = float(np.linalg.norm(truth) ** 2)
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(truth - estimate) ** 2 / denom)
