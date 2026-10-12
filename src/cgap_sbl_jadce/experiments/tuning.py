from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ThresholdChoice:
    tau: float
    mean_precision: float
    mean_recall: float
    mean_f1: float


def threshold_candidates() -> np.ndarray:
    """Shared fixed grid: 0 through 0.2 inclusive, in steps of 0.001."""
    return np.arange(201, dtype=float) / 1000


def threshold_curve(a_true: np.ndarray, gammas: np.ndarray) -> list[dict[str, float]]:
    a_true = np.asarray(a_true, dtype=bool)
    gammas = np.asarray(gammas, dtype=float)
    if a_true.shape != gammas.shape:
        raise ValueError("a_true and gammas must have matching (samples, N) shapes.")
    candidates = threshold_candidates()
    predicted = gammas[None, :, :] >= candidates[:, None, None]
    tp = np.sum(predicted & a_true[None, :, :], axis=2)
    detected = np.sum(predicted, axis=2)
    actual = np.sum(a_true, axis=1)[None, :]
    precision = np.divide(tp, detected, out=np.zeros_like(tp, dtype=float), where=detected > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp, dtype=float), where=actual > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(precision), where=denominator > 0)
    return [{"tau": float(tau), "precision": float(p), "recall": float(r), "f1": float(f)}
            for tau, p, r, f in zip(candidates, precision.mean(axis=1), recall.mean(axis=1), f1.mean(axis=1))]


def choose_gamma_threshold(a_true: np.ndarray, gammas: np.ndarray) -> ThresholdChoice:
    rows = threshold_curve(a_true, gammas)
    best = max(rows, key=lambda x: (x["f1"], x["recall"], x["precision"]))
    return ThresholdChoice(best["tau"], best["precision"], best["recall"], best["f1"])
