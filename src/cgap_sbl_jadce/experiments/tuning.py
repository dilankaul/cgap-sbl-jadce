from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from cgap_sbl_jadce.metrics import precision_recall_f1, support_from_gamma


@dataclass(frozen=True)
class ThresholdChoice:
    tau: float
    mean_precision: float
    mean_recall: float
    mean_f1: float


def threshold_candidates() -> np.ndarray:
    return np.arange(201, dtype=float) / 1000


def threshold_curve(a_true: np.ndarray, gammas: np.ndarray, tau_values=None) -> list[dict[str, float]]:
    a_true = np.asarray(a_true, dtype=bool)
    gammas = np.asarray(gammas, dtype=float)
    if a_true.shape != gammas.shape:
        raise ValueError("a_true and gammas must have matching (samples, N) shapes.")
    rows: list[dict[str, float]] = []
    for tau in threshold_candidates() if tau_values is None else tau_values:
        metrics = [precision_recall_f1(a_true[i], support_from_gamma(gammas[i], float(tau))) for i in range(len(a_true))]
        p, r, f = np.mean(metrics, axis=0)
        rows.append({"tau": float(tau), "precision": float(p), "recall": float(r), "f1": float(f)})
    return rows


def choose_gamma_threshold(a_true: np.ndarray, gammas: np.ndarray) -> ThresholdChoice:
    rows = threshold_curve(a_true, gammas)
    best = max(rows, key=lambda x: (x["f1"], x["recall"], x["precision"]))
    return ThresholdChoice(best["tau"], best["precision"], best["recall"], best["f1"])
