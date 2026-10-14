from __future__ import annotations
import numpy as np
from scipy.linalg import lstsq


def mmv_omp(Theta: np.ndarray, Y: np.ndarray, B_q: int) -> np.ndarray:
    L, N = Theta.shape
    if not (0 <= B_q <= min(L, N)):
        raise ValueError("B_q must satisfy 0 <= B_q <= min(L, N).")
    if B_q == 0:
        return np.zeros((N, Y.shape[1]), dtype=np.complex128)
    residual = Y.copy()
    support: list[int] = []
    Xs = np.empty((0, Y.shape[1]), dtype=np.complex128)
    for _ in range(B_q):
        proxy = Theta.conj().T @ residual
        energy = np.sum(np.abs(proxy) ** 2, axis=1)
        if support:
            energy[np.asarray(support)] = -np.inf
        support.append(int(np.argmax(energy)))
        Xs, *_ = lstsq(Theta[:, support], Y, lapack_driver="gelsd")
        residual = Y - Theta[:, support] @ Xs
    Z_hat = np.zeros((N, Y.shape[1]), dtype=np.complex128)
    Z_hat[np.asarray(support), :] = Xs
    return Z_hat
