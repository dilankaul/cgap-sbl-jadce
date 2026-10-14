from __future__ import annotations
import numpy as np
from scipy.linalg import lstsq


def mmv_cosamp(Theta: np.ndarray, Y: np.ndarray, B_q: int, max_iter: int = 500, tol: float = 1e-4) -> np.ndarray:
    """MMV CoSaMP with unregularized SVD least-squares support estimation."""
    L, N = Theta.shape
    if not (0 <= B_q <= min(L, N)):
        raise ValueError("B_q must satisfy 0 <= B_q <= min(L, N).")
    Z_hat = np.zeros((N, Y.shape[1]), dtype=np.complex128)
    if B_q == 0:
        return Z_hat
    residual = Y.copy()
    support = np.array([], dtype=int)
    y_norm = max(np.linalg.norm(Y), np.finfo(float).eps)

    for _ in range(max_iter):
        proxy = Theta.conj().T @ residual
        energy = np.sum(np.abs(proxy) ** 2, axis=1)
        take = min(2 * B_q, N)
        omega = np.argpartition(energy, -take)[-take:]
        merged = np.union1d(support, omega)

        B, *_ = lstsq(Theta[:, merged], Y, lapack_driver="gelsd")
        row_energy = np.sum(np.abs(B) ** 2, axis=1)
        keep = min(B_q, len(merged))
        local = np.argpartition(row_energy, -keep)[-keep:]
        support = np.sort(merged[local])

        Xs, *_ = lstsq(Theta[:, support], Y, lapack_driver="gelsd")
        Z_new = np.zeros_like(Z_hat)
        Z_new[support, :] = Xs
        residual = Y - Theta @ Z_new
        Z_hat = Z_new
        if np.linalg.norm(residual) / y_norm <= tol:
            break
    return Z_hat
