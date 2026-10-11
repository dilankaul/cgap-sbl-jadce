from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CommunicationSample:
    a: np.ndarray
    H: np.ndarray
    Z: np.ndarray
    Theta: np.ndarray
    W: np.ndarray
    Y: np.ndarray
    sigma2: float


def generate_H(rng: np.random.Generator, M: int, N: int) -> np.ndarray:
    return (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))) / np.sqrt(2.0)


def generate_Theta(rng: np.random.Generator, L: int, N: int) -> np.ndarray:
    """L x N unit-norm QPSK pilot matrix."""
    symbols = rng.integers(0, 4, size=(L, N))
    Theta = np.exp(1j * (np.pi / 2.0) * (symbols + 0.5))
    Theta /= np.sqrt(L)
    return Theta.astype(np.complex128, copy=False)


def build_Z(a: np.ndarray, H: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float)[:, None] * np.asarray(H).T


def add_noise_for_snr(rng: np.random.Generator, signal: np.ndarray, snr_db: float) -> tuple[np.ndarray, float]:
    signal_power = float(np.mean(np.abs(signal) ** 2))
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    sigma2 = signal_power / snr_linear if signal_power > 0 else 1.0 / snr_linear
    W = np.sqrt(sigma2 / 2.0) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return W, sigma2


def simulate_communication(
    rng: np.random.Generator,
    a: np.ndarray,
    M: int,
    L: int,
    snr_db: float,
    Theta: np.ndarray | None = None,
    H: np.ndarray | None = None,
) -> CommunicationSample:
    N = len(a)
    H = generate_H(rng, M, N) if H is None else H
    Theta = generate_Theta(rng, L, N) if Theta is None else Theta
    Z = build_Z(a, H)
    signal = Theta @ Z
    W, sigma2 = add_noise_for_snr(rng, signal, snr_db)
    return CommunicationSample(a=np.asarray(a, dtype=bool), H=H, Z=Z, Theta=Theta, W=W, Y=signal + W, sigma2=sigma2)
