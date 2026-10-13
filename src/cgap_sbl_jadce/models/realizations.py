from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from cgap_sbl_jadce.models.activity import ActivitySample
from cgap_sbl_jadce.models.communication import generate_H, generate_Theta, build_Z, add_noise_for_snr


@dataclass(frozen=True)
class Realization:
    a: np.ndarray
    H: np.ndarray
    Z: np.ndarray
    Theta: np.ndarray
    W: np.ndarray
    Y: np.ndarray
    sigma2: float


def communication_realization(
    activity: ActivitySample,
    master_seed: int,
    sample_index: int,
    M: int,
    L: int,
    snr_db: float,
) -> Realization:
    """Deterministically reproduce one sample/condition.

    H depends only on sample_index. Theta depends on (sample_index, L), and W
    on (sample_index, L, snr_db). Therefore all estimators see identical inputs,
    and communication generation can save identical inputs for later tasks.
    """
    N = len(activity.a)
    H_rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), int(sample_index), 11]))
    H = generate_H(H_rng, M, N)

    theta_rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), int(sample_index), 101, int(L)]))
    Theta = generate_Theta(theta_rng, L, N)
    Z = build_Z(activity.a, H)
    signal = Theta @ Z

    snr_key = int(round((float(snr_db) + 1000.0) * 1000.0))
    noise_rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), int(sample_index), 202, int(L), snr_key]))
    W, sigma2 = add_noise_for_snr(noise_rng, signal, snr_db)
    return Realization(a=activity.a, H=H, Z=Z, Theta=Theta, W=W, Y=signal + W, sigma2=sigma2)
