from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ActivitySample:
    a: np.ndarray
    device_locations: np.ndarray
    event_locations: np.ndarray
    activation_probabilities: np.ndarray
    attempts: int = 1


def randomize_activity_sample(sample: ActivitySample, seed: int, q: int) -> ActivitySample:
    """Uniform fixed-count support on the original geometry (paper VIII-F)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, q]))
    S_q = int(np.sum(sample.a))
    a = np.zeros_like(sample.a, dtype=bool)
    a[rng.choice(len(a), size=S_q, replace=False)] = True
    return replace(sample, a=a, activation_probabilities=np.full(len(a), S_q / len(a)), attempts=1)


@dataclass(frozen=True)
class ActivityCalibrationRow:
    kappa: float
    mean_expected_active: float
    mean_realized_active: float
    std_realized_active: float
    min_realized_active: int
    max_realized_active: int
    std_expected_active: float = 0.0
    se_expected_active: float = 0.0


def uniform_disk(rng: np.random.Generator, count: int, R: float) -> np.ndarray:
    r = float(R) * np.sqrt(rng.random(count))
    theta = 2.0 * np.pi * rng.random(count)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def _uniform_disk_batch(rng: np.random.Generator, batch_size: int, count: int, R: float) -> np.ndarray:
    r = float(R) * np.sqrt(rng.random((batch_size, count)))
    theta = 2.0 * np.pi * rng.random((batch_size, count))
    return np.stack((r * np.cos(theta), r * np.sin(theta)), axis=-1)


def event_activation_probabilities(device_locations: np.ndarray, event_locations: np.ndarray, kappa: float, D: float) -> np.ndarray:
    delta = device_locations[:, None, :] - event_locations[None, :, :]
    d = np.sqrt(np.sum(delta * delta, axis=2))
    if kappa <= 0:
        p_iv = (d == 0.0).astype(float)
    else:
        denom = 1.0 - np.exp(-D / kappa)
        raw = (np.exp(-d / kappa) - np.exp(-D / kappa)) / denom
        p_iv = np.where(d <= D, raw, 0.0)
        p_iv = np.clip(p_iv, 0.0, 1.0)
    return 1.0 - np.prod(1.0 - p_iv, axis=1)


def _event_probabilities_from_distances(d: np.ndarray, kappa: float, D: float) -> np.ndarray:
    if kappa <= 0:
        p_iv = (d == 0.0).astype(float)
    else:
        denom = 1.0 - np.exp(-D / kappa)
        raw = (np.exp(-d / kappa) - np.exp(-D / kappa)) / denom
        p_iv = np.where(d <= D, raw, 0.0)
        p_iv = np.clip(p_iv, 0.0, 1.0)
    return 1.0 - np.prod(1.0 - p_iv, axis=-1)


def _event_probabilities_batch(device_locations: np.ndarray, event_locations: np.ndarray, kappa: float, D: float) -> np.ndarray:
    delta = device_locations[:, :, None, :] - event_locations[:, None, :, :]
    d = np.sqrt(np.sum(delta * delta, axis=3))
    return _event_probabilities_from_distances(d, kappa=kappa, D=D)


def generate_activity_sample(
    rng: np.random.Generator,
    N: int,
    V: int,
    R: float,
    kappa: float,
    D: float,
    S: int,
    exact_sparsity: bool = True,
    max_attempts: int = 100000,
    batch_size: int = 128,
) -> ActivitySample:
    """Generate one spatial activity realization.

    When ``exact_sparsity`` is false, the event model is sampled once and the
    number of active MTDs is allowed to vary naturally. This is the mode used
    when S denotes a *target mean* active count.

    When ``exact_sparsity`` is true, batched rejection sampling is used until
    exactly S active MTDs are observed. That mode is retained for controlled
    experiments, but it should not be used to calibrate the average sparsity.
    """
    if not exact_sparsity:
        device_locations = uniform_disk(rng, N, R)
        event_locations = uniform_disk(rng, V, R)
        p = event_activation_probabilities(device_locations, event_locations, kappa, D)
        a = rng.random(N) < p
        return ActivitySample(
            a=a,
            device_locations=device_locations,
            event_locations=event_locations,
            activation_probabilities=p,
            attempts=1,
        )

    batch_size = max(1, int(batch_size))
    attempts = 0
    while attempts < max_attempts:
        B = min(batch_size, max_attempts - attempts)
        device_locations = _uniform_disk_batch(rng, B, N, R)
        event_locations = _uniform_disk_batch(rng, B, V, R)
        p = _event_probabilities_batch(device_locations, event_locations, kappa, D)
        a = rng.random((B, N)) < p
        matches = np.flatnonzero(np.sum(a, axis=1) == int(S))
        if matches.size == 0:
            attempts += B
            continue
        j = int(matches[0])
        attempts += j + 1
        return ActivitySample(
            a=a[j],
            device_locations=device_locations[j],
            event_locations=event_locations[j],
            activation_probabilities=p[j],
            attempts=attempts,
        )
    raise RuntimeError("Could not obtain requested sparsity. Check activity parameters or increase max_attempts.")


def generate_activity_dataset(
    seed: int,
    num_samples: int,
    N: int,
    V: int,
    R: float,
    kappa: float,
    D: float,
    S: int,
    exact_sparsity: bool = True,
    max_attempts: int = 100000,
    batch_size: int = 128,
    show_progress: bool = True,
    desc: str = "activity samples",
) -> list[ActivitySample]:
    seq = np.random.SeedSequence(seed)
    children = seq.spawn(num_samples)
    samples: list[ActivitySample] = []
    iterator = tqdm(children, total=num_samples, desc=desc, unit="sample", disable=not show_progress)
    for child in iterator:
        sample = generate_activity_sample(
            np.random.default_rng(child), N=N, V=V, R=R, kappa=kappa, D=D,
            S=S, exact_sparsity=exact_sparsity, max_attempts=max_attempts,
            batch_size=batch_size,
        )
        samples.append(sample)
        if show_progress:
            if exact_sparsity:
                iterator.set_postfix(last_attempts=sample.attempts, active=int(np.sum(sample.a)), refresh=False)
            else:
                iterator.set_postfix(active=int(np.sum(sample.a)), refresh=False)
    return samples


def calibrate_kappa_grid(
    seed: int,
    num_samples: int,
    N: int,
    V: int,
    R: float,
    D: float,
    kappa_values: list[float] | np.ndarray,
    batch_size: int = 128,
    show_progress: bool = True,
) -> list[ActivityCalibrationRow]:
    """Estimate average active MTD count for a grid of kappa values.

    The same geometries and the same Bernoulli uniforms are shared across all
    kappa values. This common-random-number design makes the kappa comparison
    much less noisy than independently regenerating every point on the curve.

    ``mean_expected_active`` is the Monte-Carlo estimate of sum_i P_i and is
    the preferred quantity for selecting kappa. ``mean_realized_active`` also
    reports the actual Bernoulli activity count that would be observed.
    """
    kappas = np.asarray(kappa_values, dtype=float)
    if kappas.ndim != 1 or len(kappas) == 0 or not np.all(np.isfinite(kappas)) or np.any(kappas <= 0):
        raise ValueError("kappa_values must be a non-empty sequence of positive values.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    rng = np.random.default_rng(seed)
    expected_sum = np.zeros(len(kappas), dtype=float)
    expected = np.empty((len(kappas), num_samples), dtype=float)
    realized = np.empty((len(kappas), num_samples), dtype=np.int32)
    batch_size = max(1, int(batch_size))

    progress = tqdm(total=num_samples, desc="Activity kappa calibration", unit="realization", disable=not show_progress)
    offset = 0
    while offset < num_samples:
        B = min(batch_size, num_samples - offset)
        devices = _uniform_disk_batch(rng, B, N, R)
        events = _uniform_disk_batch(rng, B, V, R)
        uniforms = rng.random((B, N))
        distances = np.sqrt(np.sum((devices[:, :, None, :] - events[:, None, :, :]) ** 2, axis=3))

        for j, kappa in enumerate(kappas):
            p = _event_probabilities_from_distances(distances, float(kappa), D)
            expected[j, offset:offset + B] = np.sum(p, axis=1)
            expected_sum[j] += float(np.sum(p))
            realized[j, offset:offset + B] = np.sum(uniforms < p, axis=1)

        offset += B
        progress.update(B)
    progress.close()

    rows: list[ActivityCalibrationRow] = []
    for j, kappa in enumerate(kappas):
        counts = realized[j]
        rows.append(ActivityCalibrationRow(
            kappa=float(kappa),
            mean_expected_active=float(expected_sum[j] / num_samples),
            mean_realized_active=float(np.mean(counts)),
            std_realized_active=float(np.std(counts, ddof=1)) if num_samples > 1 else 0.0,
            min_realized_active=int(np.min(counts)),
            max_realized_active=int(np.max(counts)),
            std_expected_active=float(np.std(expected[j], ddof=1)) if num_samples > 1 else 0.0,
            se_expected_active=float(np.std(expected[j], ddof=1) / np.sqrt(num_samples)) if num_samples > 1 else 0.0,
        ))
    return rows


def best_kappa_for_target(rows: list[ActivityCalibrationRow], target_mean_active: float) -> ActivityCalibrationRow:
    if not rows:
        raise ValueError("rows must not be empty.")
    return min(rows, key=lambda row: abs(row.mean_expected_active - float(target_mean_active)))
