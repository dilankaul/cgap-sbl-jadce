from __future__ import annotations
from pathlib import Path
import numpy as np
from cgap_sbl_jadce.models.activity import ActivitySample
from cgap_sbl_jadce.io import ensure_dir, save_npz


def save_activity_samples(path: str | Path, samples: list[ActivitySample]) -> None:
    p = Path(path); ensure_dir(p.parent)
    save_npz(
        p,
        a=np.stack([s.a for s in samples]),
        device_locations=np.stack([s.device_locations for s in samples]),
        event_locations=np.stack([s.event_locations for s in samples]),
        activation_probabilities=np.stack([s.activation_probabilities for s in samples]),
        attempts=np.asarray([s.attempts for s in samples], dtype=np.int64),
    )


def load_activity_samples(path: str | Path) -> list[ActivitySample]:
    # NPZ members are decompressed on every lookup, so read each array once.
    # Close the archive before constructing per-realization views.
    with np.load(path) as data:
        a = data["a"].astype(bool)
        device_locations = data["device_locations"]
        event_locations = data["event_locations"]
        activation_probabilities = data["activation_probabilities"]
        attempts = data["attempts"] if "attempts" in data.files else np.ones(a.shape[0], dtype=int)
    return [
        ActivitySample(
            a=a[i],
            device_locations=device_locations[i],
            event_locations=event_locations[i],
            activation_probabilities=activation_probabilities[i],
            attempts=int(attempts[i]),
        )
        for i in range(a.shape[0])
    ]
