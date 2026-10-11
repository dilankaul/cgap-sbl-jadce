from pathlib import Path

import pytest
import yaml

from casbl_jadce.config import load_config


def test_explicit_seeds_are_simple_and_distinct():
    cfg = load_config(Path("configs/experiment.yaml"))
    assert cfg["seeds"] == {
        "kappa_calibration": 1379,
        "kappa_validation": 8641,
        "activity_tuning": 2953,
        "activity_evaluation": 7069,
        "communication_tuning": 4283,
        "communication_evaluation": 9517,
        "spatial_randomization": 6031,
    }
    assert "activity_seed" not in cfg["run"]
    assert not any("seed_offset" in name for name in cfg["activity"])


@pytest.mark.parametrize("invalid", [2953, -1, 1.5, True])
def test_invalid_or_duplicate_stream_seed_rejected(tmp_path, invalid):
    cfg = load_config("configs/experiment.yaml")
    cfg["seeds"]["communication_tuning"] = invalid
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="seed"):
        load_config(path)
