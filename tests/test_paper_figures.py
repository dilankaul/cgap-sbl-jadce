from pathlib import Path
import runpy

import pytest

from cgap_sbl_jadce.figures import paper_activity_realizations as plotting
from cgap_sbl_jadce.figures import task02_activity_realizations as activity
from cgap_sbl_jadce.config import load_config


@pytest.mark.parametrize("split", ["tuning", "evaluation"])
def test_paper_export_uses_one_selected_saved_sample(monkeypatch, split):
    cfg = load_config("configs/experiment.yaml")
    samples = [object(), object()]
    calls = []
    loaded = []
    monkeypatch.setattr(activity, "_saved_activity_parameters", lambda cfg: (40, 15, 3.78))
    monkeypatch.setattr(plotting, "load_activity_samples", lambda path: loaded.append(path) or samples)
    monkeypatch.setattr(plotting, "plot_activity_sample", lambda sample, **kw: calls.append((sample, kw)) or [kw["stem"].with_suffix(".pdf")])
    path = plotting.plot_paper_activity_figure(cfg, split, 1)
    assert len(calls) == 1
    assert loaded[0].name == f"{split}_activity.npz"
    sample, kw = calls[0]
    assert sample is samples[1]
    assert kw["title"] is None
    assert kw["kappa"] == 3.78
    assert kw["style"]["axes.labelsize"] == 15
    assert kw["figsize"] == (6.8, 6.2)
    assert kw["formats"] == "pdf"
    assert path == Path(cfg["run"]["output_dir"]) / "paper_figures" / "activity" / split / f"event_driven_activation_model_{split}_0001.pdf"


@pytest.mark.parametrize("split,index", [("both", 0), ("tuning", -1), ("evaluation", 1), ("evaluation", True)])
def test_invalid_paper_selection_rejected(monkeypatch, split, index):
    monkeypatch.setattr(plotting, "load_activity_samples", lambda path: [object()])
    with pytest.raises(ValueError):
        plotting.plot_paper_activity_figure(load_config("configs/experiment.yaml"), split, index)


def test_paper_script_passes_single_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(plotting, "plot_paper_activity_figure", lambda cfg, split, index: calls.append((split, index)) or Path("figure.pdf"))
    script = runpy.run_path("scripts/paper_figures/activity_realization.py", run_name="import_check")
    assert script["main"](["--split", "evaluation", "--index", "5"]) == Path("figure.pdf")
    assert calls == [("evaluation", 5)]
