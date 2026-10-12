from cgap_sbl_jadce.paths import result_path
from cgap_sbl_jadce.figures.common import figure_formats


def test_figure_format_options():
    assert figure_formats("none") == ()
    assert figure_formats("png") == ("png",)
    assert figure_formats("pdf") == ("pdf",)
    assert figure_formats("both") == ("png", "pdf")


def test_activity_s_distribution_uses_saved_task_data(tmp_path):
    import numpy as np
    from cgap_sbl_jadce.models.activity import ActivitySample
    from cgap_sbl_jadce.dataset import save_activity_samples
    from cgap_sbl_jadce.figures.task02_active_mtd_count_distribution import plot_activity_s_distribution

    def sample(active_count):
        a = np.zeros(6, dtype=bool); a[:active_count] = True
        return ActivitySample(
            a=a,
            device_locations=np.zeros((6, 2)),
            event_locations=np.zeros((1, 2)),
            activation_probabilities=np.full(6, active_count / 6),
        )

    out = tmp_path / "run"
    save_activity_samples(result_path(out, "activity", "tuning_activity.npz"), [sample(1), sample(2)])
    save_activity_samples(result_path(out, "activity", "evaluation_activity.npz"), [sample(2), sample(3)])
    cfg = {"run": {"output_dir": str(out)}, "system": {"S": 2}}
    paths = plot_activity_s_distribution(cfg, formats="png")
    assert len(paths) == 1
    assert paths[0].exists()


def test_activity_probability_grid_matches_event_model():
    import numpy as np
    from cgap_sbl_jadce.figures.task02_activity_realizations import activity_probability_grid

    X, Y, P = activity_probability_grid(
        np.array([[0.0, 0.0]]), R=20.0, D=10.0, kappa=3.0, resolution=101
    )
    center = 50
    assert np.isclose(P[center, center], 1.0)
    # x=12 m is inside the BS cell but outside the event cutoff D=10 m.
    j = int(np.argmin(np.abs(X[center] - 12.0)))
    assert np.isclose(P[center, j], 0.0)
    # A corner is outside the circular BS cell and must be masked.
    assert bool(P.mask[0, 0])


def test_activity_probability_figure_can_be_saved(tmp_path):
    import numpy as np
    from cgap_sbl_jadce.models.activity import ActivitySample
    from cgap_sbl_jadce.figures.task02_activity_realizations import plot_activity_sample

    sample = ActivitySample(
        a=np.array([True, False]),
        device_locations=np.array([[0.0, 0.0], [8.0, 0.0]]),
        event_locations=np.array([[0.0, 0.0]]),
        activation_probabilities=np.array([1.0, 0.1]),
    )
    paths = plot_activity_sample(
        sample, R=20.0, D=10.0, kappa=3.0,
        stem=tmp_path / "probability", formats="png",
        show_probability_field=True, probability_resolution=101,
    )
    assert len(paths) == 1
    assert paths[0].exists()

def test_probability_field_uses_vector_contours(tmp_path, monkeypatch):
    import matplotlib.axes
    import numpy as np
    from cgap_sbl_jadce.models.activity import ActivitySample
    from cgap_sbl_jadce.figures.task02_activity_realizations import plot_activity_sample

    called = {"contourf": False}
    original_contourf = matplotlib.axes.Axes.contourf

    def spy_contourf(self, *args, **kwargs):
        called["contourf"] = True
        return original_contourf(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "contourf", spy_contourf)
    sample = ActivitySample(
        a=np.array([True, False]),
        device_locations=np.array([[0.0, 0.0], [8.0, 0.0]]),
        event_locations=np.array([[0.0, 0.0]]),
        activation_probabilities=np.array([1.0, 0.1]),
    )
    paths = plot_activity_sample(
        sample, R=20.0, D=10.0, kappa=3.0,
        stem=tmp_path / "vector_probability", formats="pdf",
        show_probability_field=True, probability_resolution=101,
        probability_levels=32,
    )
    assert called["contourf"]
    assert len(paths) == 1
    assert paths[0].exists()


def test_probability_contours_start_at_zero_and_colormap_zero_is_white(monkeypatch, tmp_path):
    import matplotlib.axes
    import numpy as np
    from cgap_sbl_jadce.models.activity import ActivitySample
    from cgap_sbl_jadce.figures.task02_activity_realizations import plot_activity_sample

    captured = {}
    original = matplotlib.axes.Axes.contourf

    def spy(self, *args, **kwargs):
        captured["levels"] = np.asarray(kwargs["levels"], dtype=float)
        captured["cmap"] = kwargs["cmap"]
        captured["norm"] = kwargs["norm"]
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "contourf", spy)
    sample = ActivitySample(
        a=np.array([True, False]),
        device_locations=np.array([[0.0, 0.0], [5.0, 5.0]]),
        event_locations=np.array([[0.0, 0.0]]),
        activation_probabilities=np.array([1.0, 0.1]),
    )
    plot_activity_sample(
        sample, R=20.0, D=10.0, kappa=3.0,
        stem=tmp_path / "probability", formats="png",
        show_probability_field=True, probability_levels=16,
    )

    assert captured["levels"][0] == 0.0
    assert captured["levels"][-1] == 1.0
    assert captured["norm"].vmin == 0.0
    assert captured["norm"].vmax == 1.0
    assert np.allclose(captured["cmap"](0.0)[:3], [1.0, 1.0, 1.0])



def test_selected_realization_exports_pdf_with_native_truetype_fonts(tmp_path):
    import numpy as np
    import matplotlib as mpl
    from cgap_sbl_jadce.models.activity import ActivitySample
    from cgap_sbl_jadce.dataset import save_activity_samples
    from cgap_sbl_jadce.figures.task02_activity_realizations import plot_activity_realization

    sample = ActivitySample(
        a=np.array([True, False]),
        device_locations=np.array([[0.0, 0.0], [8.0, 0.0]]),
        event_locations=np.array([[0.0, 0.0]]),
        activation_probabilities=np.array([1.0, 0.1]),
    )
    save_activity_samples(result_path(tmp_path, "activity", "tuning_activity.npz"), [sample])
    cfg = {
        "run": {"output_dir": str(tmp_path)},
        "system": {"R": 20.0},
        "activity": {"D": 10.0, "kappa": 3.0},
    }
    assert not mpl.rcParams["text.usetex"]
    paths = plot_activity_realization(cfg, formats="both")
    assert [p.suffix for p in paths] == [".png", ".pdf"]
    assert all(p.exists() for p in paths)
    pdf = paths[1].read_bytes()
    assert pdf.startswith(b"%PDF-")
    assert b"/Subtype /CIDFontType2" in pdf
    assert b"/FontFile2" in pdf
    assert b"/Subtype /Type3" not in pdf


def test_per_figure_font_overrides_are_isolated(tmp_path):
    from cgap_sbl_jadce.figures.common import plt, _save, _apply_figure_style

    defaults = plt.rcParams.copy()
    cfg = {"figures": {
        "style": {"axes.labelsize": 13},
        "styles": {"custom": {
            "axes.labelsize": 17, "axes.titlesize": 18,
            "xtick.labelsize": 8, "legend.fontsize": 7,
            "colorbar.labelsize": 14, "colorbar.ticksize": 6,
        }},
    }}
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="curve")
    ax.set_title("Title"); ax.set_xlabel("X")
    legend = ax.legend()
    cbar = fig.colorbar(ax.imshow([[0, 1], [1, 0]]), ax=ax, label="Probability")
    style = {**cfg["figures"]["style"], **cfg["figures"]["styles"]["custom"]}
    _apply_figure_style(fig, style)
    _save(fig, tmp_path / "custom", "png")
    assert ax.xaxis.label.get_fontsize() == 17
    assert ax.title.get_fontsize() == 18
    assert ax.get_xticklabels()[0].get_fontsize() == 8
    assert legend.get_texts()[0].get_fontsize() == 7
    assert cbar.ax.yaxis.label.get_fontsize() == 14
    assert cbar.ax.get_yticklabels()[0].get_fontsize() == 6
    other, other_ax = plt.subplots()
    other_ax.set_xlabel("X")
    _apply_figure_style(other, cfg["figures"]["style"])
    _save(other, tmp_path / "other", "png")
    assert other_ax.xaxis.label.get_fontsize() == 13
    assert plt.rcParams == defaults


def test_batch_activity_figures_include_both_splits_and_titles(tmp_path, monkeypatch):
    import numpy as np
    from cgap_sbl_jadce.models.activity import ActivitySample
    from cgap_sbl_jadce.dataset import save_activity_samples
    from cgap_sbl_jadce.figures import task02_activity_realizations as plotting

    samples = [ActivitySample(
        a=np.array([True, bool(index)]),
        device_locations=np.array([[0.0, 0.0], [8.0, 0.0]]),
        event_locations=np.array([[0.0, 0.0]]),
        activation_probabilities=np.array([1.0, 0.1]),
    ) for index in range(2)]
    save_activity_samples(result_path(tmp_path, "activity", "tuning_activity.npz"), samples)
    save_activity_samples(result_path(tmp_path, "activity", "evaluation_activity.npz"), samples[:1])
    cfg = {
        "run": {"output_dir": str(tmp_path)}, "system": {"R": 20.0},
        "activity": {"D": 10.0, "kappa": 3.0},
        "figures": {"tuning_indices": [0, 1], "evaluation_indices": [0]},
    }
    titles = []
    original_save = plotting._save

    def capture_save(fig, stem, formats):
        paths = original_save(fig, stem, formats)
        titles.append((fig.axes[0].get_title(), fig.axes[0].title.get_fontsize()))
        return paths

    monkeypatch.setattr(plotting, "_save", capture_save)
    paths = plotting.plot_activity_realizations(cfg, formats="png", show_progress=False)
    assert len(paths) == 3
    assert all(path.exists() for path in paths)
    assert len(set(paths)) == 3
    assert titles == [
        ("Tuning activity realization 0: 1 active MTDs", 15),
        ("Tuning activity realization 1: 2 active MTDs", 15),
        ("Evaluation activity realization 0: 1 active MTDs", 15),
    ]


def test_font_metadata_filter_keeps_other_warnings():
    import logging
    from cgap_sbl_jadce.figures.common import _FontTimestampFilter
    filt = _FontTimestampFilter()
    for field in ('created', 'modified'):
        record = logging.LogRecord('fontTools', logging.WARNING, '', 0,
            "'%s' timestamp seems very low; regarding as unix timestamp", (field,), None)
        assert not filt.filter(record)
    record = logging.LogRecord('fontTools', logging.WARNING, '', 0, 'Missing glyph', (), None)
    assert filt.filter(record)



def test_activity_presentation_defaults_live_in_code(tmp_path):
    from cgap_sbl_jadce.figures.task02_activity_realizations import _activity_plot_options
    from cgap_sbl_jadce.config import load_config
    cfg = {'run': {'output_dir': str(tmp_path)}, 'system': {'R': 40},
           'activity': {'D': 15, 'kappa': 3.78}, 'figures': {}}
    options = _activity_plot_options(cfg)
    assert options['show_event_radius'] and options['show_probability_field']
    assert options['probability_resolution'] == 301
    assert options['probability_alpha'] == 1.0
    assert options['probability_levels'] == 128
    assert not _activity_plot_options(cfg, False, False)['show_probability_field']
    assert set(load_config('configs/experiment.yaml')['figures']) == {'tuning_indices', 'evaluation_indices'}
