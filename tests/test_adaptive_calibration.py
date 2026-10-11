import pytest
from cgap_sbl_jadce.experiments import calibration
from cgap_sbl_jadce.models.activity import ActivityCalibrationRow, calibrate_kappa_grid


def test_refinement_grid_selection_and_independent_validation(monkeypatch):
    calls = []

    def scan(**kwargs):
        calls.append(kwargs)
        return [ActivityCalibrationRow(k, 2 * k, 2 * k, 1, 0, 3, 0.5, 0.05)
                for k in kwargs['kappa_values']]

    monkeypatch.setattr(calibration, 'calibrate_kappa_grid', scan)
    args = dict(target=1.57, coarse_values=[2, 0.5, 1],
                refinement_steps=[0.1, 0.02, 0.005], validation_seed=92,
                validation_samples=100, seed=91, num_samples=100,
                N=10, V=1, R=40, D=15, batch_size=16, show_progress=False)
    rows, best, records, info = calibration.calibrate_kappa_adaptive(**args)
    assert best.kappa == 0.785
    assert info['final_step'] == 0.005
    assert info['validation']['within_tolerance']
    assert [c['seed'] for c in calls[:-1]] == [91] * (len(calls) - 1)
    assert calls[-1]['seed'] == 92
    assert calls[-1]['kappa_values'] == [0.785]
    assert calls[0]['kappa_values'] == [0.5, 1.0, 2.0]
    assert any(record['stage'] == 'refine_3' for record in records)
    assert calibration.calibrate_kappa_adaptive(**args) == (rows, best, records, info)


def test_no_bracket_reports_failure(monkeypatch):
    monkeypatch.setattr(calibration, 'calibrate_kappa_grid', lambda **kw: [
        ActivityCalibrationRow(k, k, k, 0, 0, 1) for k in kw['kappa_values']])
    with pytest.raises(calibration.CalibrationBracketError, match='not bracketed') as error:
        calibration.calibrate_kappa_adaptive(target=10, coarse_values=[0.5, 1],
            refinement_steps=[0.005], validation_seed=2, validation_samples=10,
            seed=1, num_samples=10, show_progress=False)
    assert len(error.value.records) == 2


def test_replayed_geometry_is_identical_across_scans():
    args = dict(seed=321, num_samples=100, N=40, V=2, R=40,
                D=15, batch_size=16, show_progress=False)
    together = calibrate_kappa_grid(kappa_values=[1, 3], **args)
    assert together[1] == calibrate_kappa_grid(kappa_values=[3], **args)[0]
    assert together[1].se_expected_active > 0
