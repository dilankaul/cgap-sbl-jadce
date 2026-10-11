import numpy as np
from cgap_sbl_jadce.models.activity import calibrate_kappa_grid, best_kappa_for_target


def test_kappa_calibration_is_reproducible_and_increases_activity():
    kwargs = dict(
        seed=77, num_samples=120, N=80, V=2, R=40.0, D=15.0,
        kappa_values=[2.0, 4.0], batch_size=24, show_progress=False,
    )
    a = calibrate_kappa_grid(**kwargs)
    b = calibrate_kappa_grid(**kwargs)
    assert a == b
    assert a[1].mean_expected_active > a[0].mean_expected_active
    best = best_kappa_for_target(a, target_mean_active=a[1].mean_expected_active)
    assert np.isclose(best.kappa, 4.0)
