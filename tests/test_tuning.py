import numpy as np
from cgap_sbl_jadce.experiments.tuning import threshold_curve, choose_gamma_threshold


def test_threshold_curve_contains_selected_threshold():
    a = np.array([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=bool)
    gamma = np.array([[0.01, 0.8, 0.02, 0.7], [0.9, 0.01, 0.03, 0.6]])
    rows = threshold_curve(a, gamma)
    choice = choose_gamma_threshold(a, gamma)
    best = max(rows, key=lambda x: (x["f1"], x["recall"], x["precision"]))
    assert np.isclose(choice.tau, best["tau"])
    assert np.isclose(choice.mean_f1, best["f1"])


def test_fixed_tau_grid_and_inclusive_boundary():
    from cgap_sbl_jadce.experiments.tuning import threshold_candidates
    grid = threshold_candidates()
    np.testing.assert_allclose(grid, np.linspace(0, .2, 201))
    rows = threshold_curve(np.array([[True, False]]), np.array([[.05, .01]]))
    assert len(rows) == 201
    assert rows[11]['f1'] == 1.0
    assert rows[50]['recall'] == 1.0  # gamma equal to tau is active
    assert rows[51]['recall'] == 0.0
