import numpy as np
from cgap_sbl_jadce.experiments.tuning import threshold_curve, choose_gamma_threshold


def test_threshold_curve_contains_selected_threshold():
    a = np.array([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=bool)
    gamma = np.array([[0.01, 0.8, 0.02, 0.7], [0.9, 0.01, 0.03, 0.6]])
    rows = threshold_curve(a, gamma, num_thresholds=12)
    choice = choose_gamma_threshold(a, gamma, num_thresholds=12)
    best = max(rows, key=lambda x: (x["f1"], x["recall"], x["precision"]))
    assert np.isclose(choice.tau, best["tau"])
    assert np.isclose(choice.mean_f1, best["f1"])
