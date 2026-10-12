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


def test_vectorized_threshold_scores_match_per_realization_metrics():
    from cgap_sbl_jadce.metrics import precision_recall_f1, support_from_gamma
    a = np.array([[False, False, False], [True, False, True], [False, True, False]])
    gamma = np.array([[0, .001, .02], [.05, .03, .18], [.002, .2, .1]])
    for row in threshold_curve(a, gamma):
        expected = np.mean([precision_recall_f1(truth, support_from_gamma(g, row['tau']))
                            for truth, g in zip(a, gamma)], axis=0)
        np.testing.assert_allclose([row['precision'], row['recall'], row['f1']], expected)


def test_missing_condition_selection_rejected():
    import pytest
    from cgap_sbl_jadce.experiments.pipeline import selection_for_condition
    with pytest.raises(ValueError, match='Rerun Task 05'):
        selection_for_condition({'conditions': {}}, 25, 12)
