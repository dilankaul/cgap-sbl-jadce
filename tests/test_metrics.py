import numpy as np
import pytest
from cgap_sbl_jadce.metrics import nmse


def test_nmse_ignores_inactive_row_leakage():
    truth = np.array([[1+1j, 2], [0, 0], [3, -1j]])
    estimate = truth.copy()
    estimate[1] = [1e6, -1e6j]
    assert nmse(truth, estimate, np.array([True, False, True])) == 0
    assert nmse(truth, estimate) == 0


def test_nmse_penalizes_missed_active_rows():
    truth = np.array([[1+1j, 2], [0, 0], [3, -1j]])
    estimate = truth.copy()
    estimate[2] = 0
    estimate[1] = 100
    expected = np.linalg.norm(truth[2])**2 / np.linalg.norm(truth)**2
    assert nmse(truth, estimate, [True, False, True]) == pytest.approx(expected)


def test_nmse_explicit_activity_includes_zero_fading_active_row():
    truth = np.array([[1., 0], [0, 0]])
    estimate = np.array([[1., 0], [2, 0]])
    assert nmse(truth, estimate, [True, True]) == 4


def test_nmse_zero_true_energy_is_undefined():
    assert np.isnan(nmse(np.zeros((3, 2)), np.ones((3, 2)), [False]*3))
