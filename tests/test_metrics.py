import numpy as np
import pytest

from cgap_sbl_jadce.metrics import nmse


def test_nmse_ignores_inactive_estimates_but_counts_missed_active_rows():
    true = np.array([[1 + 1j, 2], [0, 0], [2j, 1]], dtype=complex)
    active = np.array([True, False, True])
    estimate = true.copy()
    estimate[1] = [1000, -1000j]
    assert nmse(true, estimate, active) == 0
    assert nmse(true, estimate) == 0
    estimate[0] = 0
    assert nmse(true, estimate, active) == pytest.approx(6 / 11)
    estimate[2] = 0
    assert nmse(true, estimate, active) == pytest.approx(1)


def test_nmse_uses_explicit_activity_even_when_an_active_true_row_is_zero():
    true = np.array([[1.0], [0.0], [0.0]])
    estimate = np.array([[1.0], [2.0], [1000.0]])
    assert nmse(true, estimate, [True, True, False]) == pytest.approx(4)


def test_nmse_zero_reference_energy_is_undefined():
    assert np.isnan(nmse(np.zeros((3, 2)), np.ones((3, 2))))
    assert np.isnan(nmse(np.ones((3, 2)), np.ones((3, 2)), np.zeros(3, dtype=bool)))


@pytest.mark.parametrize('estimate,active', [(np.zeros((2, 2)), None), (np.zeros((3, 2)), [True, False])])
def test_nmse_rejects_mismatched_dimensions(estimate, active):
    with pytest.raises(ValueError, match='shape'):
        nmse(np.zeros((3, 2)), estimate, active)


@pytest.mark.parametrize('method', ['sbl', 'cgap'])
def test_convergence_nmse_history_uses_true_active_rows(method):
    from cgap_sbl_jadce.algorithms.sbl import sbl
    from cgap_sbl_jadce.algorithms.cgap_sbl import cgap_sbl_anc
    from cgap_sbl_jadce.algorithms.posterior import posterior_mmv
    theta = np.array([[1, 1, 1], [1, 1j, -1]], dtype=complex) / np.sqrt(2)
    true = np.array([[1 + 1j], [0], [0]], dtype=complex)
    y = theta @ true + np.array([[0.1j], [0.1]])
    options = dict(K=3, delta=1e-12, keep_history=True, Z_true=true)
    result = (sbl(theta, y, .1, **options) if method == 'sbl'
              else cgap_sbl_anc(theta, y, .1, np.eye(3), 2., .1, **options))
    previous = np.vstack([np.full(3, .1), result.gamma_history[:-1]])
    expected = []
    for gamma in previous:
        mu = posterior_mmv(theta, y, .1, gamma).mu
        expected.append(np.linalg.norm(true[0] - mu[0]) ** 2 / np.linalg.norm(true[0]) ** 2)
    np.testing.assert_allclose(result.nmse_history, expected)
