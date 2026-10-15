import numpy as np
import pytest
from cgap_sbl_jadce.algorithms.omp import mmv_omp
from cgap_sbl_jadce.algorithms.cosamp import mmv_cosamp
from cgap_sbl_jadce.metrics import oracle_sparsity_budget


def test_mmv_baselines_recover_identity_support():
    N = L = 12; M = 2; S = 3
    Theta = np.eye(N, dtype=np.complex128)
    Z = np.zeros((N, M), dtype=np.complex128)
    support = np.array([1, 5, 9])
    Z[support] = np.array([[1, 0.5j], [0.7, -1j], [1.2j, 0.3]])
    Y = Theta @ Z
    omp = mmv_omp(Theta, Y, S)
    co = mmv_cosamp(Theta, Y, S, max_iter=10, tol=1e-12, lambda_reg=0)
    assert set(np.flatnonzero(np.linalg.norm(omp, axis=1) > 0)) == set(support)
    assert set(np.flatnonzero(np.linalg.norm(co, axis=1) > 0)) == set(support)
    assert np.allclose(omp, Z)
    assert np.allclose(co, Z)


def test_mmv_baselines_accept_zero_oracle_sparsity():
    Theta = np.eye(6, dtype=np.complex128)
    Y = np.zeros((6, 2), dtype=np.complex128)
    assert np.allclose(mmv_omp(Theta, Y, 0), 0.0)
    assert np.allclose(mmv_cosamp(Theta, Y, 0), 0.0)


def test_oracle_budget_equals_realized_s_when_feasible():
    assert oracle_sparsity_budget(S_q=10, L=20, N=400) == 10


def test_oracle_budget_is_capped_by_pilot_dimension():
    assert oracle_sparsity_budget(S_q=22, L=20, N=400) == 20


def test_cosamp_ridge_matches_independent_two_stage_reference():
    rng = np.random.default_rng(42)
    Theta = rng.normal(size=(5, 12)) + 1j * rng.normal(size=(5, 12))
    Y = rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2))
    B_q = 3
    lambda_reg = .01
    energy = np.sum(np.abs(Theta.conj().T @ Y) ** 2, axis=1)
    merged = np.sort(np.argpartition(energy, -2 * B_q)[-2 * B_q:])
    A = Theta[:, merged]
    B = np.linalg.solve(A.conj().T @ A + lambda_reg * np.eye(len(merged)), A.conj().T @ Y)
    local = np.argpartition(np.sum(np.abs(B) ** 2, axis=1), -B_q)[-B_q:]
    support = np.sort(merged[local])
    A = Theta[:, support]
    expected = np.zeros((12, 2), dtype=complex)
    expected[support] = np.linalg.solve(A.conj().T @ A + lambda_reg * np.eye(B_q), A.conj().T @ Y)
    actual = mmv_cosamp(Theta, Y, B_q, max_iter=1, lambda_reg=lambda_reg)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_cosamp_default_ridge_bounds_ill_conditioned_square_fit():
    Theta = np.diag([1., 1e-10]).astype(complex)
    Y = np.array([[1., 1j], [1., -1j]])
    actual = mmv_cosamp(Theta, Y, 2, max_iter=1)
    factors = np.array([1., 1e-10]) / (np.array([1., 1e-10]) ** 2 + .01)
    np.testing.assert_allclose(actual, factors[:, None] * Y, rtol=1e-10, atol=1e-12)
    assert np.linalg.norm(actual) < 2
    assert np.linalg.norm(mmv_cosamp(Theta, Y, 2, max_iter=1, lambda_reg=0)) > 1e9


@pytest.mark.parametrize('lambda_reg', [-.01, float('nan'), float('inf')])
def test_cosamp_rejects_invalid_ridge(lambda_reg):
    with pytest.raises(ValueError, match='lambda_reg'):
        mmv_cosamp(np.eye(2), np.ones((2, 1)), 1, lambda_reg=lambda_reg)


def test_ridge_setting_blocks_old_cosamp_results_and_preserves_omp(saved_data):
    from cgap_sbl_jadce.experiments.estimators import run_estimator_stage, completed_methods
    from cgap_sbl_jadce.gui.setups import validate_setup
    from cgap_sbl_jadce.paths import run_path
    from cgap_sbl_jadce.settings import grouped_config

    cfg = grouped_config(saved_data)
    cfg['algorithms']['cosamp']['solver']['cosamp_lambda_reg'] = 0.
    validate_setup(cfg)
    for method in ('cosamp', 'omp'):
        run_estimator_stage(cfg, method, show_progress=False, conditions=[[7, 8]])
    assert set(completed_methods(cfg, [(7, 8)])) == {'cosamp', 'omp'}
    path = run_path(cfg, 'cosamp', 'per_sample.csv', (7, 8))
    before = path.read_bytes()
    cfg['algorithms']['cosamp']['solver']['cosamp_lambda_reg'] = .01
    assert completed_methods(cfg, [(7, 8)]) == ['omp']
    assert path.read_bytes() == before


@pytest.mark.parametrize('lambda_reg', [-.01, float('nan'), float('inf')])
def test_setup_rejects_invalid_ridge(lambda_reg):
    from cgap_sbl_jadce.config import load_config
    from cgap_sbl_jadce.gui.setups import validate_setup

    cfg = load_config('configs/experiment.yaml')
    cfg['algorithms']['cosamp']['solver']['cosamp_lambda_reg'] = lambda_reg
    with pytest.raises(ValueError, match='cosamp_lambda_reg'):
        validate_setup(cfg)
