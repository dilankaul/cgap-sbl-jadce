import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from cgap_sbl_jadce.algorithms.cgap_sbl import cgap_sbl
from cgap_sbl_jadce.algorithms.cgap_variants import cgap_sbl_variant
from cgap_sbl_jadce.algorithms.graph_interactions import spatial_penalty
from cgap_sbl_jadce.algorithms.sbl import sbl
from cgap_sbl_jadce.variants import SPATIAL_METHODS

C = np.array([[1., .7, .1], [.7, 1., .3], [.1, .3, 1.]])
OPTIONS = dict(alpha=2.3, beta=4.1, graph_kappa=.6, diffusion_time=1.7, affinity_bandwidth=.8)


def interaction(method, gamma, C=C, **options):
    p = {**OPTIONS, **options}
    N = len(gamma)
    J = np.ones((N, N)) / N
    if method == 'glc':
        return p['alpha'] * (p['beta'] * N * J - C)
    A = C.copy()
    if method == 'aga':
        log_gamma = np.log(np.maximum(gamma, 1e-12))
        A *= np.exp(-((log_gamma[:, None] - log_gamma[None, :]) / p['affinity_bandwidth']) ** 2 / 2)
    r = 1 / np.sqrt(A.sum(axis=1))
    F = r[:, None] * A * r[None, :]
    if method in {'nglc', 'aga'}:
        return p['alpha'] * (p['beta'] * J - F)
    if method == 'gd':
        return p['alpha'] * (np.eye(N) - expm(-p['diffusion_time'] * (np.eye(N) - F)))
    return p['alpha'] * (np.eye(N) - (p['graph_kappa'] if method == 'rgl' else 1.) * F)


@pytest.mark.parametrize('method', SPATIAL_METHODS)
@pytest.mark.parametrize('sparse', [False, True])
def test_penalties_equal_paper_interactions_and_preserve_diagonal(method, sparse):
    gamma = np.array([.8, .002, .1])
    actual = spatial_penalty(csr_matrix(C) if sparse else C, 3, method, **OPTIONS)(gamma)
    np.testing.assert_allclose(actual, np.maximum(interaction(method, gamma) @ gamma, 0.), atol=1e-14)
    assert np.any(actual > 0)
    # Nonzero diagonals contribute to both degrees and affinity.
    assert not np.allclose(actual, np.maximum(interaction(method, gamma, C=C - np.eye(3)) @ gamma, 0.))


def observations():
    rng = np.random.default_rng(1379)
    Theta = (rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))) / 2
    Y = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    return Theta, Y


@pytest.mark.parametrize('method', SPATIAL_METHODS)
def test_each_iteration_uses_previous_gamma_and_shared_m_step(method):
    Theta, Y = observations()
    result = cgap_sbl_variant(Theta, Y, .1, C, method, **OPTIONS, K=6, delta=1e-15, keep_history=True)
    previous = np.full(3, .1)
    from cgap_sbl_jadce.algorithms.posterior import posterior_mmv
    for gamma, lambda_ in zip(result.gamma_history, result.lambda_history):
        expected_penalty = np.maximum(interaction(method, previous) @ previous, 0.)
        np.testing.assert_allclose(lambda_, expected_penalty, atol=1e-13)
        posterior = posterior_mmv(Theta, Y, .1, previous)
        eta = posterior.Sigma_diag + np.sum(abs(posterior.mu) ** 2, axis=1) / 2
        np.testing.assert_allclose(gamma, 2 * eta / (1 + np.sqrt(1 + 4 * expected_penalty * eta / 2)))
        previous = gamma
    np.testing.assert_allclose(result.mu, posterior_mmv(Theta, Y, .1, previous).mu)
    if method != 'aga':
        reference = cgap_sbl(Theta, Y, .1, interaction(method, np.full(3, .1)), K=6, delta=1e-15)
        np.testing.assert_allclose(result.gamma, reference.gamma, atol=1e-13)


@pytest.mark.parametrize('method', SPATIAL_METHODS)
def test_alpha_zero_is_exactly_conventional_sbl(method):
    Theta, Y = observations()
    options = {**OPTIONS, 'alpha': 0.}
    actual = cgap_sbl_variant(Theta, Y, .1, C, method, **options, K=7, keep_history=True)
    reference = sbl(Theta, Y, .1, K=7, keep_history=True)
    np.testing.assert_array_equal(actual.gamma, reference.gamma)
    np.testing.assert_array_equal(actual.mu, reference.mu)
    np.testing.assert_array_equal(actual.gamma_history, reference.gamma_history)
    assert not actual.lambda_history.any()


def test_gd_zero_time_and_rgl_boundary_and_aga_limiting_behavior():
    gamma = np.array([.8, .002, .1])
    assert not spatial_penalty(C, 3, 'gd', alpha=5, diffusion_time=0)(gamma).any()
    gl = spatial_penalty(C, 3, 'gl', alpha=5)(gamma)
    rgl = spatial_penalty(C, 3, 'rgl', alpha=5, graph_kappa=1-1e-12)(gamma)
    np.testing.assert_allclose(rgl, gl, atol=1e-11)
    with pytest.raises(ValueError, match='boundary'):
        spatial_penalty(C, 3, 'rgl', alpha=5, graph_kappa=1)
    aga = spatial_penalty(C, 3, 'aga', alpha=5, beta=4, affinity_bandwidth=1e10)(gamma)
    nglc = spatial_penalty(C, 3, 'nglc', alpha=5, beta=4)(gamma)
    np.testing.assert_allclose(aga, nglc, atol=1e-13)
    log_floor_gamma = np.array([0., 1e-30, .1])
    saved = log_floor_gamma.copy()
    penalty = spatial_penalty(C, 3, 'aga', alpha=5, beta=4, affinity_bandwidth=.1)(log_floor_gamma)
    np.testing.assert_allclose(penalty, np.maximum(interaction('aga', log_floor_gamma, alpha=5, beta=4, affinity_bandwidth=.1) @ log_floor_gamma, 0.))
    np.testing.assert_array_equal(saved, log_floor_gamma)


@pytest.mark.parametrize('method', SPATIAL_METHODS)
def test_sparse_solvers_never_build_dense_interaction_or_inverse(method, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('dense spatial matrix or inverse')
    monkeypatch.setattr(csr_matrix, 'toarray', forbidden)
    monkeypatch.setattr(np.linalg, 'inv', forbidden)
    Theta, Y = observations()
    result = cgap_sbl_variant(Theta, Y, .1, csr_matrix(C), method, **OPTIONS, K=3)
    assert np.isfinite(result.gamma).all()


@pytest.mark.parametrize('method', SPATIAL_METHODS)
@pytest.mark.parametrize('bad', [np.zeros((3, 3)), np.eye(2), np.array([[1., 2, 0], [0, 1, 0], [0, 0, 1]]),
    C * np.nan, -C, C.astype(complex) + 1j])
def test_invalid_graphs_are_rejected(method, bad):
    with pytest.raises(ValueError):
        spatial_penalty(bad, 3, method, **OPTIONS)


@pytest.mark.parametrize('method, updates', [('rgl', {'graph_kappa_high': 1.01}), ('gd', {'diffusion_time_low': -1}),
    ('aga', {'affinity_bandwidth_low': 0}), ('aga', {'affinity_bandwidth_high': float('inf')}), ('nglc', {'beta_low': -1})])
def test_variant_search_domains(method, updates):
    from cgap_sbl_jadce.config import load_config
    from cgap_sbl_jadce.settings import validate_spatial_search
    cfg = load_config('configs/experiment.yaml')
    cfg['algorithms'][method]['tuning'].update(updates)
    with pytest.raises(ValueError):
        validate_spatial_search(cfg, method)
