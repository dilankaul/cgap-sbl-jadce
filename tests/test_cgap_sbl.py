import numpy as np
import pytest
from cgap_sbl_jadce.algorithms.cgap_sbl import cgap_sbl, cgap_sbl_anc, stable_gamma_update, relative_gamma_change
from cgap_sbl_jadce.algorithms.sbl import sbl
from cgap_sbl_jadce.models.communication import simulate_communication
from cgap_sbl_jadce.models.spatial_dependence import build_Omega_ANC


def example():
    rng = np.random.default_rng(4)
    a = np.zeros(20, dtype=bool)
    a[[1, 7, 13]] = True
    return simulate_communication(rng, a, M=3, L=10, snr_db=15)


def test_stable_root_satisfies_stationarity_and_zero_penalty():
    eta = np.array([.01, .2, 2., 20.])
    lambda_ = np.array([0., 1e-8, .5, 12.])
    gamma = stable_gamma_update(eta, lambda_, 4)
    assert gamma[0] == eta[0]
    np.testing.assert_allclose(lambda_ * gamma**2 + 4*gamma - 4*eta, 0, atol=1e-12)
    assert np.all(gamma <= eta)
    with pytest.raises(ValueError, match="non-negative"):
        stable_gamma_update(eta, -np.ones(4), 4)


def test_zero_omega_and_zero_alpha_recover_sbl():
    r = example()
    options = dict(K=25, delta=1e-8, keep_history=True)
    reference = sbl(r.Theta, r.Y, r.sigma2, **options)
    for result in [cgap_sbl(r.Theta, r.Y, r.sigma2, np.zeros((20,20)), **options),
                   cgap_sbl_anc(r.Theta, r.Y, r.sigma2, np.eye(20), 0., .1, **options)]:
        np.testing.assert_allclose(result.gamma_history, reference.gamma_history)
        np.testing.assert_allclose(result.mu, reference.mu)
        assert result.iterations == reference.iterations
        assert result.converged == reference.converged
        assert not np.any(result.lambda_history)


def test_anc_matches_general_and_lagged_penalty():
    r = example()
    C = np.eye(20)
    Omega = build_Omega_ANC(C, 2., .1)
    assert np.linalg.eigvalsh(Omega).min() < 0
    options = dict(K=8, delta=1e-12, keep_history=True)
    general = cgap_sbl(r.Theta, r.Y, r.sigma2, Omega, **options)
    anc = cgap_sbl_anc(r.Theta, r.Y, r.sigma2, C, 2., .1, **options)
    np.testing.assert_allclose(anc.gamma_history, general.gamma_history)
    previous = np.vstack([np.full(20,.1), anc.gamma_history[:-1]])
    np.testing.assert_allclose(anc.lambda_history, np.maximum(previous @ Omega.T, 0))


def test_relative_stopping_is_scale_invariant():
    old = np.array([.1,.2]); new = old*1.01
    assert np.isclose(relative_gamma_change(new, old), .01)
    assert np.isclose(relative_gamma_change(new*100, old*100), .01)


def test_final_estimate_uses_final_gamma_and_sparse_anc_matches_dense():
    from scipy.sparse import csr_matrix
    from cgap_sbl_jadce.algorithms.posterior import posterior_mmv
    r = example()
    C = np.eye(20)
    dense = cgap_sbl_anc(r.Theta, r.Y, r.sigma2, C, 2., .1, K=5)
    sparse = cgap_sbl_anc(r.Theta, r.Y, r.sigma2, csr_matrix(C), 2., .1, K=5)
    np.testing.assert_allclose(dense.gamma, sparse.gamma)
    posterior = posterior_mmv(r.Theta, r.Y, r.sigma2, dense.gamma)
    np.testing.assert_allclose(dense.mu, posterior.mu)


def test_penalized_update_maximizes_frozen_scalar_objective():
    eta = np.array([.2, 1., 3.])
    penalty = np.array([0., .5, 8.])
    gamma = stable_gamma_update(eta, penalty, 4)
    def objective(value):
        return -4*np.log(value) - 4*eta/value - penalty*value
    for scale in [.01, .5, .99, 1.01, 2., 100.]:
        assert np.all(objective(gamma) > objective(gamma*scale))
