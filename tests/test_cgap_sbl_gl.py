import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cgap_sbl_jadce.algorithms.cgap_sbl import cgap_sbl
from cgap_sbl_jadce.algorithms.cgap_sbl_gl import cgap_sbl_gl, gl_penalty, normalized_connectivity
from cgap_sbl_jadce.algorithms.sbl import sbl


def example():
    rng = np.random.default_rng(9182)
    Theta = (rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6))) / np.sqrt(8)
    Y = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    C = np.eye(6)
    for i, j, weight in [(0, 1, .7), (1, 2, .4), (2, 3, .2), (3, 4, .9)]:
        C[i, j] = C[j, i] = weight
    return Theta, Y, C


def test_gl_matches_explicit_laplacian_and_uses_previous_gamma():
    Theta, Y, C = example()
    xi = C.sum(axis=1)
    L = np.eye(6) - C / np.sqrt(xi[:, None] * xi[None, :])
    options = dict(K=8, delta=1e-10, keep_history=True)
    reference = cgap_sbl(Theta, Y, .1, 7 * L, **options)
    result = cgap_sbl_gl(Theta, Y, .1, csr_matrix(C), 7, **options)
    np.testing.assert_allclose(result.gamma_history, reference.gamma_history, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.mu, reference.mu, rtol=1e-12, atol=1e-14)
    previous = np.vstack([np.full(6, .1), result.gamma_history[:-1]])
    np.testing.assert_allclose(result.lambda_history, 7 * np.maximum(previous @ L.T, 0), atol=1e-14)
    assert np.all(result.lambda_history >= 0)
    assert result.iterations == reference.iterations


@pytest.mark.parametrize('alpha,C_identity', [(0., False), (10., True)])
def test_gl_recovers_sbl_with_zero_penalty(alpha, C_identity):
    Theta, Y, C = example()
    if C_identity:
        C = np.eye(6)
    options = dict(K=8, delta=1e-10, keep_history=True)
    result = cgap_sbl_gl(Theta, Y, .1, C, alpha, **options)
    reference = sbl(Theta, Y, .1, **options)
    np.testing.assert_array_equal(result.gamma, reference.gamma)
    np.testing.assert_array_equal(result.mu, reference.mu)
    assert not result.lambda_history.any()


def test_gl_retains_self_connections_and_normalizes_connectivity_without_dense_conversion(monkeypatch):
    _, _, C = example()
    xi = C.sum(axis=1)
    def forbidden(*args, **kwargs):
        pytest.fail('GL constructed a dense graph matrix')
    monkeypatch.setattr(csr_matrix, 'toarray', forbidden)
    matrix, r = normalized_connectivity(csr_matrix(C), 6)
    np.testing.assert_allclose(r, 1 / np.sqrt(xi))
    np.testing.assert_array_equal(matrix.diagonal(), np.ones(6))
    np.testing.assert_allclose(gl_penalty(matrix, r, np.sqrt(xi), 5), 0, atol=1e-14)
    gamma = np.array([1., .02, .1, .03, .01, 2.])
    penalty = gl_penalty(matrix, r, gamma, 5)
    assert penalty[0] > 0 and penalty[5] == 0
    scaled, r_scaled = normalized_connectivity(C * 3, 6)
    np.testing.assert_allclose(gl_penalty(scaled, r_scaled, gamma, 5), penalty, atol=1e-14)
    # Equal raw variances are not the null vector on an irregular graph.
    assert np.any(gl_penalty(matrix, r, np.ones(6), 5) > 0)


@pytest.mark.parametrize('kind', ['shape', 'asymmetric', 'negative', 'nan', 'zero_degree', 'complex', 'alpha_negative', 'alpha_nan'])
def test_gl_rejects_invalid_graphs_and_strength(kind):
    Theta, Y, C = example()
    alpha = 2.
    if kind == 'shape': C = C[:-1]
    if kind == 'asymmetric': C[0, 1] = 0
    if kind == 'negative': C[0, 1] = C[1, 0] = -.1
    if kind == 'nan': C[0, 0] = np.nan
    if kind == 'zero_degree': C[5] = 0; C[:, 5] = 0
    if kind == 'complex': C = C.astype(complex); C[0, 1] += 1j
    if kind == 'alpha_negative': alpha = -1
    if kind == 'alpha_nan': alpha = np.nan
    with pytest.raises(ValueError):
        cgap_sbl_gl(Theta, Y, .1, C, alpha)
