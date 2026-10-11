import numpy as np
from casbl_jadce.algorithms.posterior import posterior_mmv, posterior_mmv_full_reference


def test_optimized_posterior_matches_full_reference():
    rng = np.random.default_rng(2)
    L, N, M = 8, 15, 3
    Theta = (rng.standard_normal((L, N)) + 1j * rng.standard_normal((L, N))) / np.sqrt(2 * L)
    Y = rng.standard_normal((L, M)) + 1j * rng.standard_normal((L, M))
    gamma = rng.uniform(0.05, 1.5, N)
    sigma2 = 0.2
    fast = posterior_mmv(Theta, Y, sigma2, gamma)
    mu, Sigma = posterior_mmv_full_reference(Theta, Y, sigma2, gamma)
    assert np.allclose(fast.mu, mu, atol=1e-10, rtol=1e-9)
    assert np.allclose(fast.Sigma_diag, np.real(np.diag(Sigma)), atol=1e-10, rtol=1e-9)
