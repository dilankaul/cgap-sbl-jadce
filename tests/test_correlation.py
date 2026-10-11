import numpy as np
from cgap_sbl_jadce.models.correlation import build_C, build_Omega_ANC, anc_penalty


def test_omega_and_fast_penalty_match_explicit_expression():
    loc = np.array([[0., 0.], [1., 0.], [30., 0.]])
    C = build_C(loc, rho=7, U=20)
    alpha, beta, M = 2.5, 0.1, 4
    gamma = np.array([0.2, 0.4, 0.1])
    Omega = build_Omega_ANC(C, alpha, beta)
    assert np.allclose(anc_penalty(C, gamma, alpha, beta), np.maximum(Omega @ gamma, 0))
    assert C[0, 2] == 0.0
    assert np.allclose(np.diag(C), 1.0)
