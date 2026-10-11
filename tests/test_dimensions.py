import numpy as np
from cgap_sbl_jadce.models.communication import generate_H, generate_Theta, build_Z


def test_paper_dimensions():
    rng = np.random.default_rng(1)
    N, M, L = 30, 4, 12
    a = np.zeros(N, dtype=bool); a[:5] = True
    H = generate_H(rng, M, N)
    Theta = generate_Theta(rng, L, N)
    Z = build_Z(a, H)
    Y = Theta @ Z
    assert H.shape == (M, N)
    assert Z.shape == (N, M)
    assert Theta.shape == (L, N)
    assert Y.shape == (L, M)
    assert np.allclose(np.sum(np.abs(Theta) ** 2, axis=0), 1.0)
