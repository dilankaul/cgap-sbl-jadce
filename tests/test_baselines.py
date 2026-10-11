import numpy as np
from casbl_jadce.algorithms.omp import mmv_omp
from casbl_jadce.algorithms.cosamp import mmv_cosamp
from casbl_jadce.metrics import oracle_sparsity_budget


def test_mmv_baselines_recover_identity_support():
    N = L = 12; M = 2; S = 3
    Theta = np.eye(N, dtype=np.complex128)
    Z = np.zeros((N, M), dtype=np.complex128)
    support = np.array([1, 5, 9])
    Z[support] = np.array([[1, 0.5j], [0.7, -1j], [1.2j, 0.3]])
    Y = Theta @ Z
    omp = mmv_omp(Theta, Y, S)
    co = mmv_cosamp(Theta, Y, S, max_iter=10, tol=1e-12)
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
