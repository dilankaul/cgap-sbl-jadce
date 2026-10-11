import numpy as np
from cgap_sbl_jadce.models.activity import ActivitySample
from cgap_sbl_jadce.models.realizations import communication_realization


def test_condition_generation_is_reproducible_and_snr_independent_theta():
    N = 20
    a = np.zeros(N, dtype=bool); a[:3] = True
    activity = ActivitySample(a, np.zeros((N,2)), np.zeros((1,2)), np.zeros(N))
    r1 = communication_realization(activity, 123, 7, 4, 10, 8)
    r2 = communication_realization(activity, 123, 7, 4, 10, 8)
    r3 = communication_realization(activity, 123, 7, 4, 10, 16)
    assert np.array_equal(r1.H, r2.H)
    assert np.array_equal(r1.Theta, r2.Theta)
    assert np.array_equal(r1.W, r2.W)
    assert np.array_equal(r1.Theta, r3.Theta)
    assert np.array_equal(r1.H, r3.H)
    assert not np.array_equal(r1.W, r3.W)
