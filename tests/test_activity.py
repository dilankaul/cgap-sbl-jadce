import numpy as np
from cgap_sbl_jadce.models.activity import event_activation_probabilities, generate_activity_sample


def test_event_probability_is_zero_beyond_D():
    devices = np.array([[0., 0.], [21., 0.]])
    events = np.array([[0., 0.]])
    p = event_activation_probabilities(devices, events, kappa=3.0, D=20.0)
    assert np.isclose(p[0], 1.0)
    assert p[1] == 0.0


def test_batched_activity_generator_records_attempts_and_exact_sparsity():
    rng = np.random.default_rng(9)
    sample = generate_activity_sample(
        rng, N=20, V=1, R=20.0, kappa=3.0, D=8.0,
        S=1, exact_sparsity=True, max_attempts=20000, batch_size=32,
    )
    assert int(np.sum(sample.a)) == 1
    assert sample.attempts >= 1


def test_unconditioned_activity_does_not_force_S():
    rng = np.random.default_rng(123)
    sample = generate_activity_sample(
        rng, N=50, V=2, R=40.0, kappa=3.8, D=15.0,
        S=10, exact_sparsity=False, batch_size=32,
    )
    assert sample.attempts == 1
    assert sample.a.shape == (50,)
    assert sample.activation_probabilities.shape == (50,)


def test_variable_s_realizations_are_not_forced_to_target():
    counts = []
    for seed in range(20):
        sample = generate_activity_sample(
            np.random.default_rng(seed), N=80, V=2, R=40.0, kappa=3.8, D=15.0,
            S=10, exact_sparsity=False,
        )
        counts.append(int(np.sum(sample.a)))
    assert len(set(counts)) > 1
