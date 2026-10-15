import numpy as np
from cgap_sbl_jadce import dataset


def test_loader_reads_each_npz_array_once_and_closes_archive(tmp_path, monkeypatch):
    path = tmp_path / 'activity.npz'
    arrays = {
        'a': np.array([[1, 0], [0, 1]], dtype=np.uint8),
        'device_locations': np.arange(8).reshape(2, 2, 2),
        'event_locations': np.arange(4).reshape(2, 1, 2),
        'activation_probabilities': np.array([[0.8, 0.1], [0.2, 0.9]]),
        'attempts': np.array([1, 2]),
    }
    np.savez_compressed(path, **arrays)
    original_load = np.load
    reads = []
    closed = []

    class Archive:
        def __init__(self):
            self.archive = original_load(path)
            self.files = self.archive.files
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
            closed.append(True)
        def __getitem__(self, key):
            reads.append(key)
            return self.archive[key]

    monkeypatch.setattr(dataset.np, 'load', lambda path: Archive())
    samples = dataset.load_activity_samples(path)
    assert len(samples) == 2
    assert closed == [True]
    assert sorted(reads) == sorted(arrays)
    for index, sample in enumerate(samples):
        assert sample.a.dtype == bool
        assert np.array_equal(sample.a, arrays['a'][index])
        assert np.array_equal(sample.device_locations, arrays['device_locations'][index])
        assert np.array_equal(sample.event_locations, arrays['event_locations'][index])
        assert np.array_equal(sample.activation_probabilities, arrays['activation_probabilities'][index])
        assert sample.attempts == arrays['attempts'][index]


def test_missing_attempts_is_rejected(tmp_path):
    path = tmp_path / 'incomplete.npz'
    np.savez_compressed(path, a=np.array([[True]]), device_locations=np.zeros((1, 1, 2)),
        event_locations=np.zeros((1, 1, 2)), activation_probabilities=np.ones((1, 1)))
    import pytest
    with pytest.raises(KeyError, match='attempts'):
        dataset.load_activity_samples(path)
