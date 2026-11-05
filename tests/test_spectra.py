import numpy as np

from casmi.spectra import clean, entropy_similarity


def test_entropy_similarity_bounds():
    mz, it = np.array([100.0, 150.0, 200.0]), np.array([1.0, 0.5, 0.2])
    assert abs(entropy_similarity(mz, it, mz, it) - 1.0) < 1e-9
    assert entropy_similarity(mz, it, mz + 5, it) == 0.0
    s = entropy_similarity(mz, it, np.array([100.0, 150.0, 300.0]), it)
    assert 0.0 < s < 1.0
    assert abs(entropy_similarity(mz, it, mz - 14.01565, it, shift=14.01565) - 1.0) < 1e-9


def test_clean():
    mz, it = clean([50, 100, 400, 403], [0.001, 1.0, 0.5, 0.3], precursor_mz=400.0)
    assert list(mz) == [100, 400] and list(it) == [1.0, 0.5]
