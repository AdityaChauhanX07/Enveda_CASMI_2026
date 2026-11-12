import numpy as np

from casmi.network import Spec, modified_cosine, prepare


def _spec(mz, it, prec, name="m"):
    a, w = prepare(np.array(mz), np.array(it), prec)
    return Spec(name, "[M+H]+", "positive", prec, a, w)


def test_identical_spectra_score_one():
    s = _spec([100.0, 150.0, 200.0, 250.0], [1.0, 0.5, 0.8, 0.2], 300.0)
    score, n = modified_cosine(s, s)
    assert abs(score - 1.0) < 1e-9 and n == 4


def test_shifted_fragments_match_by_precursor_difference():
    a = _spec([100.0, 150.0, 200.0, 250.0], [1.0, 0.5, 0.8, 0.2], 300.0, "a")
    b = _spec([100.0, 164.016, 214.016, 264.016], [1.0, 0.5, 0.8, 0.2], 314.016, "b")  # +CH2 on 3 fragments
    score, n = modified_cosine(a, b)
    assert n == 4 and score > 0.99


def test_unrelated_spectra_score_low():
    a = _spec([100.0, 150.0, 200.0], [1.0, 1.0, 1.0], 300.0, "a")
    b = _spec([101.5, 157.3, 203.9], [1.0, 1.0, 1.0], 320.0, "b")
    assert modified_cosine(a, b)[0] == 0.0
