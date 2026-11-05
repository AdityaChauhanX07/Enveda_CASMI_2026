import pandas as pd
import pytest

from casmi.adducts import TEST_ADDUCTS, neutral_mass, precursor_mz
from casmi.metric import ik14, score
from casmi.submission import finalize


def S(ids, smi):
    return pd.DataFrame({"molecule_id": ids, "smiles": smi})


def test_official_examples():
    sol = S(["m_1", "m_2"], ["CCO", "c1ccccc1"])
    assert score(sol, S(["m_1", "m_2"], ["CCO;CC(=O)O", "CC(=O)O;c1ccccc1"])) == 0.75
    assert score(sol, S(["m_1", "m_2"], ["CC(=O)O", "c1ccccc1;c1ccccc1;c1ccccc1"])) == 0.5


def test_stereo_is_free():
    assert score(S(["m"], ["C[C@H](N)C(=O)O"]), S(["m"], ["CC(N)C(=O)O"])) == 1.0
    # stereocentre pinning an enone: must match only because stereo is stripped first
    assert score(S(["m"], ["C=C(C)C1CCC2CCC(=O)C=C2C1"]), S(["m"], ["C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1"])) == 1.0


def test_duplicates_consume_rank_and_finalize_dedupes():
    sol = S(["m"], ["c1ccccc1"])
    assert score(sol, S(["m"], ["CCO;OCC;c1ccccc1"])) == pytest.approx(1 / 3)  # duplicate still counts
    fixed = finalize(["CCO", "OCC", "not_a_smiles", "c1ccccc1"])
    assert fixed == ["CCO", "c1ccccc1"]
    assert score(sol, S(["m"], [";".join(fixed)])) == 0.5


def test_keto_enol_same_key():
    assert ik14("CC(=O)C") == ik14("CC(O)=C")


def test_adduct_roundtrip():
    m = 302.042653  # quercetin
    for a in TEST_ADDUCTS:
        assert abs(neutral_mass(precursor_mz(m, a), a) - m) < 1e-9
    assert abs(precursor_mz(m, "[M+H]+") - 303.049929) < 2e-6
