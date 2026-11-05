"""Adduct table: neutral mass <-> precursor m/z. Monoisotopic masses, electron mass included."""
from __future__ import annotations

PROTON = 1.007276467
ELECTRON = 0.000548580
H2O = 18.010564684
NH3 = 17.026549101
NA, K, CL = 22.989769282, 38.963706486, 34.968852682
FORMIC, ACETIC = 46.005479303, 60.021129367

# name -> (n_M, charge, shift) with  mz = (n_M * M + shift) / |charge|
ADDUCTS: dict[str, tuple[int, int, float]] = {
    # the ten test adducts
    "[M+H]+": (1, 1, PROTON),
    "[M+NH4]+": (1, 1, NH3 + PROTON),
    "[M-H2O+H]+": (1, 1, -H2O + PROTON),
    "[M-2H2O+H]+": (1, 1, -2 * H2O + PROTON),
    "[M+Na]+": (1, 1, NA - ELECTRON),
    "[M+K]+": (1, 1, K - ELECTRON),
    "[M-H]-": (1, -1, -PROTON),
    "[M-H2O-H]-": (1, -1, -H2O - PROTON),
    "[M+CH2O2-H]-": (1, -1, FORMIC - PROTON),
    "[M+Cl]-": (1, -1, CL + ELECTRON),
    # frequent extra adducts in train
    "[M]+": (1, 1, -ELECTRON),
    "[M]-": (1, -1, ELECTRON),
    "[M+H-H2O]+": (1, 1, -H2O + PROTON),
    "[M+C2H4O2-H]-": (1, -1, ACETIC - PROTON),
    "[M+2H]2+": (1, 2, 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON),
    "[2M+Na]+": (2, 1, NA - ELECTRON),
    "[2M+NH4]+": (2, 1, NH3 + PROTON),
    "[2M-H]-": (2, -1, -PROTON),
}
TEST_ADDUCTS = list(ADDUCTS)[:10]


def neutral_mass(precursor_mz: float, adduct: str) -> float | None:
    a = ADDUCTS.get(adduct)
    if a is None:
        return None
    n, z, shift = a
    return (precursor_mz * abs(z) - shift) / n


def precursor_mz(mass: float, adduct: str) -> float | None:
    a = ADDUCTS.get(adduct)
    if a is None:
        return None
    n, z, shift = a
    return (n * mass + shift) / abs(z)


def ppm(observed: float, theoretical: float) -> float:
    return (observed - theoretical) / theoretical * 1e6
