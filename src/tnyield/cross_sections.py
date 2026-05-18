"""Maxwellian-averaged thermonuclear reaction rates <sigma v> in cm^3 / s.

All temperature-dependent fits are from Barroso, *A Fisica dos
Explosivos Nucleares*:

  - DT      (Eq. 10.40), valid 1-500 keV.
  - DD      (Eq. 10.48), sum of both branches (D-D->He3+n and D-D->T+p).
  - D-He3   (Eq. 10.49).

These are the same fits used by Barroso's LUI1-3T thermonuclear burn
code and reproduce standard published curves to within ~30 % over the
weapons-relevant range (10-100 keV).

Li-6(n, alpha)T microscopic cross sections at the two fusion-neutron
energies (14.1 MeV and 2.45 MeV) come from Section 10.2.3.
"""

from __future__ import annotations

import math


# ---------- thermonuclear <sigma v> rates ----------------------------------


def sigma_v_DT(T_keV: float) -> float:
    """<sigma v> for D + T -> He4 + n, in cm^3/s. Eq. (10.40), 1-500 keV."""
    if T_keV <= 0.0:
        return 0.0
    T = T_keV
    num = 3.05e-13 * math.exp(-17.7 * T ** -0.348)
    den = (1.0
           + 0.1554 * T ** (1.0 / 3.0)
           - 0.1418 * T ** (2.0 / 3.0)
           - 0.05 * T
           + 0.0364 * T ** (4.0 / 3.0))
    return num / den


def sigma_v_DD(T_keV: float) -> float:
    """<sigma v> for D + D (sum of both branches), in cm^3/s. Eq. (10.48)."""
    if T_keV <= 0.0:
        return 0.0
    T = T_keV
    if T >= 10.0:
        A, B = 9.59e-14, 21.37
    else:
        A, B = 2.83e-14, 18.78
    return A * T ** (-2.0 / 3.0) * math.exp(-B * T ** (-1.0 / 3.0))


def sigma_v_DHe3(T_keV: float) -> float:
    """<sigma v> for D + He3 -> He4 + p, in cm^3/s. Eq. (10.49).

    Piecewise: zero below 10 keV, constant 2e-16 above 100 keV.
    """
    if T_keV <= 0.0:
        return 0.0
    T = T_keV
    if T < 10.0:
        return 0.0
    if T <= 100.0:
        return 4.81e-12 * T ** (-2.0 / 3.0) * math.exp(-33.4 * T ** (-1.0 / 3.0))
    return 2e-16


# ---------- Li-6(n, alpha)T cross sections (Section 10.2.3) ---------------

SIGMA_LI6_NT_14MEV = 0.026e-24    # cm^2 (14.1 MeV neutron)
SIGMA_LI6_NT_2_45MEV = 0.25e-24   # cm^2 (2.45 MeV neutron)


def sigma_Li6_nT(E_n_MeV: float) -> float:
    """Microscopic Li-6(n, alpha)T cross section in cm^2 for fast neutrons.

    Two-point log-log interpolation between the 2.45 MeV and 14.1 MeV
    anchors from Section 10.2.3. Clamped outside the range. Not valid
    for thermal neutrons (where sigma scales as 1/v and reaches ~940
    barns at 0.025 eV).
    """
    if E_n_MeV >= 14.1:
        return SIGMA_LI6_NT_14MEV
    if E_n_MeV <= 2.45:
        return SIGMA_LI6_NT_2_45MEV
    log_E1, log_E2 = math.log(2.45), math.log(14.1)
    log_s1 = math.log(SIGMA_LI6_NT_2_45MEV)
    log_s2 = math.log(SIGMA_LI6_NT_14MEV)
    frac = (math.log(E_n_MeV) - log_E1) / (log_E2 - log_E1)
    return math.exp(log_s1 + frac * (log_s2 - log_s1))


# ---------- reaction-rate helper ------------------------------------------


def reaction_time(n_cm3: float, sigma_v_cm3_s: float) -> float:
    """Characteristic single-particle reaction time tau_r = 1 / (n <sigma v>).

    Eq. (10.7). `n` is the partner-particle number density (cm^-3),
    `sigma_v_cm3_s` is <sigma v> in cm^3/s.
    """
    if n_cm3 <= 0.0 or sigma_v_cm3_s <= 0.0:
        return float("inf")
    return 1.0 / (n_cm3 * sigma_v_cm3_s)
