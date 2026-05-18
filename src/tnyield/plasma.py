"""Plasma and radiation-transport primitives from Ch. 10.

Includes:
  - mean free paths for fusion neutrons and Planck-averaged radiation
    in DT and LiDT (Eqs. 10.10-10.13);
  - blackbody energy and pressure helpers used in the ablation /
    radiation-case calculations of Section 10.4.2;
  - rho-R thresholds for self-sustained thermonuclear propagation,
    quoted by Barroso from Fraley et al.
"""

from __future__ import annotations

from .constants import BOLTZMANN_k, KEV_TO_K, RADIATION_CONST_a


# ---------- mean free paths (Eqs. 10.10-10.13) ----------------------------


def lambda_n_DT(rho_g_per_cm3: float) -> float:
    """14-MeV neutron mean free path in DT (cm). Eq. (10.10)."""
    if rho_g_per_cm3 <= 0.0:
        return float("inf")
    return 4.6 / rho_g_per_cm3


def lambda_p_DT(T_keV: float, rho_g_per_cm3: float) -> float:
    """Planck-mean radiation MFP in DT (cm). Eq. (10.11)."""
    if rho_g_per_cm3 <= 0.0 or T_keV <= 0.0:
        return float("inf")
    return 14.0 * T_keV ** 3.5 / rho_g_per_cm3 ** 2


def lambda_n_LiDT(rho_g_per_cm3: float) -> float:
    """Fusion-neutron mean free path in LiDT (cm). Eq. (10.12)."""
    if rho_g_per_cm3 <= 0.0:
        return float("inf")
    return 3.0 / rho_g_per_cm3


def lambda_p_LiDT(T_keV: float, rho_g_per_cm3: float) -> float:
    """Planck-mean radiation MFP in LiDT (cm). Eq. (10.13)."""
    if rho_g_per_cm3 <= 0.0 or T_keV <= 0.0:
        return float("inf")
    return 5.1 * T_keV ** 3.5 / rho_g_per_cm3 ** 2


# ---------- rho-R thresholds (Section 10.2.2 / 10.2.3) --------------------

# Above these values of (rho * R) the named energy-loss channel is
# substantially trapped, so the thermonuclear reaction can become
# self-sustaining. Quoted directly from the book.

RHOR_NEUTRON_REABSORPTION_DT = 4.6   # g/cm^2; lambda_n / R = 1 in DT
RHOR_NEUTRON_REABSORPTION_LIDT = 3.0  # g/cm^2; lambda_n / R = 1 in LiDT
RHOR_LIDT_PROPAGATION = 2.2          # g/cm^2; Fraley threshold for burn
RHOR_LIDT_HOTSPOT_IGNITION = 10.0    # g/cm^2; hot-spot requirement


def burn_fraction_lindl(rho_R_g_per_cm2: float, H_B_g_per_cm2: float = 7.0) -> float:
    """Approximate thermonuclear burn fraction f = rho-R / (rho-R + H_B).

    The Lindl burn-up parameterization, widely used as a back-of-envelope
    estimator for high-density, near-ignition DT or LiDT plasmas. The
    Barroso book does not give this formula explicitly but does discuss
    the rho-R thresholds it captures (Section 10.2.3).  `H_B = 7 g/cm^2`
    is the standard DT value (Lindl 1998); use ~ 20 g/cm^2 for cold
    LiD.  Returns 0 if rho-R is non-positive.
    """
    if rho_R_g_per_cm2 <= 0.0 or H_B_g_per_cm2 <= 0.0:
        return 0.0
    return rho_R_g_per_cm2 / (rho_R_g_per_cm2 + H_B_g_per_cm2)


# ---------- equation of state for the radiation case (Section 10.4.2) -----


def plasma_energy_density(T_K: float, N_cm3: float, Z: int) -> float:
    """Internal energy density of a fully-ionized ideal plasma + blackbody
    radiation, in erg/cm^3.

        u = (3/2) (Z + 1) N k T  +  a T^4

    `N` is the ion number density, `Z` the ion charge state. From the
    energy equation in Section 10.4.2 (page 318).
    """
    if T_K <= 0.0 or N_cm3 <= 0.0:
        return 0.0
    return 1.5 * (Z + 1) * N_cm3 * BOLTZMANN_k * T_K + RADIATION_CONST_a * T_K ** 4


def plasma_pressure(T_K: float, N_cm3: float, Z: int) -> float:
    """Total pressure of a fully-ionized plasma + blackbody radiation, in
    erg/cm^3 (= dyn/cm^2; 1 Mbar = 1e12 dyn/cm^2).

        P = (Z + 1) N k T  +  a T^4 / 3

    From the pressure equation in Section 10.4.2.
    """
    if T_K <= 0.0 or N_cm3 <= 0.0:
        return 0.0
    return (Z + 1) * N_cm3 * BOLTZMANN_k * T_K + RADIATION_CONST_a * T_K ** 4 / 3.0


def temperature_for_energy(
    E_erg: float,
    V_cm3: float,
    N_cm3: float,
    Z: int,
    *,
    T_low_K: float = 1e4,
    T_high_K: float = 1e10,
    tol: float = 1e-3,
) -> float:
    """Invert `plasma_energy_density` to solve for T (K) given total
    energy `E_erg` deposited in volume `V_cm3` of a plasma with ion
    density `N_cm3` and charge state `Z`.  Bisection.

    Used in the Section 10.4.2 radiation-case example: 5 kt deposited in
    a Li diffuser at solid density yields ~21e6 K.
    """
    if E_erg <= 0.0 or V_cm3 <= 0.0:
        return 0.0
    target = E_erg / V_cm3
    lo, hi = T_low_K, T_high_K
    if plasma_energy_density(lo, N_cm3, Z) >= target:
        return lo
    if plasma_energy_density(hi, N_cm3, Z) <= target:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        u = plasma_energy_density(mid, N_cm3, Z)
        if u > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * mid:
            break
    return 0.5 * (lo + hi)


def ablation_pressure_from_energy(
    E_kt: float,
    V_cm3: float,
    N_cm3: float,
    Z: int,
) -> tuple[float, float]:
    """Convenience wrapper for the Section 10.4.2 worked example.

    Given energy `E_kt` (in kilotons of TNT) deposited in a uniform
    volume `V_cm3` of a plasma with ion density `N_cm3` and ionization
    `Z`, returns (T_K, P_Mbar): the equilibrium temperature and the
    total (plasma + radiation) pressure.

    Worked example from the book: 5 kt in 9.14e4 cm^3 of Li (Z=3,
    N = 5e22 atoms/cm^3) gives T = 21e6 K and P = 1070 Mbar.
    """
    from .constants import ERG_PER_KT
    E_erg = E_kt * ERG_PER_KT
    T_K = temperature_for_energy(E_erg, V_cm3, N_cm3, Z)
    P_dyn_per_cm2 = plasma_pressure(T_K, N_cm3, Z)
    P_Mbar = P_dyn_per_cm2 / 1e12
    return T_K, P_Mbar


def kelvin_to_keV(T_K: float) -> float:
    """Convert temperature in Kelvin to keV (T_keV = k_B T)."""
    return T_K / KEV_TO_K


def keV_to_kelvin(T_keV: float) -> float:
    """Convert temperature in keV to Kelvin."""
    return T_keV * KEV_TO_K
