"""Fusion-boosted fission: Ch. 9 of Barroso.

Boosting introduces a small amount of deuterium-tritium gas (typically a
few grams) at the centre of a fissile pit.  At the temperatures reached
during the fission chain (tens of millions of degrees, Fig. 4-1) the DT
fuses; each fusion releases a 14.1 MeV neutron, and per unit mass DT
fusion releases 16x as many neutrons as fission (book, line 9108).

These fast neutrons reduce alpha (the inverse of the chain-multiplication
time) and let the fissile material burn for longer before hydrodynamic
disassembly: a small fusion energy yield (typically << 1 kt) bootstraps
a large additional fission yield (often 10x-30x the unboosted value).

The book gives the closed-form DT burn equations:

  Eq. (9.1): dn/dt = -(1/2) n^2 <sigma v>_DT
  Eq. (9.2): f = 1 - 1 / (1 + n_0 <sigma v>_DT tau / 2)
  Eq. (9.3): N_n = n_0 f / 2          (neutrons per cm^3)

It also tabulates empirical boost ratios (Table 9-2), but does not give
a closed-form expression for the boost factor itself.  This module:

  1. Computes f, N_n, and the direct fusion energy from (9.1)-(9.3).
  2. Estimates the *additional* fission yield those neutrons induce,
     using a simple "extra fissions per fusion neutron" multiplier.
     The user supplies that multiplier; a default of 8 reproduces the
     R = 5.2 cm Pu sphere of Table 9-2 (10.8 kt from 0.3 kt baseline)
     to better than a factor of 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import AVOGADRO, ERG_PER_KT, ERG_PER_MEV
from .cross_sections import sigma_v_DT
from .reactions import A_DT_EQUIMOLAR, Q_DT, Q_FISSION


# Empirical default: each fast fusion neutron causes ~8 additional
# fissions in a surrounding near-critical pit.  Calibrated against the
# R = 5.2 cm Pu sphere row of Table 9-2 (Q_F = 200 MeV, baseline yield
# 0.3 kt -> boosted 10.8 kt with 36 % DT burn of 8.48 g; the implied
# multiplier is ~8).  Real boosted weapons may use values from 5-20.
DEFAULT_EXTRA_FISSIONS_PER_FUSION_NEUTRON = 8.0


@dataclass(frozen=True)
class BoostResult:
    """Result of a boosting calculation."""

    burn_fraction_DT: float          # Eq. (9.2), dimensionless
    n_fusions: float                 # number of D+T -> He4 + n events
    n_fusion_neutrons: float         # = n_fusions (one neutron per reaction)
    Y_fusion_kt: float               # direct fusion energy in kt TNT
    Y_added_fission_kt: float        # extra fission energy induced by fusion neutrons
    Y_baseline_kt: float             # user-supplied unboosted fission yield
    Y_total_kt: float                # baseline + fusion + added fission
    boost_ratio: float               # Y_total / Y_baseline

    def __str__(self) -> str:
        return (
            f"DT burn fraction         : {self.burn_fraction_DT * 100:.2f} %\n"
            f"Total DT fusions         : {self.n_fusions:.3e}\n"
            f"14.1 MeV fusion neutrons : {self.n_fusion_neutrons:.3e}\n"
            f"Direct fusion yield      : {self.Y_fusion_kt:.4g} kt\n"
            f"Added fission yield      : {self.Y_added_fission_kt:.4g} kt\n"
            f"Baseline fission yield   : {self.Y_baseline_kt:.4g} kt\n"
            f"Boosted total yield      : {self.Y_total_kt:.4g} kt\n"
            f"Boost ratio              : {self.boost_ratio:.3g} x"
        )


def DT_burn_fraction(
    rho_DT_g_per_cm3: float,
    T_keV: float,
    tau_s: float,
) -> float:
    """DT burn fraction at confinement time `tau_s`.  Eq. (9.2).

    Solves the rate equation (9.1) assuming uniform temperature and
    density during confinement:

        f = 1 - 1 / (1 + n_0 <sigma v>_DT tau / 2)

    where n_0 is the total DT particle density (D + T) in cm^-3.
    """
    if rho_DT_g_per_cm3 <= 0.0 or T_keV <= 0.0 or tau_s <= 0.0:
        return 0.0
    n0 = rho_DT_g_per_cm3 * AVOGADRO / A_DT_EQUIMOLAR
    sv = sigma_v_DT(T_keV)
    denom = 1.0 + n0 * sv * tau_s / 2.0
    return 1.0 - 1.0 / denom


def DT_neutron_yield(
    mass_DT_g: float,
    rho_DT_g_per_cm3: float,
    T_keV: float,
    tau_s: float,
) -> tuple[float, float]:
    """Total number of DT fusions and fusion neutrons for a mass of DT.

    Returns (N_fusions, N_neutrons).  Since each D+T -> He4 + n event
    releases exactly one neutron, the two are equal.  Combines Eqs.
    (9.2) and (9.3) integrated over the DT volume.
    """
    if mass_DT_g <= 0.0:
        return 0.0, 0.0
    f = DT_burn_fraction(rho_DT_g_per_cm3, T_keV, tau_s)
    # Total DT particles: m / A_DT * N_A (each particle is either a D
    # or a T atom in the equimolar mix).  Each fusion consumes one D
    # and one T -> 2 particles, so N_fusions = N_particles * f / 2.
    n_particles_total = mass_DT_g * AVOGADRO / A_DT_EQUIMOLAR
    n_fusions = n_particles_total * f / 2.0
    return n_fusions, n_fusions


def boosted_yield(
    *,
    mass_DT_g: float,
    rho_DT_g_per_cm3: float,
    T_keV: float,
    tau_s: float,
    Y_baseline_kt: float,
    extra_fissions_per_fusion_neutron: float = DEFAULT_EXTRA_FISSIONS_PER_FUSION_NEUTRON,
) -> BoostResult:
    """Estimate the boosted fission yield from DT params and a baseline.

    Parameters
    ----------
    mass_DT_g : grams of equimolar DT initially loaded into the pit.
    rho_DT_g_per_cm3 : DT mass density at the moment of compression
        (book's worked example uses ~ 0.6 g/cm^3, ~ 3x solid DT).
    T_keV : peak ion temperature inside the DT gas during the chain
        (book uses ~ 4 keV ~ 50 MK in its primary example, rising
        to ~ 20 keV when self-heating is considered).
    tau_s : effective confinement time of the DT (~ 1e-7 s book value
        for the primary example).
    Y_baseline_kt : the unboosted fission yield of the same pit, in kt.
        Compute this with the `fissionyield` package if desired.
    extra_fissions_per_fusion_neutron : empirical multiplier.  Each
        fast fusion neutron causes this many additional fissions in
        the surrounding fissile material before disassembly.  See
        module docstring.

    Returns
    -------
    BoostResult dataclass with the full energy decomposition.
    """
    n_fusions, n_neutrons = DT_neutron_yield(mass_DT_g, rho_DT_g_per_cm3, T_keV, tau_s)
    f = DT_burn_fraction(rho_DT_g_per_cm3, T_keV, tau_s)

    Y_fusion_kt = n_fusions * Q_DT * ERG_PER_MEV / ERG_PER_KT

    extra_fissions = n_neutrons * extra_fissions_per_fusion_neutron
    Y_added_fission_kt = extra_fissions * Q_FISSION * ERG_PER_MEV / ERG_PER_KT

    Y_total = Y_baseline_kt + Y_fusion_kt + Y_added_fission_kt
    ratio = Y_total / Y_baseline_kt if Y_baseline_kt > 0 else float("inf")

    return BoostResult(
        burn_fraction_DT=f,
        n_fusions=n_fusions,
        n_fusion_neutrons=n_neutrons,
        Y_fusion_kt=Y_fusion_kt,
        Y_added_fission_kt=Y_added_fission_kt,
        Y_baseline_kt=Y_baseline_kt,
        Y_total_kt=Y_total,
        boost_ratio=ratio,
    )
