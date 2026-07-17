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


# --- Grounded defaults for the DT state (Barroso Ch. 9 worked example) ---
# The DT density, temperature, and confinement time are NOT independent
# inputs in a real device: they are consequences of the primary's implosion
# and explosion, which Barroso obtains by running the coupled hydro +
# neutronics code (RIC1). We can't close that loop in a closed form, but we
# can derive grounded, self-consistent estimates from the primary's
# compression and geometry -- see `dt_density`, `dt_temperature`, and
# `dt_confinement_time`, which `tnyield.total.yield_kt_total` wires in from
# the composite primary. These fall-back constants are the book's
# primary-example values, used when no primary context is available.
DEFAULT_RHO_DT = 0.6      # g/cm^3  (~3x solid DT; Table 9-2)
DEFAULT_T_KEV = 4.0       # keV     (~50 MK; Fig. 4-1 / Section 9.1)
DEFAULT_TAU_S = 1e-7      # s       (~10 shakes; Section 9.1)

# Disassembly-limited confinement: tau ~ R_pit / u_disassembly, calibrated
# so Barroso's ~5 cm Pu pit gives tau ~ 1e-7 s.
U_DISASSEMBLY_CM_S = 5.0e7


def dt_density(eta: float, rho_dt_fill_g_per_cm3: float = 0.2) -> float:
    """Estimate DT gas density (g/cm^3) from the pit compression `eta`.

    The central DT is compressed roughly in step with the pit. Anchored
    to Barroso's ~0.6 g/cm^3 (~3x solid DT) for an un-imploded pit
    (eta=1) and scaled with compression. Rough -- the gas is far softer
    than the surrounding metal, so this is a grounded default, not a
    first-principles derivation.
    """
    if eta <= 0.0:
        return DEFAULT_RHO_DT
    return 3.0 * rho_dt_fill_g_per_cm3 * eta


def dt_temperature(eta: float) -> float:
    """Estimate DT ion temperature (keV) from the pit compression `eta`.

    Anchored to Barroso's ~4 keV at modest compression (eta ~ 2.5,
    Fig. 4-1), rising with compression and clamped to [3, 20] keV (the
    upper end reflects alpha self-heating, Section 9.1).
    """
    if eta <= 0.0:
        return DEFAULT_T_KEV
    return min(20.0, max(3.0, DEFAULT_T_KEV * eta / 2.5))


def dt_confinement_time(R_pit_cm: float) -> float:
    """Estimate DT confinement time (s) as the primary's disassembly time.

    tau ~ R_pit / u_disassembly. This is the same hydrodynamic
    disassembly the Cochran yield model integrates the chain over; a
    bigger pit confines the DT longer (more burn). Calibrated so a ~5 cm
    pit gives ~ 1e-7 s (Barroso, Section 9.1).
    """
    if R_pit_cm <= 0.0:
        return DEFAULT_TAU_S
    return R_pit_cm / U_DISASSEMBLY_CM_S


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
    rho_DT_g_per_cm3: float          # DT density actually used
    T_keV: float                     # DT temperature actually used
    tau_s: float                     # DT confinement time actually used

    def __str__(self) -> str:
        return (
            f"DT state used            : rho={self.rho_DT_g_per_cm3:.3g} g/cm^3, "
            f"T={self.T_keV:.3g} keV, tau={self.tau_s:.2g} s\n"
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
    Y_baseline_kt: float,
    rho_DT_g_per_cm3: float | None = None,
    T_keV: float | None = None,
    tau_s: float | None = None,
    extra_fissions_per_fusion_neutron: float = DEFAULT_EXTRA_FISSIONS_PER_FUSION_NEUTRON,
) -> BoostResult:
    """Estimate the boosted fission yield from DT params and a baseline.

    Parameters
    ----------
    mass_DT_g : grams of equimolar DT initially loaded into the pit.
    Y_baseline_kt : the unboosted fission yield of the same pit, in kt.
        Compute this with the `fissionyield` package if desired.
    rho_DT_g_per_cm3, T_keV, tau_s : the DT state at burn time. These are
        physically consequences of the primary (see module docstring), so
        they are OPTIONAL: when left None they fall back to Barroso's
        primary-example values (0.6 g/cm^3, 4 keV, 1e-7 s). For a
        self-consistent estimate derived from the primary's compression
        and geometry, call via `tnyield.total.yield_kt_total`, which wires
        in `dt_density`/`dt_temperature`/`dt_confinement_time`.
    extra_fissions_per_fusion_neutron : empirical multiplier.  Each
        fast fusion neutron causes this many additional fissions in
        the surrounding fissile material before disassembly.  See
        module docstring.

    Returns
    -------
    BoostResult dataclass with the full energy decomposition and the DT
    state actually used.
    """
    rho = DEFAULT_RHO_DT if rho_DT_g_per_cm3 is None else rho_DT_g_per_cm3
    T = DEFAULT_T_KEV if T_keV is None else T_keV
    tau = DEFAULT_TAU_S if tau_s is None else tau_s

    n_fusions, n_neutrons = DT_neutron_yield(mass_DT_g, rho, T, tau)
    f = DT_burn_fraction(rho, T, tau)

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
        rho_DT_g_per_cm3=rho,
        T_keV=T,
        tau_s=tau,
    )
