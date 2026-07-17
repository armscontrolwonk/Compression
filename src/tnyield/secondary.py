"""Teller-Ulam secondary: radiation-driven implosion of a LiD capsule.

Barroso describes the standard modern thermonuclear architecture in
Sections 10.4 - 10.7 (with the W-87 worked example in Section 10.7,
Table 10-11).  The secondary is a cylinder or sphere of lithium
deuteride, surrounded by a U-natural tamper, with a small U-235 spark
plug at its centre.  When the primary detonates:

  1. X-rays from the primary fill the radiation case and heat the
     low-Z diffuser to ~ 21e6 K (Section 10.4.2 worked example).
  2. The resulting blackbody-plus-plasma pressure (~ 1000 Mbar)
     ablates the tamper, driving the LiD inwards.
  3. The shock compresses the secondary to ~ 10x-100x solid density;
     the spark plug fissions and provides the seed alpha-particle
     energy that ignites the LiD.
  4. Li-6 captures spark-plug neutrons to make tritium (Eq. 10.14);
     D + T -> He4 + n then propagates through the LiD.
  5. 14 MeV fusion neutrons fast-fission the U-238 tamper.

This module composes the closed-form pieces:

  - Section 10.4.2 ablation-temperature / ablation-pressure equations
    (from `plasma.ablation_pressure_from_energy`);
  - the Lindl burn fraction f = rho-R / (rho-R + H_B) as a function
    of compressed rho-R (`plasma.burn_fraction_lindl`);
  - the per-kg energies KT_PER_G_LI6D and KT_PER_G_U238_FISSION.

It is *not* a hydrocode and does not solve the 3-temperature equations
of Section 10.3.  It is a back-of-envelope estimator in the spirit of
the `fissionyield` Cochran model.

The LiD burn fraction is grounded to Barroso's own 3-temperature
simulations, not guessed (see `plasma.lid_burn_fraction`). His W-87
secondary (Table 10-11) burns ~90% of the D at a compressed rho-R
~150 g/cm^2, which the Lindl form reproduces with H_B ~ 20 g/cm^2. A
competent, well-tamped, Li-6-rich secondary therefore burns MOST of a
compact fuel charge -- ~50-90%, not a few percent.

Two important caveats Barroso documents:
  - The tamper is essential: a tamperless charge collapses to ~5% burn
    (Table 10-9) because energy leaks out before the fuel reacts.
  - A LiD charge much larger than the ~2-3 kg W-87 load is "bastante
    mais ineficiente" (Section 10.6): the burn wave cannot cross the
    whole mass before disassembly, so the Lindl value is an UPPER
    BOUND for big charges. For those, pass `LiD_burn_fraction`
    explicitly or band it down toward the Table 10-9 regime (~35%).

(An earlier version of this model assumed ~3.5% here; that came from
pairing the W-87 fusion yield with the LiD mass of a *different*, much
larger Barroso example, and was wrong by ~20x.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .plasma import H_B_LID_UNTAMPED, burn_fraction_lindl
from .reactions import (
    KT_PER_G_LI6D,
    KT_PER_G_U235_FISSION,
    KT_PER_G_U238_FISSION,
)


# Default Lindl burn-fraction parameters (see module docstring).
# H_B=20 for tamped LiD is calibrated to Barroso's W-87 (Table 10-11):
# ~90% D burn at compressed rho-R ~150 g/cm^2.
DEFAULT_H_B_LID = 20.0    # g/cm^2 for tamped LiD secondary (W-87-anchored)
DEFAULT_H_B_DT = 7.0      # g/cm^2 for pure DT (Lindl 1998)
# Mass burn-up fraction of the U-238 tamper from fast fission by
# 14 MeV fusion neutrons.  Thick Teller-Ulam tampers fission a smaller
# fraction of their bulk mass (~ 1 %) than thin Sloika-style layers
# (~ 10 %), because their inner-shell neutrons are not always able to
# escape the tamper before fissioning a U-238 atom.  Default tuned to
# match the W-87 numbers in Table 10-11 (~ 0.2-1 % per tamper layer).
DEFAULT_U238_BURN_FRACTION = 0.02
DEFAULT_SPARK_PLUG_BURN_FRACTION = 0.20
DEFAULT_XRAY_FRACTION = 0.80  # fraction of primary yield emerging as X-rays


@dataclass(frozen=True)
class SecondaryResult:
    """Yield decomposition for a Teller-Ulam secondary (kt)."""

    Y_primary_kt: float
    Y_spark_plug_kt: float
    Y_fusion_kt: float
    Y_tamper_kt: float
    Y_total_kt: float

    compression: float        # rho / rho_0 of the LiD at peak
    rho_R_g_per_cm2: float    # compressed product
    burn_fraction_LiD: float  # f = rho-R / (rho-R + H_B)

    fraction_primary: float
    fraction_spark_plug: float
    fraction_fusion: float
    fraction_tamper: float

    def __str__(self) -> str:
        return (
            f"Primary fission           : {self.Y_primary_kt:9.3f} kt"
            f" ({self.fraction_primary*100:5.1f} %)\n"
            f"Spark-plug fission        : {self.Y_spark_plug_kt:9.3f} kt"
            f" ({self.fraction_spark_plug*100:5.1f} %)\n"
            f"LiD fusion                : {self.Y_fusion_kt:9.3f} kt"
            f" ({self.fraction_fusion*100:5.1f} %)\n"
            f"U-238 tamper fast fission : {self.Y_tamper_kt:9.3f} kt"
            f" ({self.fraction_tamper*100:5.1f} %)\n"
            f"Total yield               : {self.Y_total_kt:9.3f} kt\n"
            f"  LiD compression rho/rho_0 : {self.compression:.2f}\n"
            f"  Compressed rho*R          : {self.rho_R_g_per_cm2:.3f} g/cm^2\n"
            f"  LiD burn fraction         : {self.burn_fraction_LiD*100:5.1f} %"
        )


def _sphere_volume(R_cm: float) -> float:
    return 4.0 / 3.0 * math.pi * R_cm ** 3


def _sphere_radius_from_mass(mass_g: float, rho_g_per_cm3: float) -> float:
    if mass_g <= 0.0 or rho_g_per_cm3 <= 0.0:
        return 0.0
    V = mass_g / rho_g_per_cm3
    return (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)


def secondary_yield(
    *,
    Y_primary_kt: float,
    mass_LiD_kg: float,
    mass_U238_tamper_kg: float,
    mass_spark_plug_kg: float = 0.0,
    li6_enrichment: float = 0.95,
    rho_LiD_solid: float = 0.82,
    compression: float = 100.0,
    LiD_burn_fraction: float | None = None,
    U238_burn_fraction: float = DEFAULT_U238_BURN_FRACTION,
    spark_plug_burn_fraction: float = DEFAULT_SPARK_PLUG_BURN_FRACTION,
    H_B_LiD: float = DEFAULT_H_B_LID,
) -> SecondaryResult:
    """Estimate the total yield and decomposition of a Teller-Ulam
    secondary stage.

    Parameters
    ----------
    Y_primary_kt : fission primary yield (kt). Drives the radiation
        case temperature; carried into the total as-is.
    mass_LiD_kg : total LiD mass in the secondary fuel charge.
    mass_U238_tamper_kg : U-238 (or U-natural) tamper mass surrounding
        the LiD.  Soaks up 14 MeV fusion neutrons and fast-fissions.
    mass_spark_plug_kg : U-235 or Pu-239 spark plug mass (often 0 for
        a pure-LiD design).
    li6_enrichment : fraction of Li that is Li-6 (0..1).
    rho_LiD_solid : solid density of LiD (g/cm^3; book uses ~0.82).
    compression : peak rho/rho_0 of the LiD during the burn.  Modern
        secondaries typically reach ~100x; Barroso's worked LiDT cases
        (Section 10.5) show ~20x-50x.  Used only when
        `LiD_burn_fraction` is None.
    LiD_burn_fraction : if supplied, use this value directly instead of
        the Barroso-grounded Lindl formula.  A well-tamped compact
        secondary burns ~50-90% (W-87 is ~90%); override toward the
        finite-media floor (~35%) for charges much larger than the
        ~2-3 kg W-87 fuel load (see module docstring).
    U238_burn_fraction : fraction of tamper U-238 that fast-fissions.
    spark_plug_burn_fraction : fraction of the spark-plug fissile mass
        that burns before disassembly.
    H_B_LiD : tamped Lindl burn-fraction scale (g/cm^2) for LiD, default
        20 (W-87-calibrated). Only used when `LiD_burn_fraction` is None;
        if no tamper is present the untamped collapse value is used
        instead (energy leaks out before the fuel reacts).

    Returns
    -------
    SecondaryResult dataclass.
    """
    if Y_primary_kt < 0:
        raise ValueError("Y_primary_kt must be >= 0")
    for name, val in (
        ("mass_LiD_kg", mass_LiD_kg),
        ("mass_U238_tamper_kg", mass_U238_tamper_kg),
        ("mass_spark_plug_kg", mass_spark_plug_kg),
    ):
        if val < 0:
            raise ValueError(f"{name} must be >= 0")
    if compression <= 0:
        raise ValueError("compression must be > 0")

    # Compressed LiD geometry.  Treat as a sphere of total LiD mass at
    # compressed density rho_c = compression * rho_solid.
    rho_c = compression * rho_LiD_solid
    R_c_cm = _sphere_radius_from_mass(mass_LiD_kg * 1000.0, rho_c)
    rho_R = rho_c * R_c_cm

    if LiD_burn_fraction is not None:
        if not 0.0 <= LiD_burn_fraction <= 1.0:
            raise ValueError("LiD_burn_fraction must be in [0, 1]")
        f_LiD = LiD_burn_fraction
    elif rho_R > 0:
        # Barroso-grounded: Lindl saturation, but the tamper is essential
        # (Table 10-9) -- an untamped charge collapses to ~5% burn.
        H_B_eff = H_B_LiD if mass_U238_tamper_kg > 0.0 else H_B_LID_UNTAMPED
        f_LiD = burn_fraction_lindl(rho_R, H_B_eff)
    else:
        f_LiD = 0.0

    # Per-kg energy of fully burnt LiD with the given Li-6 enrichment.
    # Inline the formula from `layer_cake.lid_yield_per_kg` to avoid an
    # import cycle.
    from .reactions import M_D, M_LI6, M_LI7
    A_Li_avg = li6_enrichment * M_LI6 + (1.0 - li6_enrichment) * M_LI7
    active_frac = (li6_enrichment * M_LI6 + M_D) / (A_Li_avg + M_D)
    Y_fusion = KT_PER_G_LI6D * 1000.0 * mass_LiD_kg * f_LiD * active_frac

    Y_tamper = KT_PER_G_U238_FISSION * 1000.0 * mass_U238_tamper_kg * U238_burn_fraction
    Y_spark = (
        KT_PER_G_U235_FISSION * 1000.0 * mass_spark_plug_kg * spark_plug_burn_fraction
    )

    Y_total = Y_primary_kt + Y_spark + Y_fusion + Y_tamper

    if Y_total > 0:
        fp = Y_primary_kt / Y_total
        fs = Y_spark / Y_total
        ff = Y_fusion / Y_total
        ft = Y_tamper / Y_total
    else:
        fp = fs = ff = ft = 0.0

    return SecondaryResult(
        Y_primary_kt=Y_primary_kt,
        Y_spark_plug_kt=Y_spark,
        Y_fusion_kt=Y_fusion,
        Y_tamper_kt=Y_tamper,
        Y_total_kt=Y_total,
        compression=compression,
        rho_R_g_per_cm2=rho_R,
        burn_fraction_LiD=f_LiD,
        fraction_primary=fp,
        fraction_spark_plug=fs,
        fraction_fusion=ff,
        fraction_tamper=ft,
    )
