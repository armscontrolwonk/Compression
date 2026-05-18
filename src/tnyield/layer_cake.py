"""Layer-cake / Sloika fission-fusion-fission energy decomposition.

The "Sloika" (layer cake) is a thin-tamper design in which a fission
primary is surrounded by a lithium-deuteride layer and then a fast-
fissionable outer shell (typically U-238).  Three energy channels:

  1. Primary fission: Y_p (user supplied; compute with `fissionyield`).
  2. LiD fusion: primary neutrons hit Li-6 to make tritium (Eq. 10.14),
     and the tritium fuses with the deuterium also in the LiD layer
     (D + T -> He4 + n, Eq. 10.2).  Net energy per Li6+D pair burnt is
     Q_Li6_nT + Q_DT ~ 22.4 MeV.
  3. U-238 fast fission: the 14.1 MeV fusion neutrons (plus some
     scattered primary fission neutrons) fast-fission the U-238 tamper.

Barroso does not give a dedicated Sloika model -- the book focuses on
modern Teller-Ulam two-stage designs (Section 10.4 onwards) -- but does
discuss the three-phase fission-fusion-fission energy decomposition
(section 10.4.2, page 319: "Daí a importância do processo trifásico de
fissão-fusão-fissão nos explosivos termonucleares de alta potência").
The numbers used in this module come from that section and from the
W-87 worked example (Table 10-11).
"""

from __future__ import annotations

from dataclasses import dataclass

from .reactions import (
    KT_PER_G_LI6D,
    KT_PER_G_U238_FISSION,
    M_D,
    M_LI6,
    M_LI7,
)


# Default burn fractions for a layered (Sloika-style) device.  These are
# representative of historical "RDS-6s" / "Joe-4" -type devices and the
# discussion in Barroso Section 10.4.2: a thin LiD layer driven by a
# fission primary burns only a small fraction of its fuel because rho-R
# is below the propagation threshold (Eq. 10.10, RHOR_LIDT_PROPAGATION).
# The U-238 tamper captures roughly 15 % of the fusion neutrons (book,
# line 12586).
DEFAULT_LID_BURN_FRACTION = 0.20         # 20 % typical for a thin layer
DEFAULT_U238_BURN_FRACTION = 0.15        # ~15 % of fusion neutrons cause fast fission


@dataclass(frozen=True)
class LayerCakeResult:
    """Energy decomposition of a layer-cake (Sloika) device, in kt."""

    Y_primary_kt: float
    Y_fusion_kt: float
    Y_tamper_kt: float
    Y_total_kt: float
    fraction_primary: float
    fraction_fusion: float
    fraction_tamper: float
    LiD_burn_fraction: float
    U238_burn_fraction: float

    def __str__(self) -> str:
        return (
            f"Primary fission yield : {self.Y_primary_kt:8.3f} kt"
            f" ({self.fraction_primary*100:5.1f} %)\n"
            f"LiD fusion yield      : {self.Y_fusion_kt:8.3f} kt"
            f" ({self.fraction_fusion*100:5.1f} %)\n"
            f"U-238 tamper fission  : {self.Y_tamper_kt:8.3f} kt"
            f" ({self.fraction_tamper*100:5.1f} %)\n"
            f"Total yield           : {self.Y_total_kt:8.3f} kt\n"
            f"LiD burn fraction     : {self.LiD_burn_fraction*100:5.1f} %\n"
            f"U-238 burn fraction   : {self.U238_burn_fraction*100:5.1f} %"
        )


def lid_yield_per_kg(li6_enrichment: float = 0.95) -> float:
    """Per-kg energy of fully burnt LiD, in kt/kg, as a function of Li-6
    enrichment (fraction).  Pure Li-6 D burns via Li6+n->T+alpha then
    D+T->He4+n (net ~22.4 MeV per pair, ~67 kt/kg).  Li-7 contributes
    much less below the (n, n'+alpha+T) threshold of ~3.5 MeV; treated
    here as a dead mass.
    """
    if not 0.0 <= li6_enrichment <= 1.0:
        raise ValueError("li6_enrichment must be in [0, 1]")
    # Mass-weighted average atomic mass of the lithium component.
    A_Li_avg = li6_enrichment * M_LI6 + (1.0 - li6_enrichment) * M_LI7
    # Active mass fraction = (Li-6 mass + D mass) / (Li_avg mass + D mass)
    active_mass_frac = (li6_enrichment * M_LI6 + M_D) / (A_Li_avg + M_D)
    return KT_PER_G_LI6D * 1000.0 * active_mass_frac


def layer_cake_yield(
    *,
    Y_primary_kt: float,
    mass_LiD_kg: float,
    mass_U238_kg: float,
    li6_enrichment: float = 0.95,
    LiD_burn_fraction: float = DEFAULT_LID_BURN_FRACTION,
    U238_burn_fraction: float = DEFAULT_U238_BURN_FRACTION,
) -> LayerCakeResult:
    """Compute the yield decomposition of a Sloika-type device.

    Parameters
    ----------
    Y_primary_kt : yield of the fission primary alone (compute with
        the `fissionyield` package or measure).
    mass_LiD_kg : total mass of lithium-deuteride in the LiD layer.
    mass_U238_kg : total mass of U-238 (or U-natural ~ 99.3 % U-238)
        in the outer tamper.
    li6_enrichment : Li-6 atomic fraction of the lithium.  Natural Li
        is 7.5 %, Sloika designs used 20-40 %, modern thermonuclears
        use up to 95 %.
    LiD_burn_fraction : fraction of the active Li6D mass that burns.
        Realistic values for a thin pure-Sloika layer are 5-25 %.
    U238_burn_fraction : fraction of the tamper U-238 that undergoes
        fast fission from fusion neutrons.  Typical values 5-20 %.
    """
    if Y_primary_kt < 0:
        raise ValueError("Y_primary_kt must be >= 0")
    if mass_LiD_kg < 0 or mass_U238_kg < 0:
        raise ValueError("masses must be >= 0")
    if not 0.0 <= LiD_burn_fraction <= 1.0:
        raise ValueError("LiD_burn_fraction must be in [0, 1]")
    if not 0.0 <= U238_burn_fraction <= 1.0:
        raise ValueError("U238_burn_fraction must be in [0, 1]")

    Y_fusion = lid_yield_per_kg(li6_enrichment) * mass_LiD_kg * LiD_burn_fraction
    Y_tamper = KT_PER_G_U238_FISSION * 1000.0 * mass_U238_kg * U238_burn_fraction

    Y_total = Y_primary_kt + Y_fusion + Y_tamper
    if Y_total > 0:
        fp = Y_primary_kt / Y_total
        ff = Y_fusion / Y_total
        ft = Y_tamper / Y_total
    else:
        fp = ff = ft = 0.0

    return LayerCakeResult(
        Y_primary_kt=Y_primary_kt,
        Y_fusion_kt=Y_fusion,
        Y_tamper_kt=Y_tamper,
        Y_total_kt=Y_total,
        fraction_primary=fp,
        fraction_fusion=ff,
        fraction_tamper=ft,
        LiD_burn_fraction=LiD_burn_fraction,
        U238_burn_fraction=U238_burn_fraction,
    )
