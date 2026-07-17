"""Crude device -> compression (eta) front-end, calibrated to Barroso.

EXPERIMENTAL / ORDER-OF-MAGNITUDE. Estimating the compression an implosion
actually achieves is precisely what Barroso does with a Lagrangian
radiation-hydrodynamics code (his Ch. 6 and 8): a convergent chemical
detonation drives a shock through the tamper into the fissile pit, with
real equations of state, shock focusing, rarefaction, and allotropic phase
changes. There is no honest closed form for it.

What this module does instead is fit a simple 3-parameter log law to
Barroso's OWN hydro results. His Tables 8-2 and 8-3 implode a 6 kg hollow
alpha-Pu pit (2 cm U-natural tamper) with variable Composite B and variable
tamper, and report the yield; we invert those yields to the effective
compression his implosion achieved (through the Cochran yield inverse,
`fissionyield.compression`). The resulting eta values (~1.9-2.6, rising
with both HE mass and tamper) are fit by

    eta ~ C0 + A*ln(m_HE_eq / m_fissile) + B*ln(1 + m_tamper / m_fissile)
              + geometry_offset

with C0=0.59, A=0.43, B=0.22 for Composite B and hollow alpha-Pu. This
reproduces the calibration set to +/- 0.06 in eta.

Three loud caveats:
  1. OFF the calibration set -- other pit masses, materials (HEU is stiffer
     and compresses less; delta-Pu is softer), geometries, or explosives --
     this is an extrapolation, i.e. a guess. The `geometry` offsets and the
     returned band are rough.
  2. Barroso reports only ~5% of the chemical energy reaches the fissile
     mass in spherical implosion, and that low implosion velocity gives
     modest compression -- both are baked into the calibration, not
     derived, so don't push the inputs far from weapon-like values.
  3. Yield is ~10x elastic in eta, so even a good eta estimate leaves the
     yield uncertain to ~2x. ALWAYS use the returned band, never the point.

The tamper's role is confinement, not inertia: in Barroso's Table 8-3 a
heavier tamper RAISES eta despite adding driven mass, so `B > 0`.
"""

from __future__ import annotations

import math
from collections import namedtuple

from .composite import yield_kt_composite
from .model import yield_kt

# --- calibration constants (Composite B, hollow alpha-Pu; Tables 8-2/8-3) --
_C0 = 0.59
_A = 0.43   # sensitivity to ln(HE mass / fissile mass)
_B = 0.22   # sensitivity to ln(1 + tamper mass / fissile mass); confinement

# Geometry offset in eta. Hollow is the calibration point (0.0); Barroso
# shows a solid pit compresses less and a porous/collapsing one more.
GEOMETRY_OFFSET = {"solid": -0.25, "hollow": 0.0, "porous": 0.10}

# Default band half-width in eta -- a floor on the model uncertainty for
# near-calibration devices; treat it as larger the further you extrapolate.
DEFAULT_ETA_UNCERTAINTY = 0.25

# Composite B is the reference explosive (Barroso item 5.1: rho=1.70 g/cm^3,
# D=8 km/s, P_CJ~292 kbar). `he_relative_power` expresses another explosive
# as a Composite-B-equivalent mass multiplier (roughly its detonation-energy
# or CJ-pressure ratio); 1.0 = Composite B.

ImplosionEstimate = namedtuple(
    "ImplosionEstimate", ["eta", "eta_low", "eta_high", "geometry"]
)


def implosion_compression(
    m_he_kg: float,
    m_fissile_kg: float,
    m_tamper_kg: float = 0.0,
    *,
    geometry: str = "hollow",
    he_relative_power: float = 1.0,
    eta_uncertainty: float = DEFAULT_ETA_UNCERTAINTY,
) -> ImplosionEstimate:
    """Crude estimate of the compression eta an implosion achieves.

    Parameters
    ----------
    m_he_kg : chemical high-explosive mass (kg). `he_relative_power`
        rescales it to a Composite-B-equivalent mass for other explosives.
    m_fissile_kg : total fissile pit mass (kg); for a composite pit use
        m_pu + m_heu.
    m_tamper_kg : tamper/reflector mass (kg), e.g. U-natural. Confines the
        implosion -- more tamper raises eta (Barroso Table 8-3).
    geometry : 'solid', 'hollow' (default, the calibration), or 'porous'.
    he_relative_power : Composite-B-equivalent power of the explosive.
    eta_uncertainty : band half-width in eta (see module caveats).

    Returns
    -------
    ImplosionEstimate(eta, eta_low, eta_high, geometry). Read the band --
    the point is not trustworthy to better than the band, and off the
    calibration set the true uncertainty is larger.
    """
    if m_fissile_kg <= 0.0:
        raise ValueError("m_fissile_kg must be > 0")
    if m_he_kg < 0.0 or m_tamper_kg < 0.0:
        raise ValueError("masses must be >= 0")
    if geometry not in GEOMETRY_OFFSET:
        raise ValueError(f"geometry must be one of {tuple(GEOMETRY_OFFSET)}")
    if he_relative_power <= 0.0:
        raise ValueError("he_relative_power must be > 0")

    m_he_eq = m_he_kg * he_relative_power
    if m_he_eq <= 0.0:
        # No explosive -> no implosion -> no compression.
        return ImplosionEstimate(1.0, 1.0, 1.0, geometry)

    eta = (
        _C0
        + _A * math.log(m_he_eq / m_fissile_kg)
        + _B * math.log(1.0 + m_tamper_kg / m_fissile_kg)
        + GEOMETRY_OFFSET[geometry]
    )
    eta = max(1.0, eta)
    band = max(0.06, eta_uncertainty)
    return ImplosionEstimate(eta, max(1.0, eta - band), eta + band, geometry)


# A yield band chains the (crude) compression estimate through the Cochran
# composite yield -- the full "device in, yield band out" front end.
DeviceYield = namedtuple(
    "DeviceYield",
    ["Y_low", "Y_nominal", "Y_high", "eta", "eta_low", "eta_high"],
)


def device_yield_band(
    *,
    m_he_kg: float,
    m_pu_kg: float = 0.0,
    m_heu_kg: float = 0.0,
    m_tamper_kg: float = 0.0,
    geometry: str = "hollow",
    he_relative_power: float = 1.0,
    eta_uncertainty: float = DEFAULT_ETA_UNCERTAINTY,
    pu_material: str = "alpha-WGPu",
    heu_material: str = "WGU",
) -> DeviceYield:
    """Estimate a bare-primary yield BAND directly from device parameters.

    Chains `implosion_compression` (device -> eta band) into the Cochran
    yield (eta -> yield). The eta calibration comes from Barroso's
    *alpha*-phase Pu implosions, so pure-Pu yields default to `alpha-WGPu`
    to stay self-consistent (that reproduces his Table 8-2/8-3 yields);
    pass `pu_material='delta-WGPu'` for a delta-stabilized pit, which will
    yield less at the same compression. A mixed Pu+HEU pit routes through
    the two-region `yield_kt_composite` (delta-Pu basis), which is a rougher
    match to the alpha-Pu eta calibration.

    The width of the yield band -- typically a factor of ~2 or more, often
    much larger near the criticality cliff -- is the honest message: a
    device-level yield estimate is an order-of-magnitude, not a number.
    """
    m_fissile = m_pu_kg + m_heu_kg
    if m_fissile <= 0.0:
        raise ValueError("need a positive fissile mass (m_pu_kg + m_heu_kg)")
    imp = implosion_compression(
        m_he_kg, m_fissile, m_tamper_kg,
        geometry=geometry, he_relative_power=he_relative_power,
        eta_uncertainty=eta_uncertainty,
    )

    def Y(eta: float) -> float:
        if m_heu_kg <= 0.0:
            return yield_kt(m_pu_kg, eta, pu_material, model="6.49")
        if m_pu_kg <= 0.0:
            return yield_kt(m_heu_kg, eta, heu_material, model="6.49")
        return yield_kt_composite(m_pu_kg, m_heu_kg, eta)

    return DeviceYield(
        Y(imp.eta_low), Y(imp.eta), Y(imp.eta_high),
        imp.eta, imp.eta_low, imp.eta_high,
    )
