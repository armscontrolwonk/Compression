"""Multi-stage thermonuclear weapon yield estimator.

Single-call composition of the five energy channels:

  1. Primary fission (Pu and/or HEU)            -- fissionyield.yield_kt_composite
  2. Boost (DT gas in the primary pit)          -- tnyield.boost
  3. Layer cake / Sloika (LiD + U-238 layer)    -- tnyield.layer_cake
  4. Secondary (LiD fuel + spark plug + tamper) -- tnyield.secondary
  5. Tertiary FFF jacket (U-238 outer casing)   -- per-mass fast-fission

Stages you don't have just stay at zero mass; the function handles
any subset.  The "primary" composite uses Cochran's two-region one-
group diffusion theory with the Serber-b calibration (b=1.0 for Pu,
b=0.5 for HEU, mass-weighted).

Forward and inverse:

    Y = yield_kt_total(m_pu_kg=4.2, m_heu_kg=6.8, eta=3.0,
                       m_dt_g=4.0, m_li6d_layer_kg=20.0, ...)

    m_li6d = solve_mass(target_Y_kt=400.0, unknown="m_li6d_layer_kg",
                        m_pu_kg=4.2, m_heu_kg=6.8, eta=3.0, ...)

The inverse fixes everything except the named mass and bisects.
"""

from __future__ import annotations

from dataclasses import dataclass

from fissionyield import pit_radius_cm, yield_kt_composite

from .boost import (
    DEFAULT_EXTRA_FISSIONS_PER_FUSION_NEUTRON,
    boosted_yield,
    dt_confinement_time,
    dt_density,
    dt_temperature,
)
from .layer_cake import (
    DEFAULT_LID_BURN_FRACTION as _LC_LID_BURN,
)
from .layer_cake import (
    DEFAULT_U238_BURN_FRACTION as _LC_U238_BURN,
)
from .layer_cake import layer_cake_yield
from .reactions import (
    KT_PER_G_U235_FISSION,
    KT_PER_G_U238_FISSION,
    KT_PER_G_PU239_FISSION,
)
from .secondary import (
    DEFAULT_FISSILE_SECONDARY_BURN_FRACTION,
    DEFAULT_H_B_LID,
    DEFAULT_SPARK_PLUG_BURN_FRACTION,
)
from .secondary import (
    DEFAULT_U238_BURN_FRACTION as _SEC_U238_BURN,
)
from .secondary import secondary_yield


# Default jacket burn fraction: thick outer U-238 (or U-natural) casing
# fast-fissioned by fusion neutrons from any earlier stage.  Real-world
# Teller-Ulam jackets fission ~0.5-2 % of their bulk mass.
DEFAULT_JACKET_BURN_FRACTION = 0.01


SOLVABLE_MASSES = (
    "m_pu_kg",
    "m_heu_kg",
    "m_dt_g",
    "m_li6d_layer_kg",
    "m_u238_layer_kg",
    "m_lid_secondary_kg",
    "m_spark_plug_kg",
    "m_u238_tamper_kg",
    "m_fissile_secondary_kg",
    "m_u238_jacket_kg",
)


@dataclass(frozen=True)
class TotalYieldResult:
    """Total yield in kt plus the breakdown by energy channel."""

    Y_total_kt: float

    Y_primary_kt: float
    Y_boost_fusion_kt: float
    Y_boost_added_fission_kt: float

    Y_layer_cake_fusion_kt: float
    Y_layer_cake_tamper_kt: float

    Y_spark_plug_kt: float
    Y_secondary_fusion_kt: float
    Y_secondary_tamper_kt: float
    Y_fissile_secondary_kt: float

    Y_jacket_kt: float

    DT_burn_fraction: float
    secondary_LiD_burn_fraction: float

    def __str__(self) -> str:
        rows = [
            ("Primary fission (Pu+HEU)", self.Y_primary_kt),
            ("Boost: direct DT fusion", self.Y_boost_fusion_kt),
            ("Boost: extra fission", self.Y_boost_added_fission_kt),
            ("Layer cake: LiD fusion", self.Y_layer_cake_fusion_kt),
            ("Layer cake: U-238 layer", self.Y_layer_cake_tamper_kt),
            ("Secondary: spark plug", self.Y_spark_plug_kt),
            ("Secondary: LiD fusion", self.Y_secondary_fusion_kt),
            ("Secondary: U-238 tamper", self.Y_secondary_tamper_kt),
            ("Secondary: fissile fission", self.Y_fissile_secondary_kt),
            ("Tertiary FFF jacket", self.Y_jacket_kt),
        ]
        lines = [f"Total yield                : {self.Y_total_kt:10.3f} kt"]
        for name, Y in rows:
            if Y > 0:
                frac = Y / self.Y_total_kt * 100 if self.Y_total_kt > 0 else 0.0
                lines.append(f"  {name:<26}: {Y:8.3f} kt ({frac:5.1f} %)")
        if self.DT_burn_fraction > 0:
            lines.append(f"  (DT burn fraction         : {self.DT_burn_fraction * 100:.2f} %)")
        if self.secondary_LiD_burn_fraction > 0:
            lines.append(f"  (secondary LiD burn frac. : {self.secondary_LiD_burn_fraction * 100:.2f} %)")
        return "\n".join(lines)


def _spark_plug_kt_per_g(material: str) -> float:
    """kt/g for the spark plug material, by name or substring."""
    m = material.lower()
    if "pu" in m:
        return KT_PER_G_PU239_FISSION
    return KT_PER_G_U235_FISSION


def yield_kt_total(
    *,
    # ---- primary (composite Pu + HEU) -----------------------------------
    m_pu_kg: float = 0.0,
    m_heu_kg: float = 0.0,
    eta: float = 2.5,
    # ---- boost ----------------------------------------------------------
    m_dt_g: float = 0.0,
    dt_rho_g_per_cm3: float | None = None,   # None -> derived from eta
    dt_T_keV: float | None = None,           # None -> derived from eta
    dt_tau_s: float | None = None,           # None -> derived from pit radius
    extra_fissions_per_fusion_neutron: float = DEFAULT_EXTRA_FISSIONS_PER_FUSION_NEUTRON,
    # ---- layer cake / Sloika --------------------------------------------
    m_li6d_layer_kg: float = 0.0,
    li6_enrichment_layer: float = 0.95,
    m_u238_layer_kg: float = 0.0,
    lid_layer_burn_fraction: float = _LC_LID_BURN,
    u238_layer_burn_fraction: float = _LC_U238_BURN,
    # ---- secondary (Teller-Ulam) ----------------------------------------
    m_lid_secondary_kg: float = 0.0,
    li6_enrichment_secondary: float = 0.95,
    m_spark_plug_kg: float = 0.0,
    spark_plug_material: str = "delta-WGPu",
    m_u238_tamper_kg: float = 0.0,
    secondary_compression: float = 80.0,
    secondary_LiD_burn_fraction: float | None = None,
    u238_tamper_burn_fraction: float = _SEC_U238_BURN,
    spark_plug_burn_fraction: float = DEFAULT_SPARK_PLUG_BURN_FRACTION,
    secondary_H_B_LiD: float = DEFAULT_H_B_LID,
    m_fissile_secondary_kg: float = 0.0,
    fissile_secondary_material: str = "WGU",
    fissile_secondary_burn_fraction: float = DEFAULT_FISSILE_SECONDARY_BURN_FRACTION,
    # ---- tertiary FFF jacket --------------------------------------------
    m_u238_jacket_kg: float = 0.0,
    jacket_burn_fraction: float = DEFAULT_JACKET_BURN_FRACTION,
) -> TotalYieldResult:
    """Total yield of a multi-stage device, summed over all channels.

    All masses default to zero -- a pure-primary calculation just passes
    `m_pu_kg`, `m_heu_kg`, and `eta`.  Spark-plug material is "delta-WGPu"
    (any string containing "pu") or "WGU" (anything else).

    Caveats:
      - Mass alone underdetermines yield; compressions and burn fractions
        are knobs you can tune.  Defaults are book/literature values.
      - The Lindl formula is used for `secondary_LiD_burn_fraction` only
        when you do NOT supply one; real Teller-Ulam burn fractions are
        a few per cent, far below what Lindl predicts at 100x compression.
      - The U-238 jacket contribution is purely mass-times-burn-fraction.
        Real designs couple it to the actual fusion-neutron flux, which
        this estimator does not track.
    """
    for name, val in {
        "m_pu_kg": m_pu_kg, "m_heu_kg": m_heu_kg, "m_dt_g": m_dt_g,
        "m_li6d_layer_kg": m_li6d_layer_kg, "m_u238_layer_kg": m_u238_layer_kg,
        "m_lid_secondary_kg": m_lid_secondary_kg, "m_spark_plug_kg": m_spark_plug_kg,
        "m_u238_tamper_kg": m_u238_tamper_kg, "m_u238_jacket_kg": m_u238_jacket_kg,
        "m_fissile_secondary_kg": m_fissile_secondary_kg,
    }.items():
        if val < 0:
            raise ValueError(f"{name} must be >= 0")
    if eta <= 0:
        raise ValueError("eta must be > 0")

    # 1. Primary composite fission
    Y_primary = yield_kt_composite(m_pu_kg, m_heu_kg, eta)

    # 2. Boost
    Y_boost_fusion = 0.0
    Y_boost_added = 0.0
    f_DT = 0.0
    if m_dt_g > 0.0:
        # The DT state is a consequence of the primary: derive it from the
        # pit's compression and geometry unless the caller overrides.
        rho = dt_density(eta) if dt_rho_g_per_cm3 is None else dt_rho_g_per_cm3
        T = dt_temperature(eta) if dt_T_keV is None else dt_T_keV
        tau = (
            dt_confinement_time(pit_radius_cm(m_pu_kg, m_heu_kg, eta))
            if dt_tau_s is None else dt_tau_s
        )
        br = boosted_yield(
            mass_DT_g=m_dt_g,
            rho_DT_g_per_cm3=rho,
            T_keV=T,
            tau_s=tau,
            Y_baseline_kt=Y_primary,
            extra_fissions_per_fusion_neutron=extra_fissions_per_fusion_neutron,
        )
        Y_boost_fusion = br.Y_fusion_kt
        Y_boost_added = br.Y_added_fission_kt
        f_DT = br.burn_fraction_DT

    # 3. Layer cake (pass Y_primary=0 so we only get its own contribution)
    if m_li6d_layer_kg > 0.0 or m_u238_layer_kg > 0.0:
        lc = layer_cake_yield(
            Y_primary_kt=0.0,
            mass_LiD_kg=m_li6d_layer_kg,
            mass_U238_kg=m_u238_layer_kg,
            li6_enrichment=li6_enrichment_layer,
            LiD_burn_fraction=lid_layer_burn_fraction,
            U238_burn_fraction=u238_layer_burn_fraction,
        )
        Y_lc_fusion = lc.Y_fusion_kt
        Y_lc_tamper = lc.Y_tamper_kt
    else:
        Y_lc_fusion = 0.0
        Y_lc_tamper = 0.0

    # 4. Secondary
    if (m_lid_secondary_kg > 0.0 or m_u238_tamper_kg > 0.0
            or m_spark_plug_kg > 0.0 or m_fissile_secondary_kg > 0.0):
        kt_per_g_spark = _spark_plug_kt_per_g(spark_plug_material)
        Y_spark = (
            kt_per_g_spark * 1000.0 * m_spark_plug_kg * spark_plug_burn_fraction
        )
        sec = secondary_yield(
            Y_primary_kt=0.0,
            mass_LiD_kg=m_lid_secondary_kg,
            mass_U238_tamper_kg=m_u238_tamper_kg,
            mass_spark_plug_kg=0.0,         # accounted for explicitly above
            li6_enrichment=li6_enrichment_secondary,
            compression=secondary_compression,
            LiD_burn_fraction=secondary_LiD_burn_fraction,
            U238_burn_fraction=u238_tamper_burn_fraction,
            spark_plug_burn_fraction=spark_plug_burn_fraction,
            H_B_LiD=secondary_H_B_LiD,
            mass_fissile_secondary_kg=m_fissile_secondary_kg,
            fissile_secondary_material=fissile_secondary_material,
            fissile_secondary_burn_fraction=fissile_secondary_burn_fraction,
        )
        Y_sec_fusion = sec.Y_fusion_kt
        Y_sec_tamper = sec.Y_tamper_kt
        Y_fissile_secondary = sec.Y_fissile_secondary_kt
        f_sec_LiD = sec.burn_fraction_LiD
    else:
        Y_spark = 0.0
        Y_sec_fusion = 0.0
        Y_sec_tamper = 0.0
        Y_fissile_secondary = 0.0
        f_sec_LiD = 0.0

    # 5. Tertiary FFF jacket
    Y_jacket = (
        KT_PER_G_U238_FISSION * 1000.0 * m_u238_jacket_kg * jacket_burn_fraction
    )

    Y_total = (
        Y_primary
        + Y_boost_fusion + Y_boost_added
        + Y_lc_fusion + Y_lc_tamper
        + Y_spark + Y_sec_fusion + Y_sec_tamper + Y_fissile_secondary
        + Y_jacket
    )

    return TotalYieldResult(
        Y_total_kt=Y_total,
        Y_primary_kt=Y_primary,
        Y_boost_fusion_kt=Y_boost_fusion,
        Y_boost_added_fission_kt=Y_boost_added,
        Y_layer_cake_fusion_kt=Y_lc_fusion,
        Y_layer_cake_tamper_kt=Y_lc_tamper,
        Y_spark_plug_kt=Y_spark,
        Y_secondary_fusion_kt=Y_sec_fusion,
        Y_secondary_tamper_kt=Y_sec_tamper,
        Y_fissile_secondary_kt=Y_fissile_secondary,
        Y_jacket_kt=Y_jacket,
        DT_burn_fraction=f_DT,
        secondary_LiD_burn_fraction=f_sec_LiD,
    )


def solve_mass(
    *,
    target_Y_kt: float,
    unknown: str,
    bracket: tuple[float, float] | None = None,
    tol_kt: float = 1e-4,
    max_iter: int = 200,
    **kwargs,
) -> float:
    """Solve for one mass parameter given a target total yield.

    Parameters
    ----------
    target_Y_kt : Target Y_total in kt.
    unknown : Name of the mass parameter to solve for.  Must be one of
        the strings in `SOLVABLE_MASSES`.
    bracket : Optional (lo, hi) search bracket in the same units as the
        target parameter (kg for masses, g for `m_dt_g`).  If omitted,
        the function picks a default range and expands upward as needed.
    tol_kt : Convergence tolerance on the yield (kt).
    max_iter : Max bisection iterations.
    **kwargs : All the other `yield_kt_total` keyword arguments.  Any
        explicit value passed for `unknown` is ignored.

    Returns
    -------
    The value of the unknown mass that hits `target_Y_kt`, or 0.0 if
    the target is already exceeded with the unknown at zero.

    Raises
    ------
    ValueError, RuntimeError on bad input or unbracketable target.
    """
    if unknown not in SOLVABLE_MASSES:
        raise ValueError(
            f"unknown must be one of {SOLVABLE_MASSES}, got {unknown!r}"
        )
    if target_Y_kt <= 0:
        raise ValueError("target_Y_kt must be > 0")
    kwargs.pop(unknown, None)

    def Y_of(m: float) -> float:
        return yield_kt_total(**{**kwargs, unknown: m}).Y_total_kt

    Y0 = Y_of(0.0)
    if Y0 > target_Y_kt:
        # Target already exceeded with this mass at zero -- can't solve
        # by adding more.  Return 0 with a hint via runtime error?
        # Returning 0 is the most useful behaviour for "the answer is
        # less than zero" cases; user can detect by checking that
        # Y_of(0) > target.
        return 0.0

    # Default bracket per parameter scale
    if bracket is not None:
        lo, hi = bracket
    else:
        lo = 0.0
        hi = 100.0 if unknown == "m_dt_g" else 50.0  # kg

    # Expand hi upward until we exceed the target
    for _ in range(60):
        if Y_of(hi) >= target_Y_kt:
            break
        hi *= 2.0
    else:
        raise RuntimeError(
            f"could not bracket {unknown}: target {target_Y_kt} kt unreachable "
            f"with given parameters (tried up to {hi})"
        )

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        Y_mid = Y_of(mid)
        if abs(Y_mid - target_Y_kt) < tol_kt:
            return mid
        if Y_mid < target_Y_kt:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)
