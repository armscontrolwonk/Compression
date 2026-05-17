"""Two-region one-group diffusion yield model for a composite fissile sphere.

An inner sphere of one fissile material (e.g., delta-WGPu) surrounded by an
outer shell of a second fissile material (e.g., WGU). Both regions are
uniformly compressed by the same factor eta (the user-accepted simplification).

Theory: extends Cochran's bare-sphere one-group derivation to two regions.
In each region, the time-dependent diffusion equation reduces to the
Helmholtz form with buckling

    B_i^2(alpha, eta) = eta * (alpha_inf_o_i * eta - alpha) / D_o_i

where D_o_i = alpha_inf_o_i * R_o_i^2 / pi^2 is the diffusion coefficient at
normal density, derived from Cochran's eq. 5.8.

The radial flux solutions, regular at the origin and with continuity of flux
and current at the inner/outer interface plus zero flux at the outer surface
(matching Cochran's simplified B = pi/R boundary condition), give the
transcendental criticality determinant

    D_o_1 [B_1 R_1 cos(B_1 R_1) - sin(B_1 R_1)] sin(B_2 t)
      + D_o_2 sin(B_1 R_1) [B_2 R_1 cos(B_2 t) + sin(B_2 t)]  = 0

where t = R_2 - R_1. The fundamental mode (largest alpha root) is the physical
eigenvalue at compression eta.

Yield uses Cochran's hydrodynamic expression (eq. 6.36 in its general
pre-simplification form):

    Y = (9/10) * M_total * (Delta_R * alpha)^2 / (1 - (R_init/R_crit)^3)

with Delta_R = R_2(eta_c) - R_2(eta) the expansion of the outer surface
between operational compression eta and the second-criticality compression
eta_c = sqrt(M_c0 / M_total) (the latter following from M_c ~ 1/rho^2 under
uniform compression).

In the single-material limit (mass_inner = 0 or mass_outer = 0) this
reduces exactly to Cochran's eq. 6.49.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from .materials import SERBER_B, Material, get_material
from .model import KT_PER_RAW, yield_kt as _yield_kt_single

# A calibration value: either a scalar applied uniformly to the yield,
# or a dict mapping Material.key (or alias) to a per-material b value,
# mass-weighted for composites. See materials.SERBER_B.
#
# Two distinct correction concepts are layered in this module:
#
#   * calibration       -- foundational physics calibration of the
#                          bare-sphere one-group model (e.g., Serber's
#                          K ~ 1/4 to 1/2 vs the rigorous K ~ 1.1).
#                          Default is SERBER_B (per-material b values
#                          from Cochran eq. 6.60 / Primer Sec. 13).
#                          Pass calibration=1.0 to recover the
#                          uncorrected rigorous result.
#
#   * correction_factor -- additional empirical/data-driven scalar
#                          multiplier applied AFTER the physics
#                          calibration. Default 1.0. Use this for
#                          ad-hoc fits to observed test yields without
#                          touching the foundational physics.
#
# Final yield = correction_factor * (mass-weighted calibration) * Y_rigorous.
Calibration = float | dict[str, float]

_KG_TO_G = 1000.0


def _resolve_b(mapping: dict[str, float], mat: Material) -> float:
    """Look up per-material b in a correction-factor dict, by key or alias.
    Missing entries default to 1.0 (no correction)."""
    if mat.key in mapping:
        return mapping[mat.key]
    for alias in mat.aliases:
        if alias in mapping:
            return mapping[alias]
    return 1.0


def _effective_calibration(
    calibration: Calibration,
    mass_inner_kg: float,
    mass_outer_kg: float,
    inner_mat: Material,
    outer_mat: Material,
) -> float:
    """Reduce a calibration (scalar or dict) to a single scalar multiplier,
    mass-weighting across the two regions if the input is a per-material dict."""
    if not isinstance(calibration, dict):
        if calibration <= 0:
            raise ValueError("calibration must be positive")
        return float(calibration)
    total = mass_inner_kg + mass_outer_kg
    if total <= 0:
        return 1.0
    if mass_inner_kg == 0:
        b = _resolve_b(calibration, outer_mat)
    elif mass_outer_kg == 0:
        b = _resolve_b(calibration, inner_mat)
    else:
        b_in = _resolve_b(calibration, inner_mat)
        b_out = _resolve_b(calibration, outer_mat)
        b = (mass_inner_kg * b_in + mass_outer_kg * b_out) / total
    if b <= 0:
        raise ValueError("per-material calibration values must be positive")
    return b


def D0(mat: Material) -> float:
    """Diffusion coefficient D_o at normal density, in cm^2/shake.

    From Cochran eq. 5.8 / 5.9: R_c^2 = pi^2 D_o / alpha_inf,o, so
    D_o = alpha_inf,o * R_o^2 / pi^2.
    """
    return mat.alpha_inf * mat.R0 * mat.R0 / (math.pi * math.pi)


def _radii(
    mass_inner_kg: float,
    mass_outer_kg: float,
    eta: float,
    inner_mat: Material,
    outer_mat: Material,
) -> tuple[float, float]:
    """Inner and outer radii (cm) at compression eta."""
    rho_inner = inner_mat.density * eta
    rho_outer = outer_mat.density * eta
    V_inner = mass_inner_kg * _KG_TO_G / rho_inner if mass_inner_kg > 0 else 0.0
    V_shell = mass_outer_kg * _KG_TO_G / rho_outer if mass_outer_kg > 0 else 0.0
    R1 = (3.0 * V_inner / (4.0 * math.pi)) ** (1.0 / 3.0) if V_inner > 0 else 0.0
    R2 = (3.0 * (V_inner + V_shell) / (4.0 * math.pi)) ** (1.0 / 3.0)
    return R1, R2


def _B_squared(mat: Material, alpha: float, eta: float) -> float:
    """One-group buckling B^2 (1/cm^2) for region at compression eta."""
    return eta * (mat.alpha_inf * eta - alpha) / D0(mat)


def _criticality_F(
    alpha: float,
    eta: float,
    R1: float,
    R2: float,
    inner_mat: Material,
    outer_mat: Material,
) -> float:
    """Two-region criticality determinant. Roots in alpha are the eigenvalues.

    A region with B^2 > 0 uses sin/cos in r; a region with B^2 < 0 uses
    sinh/cosh (its infinite-medium production rate is below alpha and the
    spatial flux decays). The composite alpha can exceed the smaller of the
    two alpha_inf values when the more reactive region drives the system.
    """
    B1_sq = _B_squared(inner_mat, alpha, eta)
    B2_sq = _B_squared(outer_mat, alpha, eta)
    t = R2 - R1
    D1 = D0(inner_mat)
    D2 = D0(outer_mat)

    if B1_sq >= 0.0:
        b1 = math.sqrt(B1_sq)
        s1 = math.sin(b1 * R1)
        diff1 = b1 * R1 * math.cos(b1 * R1) - s1
    else:
        b1 = math.sqrt(-B1_sq)
        s1 = math.sinh(b1 * R1)
        diff1 = b1 * R1 * math.cosh(b1 * R1) - s1

    if B2_sq >= 0.0:
        b2 = math.sqrt(B2_sq)
        sBt = math.sin(b2 * t)
        cBt = math.cos(b2 * t)
    else:
        b2 = math.sqrt(-B2_sq)
        sBt = math.sinh(b2 * t)
        cBt = math.cosh(b2 * t)
    sum2 = b2 * R1 * cBt + sBt

    return D1 * diff1 * sBt + D2 * s1 * sum2


def _alpha_single_cochran(mass_kg: float, eta: float, mat: Material) -> float:
    """Cochran eq. 5.19: alpha = alpha_inf,o * eta * (1 - kappa^(-2/3))."""
    k = (mass_kg / mat.M0) * eta * eta
    if k <= 0.0:
        return -mat.alpha_inf * eta * 1e6
    return mat.alpha_inf * eta * (1.0 - k ** (-2.0 / 3.0))


def alpha_eigenvalue(
    mass_inner_kg: float,
    mass_outer_kg: float,
    eta: float,
    inner_mat: str | Material,
    outer_mat: str | Material,
) -> float:
    """Fundamental alpha (1/shake) for the composite at compression eta.

    Positive for super-critical configurations, negative for sub-critical,
    zero at criticality.
    """
    inner = get_material(inner_mat)
    outer = get_material(outer_mat)
    if mass_inner_kg < 0 or mass_outer_kg < 0:
        raise ValueError("masses must be non-negative")
    if eta <= 0:
        raise ValueError("compression must be positive")
    if mass_inner_kg == 0 and mass_outer_kg == 0:
        return 0.0
    if mass_inner_kg == 0:
        return _alpha_single_cochran(mass_outer_kg, eta, outer)
    if mass_outer_kg == 0:
        return _alpha_single_cochran(mass_inner_kg, eta, inner)

    R1, R2 = _radii(mass_inner_kg, mass_outer_kg, eta, inner, outer)
    # alpha is physically bounded above by max(alpha_inf)*eta (can't beat the
    # most reactive material's infinite-medium production rate). At that bound
    # F has a structural zero, so use a small margin.
    a_upper = max(inner.alpha_inf, outer.alpha_inf) * eta * (1.0 - 1e-12)
    a_lower = -10.0 * max(inner.alpha_inf, outer.alpha_inf) * eta - 1.0

    # Scan downward from just below alpha_max; first sign change is the
    # fundamental mode (largest alpha root of F).
    def F(a: float) -> float:
        return _criticality_F(a, eta, R1, R2, inner, outer)

    n_scan = 400
    prev_a = a_upper
    prev_F = F(prev_a)
    for i in range(1, n_scan + 1):
        a = a_upper - (a_upper - a_lower) * (i / n_scan)
        cur_F = F(a)
        if prev_F * cur_F < 0.0:
            return brentq(F, a, prev_a, xtol=1e-12, rtol=1e-12)
        prev_a, prev_F = a, cur_F

    raise RuntimeError(
        f"Could not bracket alpha eigenvalue "
        f"(M_in={mass_inner_kg}, M_out={mass_outer_kg}, eta={eta}, "
        f"F at upper={F(a_upper)}, F at lower={F(a_lower)})"
    )


def _critical_mass_at_normal_density(
    fraction_inner: float, inner_mat: Material, outer_mat: Material
) -> float:
    """Total mass M_c at eta=1 for a composite with given inner mass fraction.

    The criticality determinant F(M; alpha=0, eta=1) has multiple roots, one
    per spatial harmonic (fundamental at the smallest M, higher modes at
    successively larger M with internal flux nodes). The physical critical
    mass is the fundamental, so we scan from small M upward and take the
    first sign change.
    """
    f = fraction_inner
    if f <= 0.0:
        return outer_mat.M0
    if f >= 1.0:
        return inner_mat.M0

    def F_at_crit(M: float) -> float:
        m_in = f * M
        m_out = (1.0 - f) * M
        R1, R2 = _radii(m_in, m_out, 1.0, inner_mat, outer_mat)
        return _criticality_F(0.0, 1.0, R1, R2, inner_mat, outer_mat)

    M_lo = 0.01 * min(inner_mat.M0, outer_mat.M0)
    M_hi = 2.0 * max(inner_mat.M0, outer_mat.M0)
    n_scan = 200
    prev_M = M_lo
    prev_F = F_at_crit(prev_M)
    for i in range(1, n_scan + 1):
        M = M_lo + (M_hi - M_lo) * (i / n_scan)
        cur_F = F_at_crit(M)
        if prev_F * cur_F < 0.0:
            return brentq(F_at_crit, prev_M, M, xtol=1e-9, rtol=1e-12)
        prev_M, prev_F = M, cur_F

    # Fallback: extend upward if the fundamental sits beyond the initial range.
    for _ in range(20):
        M_hi *= 2.0
        cur_F = F_at_crit(M_hi)
        if prev_F * cur_F < 0.0:
            return brentq(F_at_crit, prev_M, M_hi, xtol=1e-9, rtol=1e-12)
        prev_M, prev_F = M_hi, cur_F
    raise RuntimeError(f"Could not find fundamental M_c at f={f}")


def critical_compression(
    mass_inner_kg: float,
    mass_outer_kg: float,
    inner_mat: str | Material,
    outer_mat: str | Material,
) -> float:
    """Compression eta_c at which a composite of given masses is just critical.

    Uses M_c(eta) = M_c(1)/eta^2 (the M_c ~ 1/rho^2 scaling for any uniformly
    compressed assembly under one-group diffusion).
    """
    inner = get_material(inner_mat)
    outer = get_material(outer_mat)
    M_total = mass_inner_kg + mass_outer_kg
    if M_total <= 0:
        raise ValueError("total mass must be positive")
    if mass_inner_kg < 0 or mass_outer_kg < 0:
        raise ValueError("masses must be non-negative")
    if mass_inner_kg == 0:
        return math.sqrt(outer.M0 / mass_outer_kg)
    if mass_outer_kg == 0:
        return math.sqrt(inner.M0 / mass_inner_kg)

    f = mass_inner_kg / M_total
    M_c0 = _critical_mass_at_normal_density(f, inner, outer)
    return math.sqrt(M_c0 / M_total)


def critical_mass_composite(
    fraction_inner: float,
    eta: float,
    inner_mat: str | Material,
    outer_mat: str | Material,
) -> float:
    """Total composite mass at criticality for a given inner mass fraction
    and compression eta. M_c(eta, f) = M_c(1, f) / eta^2."""
    inner = get_material(inner_mat)
    outer = get_material(outer_mat)
    if eta <= 0:
        raise ValueError("compression must be positive")
    M_c0 = _critical_mass_at_normal_density(fraction_inner, inner, outer)
    return M_c0 / (eta * eta)


def yield_kt_composite(
    mass_inner_kg: float,
    mass_outer_kg: float,
    eta: float,
    inner_mat: str | Material,
    outer_mat: str | Material,
    calibration: Calibration = SERBER_B,
    correction_factor: float = 1.0,
) -> float:
    """Composite-core yield in kilotons.

    Uses Cochran's hydrodynamic formula with alpha from the two-region
    eigenvalue solve. Reduces exactly to yield_kt(..., model='6.49')
    in the single-material limit when ``calibration=1.0`` and
    ``correction_factor=1.0``.

    Two layered corrections to the rigorous bare-sphere prediction:

    ``calibration`` : foundational physics calibration. Default is the
        Serber/Cochran SERBER_B preset (Primer Sec. 13, Cochran eq. 6.60):
        b = 1.0 for WGPu, 0.8 for U-233, 0.5 for HEU. Mass-weighted for
        composites. Accepts a scalar (applied uniformly) or a dict
        mapping Material.key (or alias) to a per-material b value.
        Pass ``calibration=1.0`` to recover the uncorrected rigorous one-
        group prediction.

    ``correction_factor`` : additional empirical/data-driven scalar
        multiplier applied AFTER the physics calibration. Default 1.0.
        Reserve this for ad-hoc fits to observed test yields without
        modifying the foundational physics.

    Final yield = correction_factor * (mass-weighted calibration) * Y_rigorous.
    """
    inner = get_material(inner_mat)
    outer = get_material(outer_mat)
    if mass_inner_kg < 0 or mass_outer_kg < 0:
        raise ValueError("masses must be non-negative")
    if eta <= 0:
        raise ValueError("compression must be positive")
    if correction_factor <= 0:
        raise ValueError("correction_factor must be positive")
    M_total = mass_inner_kg + mass_outer_kg
    if M_total == 0:
        return 0.0
    cal_eff = _effective_calibration(
        calibration, mass_inner_kg, mass_outer_kg, inner, outer
    )
    total_factor = correction_factor * cal_eff
    if mass_inner_kg == 0:
        return total_factor * _yield_kt_single(
            mass_outer_kg, eta, outer, model="6.49"
        )
    if mass_outer_kg == 0:
        return total_factor * _yield_kt_single(
            mass_inner_kg, eta, inner, model="6.49"
        )

    eta_c = critical_compression(mass_inner_kg, mass_outer_kg, inner, outer)
    if eta <= eta_c:
        return 0.0

    alpha = alpha_eigenvalue(mass_inner_kg, mass_outer_kg, eta, inner, outer)
    if alpha <= 0.0:
        return 0.0

    _, R2_init = _radii(mass_inner_kg, mass_outer_kg, eta, inner, outer)
    _, R2_crit = _radii(mass_inner_kg, mass_outer_kg, eta_c, inner, outer)
    delta_R = R2_crit - R2_init
    V_over_dV = 1.0 / (1.0 - (R2_init / R2_crit) ** 3)

    Y_raw = 0.9 * M_total * (delta_R * alpha) ** 2 * V_over_dV
    return total_factor * KT_PER_RAW * Y_raw


def compression_composite(
    mass_inner_kg: float,
    mass_outer_kg: float,
    Y_kt: float,
    inner_mat: str | Material,
    outer_mat: str | Material,
    calibration: Calibration = SERBER_B,
    correction_factor: float = 1.0,
) -> float:
    """Compression eta consistent with an observed yield Y_kt for the given
    composite masses.

    Inverts yield_kt_composite. Useful for inferring effective compression
    from historical test data (the "Path 1" calibration in the README):
    given the configuration and observed yield, returns the eta the
    model assigns to that test.

    Below criticality (Y_kt = 0) returns the critical compression eta_c.

    ``calibration`` and ``correction_factor`` have the same meaning and
    defaults as in yield_kt_composite: the foundational physics
    calibration (default SERBER_B) and an additional empirical multiplier
    (default 1.0) are applied in that order.
    """
    inner = get_material(inner_mat)
    outer = get_material(outer_mat)
    if mass_inner_kg < 0 or mass_outer_kg < 0:
        raise ValueError("masses must be non-negative")
    if Y_kt < 0:
        raise ValueError("yield must be non-negative")
    if correction_factor <= 0:
        raise ValueError("correction_factor must be positive")
    if mass_inner_kg + mass_outer_kg <= 0:
        raise ValueError("at least one mass must be positive")
    cal_eff = _effective_calibration(
        calibration, mass_inner_kg, mass_outer_kg, inner, outer
    )
    total_factor = correction_factor * cal_eff

    # Single-material reductions: delegate to the existing 6.49 inverse,
    # adjusting the target yield for the combined multiplier.
    from .model import compression as _compression_single

    Y_target = Y_kt / total_factor
    if mass_inner_kg == 0:
        if Y_target == 0:
            return math.sqrt(outer.M0 / mass_outer_kg)
        return _compression_single(mass_outer_kg, Y_target, outer, model="6.49")
    if mass_outer_kg == 0:
        if Y_target == 0:
            return math.sqrt(inner.M0 / mass_inner_kg)
        return _compression_single(mass_inner_kg, Y_target, inner, model="6.49")

    eta_c = critical_compression(mass_inner_kg, mass_outer_kg, inner, outer)
    if Y_kt == 0:
        return eta_c

    def F(e: float) -> float:
        return (
            yield_kt_composite(
                mass_inner_kg,
                mass_outer_kg,
                e,
                inner,
                outer,
                calibration=calibration,
                correction_factor=correction_factor,
            )
            - Y_kt
        )

    lo = eta_c * (1.0 + 1e-9)
    hi = max(lo * 2.0, 20.0)
    F_hi = F(hi)
    for _ in range(40):
        if F_hi > 0:
            break
        hi *= 2.0
        F_hi = F(hi)
    else:
        raise RuntimeError(
            f"Could not bracket compression for Y={Y_kt} kt "
            f"(M_in={mass_inner_kg}, M_out={mass_outer_kg})"
        )
    return brentq(F, lo, hi, xtol=1e-9, rtol=1e-12)
