"""Tests for the two-region composite-core yield model."""

import math

import pytest

from fissionyield import (
    alpha_eigenvalue,
    compression,
    compression_composite,
    critical_compression,
    critical_mass_composite,
    get_material,
    yield_kt,
    yield_kt_composite,
)
from fissionyield.composite import D0, _criticality_F, _radii


# ---------------------------------------------------------------------------
# Apples-to-apples: single-material limit must equal Cochran eq. 6.49 exactly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inner_key,outer_key,M,eta",
    [
        ("delta-WGPu", "WGU", 6.1, 2.5),
        ("delta-WGPu", "WGU", 3.0, 4.0),
        ("delta-WGPu", "WGU", 10.0, 1.5),
        ("alpha-WGPu", "WGU", 4.0, 3.0),
        ("WGU", "delta-WGPu", 20.0, 2.0),
        ("WGU", "delta-WGPu", 60.0, 2.5),
    ],
)
def test_single_inner_limit_equals_cochran_649(inner_key, outer_key, M, eta):
    """Mass entirely in the inner region: composite == single (eq 6.49)."""
    Y_composite = yield_kt_composite(M, 0.0, eta, inner_key, outer_key)
    Y_cochran = yield_kt(M, eta, inner_key, model="6.49")
    assert Y_composite == pytest.approx(Y_cochran, rel=1e-12)


@pytest.mark.parametrize(
    "inner_key,outer_key,M,eta",
    [
        ("delta-WGPu", "WGU", 30.0, 2.0),
        ("delta-WGPu", "WGU", 60.0, 2.5),
        ("WGU", "delta-WGPu", 6.0, 3.0),
        ("WGU", "delta-WGPu", 10.0, 2.0),
    ],
)
def test_single_outer_limit_equals_cochran_649(inner_key, outer_key, M, eta):
    """Mass entirely in the outer region: composite == single (eq 6.49)."""
    Y_composite = yield_kt_composite(0.0, M, eta, inner_key, outer_key)
    Y_cochran = yield_kt(M, eta, outer_key, model="6.49")
    assert Y_composite == pytest.approx(Y_cochran, rel=1e-12)


@pytest.mark.parametrize(
    "inner_key,outer_key,M,eta",
    [
        ("delta-WGPu", "WGU", 6.1, 2.5),
        ("WGU", "delta-WGPu", 30.0, 2.0),
    ],
)
def test_alpha_single_limit_equals_cochran(inner_key, outer_key, M, eta):
    """alpha_eigenvalue in single-material limit matches Cochran eq. 5.19."""
    inner = get_material(inner_key)
    a_composite_inner = alpha_eigenvalue(M, 0.0, eta, inner_key, outer_key)
    k = (M / inner.M0) * eta * eta
    a_cochran = inner.alpha_inf * eta * (1.0 - k ** (-2.0 / 3.0))
    assert a_composite_inner == pytest.approx(a_cochran, rel=1e-12)


# ---------------------------------------------------------------------------
# Critical compression / mass: single-material limits.
# ---------------------------------------------------------------------------


def test_critical_compression_single_inner():
    inner = get_material("delta-WGPu")
    M = 6.1
    eta_c = critical_compression(M, 0.0, "delta-WGPu", "WGU")
    assert eta_c == pytest.approx(math.sqrt(inner.M0 / M), rel=1e-12)


def test_critical_compression_single_outer():
    outer = get_material("WGU")
    M = 60.0
    eta_c = critical_compression(0.0, M, "delta-WGPu", "WGU")
    assert eta_c == pytest.approx(math.sqrt(outer.M0 / M), rel=1e-12)


def test_critical_mass_composite_single_limits():
    pu = get_material("delta-WGPu")
    heu = get_material("WGU")
    assert critical_mass_composite(1.0, 1.0, pu, heu) == pytest.approx(pu.M0)
    assert critical_mass_composite(0.0, 1.0, pu, heu) == pytest.approx(heu.M0)
    assert critical_mass_composite(1.0, 2.0, pu, heu) == pytest.approx(pu.M0 / 4.0)


# ---------------------------------------------------------------------------
# Physical sanity: critical mass for a 1:1 composite is intermediate.
# ---------------------------------------------------------------------------


def test_composite_critical_mass_between_pure_values():
    pu = get_material("delta-WGPu")
    heu = get_material("WGU")
    # At f = 0.5 the bare critical mass should lie between the two pure values.
    M_c = critical_mass_composite(0.5, 1.0, pu, heu)
    M_min = min(pu.M0, heu.M0)
    M_max = max(pu.M0, heu.M0)
    assert M_min < M_c < M_max


def test_composite_critical_mass_monotone_in_pu_fraction():
    pu = get_material("delta-WGPu")
    heu = get_material("WGU")
    fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    Mcs = [critical_mass_composite(f, 1.0, pu, heu) for f in fractions]
    # Pu has the smaller bare critical mass; M_c should decrease with f.
    assert Mcs == sorted(Mcs, reverse=True)


# ---------------------------------------------------------------------------
# Yield sanity: monotonicity, criticality threshold, NRDC anchor.
# ---------------------------------------------------------------------------


def test_yield_zero_below_critical():
    pu = get_material("delta-WGPu")
    heu = get_material("WGU")
    M_c0 = critical_mass_composite(0.5, 1.0, pu, heu)
    # Below criticality at eta=1
    Y = yield_kt_composite(0.4 * M_c0, 0.4 * M_c0, 1.0, pu, heu)
    assert Y == 0.0


def test_yield_monotone_in_mass_at_fixed_split_and_eta():
    pu = get_material("delta-WGPu")
    heu = get_material("WGU")
    eta = 2.5
    masses = [(2.0, 4.0), (4.0, 8.0), (6.0, 12.0), (10.0, 20.0)]
    ys = [yield_kt_composite(mi, mo, eta, pu, heu) for mi, mo in masses]
    assert ys == sorted(ys)


def test_yield_monotone_in_compression():
    pu = get_material("delta-WGPu")
    heu = get_material("WGU")
    M_in, M_out = 3.0, 6.0
    etas = [1.5, 2.0, 2.5, 3.0, 4.0]
    ys = [yield_kt_composite(M_in, M_out, e, pu, heu) for e in etas]
    assert ys == sorted(ys)


def test_fat_man_anchor_through_composite_path():
    """6.1 kg delta-WGPu, eta=2.5, no HEU: same Fat Man ballpark as pure 6.49."""
    Y = yield_kt_composite(6.1, 0.0, 2.5, "delta-WGPu", "WGU")
    Y_cochran = yield_kt(6.1, 2.5, "delta-WGPu", model="6.49")
    assert Y == pytest.approx(Y_cochran)
    # Loose NRDC sanity: yield is in the kt-to-tens-of-kt range
    assert 1.0 < Y < 100.0


# ---------------------------------------------------------------------------
# Diffusion coefficient consistency.
# ---------------------------------------------------------------------------


def test_D0_recovers_cochran_R0_alpha_sq():
    """alpha_inf^2 * R0^2 = pi^4 * D0^2 / alpha_inf -- check the link to
    Cochran's (R0*alpha_inf)^2 packaged in materials.R0_alpha_sq."""
    for key in ("delta-WGPu", "alpha-WGPu", "WGU", "U-233"):
        mat = get_material(key)
        # D0 = alpha_inf * R0^2 / pi^2 by construction
        assert D0(mat) == pytest.approx(mat.alpha_inf * mat.R0 ** 2 / math.pi**2)
        # Recover (R0 * alpha_inf)^2
        recovered = (mat.alpha_inf * mat.R0) ** 2
        assert recovered == pytest.approx(mat.R0_alpha_sq, rel=0.05)


# ---------------------------------------------------------------------------
# Determinant degenerate-limit checks (structural).
# ---------------------------------------------------------------------------


def test_F_reduces_to_outer_critical_when_R1_to_zero():
    """As R_1 -> 0 with fixed R_2, F factors through sin(B_2 R_2)."""
    outer = get_material("WGU")
    inner = get_material("delta-WGPu")
    # Use a vanishingly small inner radius and a finite outer radius.
    # At alpha=0, eta=1, R_2 = R_o,outer should satisfy sin(B_2 R_2) = 0
    # since B_2 = pi/R_o for material 'outer' by Cochran 5.8.
    R2 = outer.R0
    F_small_R1 = _criticality_F(0.0, 1.0, R2 * 1e-6, R2, inner, outer)
    # Should be very near zero (sin(pi) = 0)
    assert abs(F_small_R1) < 1e-6 * abs(
        _criticality_F(0.0, 1.0, R2 * 1e-6, R2 * 0.5, inner, outer)
    )


def test_F_reduces_to_inner_critical_when_R2_to_R1():
    """As R_2 -> R_1 (no outer shell), F = 0 reduces to sin(B_1 R_1) = 0."""
    inner = get_material("delta-WGPu")
    outer = get_material("WGU")
    # At inner critical radius and zero shell thickness, F should vanish.
    R1 = inner.R0
    F_zero_shell = _criticality_F(0.0, 1.0, R1, R1, inner, outer)
    assert abs(F_zero_shell) < 1e-9


# ---------------------------------------------------------------------------
# Bad-input handling.
# ---------------------------------------------------------------------------


def test_negative_mass_raises():
    with pytest.raises(ValueError):
        yield_kt_composite(-1.0, 1.0, 2.0, "delta-WGPu", "WGU")
    with pytest.raises(ValueError):
        yield_kt_composite(1.0, -1.0, 2.0, "delta-WGPu", "WGU")


def test_nonpositive_compression_raises():
    with pytest.raises(ValueError):
        yield_kt_composite(1.0, 1.0, 0.0, "delta-WGPu", "WGU")
    with pytest.raises(ValueError):
        alpha_eigenvalue(1.0, 1.0, -1.0, "delta-WGPu", "WGU")


# ---------------------------------------------------------------------------
# correction_factor knob (Path 2 escape hatch -- ad-hoc damping)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [0.1, 0.5, 1.0, 2.0])
def test_correction_factor_scales_yield_linearly(factor):
    Y_base = yield_kt_composite(3.0, 5.0, 2.5, "delta-WGPu", "WGU")
    Y_scaled = yield_kt_composite(
        3.0, 5.0, 2.5, "delta-WGPu", "WGU", correction_factor=factor
    )
    assert Y_scaled == pytest.approx(factor * Y_base)


def test_correction_factor_applies_in_single_material_limits():
    Y_base = yield_kt_composite(6.1, 0.0, 2.5, "delta-WGPu", "WGU")
    Y_half = yield_kt_composite(
        6.1, 0.0, 2.5, "delta-WGPu", "WGU", correction_factor=0.5
    )
    assert Y_half == pytest.approx(0.5 * Y_base)


def test_correction_factor_bad_inputs():
    with pytest.raises(ValueError):
        yield_kt_composite(1.0, 1.0, 2.0, "delta-WGPu", "WGU", correction_factor=0.0)
    with pytest.raises(ValueError):
        yield_kt_composite(1.0, 1.0, 2.0, "delta-WGPu", "WGU", correction_factor=-1.0)


# ---------------------------------------------------------------------------
# compression_composite: invert yield -> eta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m_in,m_out,eta",
    [
        (3.0, 5.0, 2.5),
        (4.0, 8.0, 3.0),
        (1.5, 2.5, 4.0),
        (6.0, 4.0, 2.0),
    ],
)
def test_compression_composite_round_trip(m_in, m_out, eta):
    Y = yield_kt_composite(m_in, m_out, eta, "delta-WGPu", "WGU")
    eta_back = compression_composite(m_in, m_out, Y, "delta-WGPu", "WGU")
    assert eta_back == pytest.approx(eta, rel=1e-6)


@pytest.mark.parametrize("factor", [0.1, 0.5, 2.0])
def test_compression_composite_round_trip_with_correction(factor):
    """The correction factor on the inverse must match the forward."""
    m_in, m_out, eta = 3.0, 5.0, 2.5
    Y = yield_kt_composite(
        m_in, m_out, eta, "delta-WGPu", "WGU", correction_factor=factor
    )
    eta_back = compression_composite(
        m_in, m_out, Y, "delta-WGPu", "WGU", correction_factor=factor
    )
    assert eta_back == pytest.approx(eta, rel=1e-6)


def test_compression_composite_single_inner_matches_cochran():
    """In the single-material limit, the composite inverse equals the
    Cochran 6.49 inverse from model.compression."""
    m = 6.1
    Y = 5.0
    eta_comp = compression_composite(m, 0.0, Y, "delta-WGPu", "WGU")
    eta_single = compression(m, Y, "delta-WGPu", model="6.49")
    assert eta_comp == pytest.approx(eta_single, rel=1e-12)


def test_compression_composite_single_outer_matches_cochran():
    m = 30.0
    Y = 2.0
    eta_comp = compression_composite(0.0, m, Y, "delta-WGPu", "WGU")
    eta_single = compression(m, Y, "WGU", model="6.49")
    assert eta_comp == pytest.approx(eta_single, rel=1e-12)


def test_compression_composite_below_critical_returns_eta_c():
    """Y_kt = 0 means the system is at threshold; return the critical eta."""
    eta_c = critical_compression(2.0, 3.0, "delta-WGPu", "WGU")
    eta_zero = compression_composite(2.0, 3.0, 0.0, "delta-WGPu", "WGU")
    assert eta_zero == pytest.approx(eta_c, rel=1e-9)


def test_compression_composite_monotone_in_yield():
    """Higher observed yield should imply higher fitted compression."""
    m_in, m_out = 3.0, 5.0
    ys = [0.1, 1.0, 5.0, 20.0, 100.0]
    etas = [compression_composite(m_in, m_out, y, "delta-WGPu", "WGU") for y in ys]
    assert etas == sorted(etas)


def test_compression_composite_bad_inputs():
    with pytest.raises(ValueError):
        compression_composite(1.0, 1.0, -1.0, "delta-WGPu", "WGU")
    with pytest.raises(ValueError):
        compression_composite(-1.0, 1.0, 1.0, "delta-WGPu", "WGU")
    with pytest.raises(ValueError):
        compression_composite(0.0, 0.0, 1.0, "delta-WGPu", "WGU")


# ---------------------------------------------------------------------------
# Historical-test calibration anchors. These document expected fit-eta values
# for the three data points used in the README's Path 1 calibration. Failing
# this test means the model output has shifted relative to those documented
# values -- either an honest fix or a regression.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,m_pu,m_heu,Y_obs,eta_expected",
    [
        ("RDS-4",    4.2, 6.8, 28.0, 2.34),
        ("Low Tony", 0.9, 1.4,  1.0, 4.04),
        ("CHIC-12",  2.0, 0.5, 15.0, 5.35),
    ],
)
def test_historical_fit_eta(label, m_pu, m_heu, Y_obs, eta_expected):
    eta_fit = compression_composite(m_pu, m_heu, Y_obs, "delta-WGPu", "WGU")
    assert eta_fit == pytest.approx(eta_expected, rel=0.01), (
        f"{label}: expected eta~{eta_expected}, got {eta_fit:.3f}"
    )


# ---------------------------------------------------------------------------
# Same-material structural sanity: a composite with both regions the same
# material must reproduce the bare-sphere single-material result. This is
# the genuine apples-to-apples test (not the special-case mass=0 dispatch),
# and exercises the full two-region eigenvalue solve.
#
# Residual is due to the small rounding inconsistency in Cochran Table 3.1
# (his tabulated M_c, R_c, rho do not exactly satisfy M_c = (4/3) pi R_c^3 rho
# at the 3-decimal precision he reports). The structural reduction is exact
# in the math.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "material,M,eta",
    [
        ("delta-WGPu", 6.0, 2.5),
        ("delta-WGPu", 6.0, 3.0),
        ("delta-WGPu", 6.1, 3.0),
        ("delta-WGPu", 10.0, 2.5),
        ("WGU", 20.0, 2.5),
        ("WGU", 20.0, 3.0),
        ("WGU", 60.0, 2.0),
        ("alpha-WGPu", 5.0, 3.0),
    ],
)
def test_same_material_composite_matches_single(material, M, eta):
    """Pu/Pu and HEU/HEU composites at total mass M must give the same yield
    as a single-material sphere of mass M (Cochran eq. 6.49) to within ~1%."""
    Y_single = yield_kt(M, eta, material, model="6.49")
    Y_comp = yield_kt_composite(M / 2, M / 2, eta, material, material)
    if Y_single > 0:
        rel = abs(Y_comp - Y_single) / Y_single
        assert rel < 0.01, f"{material} M={M} eta={eta}: rel={rel:.4%}"
    else:
        assert Y_comp == 0.0


def test_fat_man_pu_pu_composite_consistency():
    """6.1 kg Pu yielding 20 kt: the fit eta should be the same whether
    treated as a bare sphere or as a 3.05+3.05 Pu/Pu composite."""
    eta_single = compression(6.1, 20.0, "delta-WGPu", model="6.49")
    eta_comp = compression_composite(
        3.05, 3.05, 20.0, "delta-WGPu", "delta-WGPu"
    )
    assert eta_comp == pytest.approx(eta_single, rel=0.001)
    # And both should sit in Cochran's nominal Fat Man window (2 < eta < 4).
    assert 2.0 < eta_single < 4.0
    assert 2.0 < eta_comp < 4.0


def test_same_material_critical_mass_matches():
    """f=0.5 Pu/Pu critical mass at eta=1 == bare Pu critical mass."""
    pu = get_material("delta-WGPu")
    M_c_comp = critical_mass_composite(0.5, 1.0, pu, pu)
    assert M_c_comp == pytest.approx(pu.M0, rel=0.01)
