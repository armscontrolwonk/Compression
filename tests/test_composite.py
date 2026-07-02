"""Tests for the composite two-region Pu+HEU yield function.

The expected values are pinned from the standalone extract's own
smoke-test cases (which were cross-checked against the full
fissionyield package), so any regression in the eigenvalue solver
or the radii calculation will trip these.
"""

import math

import pytest

from fissionyield import (
    effective_compression,
    yield_band,
    yield_kt_composite,
)


@pytest.mark.parametrize(
    "m_pu, m_heu, eta, expected",
    [
        (6.1, 0.0, 2.5, 5.530),       # pure Pu, Fat Man path
        (6.1, 0.0, 3.0, 19.190),      # pure Pu, eta=3
        (0.0, 60.0, 2.0, 104.403),    # pure HEU, Serber halves
        (4.2, 6.8, 5.0, 532.192),     # RDS-4 composite at eta=5
        (4.0, 8.0, 2.5, 34.464),      # 4 Pu + 8 HEU, eta=2.5
        (2.0, 0.0, 4.67, 3.002),      # Buffalo Kite anchor (UK 1956)
    ],
)
def test_composite_smoke_values(m_pu, m_heu, eta, expected):
    """The standalone extract's six anchor cases."""
    Y = yield_kt_composite(m_pu_kg=m_pu, m_heu_kg=m_heu, eta=eta)
    assert Y == pytest.approx(expected, rel=0.01)


def test_zero_masses_zero_yield():
    assert yield_kt_composite(m_pu_kg=0.0, m_heu_kg=0.0, eta=3.0) == 0.0


def test_subcritical_returns_zero():
    # 1 kg of Pu at eta=1 is subcritical
    assert yield_kt_composite(m_pu_kg=1.0, m_heu_kg=0.0, eta=1.0) == 0.0


def test_negative_mass_raises():
    with pytest.raises(ValueError):
        yield_kt_composite(m_pu_kg=-1.0, m_heu_kg=0.0, eta=2.0)


def test_zero_compression_raises():
    with pytest.raises(ValueError):
        yield_kt_composite(m_pu_kg=6.0, m_heu_kg=0.0, eta=0.0)


def test_serber_b_calibration_disable_changes_pure_HEU():
    # b=1.0 should give the uncalibrated (larger) prediction
    Y_default = yield_kt_composite(m_pu_kg=0.0, m_heu_kg=60.0, eta=2.0)
    Y_no_b = yield_kt_composite(m_pu_kg=0.0, m_heu_kg=60.0, eta=2.0,
                                b_pu=1.0, b_heu=1.0)
    assert Y_no_b == pytest.approx(2.0 * Y_default, rel=0.01)


def test_serber_b_no_effect_on_pure_Pu():
    # b=1.0 for Pu in both defaults and disabled state -> same value
    Y1 = yield_kt_composite(m_pu_kg=6.0, m_heu_kg=0.0, eta=3.0)
    Y2 = yield_kt_composite(m_pu_kg=6.0, m_heu_kg=0.0, eta=3.0,
                            b_pu=1.0, b_heu=1.0)
    assert Y1 == pytest.approx(Y2, rel=1e-9)


def test_correction_factor_scales_linearly():
    Y0 = yield_kt_composite(m_pu_kg=4.0, m_heu_kg=8.0, eta=2.5)
    Y_scaled = yield_kt_composite(m_pu_kg=4.0, m_heu_kg=8.0, eta=2.5,
                                  correction_factor=1.5)
    assert Y_scaled == pytest.approx(1.5 * Y0, rel=1e-9)


# ---------------------------------------------------------------------------
# effective_compression (inverse: known yield -> effective eta)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "m_pu, m_heu, eta_true",
    [
        (0.9, 5.6, 2.7),    # Low Tony-like composite
        (4.0, 8.0, 2.5),    # balanced composite
        (6.1, 0.0, 2.5),    # pure Pu
        (0.0, 60.0, 2.0),   # pure HEU
    ],
)
def test_effective_compression_round_trips(m_pu, m_heu, eta_true):
    """Forward then inverse recovers the compression that produced Y."""
    Y = yield_kt_composite(m_pu, m_heu, eta_true)
    fit = effective_compression(m_pu, m_heu, Y)
    assert fit.eta_eff == pytest.approx(eta_true, rel=1e-4)


def test_effective_compression_low_tony_anchor():
    """Low Tony (0.9 Pu + 5.6 HEU, published ~1.0 kt) -> eta_eff ~ 2.68,
    barely supercritical (a few crits) with high elasticity."""
    fit = effective_compression(0.9, 5.6, 1.0)
    assert fit.eta_eff == pytest.approx(2.68, abs=0.02)
    assert fit.eta_c == pytest.approx(1.96, abs=0.02)
    assert 1.5 < fit.crits < 2.5          # only ~2 crits above critical
    assert fit.elasticity > 5.0           # knife-edge


def test_effective_compression_result_reproduces_yield():
    fit = effective_compression(4.2, 6.8, 200.0)
    assert yield_kt_composite(4.2, 6.8, fit.eta_eff) == pytest.approx(200.0, rel=1e-3)


def test_effective_compression_invalid_inputs():
    with pytest.raises(ValueError):
        effective_compression(-1.0, 5.0, 1.0)
    with pytest.raises(ValueError):
        effective_compression(0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        effective_compression(1.0, 1.0, 0.0)       # non-positive yield
    with pytest.raises(ValueError):
        effective_compression(1.0, 1.0, -5.0)


def test_effective_compression_no_yield_ceiling():
    # Cochran's closed form is unbounded in eta: even a tiny device can
    # reach a large yield at (absurd) compression, so there is no
    # "unreachable" finite yield -- the inverse returns a large eta_eff
    # rather than raising, and it still round-trips.
    fit = effective_compression(0.1, 0.0, 1000.0)
    assert fit.eta_eff > 10.0
    assert yield_kt_composite(0.1, 0.0, fit.eta_eff) == pytest.approx(1000.0, rel=1e-3)


# ---------------------------------------------------------------------------
# yield_band (forward: eta range -> yield band + elasticity)
# ---------------------------------------------------------------------------

def test_yield_band_brackets_nominal_and_is_monotonic():
    b = yield_band(0.9, 5.6, 3.0)
    assert b.y_low <= b.y_nominal <= b.y_high
    assert b.eta_low < b.eta_nominal < b.eta_high
    assert b.y_nominal == pytest.approx(yield_kt_composite(0.9, 5.6, 3.0))


def test_yield_band_default_frac_is_ten_percent():
    b = yield_band(4.0, 8.0, 2.5)
    assert b.eta_low == pytest.approx(2.25)
    assert b.eta_high == pytest.approx(2.75)


def test_yield_band_explicit_eta_range():
    b = yield_band(0.9, 5.6, 3.0, eta_low=2.5, eta_high=3.5)
    assert b.eta_low == 2.5 and b.eta_high == 3.5
    assert b.y_low == pytest.approx(yield_kt_composite(0.9, 5.6, 2.5))
    assert b.y_high == pytest.approx(yield_kt_composite(0.9, 5.6, 3.5))


def test_yield_band_elasticity_positive_when_supercritical():
    b = yield_band(4.0, 8.0, 2.5)
    assert b.elasticity > 0.0
    assert not math.isnan(b.elasticity)


def test_yield_band_barely_critical_is_wider_than_deep_supercritical():
    """Relative band width shrinks as a device gets further above critical."""
    shallow = yield_band(0.9, 5.6, 2.2)     # ~1.25 crits, knife-edge
    deep = yield_band(0.9, 5.6, 5.0)        # deeply supercritical
    width_shallow = (shallow.y_high - shallow.y_low) / shallow.y_nominal
    width_deep = (deep.y_high - deep.y_low) / deep.y_nominal
    assert width_shallow > width_deep


def test_yield_band_invalid_inputs():
    with pytest.raises(ValueError):
        yield_band(1.0, 1.0, 0.0)                 # non-positive nominal
    with pytest.raises(ValueError):
        yield_band(1.0, 1.0, 2.5, frac=1.0)       # frac out of range
    with pytest.raises(ValueError):
        yield_band(1.0, 1.0, 2.5, eta_low=3.0, eta_high=2.0)   # low > high
