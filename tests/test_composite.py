"""Tests for the composite two-region Pu+HEU yield function.

The expected values are pinned from the standalone extract's own
smoke-test cases (which were cross-checked against the full
fissionyield package), so any regression in the eigenvalue solver
or the radii calculation will trip these.
"""

import pytest

from fissionyield import yield_kt_composite


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
