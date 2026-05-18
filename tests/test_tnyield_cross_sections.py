"""Tests for the <sigma v> formulas (Eqs. 10.40, 10.48, 10.49)."""

import math

import pytest

from tnyield.cross_sections import (
    reaction_time,
    sigma_Li6_nT,
    sigma_v_DD,
    sigma_v_DHe3,
    sigma_v_DT,
)


# DT cross section, eq. 10.40 -------------------------------------------------


def test_sigma_v_DT_zero_temperature_is_zero():
    assert sigma_v_DT(0.0) == 0.0
    assert sigma_v_DT(-1.0) == 0.0


def test_sigma_v_DT_at_10keV_matches_published():
    # Published <sigma v>_DT at 10 keV is ~ 1.1e-16 cm^3/s.
    val = sigma_v_DT(10.0)
    assert 0.9e-16 < val < 1.3e-16


def test_sigma_v_DT_increases_to_peak_then_falls():
    # DT peak is near 64 keV; should be monotonic up to ~50 keV.
    v_10 = sigma_v_DT(10.0)
    v_30 = sigma_v_DT(30.0)
    v_60 = sigma_v_DT(60.0)
    v_400 = sigma_v_DT(400.0)
    assert v_10 < v_30 < v_60
    assert v_60 > v_400  # past the peak


def test_sigma_v_DT_peak_value():
    # At ~ 64 keV expect ~ 7-9e-16 cm^3/s.
    val = sigma_v_DT(64.0)
    assert 5e-16 < val < 1e-15


# DD cross section, eq. 10.48 -------------------------------------------------


def test_sigma_v_DD_zero_temperature_is_zero():
    assert sigma_v_DD(0.0) == 0.0


def test_sigma_v_DD_continuity_across_10_keV():
    # The piecewise definition should be reasonably continuous at the
    # cross-over.  Tolerate within a factor of 2.
    lo = sigma_v_DD(9.9)
    hi = sigma_v_DD(10.0)
    assert 0.5 < lo / hi < 2.0


def test_sigma_v_DD_at_100_keV_in_ballpark():
    val = sigma_v_DD(100.0)
    # Reference (sum of branches) ~ 5e-17 cm^3/s, Barroso parametrization
    # is within an order of magnitude.
    assert 1e-17 < val < 5e-16


def test_sigma_v_DD_increases_with_T():
    for T1, T2 in [(2.0, 5.0), (5.0, 10.0), (20.0, 50.0)]:
        assert sigma_v_DD(T1) < sigma_v_DD(T2)


# D-He3 cross section, eq. 10.49 ----------------------------------------------


def test_sigma_v_DHe3_zero_below_10_keV():
    assert sigma_v_DHe3(0.0) == 0.0
    assert sigma_v_DHe3(5.0) == 0.0
    assert sigma_v_DHe3(9.999) == 0.0


def test_sigma_v_DHe3_constant_above_100_keV():
    assert sigma_v_DHe3(101.0) == 2e-16
    assert sigma_v_DHe3(500.0) == 2e-16


def test_sigma_v_DHe3_increases_in_middle_range():
    for T1, T2 in [(10.0, 30.0), (30.0, 60.0), (60.0, 100.0)]:
        assert sigma_v_DHe3(T1) < sigma_v_DHe3(T2)


# Li-6(n, alpha) T cross section -----------------------------------------------


def test_sigma_Li6_nT_anchors():
    assert sigma_Li6_nT(2.45) == pytest.approx(0.25e-24)
    assert sigma_Li6_nT(14.1) == pytest.approx(0.026e-24)


def test_sigma_Li6_nT_monotonic_in_anchor_range():
    # Larger neutron energy -> smaller cross section over this range.
    for E1, E2 in [(2.5, 5.0), (5.0, 10.0), (10.0, 14.0)]:
        assert sigma_Li6_nT(E1) > sigma_Li6_nT(E2)


def test_sigma_Li6_nT_clamps_outside_range():
    assert sigma_Li6_nT(1.0) == sigma_Li6_nT(2.45)
    assert sigma_Li6_nT(20.0) == sigma_Li6_nT(14.1)


# Reaction time helper --------------------------------------------------------


def test_reaction_time_at_solid_DT_10keV_matches_book_example():
    # Book line 10985: "Para combustivel de DT a densidade solida
    # (n ~ 5e22 atomos/cm^3) e temperatura de 10 keV, tau_r = 2.5e-7 s."
    # n here is *each* species, so the cross-section partner density.
    n = 5e22
    sv = sigma_v_DT(10.0)
    tau = reaction_time(n, sv)
    # ~ 1 / (5e22 * 1.1e-16) ~ 1.8e-7 s; book quotes 2.5e-7 s (rounded).
    assert 1e-7 < tau < 5e-7


def test_reaction_time_zero_inputs():
    assert reaction_time(0.0, 1e-16) == math.inf
    assert reaction_time(1e22, 0.0) == math.inf
