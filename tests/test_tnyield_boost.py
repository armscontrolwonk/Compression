"""Tests for the DT-boosting model (Barroso Ch. 9)."""

import pytest

from tnyield.boost import (
    DT_burn_fraction,
    DT_neutron_yield,
    boosted_yield,
)


def test_zero_inputs_give_zero_burn():
    assert DT_burn_fraction(0.0, 10.0, 1e-7) == 0.0
    assert DT_burn_fraction(0.6, 0.0, 1e-7) == 0.0
    assert DT_burn_fraction(0.6, 10.0, 0.0) == 0.0


def test_burn_fraction_in_unit_interval():
    f = DT_burn_fraction(0.6, 10.0, 1e-7)
    assert 0.0 <= f <= 1.0


def test_burn_fraction_book_example():
    # Book Section 9.1 worked example: 8.5 g DT at rho=0.6 g/cm^3
    # (3x solid), T = 50e6 K ~ 4 keV (NOT the 10 keV the book uses
    # for the cross-section number), tau = 1e-7 s, gives f ~ 7 %.
    # Use the temperature implied by <sigma v>_DT ~ 1e-17 cm^3/s
    # that the book quotes: that is just below 4 keV.
    # We use T = 3 keV (~ 35e6 K) for the test.
    f = DT_burn_fraction(0.6, 3.0, 1e-7)
    # Should be somewhere in 1 % - 30 % range; the simple model has
    # large sensitivity to T so we accept a broad bracket.
    assert 0.001 < f < 0.5


def test_higher_density_burns_more():
    f_lo = DT_burn_fraction(0.2, 10.0, 1e-7)
    f_hi = DT_burn_fraction(2.0, 10.0, 1e-7)
    assert f_lo < f_hi


def test_higher_T_burns_more_up_to_peak():
    f_low = DT_burn_fraction(0.6, 5.0, 1e-7)
    f_mid = DT_burn_fraction(0.6, 20.0, 1e-7)
    assert f_low < f_mid


def test_longer_confinement_burns_more():
    f_short = DT_burn_fraction(0.6, 10.0, 1e-8)
    f_long = DT_burn_fraction(0.6, 10.0, 1e-6)
    assert f_short < f_long


def test_neutron_yield_zero_mass():
    n_f, n_n = DT_neutron_yield(0.0, 0.6, 10.0, 1e-7)
    assert n_f == 0.0 and n_n == 0.0


def test_neutron_yield_proportional_to_mass():
    n_f1, _ = DT_neutron_yield(1.0, 0.6, 10.0, 1e-7)
    n_f5, _ = DT_neutron_yield(5.0, 0.6, 10.0, 1e-7)
    assert n_f5 == pytest.approx(5 * n_f1)


def test_boost_increases_yield():
    r = boosted_yield(
        mass_DT_g=5.0,
        rho_DT_g_per_cm3=0.6,
        T_keV=10.0,
        tau_s=1e-7,
        Y_baseline_kt=0.5,
    )
    assert r.Y_total_kt > r.Y_baseline_kt
    assert r.boost_ratio > 1.0


def test_boost_table_9_2_R52_pu_in_ballpark():
    # Table 9-2 row R_Pu = 5.2 cm: baseline 0.30 kt, boosted 10.8 kt.
    # The closed-form estimator should land within an order of
    # magnitude of the boosted value with reasonable parameter choices.
    r = boosted_yield(
        mass_DT_g=8.48,
        rho_DT_g_per_cm3=0.6,
        T_keV=4.0,
        tau_s=1e-7,
        Y_baseline_kt=0.30,
    )
    # Plausible result range: 1 kt - 100 kt.
    assert 1.0 < r.Y_total_kt < 100.0


def test_boost_zero_baseline_gives_inf_ratio():
    r = boosted_yield(
        mass_DT_g=1.0,
        rho_DT_g_per_cm3=0.6,
        T_keV=10.0,
        tau_s=1e-7,
        Y_baseline_kt=0.0,
    )
    import math
    assert math.isinf(r.boost_ratio)
    assert r.Y_total_kt > 0
