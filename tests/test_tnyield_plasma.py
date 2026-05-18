"""Tests for the plasma / radiation primitives."""

import math

import pytest

from tnyield.plasma import (
    RHOR_LIDT_PROPAGATION,
    ablation_pressure_from_energy,
    burn_fraction_lindl,
    keV_to_kelvin,
    kelvin_to_keV,
    lambda_n_DT,
    lambda_n_LiDT,
    lambda_p_DT,
    lambda_p_LiDT,
    plasma_energy_density,
    plasma_pressure,
    temperature_for_energy,
)


# Mean free paths -----------------------------------------------------------


def test_lambda_n_DT_solid_density():
    # Solid DT ~ 0.2 g/cm^3 -> 4.6/0.2 = 23 cm.
    assert lambda_n_DT(0.2) == pytest.approx(23.0)


def test_lambda_n_LiDT_solid_density():
    # LiDT ~ 0.85 g/cm^3 (book) -> 3.0/0.85 ~ 3.53 cm.
    assert lambda_n_LiDT(0.85) == pytest.approx(3.529, rel=1e-3)


def test_lambda_p_scales_as_T_to_7_2():
    rho = 1.0
    a, b = lambda_p_DT(1.0, rho), lambda_p_DT(2.0, rho)
    assert b / a == pytest.approx(2.0 ** 3.5, rel=1e-6)
    a, b = lambda_p_LiDT(1.0, rho), lambda_p_LiDT(2.0, rho)
    assert b / a == pytest.approx(2.0 ** 3.5, rel=1e-6)


def test_lambda_zero_density_is_inf():
    assert lambda_n_DT(0.0) == math.inf
    assert lambda_p_DT(10.0, 0.0) == math.inf
    assert lambda_n_LiDT(0.0) == math.inf
    assert lambda_p_LiDT(10.0, 0.0) == math.inf


# Burn fraction (Lindl) -----------------------------------------------------


def test_burn_fraction_zero_rhoR():
    assert burn_fraction_lindl(0.0) == 0.0
    assert burn_fraction_lindl(-1.0) == 0.0


def test_burn_fraction_default_DT():
    # H_B = 7, rho-R = 7 -> 50 %.
    assert burn_fraction_lindl(7.0) == pytest.approx(0.5)


def test_burn_fraction_high_rhoR_saturates():
    assert burn_fraction_lindl(1000.0) > 0.99
    assert burn_fraction_lindl(1000.0) < 1.0


def test_burn_fraction_LiD_default_H_B():
    # rho-R = 20 g/cm^2, H_B = 20 -> 50 %.
    assert burn_fraction_lindl(20.0, 20.0) == pytest.approx(0.5)


# Section 10.4.2 ablation calculation --------------------------------------


def test_ablation_pressure_book_example():
    # Book worked example: 5 kt deposited in 9.14e4 cm^3 of Li
    # (Z = 3, N = 5e22 atoms/cm^3) -> T ~ 21e6 K, P ~ 1070 Mbar.
    T_K, P_Mbar = ablation_pressure_from_energy(
        E_kt=5.0, V_cm3=9.14e4, N_cm3=5e22, Z=3,
    )
    # Allow 10 % envelope on each.
    assert 19e6 < T_K < 23e6
    assert 950.0 < P_Mbar < 1200.0


def test_plasma_energy_density_components_book():
    # At T = 21e6 K, Z = 3, N = 5e22:
    #   plasma part = 1.5 * 4 * 5e22 * k * T,
    #   radiation part = a * T^4,
    # together comprising the 5 kt deposited in V = 9.14e4 cm^3.
    from tnyield.constants import ERG_PER_KT
    u = plasma_energy_density(21e6, 5e22, 3)
    E_erg = u * 9.14e4
    assert 4.5 * ERG_PER_KT < E_erg < 5.5 * ERG_PER_KT


def test_plasma_pressure_components_book():
    # 580 Mbar plasma + 490 Mbar radiation = 1070 Mbar total.
    P = plasma_pressure(21e6, 5e22, 3) / 1e12
    assert 1020 < P < 1120


def test_temperature_for_energy_roundtrip():
    # Pick a T, get u*V = E, then invert and expect to recover T.
    T_truth = 1.5e7
    N = 5e22
    Z = 3
    V = 1e5
    u = plasma_energy_density(T_truth, N, Z)
    E = u * V
    T_back = temperature_for_energy(E, V, N, Z)
    assert T_back == pytest.approx(T_truth, rel=1e-3)


# Temperature conversions ---------------------------------------------------


def test_keV_kelvin_roundtrip():
    for T in [1.0, 10.0, 100.0]:
        assert kelvin_to_keV(keV_to_kelvin(T)) == pytest.approx(T)


def test_one_keV_is_about_1_16e7_K():
    assert keV_to_kelvin(1.0) == pytest.approx(1.1604e7, rel=1e-3)


def test_RHOR_threshold_constant():
    # The book quotes the Fraley threshold as 2.2 g/cm^2.
    assert RHOR_LIDT_PROPAGATION == pytest.approx(2.2)
