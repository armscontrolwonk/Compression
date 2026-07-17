"""Tests for the crude Barroso-calibrated implosion (device -> eta) front-end."""

import pytest

from fissionyield import (
    device_yield_band,
    implosion_compression,
)

# Barroso Table 8-2/8-3 achieved eta (his hydro yields inverted through the
# Cochran yield inverse for a 6 kg hollow alpha-Pu pit).
TABLE_8_2 = [(54.3, 18.2, 1.895), (108, 18.2, 2.144),
             (190, 18.2, 2.409), (302, 18.2, 2.638)]
TABLE_8_3 = [(108, 7.6, 2.003), (108, 32.2, 2.210), (108, 72.2, 2.384)]


@pytest.mark.parametrize("m_he, m_tamp, eta_true", TABLE_8_2 + TABLE_8_3)
def test_reproduces_barroso_achieved_eta(m_he, m_tamp, eta_true):
    """The fit reproduces Barroso's hydro-inverted eta to <= 0.06."""
    e = implosion_compression(m_he, 6.0, m_tamp)
    assert e.eta == pytest.approx(eta_true, abs=0.06)


def test_eta_increases_with_HE_mass():
    etas = [implosion_compression(m, 6.0, 18.2).eta for m in (54.3, 108, 190, 302)]
    assert etas == sorted(etas)


def test_eta_increases_with_tamper_confinement():
    # Barroso Table 8-3: more tamper RAISES eta despite the mass penalty.
    etas = [implosion_compression(108, 6.0, t).eta for t in (7.6, 18.2, 32.2, 72.2)]
    assert etas == sorted(etas)


def test_geometry_ordering():
    solid = implosion_compression(108, 6.0, 18.2, geometry="solid").eta
    hollow = implosion_compression(108, 6.0, 18.2, geometry="hollow").eta
    porous = implosion_compression(108, 6.0, 18.2, geometry="porous").eta
    assert solid < hollow < porous


def test_band_brackets_nominal_and_floors_at_one():
    e = implosion_compression(108, 6.0, 18.2)
    assert e.eta_low < e.eta < e.eta_high
    assert e.eta_low >= 1.0
    # tiny HE -> no meaningful compression, clamped to 1
    e0 = implosion_compression(0.0, 6.0, 0.0)
    assert e0.eta == 1.0


def test_he_relative_power_scales_like_more_HE():
    weak = implosion_compression(108, 6.0, 18.2, he_relative_power=0.5).eta
    strong = implosion_compression(108, 6.0, 18.2, he_relative_power=2.0).eta
    assert weak < strong


def test_invalid_inputs():
    with pytest.raises(ValueError):
        implosion_compression(108, 0.0, 18.2)          # no fissile
    with pytest.raises(ValueError):
        implosion_compression(-1, 6.0)                 # negative HE
    with pytest.raises(ValueError):
        implosion_compression(108, 6.0, geometry="cube")
    with pytest.raises(ValueError):
        implosion_compression(108, 6.0, he_relative_power=0.0)


# --- device_yield_band: full device -> yield band chain --------------------

@pytest.mark.parametrize("m_he, Y_barroso", [(54.3, 3.1), (108, 8.4),
                                             (190, 18.0), (302, 30.0)])
def test_device_yield_band_brackets_barroso(m_he, Y_barroso):
    """The yield band contains Barroso's hydro yield for the same device."""
    d = device_yield_band(m_he_kg=m_he, m_pu_kg=6.0, m_tamper_kg=18.2)
    assert d.Y_low <= Y_barroso <= d.Y_high
    assert d.Y_low <= d.Y_nominal <= d.Y_high


def test_device_yield_band_nominal_near_barroso():
    # Material-consistent (alpha-Pu) nominal is within ~35% of Barroso.
    d = device_yield_band(m_he_kg=108, m_pu_kg=6.0, m_tamper_kg=18.2)
    assert d.Y_nominal == pytest.approx(8.4, rel=0.35)


def test_device_yield_band_delta_pu_yields_less_than_alpha():
    # Barroso 6.7: delta-Pu is less dense (higher M0), so a delta pit yields
    # less at the same compression.
    alpha = device_yield_band(m_he_kg=108, m_pu_kg=6.0, m_tamper_kg=18.2,
                              pu_material="alpha-WGPu").Y_nominal
    delta = device_yield_band(m_he_kg=108, m_pu_kg=6.0, m_tamper_kg=18.2,
                              pu_material="delta-WGPu").Y_nominal
    assert delta < alpha


def test_device_yield_band_requires_fissile():
    with pytest.raises(ValueError):
        device_yield_band(m_he_kg=108, m_tamper_kg=18.2)
