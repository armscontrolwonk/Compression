"""Tests for the Sloika / layer-cake model."""

import pytest

from tnyield.layer_cake import layer_cake_yield, lid_yield_per_kg


def test_pure_li6d_yield_per_kg():
    # Pure Li-6 D fully burnt: ~ 64-70 kt/kg.
    y = lid_yield_per_kg(li6_enrichment=1.0)
    assert 60.0 < y < 75.0


def test_natural_lithium_is_low():
    y = lid_yield_per_kg(li6_enrichment=0.075)
    assert y < lid_yield_per_kg(0.5)
    assert y < lid_yield_per_kg(1.0)


def test_li6_enrichment_out_of_range():
    with pytest.raises(ValueError):
        lid_yield_per_kg(-0.1)
    with pytest.raises(ValueError):
        lid_yield_per_kg(1.5)


def test_zero_secondary_components_returns_primary():
    r = layer_cake_yield(
        Y_primary_kt=10.0,
        mass_LiD_kg=0.0,
        mass_U238_kg=0.0,
    )
    assert r.Y_total_kt == 10.0
    assert r.Y_fusion_kt == 0.0
    assert r.Y_tamper_kt == 0.0


def test_layer_cake_total_is_sum_of_components():
    r = layer_cake_yield(
        Y_primary_kt=10.0,
        mass_LiD_kg=20.0,
        mass_U238_kg=50.0,
        li6_enrichment=0.4,
        LiD_burn_fraction=0.15,
        U238_burn_fraction=0.10,
    )
    assert r.Y_total_kt == pytest.approx(
        r.Y_primary_kt + r.Y_fusion_kt + r.Y_tamper_kt
    )


def test_fractions_sum_to_one():
    r = layer_cake_yield(
        Y_primary_kt=10.0,
        mass_LiD_kg=20.0,
        mass_U238_kg=50.0,
    )
    s = r.fraction_primary + r.fraction_fusion + r.fraction_tamper
    assert s == pytest.approx(1.0)


def test_higher_li6_more_fusion_yield():
    common = dict(Y_primary_kt=10.0, mass_LiD_kg=20.0, mass_U238_kg=0.0,
                  LiD_burn_fraction=0.20, U238_burn_fraction=0.0)
    r_low = layer_cake_yield(li6_enrichment=0.1, **common)
    r_high = layer_cake_yield(li6_enrichment=0.95, **common)
    assert r_high.Y_fusion_kt > r_low.Y_fusion_kt


def test_layer_cake_invalid_inputs():
    with pytest.raises(ValueError):
        layer_cake_yield(Y_primary_kt=-1.0, mass_LiD_kg=10.0, mass_U238_kg=10.0)
    with pytest.raises(ValueError):
        layer_cake_yield(Y_primary_kt=1.0, mass_LiD_kg=-1.0, mass_U238_kg=10.0)
    with pytest.raises(ValueError):
        layer_cake_yield(Y_primary_kt=1.0, mass_LiD_kg=10.0, mass_U238_kg=10.0,
                         LiD_burn_fraction=1.5)
    with pytest.raises(ValueError):
        layer_cake_yield(Y_primary_kt=1.0, mass_LiD_kg=10.0, mass_U238_kg=10.0,
                         U238_burn_fraction=-0.1)


def test_joe4_like_estimate():
    # RDS-6s / "Joe-4" (1953): ~ 400 kt total, 60-200 kt from primary,
    # natural-Li / Li-6 enriched LiD layer, U-238 tamper.  Just verify
    # the calculator produces sensible numbers when fed roughly the
    # right inputs (~ 50 kg LiD, ~ 300 kg U-nat tamper, ~ 100 kt primary).
    r = layer_cake_yield(
        Y_primary_kt=100.0,
        mass_LiD_kg=50.0,
        mass_U238_kg=300.0,
        li6_enrichment=0.4,
        LiD_burn_fraction=0.20,
        U238_burn_fraction=0.10,
    )
    # Should be in 200-2000 kt range.
    assert 200.0 < r.Y_total_kt < 2000.0
