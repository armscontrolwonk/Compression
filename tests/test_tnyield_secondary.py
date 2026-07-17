"""Tests for the Teller-Ulam secondary model."""

import pytest

from tnyield.secondary import secondary_yield


def test_zero_secondary_components_returns_primary():
    r = secondary_yield(
        Y_primary_kt=50.0,
        mass_LiD_kg=0.0,
        mass_U238_tamper_kg=0.0,
    )
    assert r.Y_total_kt == 50.0


def test_total_is_sum_of_components():
    r = secondary_yield(
        Y_primary_kt=50.0,
        mass_LiD_kg=50.0,
        mass_U238_tamper_kg=500.0,
        mass_spark_plug_kg=10.0,
        compression=50.0,
    )
    assert r.Y_total_kt == pytest.approx(
        r.Y_primary_kt + r.Y_spark_plug_kt + r.Y_fusion_kt + r.Y_tamper_kt
    )


def test_fractions_sum_to_one():
    r = secondary_yield(
        Y_primary_kt=50.0,
        mass_LiD_kg=50.0,
        mass_U238_tamper_kg=500.0,
        mass_spark_plug_kg=10.0,
    )
    s = (r.fraction_primary + r.fraction_spark_plug
         + r.fraction_fusion + r.fraction_tamper)
    assert s == pytest.approx(1.0)


def test_higher_compression_burns_more_LiD():
    common = dict(
        Y_primary_kt=50.0,
        mass_LiD_kg=50.0,
        mass_U238_tamper_kg=0.0,
    )
    low = secondary_yield(compression=10.0, **common)
    high = secondary_yield(compression=200.0, **common)
    assert high.burn_fraction_LiD > low.burn_fraction_LiD
    assert high.Y_fusion_kt > low.Y_fusion_kt


def test_W87_like_ballpark():
    # Barroso Table 10-11 W-87 example: ~ 298 kt total = 122 kt primary
    # + ~115 kt LiD fusion (90% D burn) + spark/tamper fission. The LiD
    # charge is COMPACT (~2-2.5 kg), and the Barroso-grounded default
    # burn fraction (~90%) reproduces it -- no hand-set burn fraction.
    r = secondary_yield(
        Y_primary_kt=122.0,
        mass_LiD_kg=2.5,
        mass_U238_tamper_kg=520.0,
        mass_spark_plug_kg=10.0,
        compression=80.0,
        U238_burn_fraction=0.0018,
        spark_plug_burn_fraction=0.20,
    )
    assert 0.8 < r.burn_fraction_LiD < 0.95    # Barroso W-87: ~90%
    assert 150.0 < r.Y_total_kt < 450.0        # near the 298 kt book number


def test_secondary_default_burn_is_high_for_tamped_compact_charge():
    # The Barroso-grounded default burns MOST of a compact tamped charge,
    # not "a few percent" (the earlier, mistaken assumption).
    r = secondary_yield(
        Y_primary_kt=10.0,
        mass_LiD_kg=3.0,
        mass_U238_tamper_kg=300.0,   # tamped
        compression=100.0,
    )
    assert r.burn_fraction_LiD > 0.85


def test_secondary_untamped_charge_collapses():
    # Barroso Table 10-9: without a tamper the burn collapses to ~5%.
    r = secondary_yield(
        Y_primary_kt=10.0,
        mass_LiD_kg=3.0,
        mass_U238_tamper_kg=0.0,     # no tamper
        compression=100.0,
    )
    assert r.burn_fraction_LiD < 0.10


def test_secondary_override_takes_precedence():
    r = secondary_yield(
        Y_primary_kt=10.0,
        mass_LiD_kg=20.0,
        mass_U238_tamper_kg=0.0,
        compression=100.0,
        LiD_burn_fraction=0.05,
    )
    assert r.burn_fraction_LiD == 0.05


def test_secondary_invalid_LiD_burn_fraction():
    with pytest.raises(ValueError):
        secondary_yield(
            Y_primary_kt=10.0,
            mass_LiD_kg=20.0,
            mass_U238_tamper_kg=0.0,
            LiD_burn_fraction=1.5,
        )


def test_invalid_inputs():
    with pytest.raises(ValueError):
        secondary_yield(Y_primary_kt=-1.0, mass_LiD_kg=10.0, mass_U238_tamper_kg=10.0)
    with pytest.raises(ValueError):
        secondary_yield(Y_primary_kt=1.0, mass_LiD_kg=-10.0, mass_U238_tamper_kg=10.0)
    with pytest.raises(ValueError):
        secondary_yield(
            Y_primary_kt=1.0, mass_LiD_kg=10.0,
            mass_U238_tamper_kg=10.0, compression=0.0,
        )


def test_zero_LiD_zero_fusion():
    r = secondary_yield(
        Y_primary_kt=10.0,
        mass_LiD_kg=0.0,
        mass_U238_tamper_kg=100.0,
    )
    assert r.Y_fusion_kt == 0.0
    assert r.burn_fraction_LiD == 0.0
