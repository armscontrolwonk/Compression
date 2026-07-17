"""Tests for the multi-stage estimator (yield_kt_total / solve_mass)."""

import math

import pytest

from tnyield import (
    SOLVABLE_MASSES,
    TotalYieldResult,
    solve_mass,
    yield_kt_total,
)


# Forward direction --------------------------------------------------------


def test_pure_primary_matches_composite():
    """No boost, no secondary -> equals yield_kt_composite."""
    from fissionyield import yield_kt_composite
    r = yield_kt_total(m_pu_kg=6.1, eta=2.5)
    expected = yield_kt_composite(m_pu_kg=6.1, m_heu_kg=0.0, eta=2.5)
    assert r.Y_total_kt == pytest.approx(expected)
    assert r.Y_primary_kt == pytest.approx(expected)
    assert r.Y_boost_fusion_kt == 0.0
    assert r.Y_layer_cake_fusion_kt == 0.0
    assert r.Y_jacket_kt == 0.0


def test_zero_masses_zero_yield():
    r = yield_kt_total()
    assert r.Y_total_kt == 0.0


def test_negative_mass_raises():
    with pytest.raises(ValueError):
        yield_kt_total(m_pu_kg=-1.0)


def test_zero_eta_raises():
    with pytest.raises(ValueError):
        yield_kt_total(m_pu_kg=6.0, eta=0.0)


def test_boost_only_adds_to_primary():
    r0 = yield_kt_total(m_pu_kg=6.1, eta=2.5)
    rb = yield_kt_total(m_pu_kg=6.1, eta=2.5, m_dt_g=4.0)
    assert rb.Y_total_kt > r0.Y_total_kt
    assert rb.Y_primary_kt == pytest.approx(r0.Y_primary_kt)
    assert rb.Y_boost_fusion_kt > 0
    assert rb.Y_boost_added_fission_kt > 0


def test_layer_cake_contribution_independent_of_primary():
    """LiD layer mass should add the same amount regardless of primary."""
    r1 = yield_kt_total(m_pu_kg=6.1, eta=2.5)
    r2 = yield_kt_total(m_pu_kg=6.1, eta=2.5, m_li6d_layer_kg=20.0)
    delta = r2.Y_layer_cake_fusion_kt
    assert delta > 0
    # Same LiD added to a different baseline gives the same LiD energy
    r3 = yield_kt_total(m_pu_kg=4.0, m_heu_kg=8.0, eta=3.0, m_li6d_layer_kg=20.0)
    assert r3.Y_layer_cake_fusion_kt == pytest.approx(delta)


def test_breakdown_sums_to_total():
    r = yield_kt_total(
        m_pu_kg=4.2, m_heu_kg=6.8, eta=3.0,
        m_dt_g=4.0,
        m_li6d_layer_kg=20.0, m_u238_layer_kg=100.0,
        m_lid_secondary_kg=30.0, m_spark_plug_kg=5.0,
        m_u238_tamper_kg=150.0, m_u238_jacket_kg=400.0,
    )
    components = (
        r.Y_primary_kt
        + r.Y_boost_fusion_kt + r.Y_boost_added_fission_kt
        + r.Y_layer_cake_fusion_kt + r.Y_layer_cake_tamper_kt
        + r.Y_spark_plug_kt + r.Y_secondary_fusion_kt + r.Y_secondary_tamper_kt
        + r.Y_jacket_kt
    )
    assert components == pytest.approx(r.Y_total_kt)


def test_joe4_style_ballpark():
    """RDS-6s / Joe-4 (1953) was ~ 400 kt total."""
    r = yield_kt_total(
        m_pu_kg=4.0, m_heu_kg=6.0, eta=2.8,
        m_li6d_layer_kg=50.0, li6_enrichment_layer=0.40,
        m_u238_jacket_kg=300.0,
    )
    assert 200.0 < r.Y_total_kt < 800.0


def test_W87_style_ballpark():
    """W-87 is ~ 298 kt. Uses the compact ~2.5 kg LiD charge and the
    Barroso-grounded default burn fraction (~90%), not a hand-set value."""
    r = yield_kt_total(
        m_pu_kg=4.2, m_heu_kg=6.8, eta=3.0, m_dt_g=4.0,
        m_lid_secondary_kg=2.5, m_spark_plug_kg=10.0,
        spark_plug_material="WGU",
        m_u238_tamper_kg=520.0, m_u238_jacket_kg=1320.0,
        u238_tamper_burn_fraction=0.0018,
        jacket_burn_fraction=0.0018,
    )
    # Within a factor of ~1.5 of 298 kt.
    assert 200.0 < r.Y_total_kt < 450.0


def test_spark_plug_material_switch():
    r_pu = yield_kt_total(m_pu_kg=4.0, eta=2.5,
                          m_spark_plug_kg=5.0, m_lid_secondary_kg=20.0,
                          spark_plug_material="delta-WGPu")
    r_u = yield_kt_total(m_pu_kg=4.0, eta=2.5,
                         m_spark_plug_kg=5.0, m_lid_secondary_kg=20.0,
                         spark_plug_material="WGU")
    # U-235 vs Pu-239 energy per gram differ slightly; check the spark
    # plug contribution moved.
    assert r_pu.Y_spark_plug_kt != r_u.Y_spark_plug_kt


# Inverse direction -------------------------------------------------------


def test_solve_li6d_layer_round_trip():
    """Forward then inverse should recover the same mass."""
    common = dict(m_pu_kg=4.0, m_heu_kg=6.0, eta=2.8,
                  li6_enrichment_layer=0.40)
    Y_target = yield_kt_total(m_li6d_layer_kg=50.0, **common).Y_total_kt
    m_solved = solve_mass(
        target_Y_kt=Y_target,
        unknown="m_li6d_layer_kg",
        **common,
    )
    assert m_solved == pytest.approx(50.0, rel=1e-3)


def test_solve_m_dt_g_round_trip():
    common = dict(m_pu_kg=6.1, eta=2.5)
    Y_target = yield_kt_total(m_dt_g=4.0, **common).Y_total_kt
    m_solved = solve_mass(
        target_Y_kt=Y_target,
        unknown="m_dt_g",
        **common,
    )
    assert m_solved == pytest.approx(4.0, rel=1e-3)


def test_solve_jacket_round_trip():
    common = dict(m_pu_kg=4.0, m_heu_kg=6.0, eta=2.8,
                  m_li6d_layer_kg=50.0, li6_enrichment_layer=0.40)
    Y_target = yield_kt_total(m_u238_jacket_kg=200.0, **common).Y_total_kt
    m_solved = solve_mass(
        target_Y_kt=Y_target,
        unknown="m_u238_jacket_kg",
        **common,
    )
    assert m_solved == pytest.approx(200.0, rel=1e-3)


def test_solve_m_pu_round_trip():
    common = dict(m_heu_kg=0.0, eta=2.5)
    Y_target = yield_kt_total(m_pu_kg=6.0, **common).Y_total_kt
    m_solved = solve_mass(
        target_Y_kt=Y_target,
        unknown="m_pu_kg",
        **common,
    )
    assert m_solved == pytest.approx(6.0, rel=1e-3)


def test_solve_unknown_passed_in_kwargs_ignored():
    """Explicit value for `unknown` in kwargs should be silently dropped."""
    Y_target = yield_kt_total(m_pu_kg=6.1, eta=2.5,
                              m_li6d_layer_kg=20.0).Y_total_kt
    m_solved = solve_mass(
        target_Y_kt=Y_target,
        unknown="m_li6d_layer_kg",
        m_pu_kg=6.1, eta=2.5,
        m_li6d_layer_kg=999.0,        # ignored
    )
    assert m_solved == pytest.approx(20.0, rel=1e-3)


def test_solve_target_already_exceeded_returns_zero():
    """If primary alone already gives more than target, solve for 0."""
    Y_target = 0.1  # less than 6.1 kg Pu eta=2.5 (= 5.5 kt)
    m_solved = solve_mass(
        target_Y_kt=Y_target,
        unknown="m_dt_g",
        m_pu_kg=6.1, eta=2.5,
    )
    assert m_solved == 0.0


def test_solve_invalid_unknown_raises():
    with pytest.raises(ValueError):
        solve_mass(target_Y_kt=10.0, unknown="not_a_mass", m_pu_kg=6.0, eta=2.5)


def test_solve_negative_target_raises():
    with pytest.raises(ValueError):
        solve_mass(target_Y_kt=-1.0, unknown="m_pu_kg", eta=2.5)


def test_all_solvable_masses_round_trip():
    """Every parameter listed in SOLVABLE_MASSES should round-trip."""
    base = dict(
        m_pu_kg=4.0, m_heu_kg=6.0, eta=3.0,
        m_dt_g=3.0,
        m_li6d_layer_kg=20.0, m_u238_layer_kg=100.0,
        m_lid_secondary_kg=30.0, m_spark_plug_kg=5.0,
        m_u238_tamper_kg=150.0, m_u238_jacket_kg=400.0,
        m_fissile_secondary_kg=25.0,
    )
    Y_target = yield_kt_total(**base).Y_total_kt
    for unknown in SOLVABLE_MASSES:
        truth = base[unknown]
        kwargs = {k: v for k, v in base.items() if k != unknown}
        solved = solve_mass(target_Y_kt=Y_target, unknown=unknown, **kwargs)
        assert solved == pytest.approx(truth, rel=5e-3), (
            f"round-trip failed for {unknown}: {truth} != {solved}"
        )
