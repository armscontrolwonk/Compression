"""Tests for the historical-anchor library."""

import pytest

from fissionyield import (
    anchors,
    composite_tests,
    compression,
    fit_eta,
    get_test,
    pure_fission_only,
)


def test_anchors_nonempty():
    ts = anchors()
    assert len(ts) >= 7
    names = {t.name for t in ts}
    assert {"Fat Man", "SANDSTONE X-Ray", "RDS-4", "Buffalo Kite",
            "Low Tony", "Kazakh effects device", "CHIC-12"}.issubset(names)


def test_composite_only_filters_correctly():
    cs = composite_tests()
    for t in cs:
        assert t.pu_kg > 0 and t.heu_kg > 0
    # Fat Man is a pure-Pu test; should not be in composite_tests
    assert "Fat Man" not in {t.name for t in cs}


def test_pure_fission_filter_excludes_boosted():
    pf = pure_fission_only()
    for t in pf:
        assert not t.boosted
    assert "CHIC-12" not in {t.name for t in pf}


def test_get_test_lookup_case_insensitive():
    assert get_test("rds-4").name == "RDS-4"
    assert get_test("FAT MAN").name == "Fat Man"


def test_get_test_unknown_raises():
    with pytest.raises(KeyError):
        get_test("nonexistent")


def test_fat_man_fit_eta_matches_single_material():
    """Fat Man is a pure-Pu test (heu_kg=0; b_Pu=1.0). The fit-eta is the
    same whether we use the default Serber calibration or rigorous, and
    matches the single-material Cochran 6.49 inverse."""
    t = get_test("Fat Man")
    eta_hist = fit_eta(t)  # default Serber, b_Pu=1.0 -> no damping
    eta_single = compression(t.pu_kg, t.yield_kt, "delta-WGPu", model="6.49")
    assert eta_hist == pytest.approx(eta_single, rel=1e-9)


@pytest.mark.parametrize(
    "name,eta_expected",
    [
        # Default Serber-b calibration.
        ("Fat Man",                3.021),  # pure Pu, unchanged vs rigorous
        ("SANDSTONE X-Ray",        3.633),  # US 1948 composite
        ("RDS-4",                  2.495),
        ("Buffalo Kite",           4.670),  # UK pure-Pu first air-drop
        ("Low Tony",               2.681),  # UK; mass split from MoD doc A/1171
        ("Kazakh effects device",  5.476),  # planned yield (untested)
        ("CHIC-12",                5.466),
    ],
)
def test_fit_eta_anchors_match_documented(name, eta_expected):
    eta = fit_eta(name)
    assert eta == pytest.approx(eta_expected, rel=0.01)


def test_fit_eta_correction_factor_lifts_eta():
    """Applying an empirical correction_factor < 1 (data-driven damping)
    raises the effective eta needed to explain a given observed yield.
    This is independent of the foundational Serber-b calibration."""
    name = "RDS-4"
    eta_base = fit_eta(name)  # default Serber + correction=1.0
    eta_damped = fit_eta(name, correction_factor=0.5)  # +50% extra damping
    assert eta_damped > eta_base


def test_fit_eta_rigorous_lowers_eta():
    """Disabling Serber's b lowers the fit-eta because the rigorous model
    predicts higher yields, so less compression is needed to reach the
    observed value (for HEU-containing composites)."""
    name = "RDS-4"
    eta_default = fit_eta(name)  # Serber default
    eta_rigorous = fit_eta(name, calibration=1.0)
    assert eta_rigorous < eta_default


def test_total_kg_property():
    t = get_test("RDS-4")
    assert t.total_kg == pytest.approx(4.2 + 6.8)


def test_is_composite_property():
    assert get_test("RDS-4").is_composite is True
    assert get_test("Fat Man").is_composite is False
