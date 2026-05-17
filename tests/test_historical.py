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
    assert len(ts) >= 4
    names = {t.name for t in ts}
    assert {"Fat Man", "RDS-4", "Low Tony", "CHIC-12"}.issubset(names)


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
    """Fat Man is a pure-Pu test (heu_kg=0). The fit-eta from the historical
    helper must equal the single-material Cochran 6.49 inverse."""
    t = get_test("Fat Man")
    eta_hist = fit_eta(t)
    eta_single = compression(t.pu_kg, t.yield_kt, "delta-WGPu", model="6.49")
    assert eta_hist == pytest.approx(eta_single, rel=1e-9)


@pytest.mark.parametrize(
    "name,eta_expected",
    [
        ("Fat Man",  3.02),
        ("RDS-4",    2.34),
        ("Low Tony", 4.04),
        ("CHIC-12",  5.35),
    ],
)
def test_fit_eta_anchors_match_documented(name, eta_expected):
    eta = fit_eta(name)
    assert eta == pytest.approx(eta_expected, rel=0.01)


def test_fit_eta_correction_factor_lowers_eta():
    """Applying a correction factor < 1 (Path 2 damping) raises the effective
    eta needed to explain a given observed yield -- the model has to work
    harder to compensate for the damping."""
    name = "RDS-4"
    eta_base = fit_eta(name, correction_factor=1.0)
    eta_damped = fit_eta(name, correction_factor=0.5)
    assert eta_damped > eta_base


def test_total_kg_property():
    t = get_test("RDS-4")
    assert t.total_kg == pytest.approx(4.2 + 6.8)


def test_is_composite_property():
    assert get_test("RDS-4").is_composite is True
    assert get_test("Fat Man").is_composite is False
