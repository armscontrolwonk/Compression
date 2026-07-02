"""Tests for the historical-anchor library and its inversion."""

import math

import pytest

from fissionyield import yield_kt_composite
from fissionyield import historical as H


def test_anchor_library_loads():
    tests = H.anchors()
    assert len(tests) == 7
    names = {t.name for t in tests}
    assert "Fat Man" in names
    assert "Low Tony" in names
    assert "CHIC-12" in names


def test_get_test_case_insensitive():
    assert H.get_test("low tony").name == "Low Tony"
    assert H.get_test("FAT MAN").name == "Fat Man"


def test_get_test_unknown_raises():
    with pytest.raises(KeyError):
        H.get_test("Ivy Mike")


def test_composite_and_pure_filters():
    comp = {t.name for t in H.composite_tests()}
    assert "SANDSTONE X-Ray" in comp   # both Pu and HEU
    assert "Fat Man" not in comp       # pure Pu
    unboosted = {t.name for t in H.pure_fission_only()}
    assert "CHIC-12" not in unboosted  # boosted excluded
    assert "Fat Man" in unboosted


def test_properties():
    fm = H.get_test("Fat Man")
    assert fm.total_kg == pytest.approx(6.1)
    assert not fm.is_composite
    lt = H.get_test("Low Tony")
    assert lt.is_composite
    assert lt.total_kg == pytest.approx(6.5)


@pytest.mark.parametrize("name", [t.name for t in H.anchors()])
def test_every_anchor_inverts_and_round_trips(name):
    """Each anchor's fitted eta reproduces its observed yield."""
    f = H.fit(name)
    t = f.test
    assert f.eta_eff > f.eta_c                    # above yield onset
    assert f.crits > 1.0                          # supercritical
    assert not math.isnan(f.elasticity)
    y = yield_kt_composite(t.pu_kg, t.heu_kg, f.eta_eff)
    assert y == pytest.approx(t.yield_kt, rel=1e-3)


def test_low_tony_fit_matches_expected():
    f = H.fit("Low Tony")
    assert f.eta_eff == pytest.approx(2.68, abs=0.02)
    assert f.elasticity > 8.0        # knife-edge


def test_buffalo_kite_fit_matches_smoke_anchor():
    # 2.0 kg Pu, 3.0 kt -> eta ~ 4.67 (matches the composite smoke case)
    f = H.fit("Buffalo Kite")
    assert f.eta_eff == pytest.approx(4.67, abs=0.02)


def test_calibration_table_sorted_by_year_year0_last():
    table = H.calibration_table()
    assert len(table) == 7
    years = [f.test.year for f in table]
    # CHIC-12 has year 0 (unknown) and must sort last
    assert years[-1] == 0
    nonzero = [y for y in years if y]
    assert nonzero == sorted(nonzero)


def test_rigorous_calibration_gives_higher_eta():
    """Disabling Serber-b (b_heu 0.5 -> 1.0) raises the raw yield for a
    fixed eta, so reproducing the same observed yield needs LESS
    compression -- rigorous eta should be <= Serber-b eta for a device
    with HEU. Check on a pure-HEU-heavy composite (Low Tony)."""
    serber = H.fit("Low Tony").eta_eff
    rigorous = H.fit("Low Tony", b_pu=1.0, b_heu=1.0).eta_eff
    assert rigorous < serber


def test_fit_accepts_test_object_or_name():
    obj = H.get_test("RDS-4")
    assert H.fit(obj).eta_eff == pytest.approx(H.fit("RDS-4").eta_eff)
