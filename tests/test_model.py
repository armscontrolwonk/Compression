"""Sanity tests for the Cochran one-group yield model."""

import math

import pytest

from fissionyield import compression, get_material, kappa, mass_kg, yield_kt


def test_kappa_definition():
    mat = get_material("delta-WGPu")
    assert kappa(mat.M0, 1.0, mat) == pytest.approx(1.0)
    assert kappa(mat.M0, 2.0, mat) == pytest.approx(4.0)
    assert kappa(2 * mat.M0, 3.0, mat) == pytest.approx(18.0)


def test_zero_yield_at_critical():
    mat = get_material("delta-WGPu")
    assert yield_kt(mat.M0, 1.0, mat) == 0.0
    assert yield_kt(mat.M0 / 4, 1.0, mat) == 0.0


def test_yield_monotonic_in_mass():
    mat = get_material("WGU")
    ys = [yield_kt(m, 2.0, mat) for m in (5.0, 10.0, 20.0, 40.0)]
    assert ys == sorted(ys)


def test_yield_monotonic_in_compression():
    mat = get_material("WGU")
    ys = [yield_kt(20.0, e, mat) for e in (1.5, 2.0, 2.5, 3.0)]
    assert ys == sorted(ys)


@pytest.mark.parametrize("model", ["6.57", "6.49"])
@pytest.mark.parametrize(
    "material,M,eta",
    [
        ("delta-WGPu", 6.1, 2.5),
        ("alpha-WGPu", 4.0, 3.0),
        ("WGU", 20.0, 2.0),
        ("U-233", 5.0, 2.5),
    ],
)
def test_round_trip_mass(model, material, M, eta):
    mat = get_material(material)
    Y = yield_kt(M, eta, mat, model=model)
    M_back = mass_kg(eta, Y, mat, model=model)
    assert M_back == pytest.approx(M, rel=1e-6)


@pytest.mark.parametrize("model", ["6.57", "6.49"])
@pytest.mark.parametrize(
    "material,M,eta",
    [
        ("delta-WGPu", 6.1, 2.5),
        ("alpha-WGPu", 4.0, 3.0),
        ("WGU", 20.0, 2.0),
        ("U-233", 5.0, 2.5),
    ],
)
def test_round_trip_compression(model, material, M, eta):
    mat = get_material(material)
    Y = yield_kt(M, eta, mat, model=model)
    eta_back = compression(M, Y, mat, model=model)
    assert eta_back == pytest.approx(eta, rel=1e-6)


def test_simplified_within_factor_of_full():
    """6.57 and 6.49 should agree to within ~30% across a super-critical range."""
    cases = [
        ("delta-WGPu", 6.0, 2.0),
        ("delta-WGPu", 10.0, 2.0),
        ("delta-WGPu", 6.0, 3.0),
        ("WGU", 30.0, 2.0),
        ("WGU", 60.0, 2.0),
        ("WGU", 30.0, 3.0),
    ]
    for material, M, eta in cases:
        mat = get_material(material)
        y57 = yield_kt(M, eta, mat, model="6.57")
        y49 = yield_kt(M, eta, mat, model="6.49")
        assert y49 > 0, (material, M, eta)
        assert 0.7 < y57 / y49 < 1.3, (material, M, eta, y57, y49)


def test_fat_man_ballpark():
    """6.1 kg delta-WGPu ~ 20 kt should require a compression in the 2-4x range."""
    mat = get_material("delta-WGPu")
    eta = compression(6.1, 20.0, mat)
    assert 2.0 < eta < 4.0


def test_below_critical_no_yield_round_trip():
    """At Y=0 the inverse solvers return the kappa=1 mass/compression."""
    mat = get_material("WGU")
    assert mass_kg(2.0, 0.0, mat) == pytest.approx(mat.M0 / 4)
    assert compression(mat.M0 / 4, 0.0, mat) == pytest.approx(2.0)


def test_unknown_material():
    with pytest.raises(KeyError):
        get_material("unobtanium")


def test_aliases_resolve():
    assert get_material("WGPu").key == "delta-WGPu"
    assert get_material("HEU").key == "WGU-93.5"
    assert get_material("aWGPu").key == "alpha-WGPu"


def test_bad_inputs():
    with pytest.raises(ValueError):
        yield_kt(-1.0, 2.0, "WGU")
    with pytest.raises(ValueError):
        yield_kt(5.0, 0.0, "WGU")
    with pytest.raises(ValueError):
        mass_kg(0.0, 1.0, "WGU")
    with pytest.raises(ValueError):
        compression(5.0, -1.0, "WGU")
    with pytest.raises(ValueError):
        yield_kt(5.0, 2.0, "WGU", model="bogus")
