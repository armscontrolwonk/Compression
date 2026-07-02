"""Historical fission-test anchors for compression-calibration.

Each :class:`HistoricalTest` records a publicly-known nuclear test -- its
fissile-material loading, observed yield, and provenance -- so the
two-region composite model can be *inverted* to recover the effective
compression (eta) the model assigns to that test. Running the inversion
across the whole library produces a calibration ladder: the effective
eta per device, which in practice tracks implosion sophistication by era.

This is the honest calibration artifact for this model. Yield is
hypersensitive to compression for barely-critical devices (elasticity
d ln Y / d ln eta ~ 5-11), so there is no single multiplicative
"correction factor" that generalizes; the per-device effective eta is
the meaningful quantity. See `effective_compression` in `composite.py`.

Numbers in the public literature often carry +/- 20% uncertainty on
masses and looser constraints on yield, and tampers/boosting are not
modeled here -- so a fit eta is a single number under the bare-sphere
one-group model, not a measurement. Treat boosted entries' eta as an
upper bound on the pure-fission compression.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

from .composite import effective_compression


@dataclass(frozen=True)
class HistoricalTest:
    """A publicly-known nuclear test used as a calibration anchor.

    Attributes
    ----------
    name : human-readable test name or designator
    year : year of test (0 if unknown to the entry author)
    country : USA / USSR / PRC / UK / FR / ...
    pu_kg : plutonium mass in kg (0 for pure-HEU designs)
    heu_kg : HEU mass in kg (0 for pure-Pu designs)
    yield_kt : observed (or, where noted, planned) yield in kilotons
    boosted : True if the device was fusion-boosted; the fit eta then
              represents an upper bound on the pure-fission compression
              (the fusion contribution is not modeled here)
    notes : free-form caveats (mass uncertainty, design type, ...)
    source : citation or URL for the figures
    """

    name: str
    year: int
    country: str
    pu_kg: float
    heu_kg: float
    yield_kt: float
    boosted: bool = False
    notes: str = ""
    source: str = ""

    @property
    def total_kg(self) -> float:
        return self.pu_kg + self.heu_kg

    @property
    def is_composite(self) -> bool:
        return self.pu_kg > 0 and self.heu_kg > 0


# A fitted anchor: the test plus the effective compression the composite
# model assigns to its observed yield, with knife-edge diagnostics.
AnchorFit = namedtuple(
    "AnchorFit", ["test", "eta_eff", "eta_c", "crits", "elasticity"]
)


# ---------------------------------------------------------------------------
# Anchor library. Add entries here with a citation. Keep the most
# well-attested tests near the top.
# ---------------------------------------------------------------------------

_ANCHORS: tuple[HistoricalTest, ...] = (
    HistoricalTest(
        name="Fat Man",
        year=1945,
        country="USA",
        pu_kg=6.1,
        heu_kg=0.0,
        yield_kt=20.0,
        notes=(
            "Trinity test (16 July 1945); same design dropped on Nagasaki "
            "(9 August 1945). 230 kg DU/Al tamper, not modeled here."
        ),
        source=(
            "Cochran & Paine, 'The Amount of Pu and HEU Needed for Pure "
            "Fission Nuclear Weapons' (NRDC, 1995). Also Coster-Mullen, "
            "'Atom Bombs' (2016) for the 6.1 kg core mass."
        ),
    ),
    HistoricalTest(
        name="SANDSTONE X-Ray",
        year=1948,
        country="USA",
        pu_kg=2.38,
        heu_kg=4.77,
        yield_kt=37.0,
        notes=(
            "First US composite-core test (14 April 1948, Eniwetok); "
            "levitated composite core in a MK III HE assembly. Yield is "
            "well-attested; mass split is Hansen's derivation from "
            "declassified efficiency figures (35% Pu, 25% HEU utilization) "
            "and the documented 2:1 HEU:Pu mass ratio (per the September "
            "1945 LASL design target), solving "
            "0.35*M_Pu*20 + 0.25*M_HEU*17 = 37 with M_HEU = 2*M_Pu. "
            "Tamper thickness varied between the three SANDSTONE shots; "
            "not modeled here."
        ),
        source=(
            "Chuck Hansen, *Swords of Armageddon*, Vol II pp. 102-105 "
            "(mass derivation in footnote 251); yield from Hewlett & "
            "Duncan, *Atomic Shield*, Appendix 4."
        ),
    ),
    HistoricalTest(
        name="RDS-4",
        year=1953,
        country="USSR",
        pu_kg=4.2,
        heu_kg=6.8,
        yield_kt=28.0,
        notes=(
            "Tatyana; air-droppable composite-core implosion device. "
            "Tamper not modeled. Mass split is the user-supplied estimate."
        ),
        source="User-supplied (see session notes).",
    ),
    HistoricalTest(
        name="Buffalo Kite",
        year=1956,
        country="UK",
        pu_kg=2.0,
        heu_kg=0.0,
        yield_kt=3.0,
        notes=(
            "First UK air-dropped nuclear weapon test (11 October 1956, "
            "Maralinga). Pure-Pu core (no HEU shell). Yield-and-mass "
            "figures declassified from a 1956 AWRE letter on file at the "
            "UK Public Record Office; corroborates the 'medium-tech' "
            "weapon design curve in NRDC's Figure 2 per Cochran 1998."
        ),
        source=(
            "Cochran, *Technological Issues Related to the Proliferation "
            "of Nuclear Weapons*, NRDC 1998, footnote 14; primary "
            "attribution: letter from R. Cook (AWRE) to Director General "
            "Atomic Weapons, 27 June 1956, UK Public Record Office (via "
            "D. Forster letter to NRDC, 28 October 1995). Also Norris et "
            "al., *Nuclear Weapons Databook Vol. V: British, French and "
            "Chinese Nuclear Weapons* (Westview, 1994), p. 400."
        ),
    ),
    HistoricalTest(
        name="Low Tony",
        year=1960,
        country="UK",
        pu_kg=0.9,
        heu_kg=5.6,
        yield_kt=1.0,
        notes=(
            "UK low-yield design. Material loadings from declassified UK "
            "MoD/AWRE table 'Table I' dated 31 December 1959 (Controllable "
            "Document A/1171, Top Secret Atomic, declassified). In the "
            "source document the U column is HEU and the AM column is "
            "tritium (boosting material); Low Tony has no AM entry, "
            "consistent with an unboosted design. Tamper not modeled. "
            "Earlier session-recorded mass split (0.9 kg Pu + 1.4 kg HEU) "
            "was the user's recollection -- the document shows that 1.4 kg "
            "HEU is actually the Tony loading, not Low Tony."
        ),
        source=(
            "UK Ministry of Defence / Atomic Weapons Research "
            "Establishment, Controllable Document A/1171 (AWPAC/4/5), "
            "31 December 1959, Table I, page 1; declassified per stamp "
            "on document. Yield per session-supplied context."
        ),
    ),
    HistoricalTest(
        name="Kazakh effects device",
        year=1991,
        country="USSR",
        pu_kg=1.0,
        heu_kg=0.0,
        yield_kt=0.3,
        notes=(
            "CAVEAT: this entry is a PLANNED yield specification, NOT an "
            "observed test result. The device was a Soviet nuclear-effects "
            "test device emplaced in a horizontal tunnel at the "
            "Semipalatinsk (Kazakh) test site in May 1991 with 'a total "
            "mass of almost 1 kg of plutonium' and a planned yield of "
            "'0.3 kilotons.' It was never detonated; Russia disclosed in "
            "May 1995 that the device's destruction (dismantled in place) "
            "was imminent, following Kazakhstan's inheritance of the test "
            "infrastructure. Cochran (1998) cites the design as 'very "
            "close to the low end of the high-tech weapon design curve' "
            "in Figure 2. Pure-Pu, no HEU."
        ),
        source=(
            "Cochran, *Technological Issues Related to the Proliferation "
            "of Nuclear Weapons*, NRDC 1998, footnote 14; original "
            "disclosure: Victor Litovkin, 'Destroy Nuclear Device!...', "
            "*Moscow Izvestiya*, 23 May 1995, p. 1."
        ),
    ),
    HistoricalTest(
        name="CHIC-12",
        year=0,  # exact year unknown to author
        country="PRC",
        pu_kg=2.0,
        heu_kg=0.5,
        yield_kt=15.0,
        boosted=True,
        notes=(
            "Possibly fusion-boosted; the fit eta is an upper bound on the "
            "pure-fission compression."
        ),
        source="User-supplied (see session notes).",
    ),
)


def anchors() -> tuple[HistoricalTest, ...]:
    """All registered historical tests."""
    return _ANCHORS


def composite_tests() -> tuple[HistoricalTest, ...]:
    """Just the composite-core tests (both Pu and HEU non-zero)."""
    return tuple(t for t in _ANCHORS if t.is_composite)


def pure_fission_only() -> tuple[HistoricalTest, ...]:
    """Tests that are not boosted (suitable for one-group calibration)."""
    return tuple(t for t in _ANCHORS if not t.boosted)


def get_test(name: str) -> HistoricalTest:
    """Look up a registered test by name (case-insensitive)."""
    target = name.strip().lower()
    for t in _ANCHORS:
        if t.name.lower() == target:
            return t
    known = ", ".join(t.name for t in _ANCHORS)
    raise KeyError(f"Unknown historical test: {name!r}. Known: {known}")


def fit(
    test: HistoricalTest | str,
    *,
    b_pu: float | None = None,
    b_heu: float | None = None,
    correction_factor: float = 1.0,
) -> AnchorFit:
    """Invert a test's observed yield to the effective compression.

    Returns an AnchorFit(test, eta_eff, eta_c, crits, elasticity) using
    `composite.effective_compression`. `b_pu`/`b_heu` override the
    Serber-b calibration (defaults keep the model's 1.0/0.5); pass 1.0/1.0
    for the uncalibrated rigorous one-group prediction. `correction_factor`
    is an extra empirical scalar, reserved for data-driven fits.
    """
    if isinstance(test, str):
        test = get_test(test)
    kw = {"correction_factor": correction_factor}
    if b_pu is not None:
        kw["b_pu"] = b_pu
    if b_heu is not None:
        kw["b_heu"] = b_heu
    cf = effective_compression(test.pu_kg, test.heu_kg, test.yield_kt, **kw)
    return AnchorFit(test, cf.eta_eff, cf.eta_c, cf.crits, cf.elasticity)


def calibration_table(
    *,
    b_pu: float | None = None,
    b_heu: float | None = None,
    correction_factor: float = 1.0,
) -> list[AnchorFit]:
    """Invert every anchor, sorted by year (the calibration ladder)."""
    fits = [
        fit(t, b_pu=b_pu, b_heu=b_heu, correction_factor=correction_factor)
        for t in _ANCHORS
    ]
    # Year 0 (unknown) sorts last.
    return sorted(fits, key=lambda f: (f.test.year == 0, f.test.year))
