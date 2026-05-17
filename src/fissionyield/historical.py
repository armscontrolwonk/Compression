"""Historical fission-test anchors for Path-1 model calibration.

Each :class:`HistoricalTest` records a publicly-known nuclear test --
its fissile-material loading, observed yield, and provenance -- so that
the bare-sphere composite model can be back-fit to recover the
"effective compression" the model assigns to that test (see the
README's Path-1 section). The fit-eta values form a calibration ladder
that, in practice, tracks implosion sophistication by era.

This module deliberately holds only data points the user has chosen to
include. Adding a test means adding an entry below with a citation; the
author of an entry is responsible for the source. Numbers in the
public literature often have +/- 20% uncertainty on masses and looser
constraints on compression. fit_eta() therefore returns a single
number under the bare-sphere one-group model, not a confidence
interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .composite import compression_composite
from .materials import Material, get_material


@dataclass(frozen=True)
class HistoricalTest:
    """A publicly-known nuclear test used as a calibration anchor.

    Attributes
    ----------
    name : human-readable test name or designator
    year : year of test
    country : USA / USSR / PRC / UK / FR / ...
    pu_kg : plutonium mass in kg (0 for pure-HEU designs)
    heu_kg : HEU mass in kg (0 for pure-Pu designs)
    yield_kt : observed yield in kilotons
    boosted : True if the device was fusion-boosted; fit-eta then
              represents an upper bound on the pure-fission
              compression (the fusion contribution is not modeled here)
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
        name="Low Tony",
        year=1960,
        country="USA",
        pu_kg=0.9,
        heu_kg=1.4,
        yield_kt=1.0,
        notes="Small-yield composite. Tamper not modeled.",
        source="User-supplied (see session notes).",
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
            "Possibly fusion-boosted; fit-eta is an upper bound on the "
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


def fit_eta(
    test: HistoricalTest | str,
    inner_mat: str | Material = "delta-WGPu",
    outer_mat: str | Material = "WGU",
    correction_factor: float = 1.0,
) -> float:
    """Effective compression eta the bare-sphere composite model assigns
    to ``test`` under the given material assumptions.

    This is the Path-1 interpretation: trust the model's relative
    predictions, back-fit eta from the observed yield. For pure-Pu or
    pure-HEU tests one of the masses is zero and the composite formula
    reduces to Cochran 6.49.
    """
    if isinstance(test, str):
        test = get_test(test)
    return compression_composite(
        test.pu_kg,
        test.heu_kg,
        test.yield_kt,
        inner_mat,
        outer_mat,
        correction_factor=correction_factor,
    )
