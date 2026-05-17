"""Cochran one-group fast-fission yield model.

Implements the yield equations from T.B. Cochran, *Bare Homogeneous Fast
Fission Device Using One-Group Diffusion Theory* (1994, rev. 2007),
specifically equations (6.49) and (6.57), with closed-form back-solves
under (6.57) and numerical back-solves under (6.49).
"""

from .composite import (
    alpha_eigenvalue,
    compression_composite,
    critical_compression,
    critical_mass_composite,
    yield_kt_composite,
)
from .historical import (
    HistoricalTest,
    anchors,
    composite_tests,
    fit_eta,
    get_test,
    pure_fission_only,
)
from .materials import MATERIALS, Material, get_material
from .model import compression, kappa, mass_kg, yield_kt

__all__ = [
    "MATERIALS",
    "HistoricalTest",
    "Material",
    "alpha_eigenvalue",
    "anchors",
    "composite_tests",
    "compression",
    "compression_composite",
    "critical_compression",
    "critical_mass_composite",
    "fit_eta",
    "get_material",
    "get_test",
    "kappa",
    "mass_kg",
    "pure_fission_only",
    "yield_kt",
    "yield_kt_composite",
]
