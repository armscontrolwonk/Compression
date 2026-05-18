"""Cochran one-group fast-fission yield model.

Implements the yield equations from T.B. Cochran, *Bare Homogeneous Fast
Fission Device Using One-Group Diffusion Theory* (1994, rev. 2007),
specifically equations (6.49) and (6.57), with closed-form back-solves
under (6.57) and numerical back-solves under (6.49).
"""

from .composite import yield_kt_composite
from .materials import MATERIALS, Material, get_material
from .model import compression, kappa, mass_kg, yield_kt

__all__ = [
    "MATERIALS",
    "Material",
    "compression",
    "get_material",
    "kappa",
    "mass_kg",
    "yield_kt",
    "yield_kt_composite",
]
