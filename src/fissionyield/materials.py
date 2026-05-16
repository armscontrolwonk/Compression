"""Material constants for the Cochran one-group fission-yield model.

Source: T.B. Cochran, *Bare Homogeneous Fast Fission Device Using
One-Group Diffusion Theory* (1994, rev. 2007), Table 3.1.

Units:
    density   : g/cm^3
    M0        : kg              (bare critical mass at normal density)
    R0        : cm              (bare critical radius at normal density)
    alpha_inf : 1/shake         (1 shake = 10 ns; neutron-multiplication
                                 alpha in an infinite medium at normal density)
    R0_alpha_sq : (cm/shake)^2  (precomputed (R0 * alpha_inf)^2)
    cochran_a : kt              (empirical coefficient from eq. 6.58:
                                 Y ~= a * kappa_0 * (kappa^(1/3) - 1)^3;
                                 None when Cochran did not supply one)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    aliases: tuple[str, ...]
    density: float
    M0: float
    R0: float
    alpha_inf: float
    R0_alpha_sq: float
    cochran_a: float | None = None


_RAW: tuple[Material, ...] = (
    Material(
        key="alpha-WGPu",
        name="alpha-WGPu (5% Pu-240)",
        aliases=("aWGPu", "alpha-WGPu", "alpha-Pu", "Pu-alpha", "alphaWGPu"),
        density=19.5,
        M0=10.50,
        R0=5.05,
        alpha_inf=2.73,
        R0_alpha_sq=190.0,
        cochran_a=300.0,
    ),
    Material(
        key="delta-WGPu",
        name="delta-WGPu (5% Pu-240)",
        aliases=("dWGPu", "delta-WGPu", "delta-Pu", "Pu-delta", "deltaWGPu", "WGPu"),
        density=15.660,
        M0=16.29,
        R0=6.285,
        alpha_inf=2.19,
        R0_alpha_sq=190.0,
        cochran_a=480.0,
    ),
    Material(
        key="WGU-93.5",
        name="WGU (93.5% U-235)",
        aliases=("WGU", "WGU-93.5", "HEU", "U-235", "U235"),
        density=18.74,
        M0=52.42,
        R0=8.74,
        alpha_inf=1.1,
        R0_alpha_sq=92.0,
        cochran_a=630.0,
    ),
    Material(
        key="WGU-95",
        name="WGU (95% U-235)",
        aliases=("WGU-95",),
        density=18.554,
        M0=51.35,
        R0=8.71,
        alpha_inf=1.1,
        R0_alpha_sq=92.0,
        cochran_a=None,
    ),
    Material(
        key="WGU-93.86",
        name="WGU (93.86% U-235)",
        aliases=("WGU-93.86",),
        density=18.81,
        M0=51.94,
        R0=8.70,
        alpha_inf=1.1,
        R0_alpha_sq=92.0,
        cochran_a=None,
    ),
    Material(
        key="U-233",
        name="U-233 (99.4%)",
        aliases=("U-233", "U233", "U-233-99.4"),
        density=18.218,
        M0=16.20,
        R0=5.965,
        alpha_inf=2.09,
        R0_alpha_sq=155.0,
        cochran_a=None,
    ),
    Material(
        key="U-233-98",
        name="U-233 (98.11%)",
        aliases=("U-233-98", "U-233-98.11"),
        density=18.42,
        M0=16.53,
        R0=5.984,
        alpha_inf=2.09,
        R0_alpha_sq=156.0,
        cochran_a=None,
    ),
)

MATERIALS: dict[str, Material] = {m.key: m for m in _RAW}


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", "-")


def get_material(name: str | Material) -> Material:
    """Look up a material by key or alias (case-insensitive)."""
    if isinstance(name, Material):
        return name
    target = _norm(name)
    for mat in MATERIALS.values():
        if _norm(mat.key) == target or _norm(mat.name) == target:
            return mat
        if any(_norm(a) == target for a in mat.aliases):
            return mat
    known = ", ".join(MATERIALS.keys())
    raise KeyError(f"Unknown material: {name!r}. Known materials: {known}")
