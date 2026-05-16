"""Cochran's one-group yield model: forward and inverse solvers.

Forward (eq. 6.49, rigorous):
    Y = (9/10) * M * (R0 * alpha_inf,0)^2
        * kappa^(2/3) * (kappa^(1/6) - 1)^2 * (1 - kappa^(-2/3))^2
        / (1 - kappa^(-1/2))

Forward (eq. 6.57, simplified):
    Y ~= 0.55 * M * (R0 * alpha_inf,0)^2 * (kappa^(1/3) - 1)^3

Where
    kappa_0 = M / M0           (crits at normal density)
    eta     = rho_i / rho_0    (compression)
    kappa   = kappa_0 * eta^2  (crits at compression)

Unit convention: M in kg, R in cm, alpha in 1/shake. To express Y in
kilotons multiply the right-hand side by 0.24 (Cochran, footnote 9).
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from .materials import Material, get_material

KT_PER_RAW = 0.24
MODELS = ("6.57", "6.49")


def kappa(mass_kg: float, eta: float, material: str | Material) -> float:
    """kappa = (M / M0) * eta^2."""
    mat = get_material(material)
    return (mass_kg / mat.M0) * eta * eta


def yield_kt(
    mass_kg: float,
    eta: float,
    material: str | Material,
    model: str = "6.57",
) -> float:
    """Forward yield in kilotons given mass (kg) and compression eta."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    mat = get_material(material)
    if mass_kg <= 0 or eta <= 0:
        raise ValueError("mass and compression must be positive")
    k = kappa(mass_kg, eta, mat)
    if k <= 1.0:
        return 0.0
    if model == "6.57":
        coeff = 0.55
        f = (k ** (1.0 / 3.0) - 1.0) ** 3
    else:  # "6.49"
        coeff = 0.9
        f = (
            k ** (2.0 / 3.0)
            * (k ** (1.0 / 6.0) - 1.0) ** 2
            * (1.0 - k ** (-2.0 / 3.0)) ** 2
            / (1.0 - k ** (-0.5))
        )
    return KT_PER_RAW * coeff * mass_kg * mat.R0_alpha_sq * f


def mass_kg(
    eta: float,
    Y_kt: float,
    material: str | Material,
    model: str = "6.57",
) -> float:
    """Mass (kg) required to produce yield Y_kt at compression eta."""
    mat = get_material(material)
    if eta <= 0:
        raise ValueError("compression must be positive")
    if Y_kt < 0:
        raise ValueError("yield must be non-negative")
    if Y_kt == 0:
        return mat.M0 / (eta * eta)

    if model == "6.57":
        # Y = K_eff * M * (kappa^(1/3) - 1)^3 with kappa = M eta^2 / M0.
        # Let x = kappa^(1/3); then M = x^3 * M0 / eta^2, and
        #   Y = K_eff * (M0 / eta^2) * [x (x - 1)]^3
        # which gives a quadratic in x once we take the cube root.
        K_eff = KT_PER_RAW * 0.55 * mat.R0_alpha_sq
        p = (Y_kt * eta * eta / (K_eff * mat.M0)) ** (1.0 / 3.0)
        x = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * p))
        return x**3 * mat.M0 / (eta * eta)

    if model == "6.49":
        # Bracket: just above the critical mass (kappa = 1 -> M = M0/eta^2).
        lo = mat.M0 / (eta * eta) * (1.0 + 1e-9)
        hi = max(lo * 2.0, lo + 1.0)
        for _ in range(60):
            if yield_kt(hi, eta, mat, model="6.49") >= Y_kt:
                break
            hi *= 2.0
        else:
            raise RuntimeError(f"Could not bracket mass for Y={Y_kt} kt at eta={eta}")
        return brentq(lambda m: yield_kt(m, eta, mat, model="6.49") - Y_kt, lo, hi, xtol=1e-9)

    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


def compression(
    mass_kg: float,
    Y_kt: float,
    material: str | Material,
    model: str = "6.57",
) -> float:
    """Compression eta required to produce yield Y_kt with mass mass_kg."""
    mat = get_material(material)
    if mass_kg <= 0:
        raise ValueError("mass must be positive")
    if Y_kt < 0:
        raise ValueError("yield must be non-negative")
    if Y_kt == 0:
        return math.sqrt(mat.M0 / mass_kg)

    if model == "6.57":
        K_eff = KT_PER_RAW * 0.55 * mat.R0_alpha_sq
        ratio = Y_kt / (K_eff * mass_kg)
        k = (1.0 + ratio ** (1.0 / 3.0)) ** 3
        return math.sqrt(k * mat.M0 / mass_kg)

    if model == "6.49":
        lo = math.sqrt(mat.M0 / mass_kg) * (1.0 + 1e-9)
        hi = max(lo * 2.0, lo + 1.0)
        for _ in range(60):
            if yield_kt(mass_kg, hi, mat, model="6.49") >= Y_kt:
                break
            hi *= 2.0
        else:
            raise RuntimeError(f"Could not bracket eta for Y={Y_kt} kt at M={mass_kg} kg")
        return brentq(lambda e: yield_kt(mass_kg, e, mat, model="6.49") - Y_kt, lo, hi, xtol=1e-9)

    raise ValueError(f"model must be one of {MODELS}, got {model!r}")
