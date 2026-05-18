"""
composite_yield_standalone.py
=============================
Single-file, dependency-light extract of the Cochran/Serber two-region
one-group composite-core fission-yield model. Given an inner Pu core mass,
outer HEU shell mass, and uniform compression ratio eta, returns the yield
in kilotons.

The model:

  1. Two-region one-group neutron diffusion theory. The composite-core
     fundamental alpha is the root of a transcendental criticality
     determinant arising from flux/current continuity at the Pu/HEU
     interface and zero flux at the outer surface.

  2. Cochran's hydrodynamic yield expression (his eq. 6.36 in general form):

         Y_raw = 0.9 * M_total * (DeltaR * alpha)^2 * V/dV

     where DeltaR is the expansion distance of the outer surface between
     the operational compression eta and the second-criticality eta_c.

  3. Serber/Cochran per-material 'b' calibration (Cochran eq. 6.60 /
     Primer Sec. 13): b = 1.0 for WGPu, b = 0.5 for HEU; mass-weighted
     for composite cores. Pass b_pu=1.0, b_heu=1.0 to disable and recover
     the uncorrected rigorous one-group prediction.

Material constants (Cochran Table 3.1, 1994 / rev. 2007):
  delta-phase weapon-grade Pu (5% Pu-240): rho = 15.66 g/cm^3,
      M0 = 16.29 kg, R0 = 6.285 cm, alpha_inf = 2.19/shake.
  Weapon-grade U (93.5% U-235): rho = 18.74 g/cm^3, M0 = 52.42 kg,
      R0 = 8.74 cm, alpha_inf = 1.10/shake.

Reduces to single-material Cochran eq. 6.49 in the limits m_heu=0 or
m_pu=0 (closed forms used in those cases for simplicity).

Reference:
  T. B. Cochran, "Bare Homogeneous Fast Fission Device Using One-Group
  Diffusion Theory," DRAFT, August 1, 1994, rev. March 17, 2007.

Dependencies: stdlib only (math). No scipy, no numpy.

Usage:
    from composite_yield_standalone import yield_kt
    Y = yield_kt(m_pu_kg=2.0, m_heu_kg=4.0, eta=3.0)

License: same as parent fissionyield package.
"""

from __future__ import annotations

import math

__all__ = ["yield_kt"]


# Cochran Table 3.1 -- only the two materials this extract uses.
# Per material: density (g/cm^3); bare critical mass at normal density (kg);
# bare critical radius at normal density (cm); infinite-medium prompt-alpha
# at normal density (1/shake); Serber/Cochran efficiency calibration b
# (eq. 6.60).
_PU = {
    "rho":  15.660,   # delta-phase weapon-grade plutonium (5% Pu-240)
    "M0":   16.29,
    "R0":   6.285,
    "ainf": 2.19,
    "b":    1.0,
}
_HEU = {
    "rho":  18.74,    # weapon-grade uranium, 93.5% U-235
    "M0":   52.42,
    "R0":   8.74,
    "ainf": 1.10,
    "b":    0.5,
}

# One raw unit (kg * (cm/shake)^2) converts to kt with this multiplier
# (Cochran eq. 6.57, footnote 9). 1 shake = 10 ns.
_KT_PER_RAW = 0.24
_KG_TO_G = 1000.0


# ---------------------------------------------------------------------------
# Region kernels
# ---------------------------------------------------------------------------

def _D0(mat):
    """Diffusion coefficient at normal density (cm^2/shake). Cochran eq. 5.8."""
    return mat["ainf"] * mat["R0"] ** 2 / (math.pi ** 2)


def _radii(m_pu, m_heu, eta):
    """Inner radius R1 (Pu sphere) and outer radius R2 (Pu+HEU shell)
    at compression eta. Both materials assumed to compress uniformly."""
    rho_in = _PU["rho"] * eta
    rho_out = _HEU["rho"] * eta
    V_in = m_pu * _KG_TO_G / rho_in if m_pu > 0 else 0.0
    V_sh = m_heu * _KG_TO_G / rho_out if m_heu > 0 else 0.0
    R1 = (3.0 * V_in / (4.0 * math.pi)) ** (1.0 / 3.0) if V_in > 0 else 0.0
    R2 = (3.0 * (V_in + V_sh) / (4.0 * math.pi)) ** (1.0 / 3.0)
    return R1, R2


def _B_sq(mat, alpha, eta):
    """One-group buckling B^2 = eta * (alpha_inf * eta - alpha) / D0."""
    return eta * (mat["ainf"] * eta - alpha) / _D0(mat)


def _det(alpha, eta, R1, R2):
    """Two-region criticality determinant. Roots in alpha are the
    eigenvalues of the boundary-value problem. Regions with B^2 < 0
    use sinh/cosh; regions with B^2 >= 0 use sin/cos."""
    B1s = _B_sq(_PU, alpha, eta)
    B2s = _B_sq(_HEU, alpha, eta)
    t = R2 - R1

    if B1s >= 0.0:
        b1 = math.sqrt(B1s)
        s1 = math.sin(b1 * R1)
        diff1 = b1 * R1 * math.cos(b1 * R1) - s1
    else:
        b1 = math.sqrt(-B1s)
        s1 = math.sinh(b1 * R1)
        diff1 = b1 * R1 * math.cosh(b1 * R1) - s1

    if B2s >= 0.0:
        b2 = math.sqrt(B2s)
        sBt = math.sin(b2 * t)
        cBt = math.cos(b2 * t)
    else:
        b2 = math.sqrt(-B2s)
        sBt = math.sinh(b2 * t)
        cBt = math.cosh(b2 * t)
    sum2 = b2 * R1 * cBt + sBt

    return _D0(_PU) * diff1 * sBt + _D0(_HEU) * s1 * sum2


# ---------------------------------------------------------------------------
# Pure-Python root finding (bisection)
# ---------------------------------------------------------------------------

def _bisect(f, a, b, xtol=1e-10, maxiter=200):
    """Bisection on [a,b] requiring f(a)*f(b) <= 0."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(f"bisect: no sign change in bracket [{a}, {b}]")
    for _ in range(maxiter):
        c = 0.5 * (a + b)
        if (b - a) < xtol:
            return c
        fc = f(c)
        if fa * fc <= 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Eigenvalue and critical compression
# ---------------------------------------------------------------------------

def _alpha_eig(m_pu, m_heu, eta):
    """Fundamental alpha (1/shake) for the composite at compression eta.
    The fundamental is the LARGEST alpha root of the criticality
    determinant -- found by scanning from alpha_max downward and taking
    the first sign change."""
    if m_pu == 0 and m_heu == 0:
        return 0.0
    # Single-material closed forms (Cochran eq. 5.19)
    if m_pu == 0:
        kappa = (m_heu / _HEU["M0"]) * eta * eta
        return _HEU["ainf"] * eta * (1.0 - kappa ** (-2.0 / 3.0)) if kappa > 0 else 0.0
    if m_heu == 0:
        kappa = (m_pu / _PU["M0"]) * eta * eta
        return _PU["ainf"] * eta * (1.0 - kappa ** (-2.0 / 3.0)) if kappa > 0 else 0.0

    R1, R2 = _radii(m_pu, m_heu, eta)
    a_max = max(_PU["ainf"], _HEU["ainf"]) * eta * (1.0 - 1e-12)
    a_min = -10.0 * max(_PU["ainf"], _HEU["ainf"]) * eta - 1.0

    def F(a):
        return _det(a, eta, R1, R2)

    n = 400
    prev_a, prev_f = a_max, F(a_max)
    for i in range(1, n + 1):
        a = a_max - (a_max - a_min) * (i / n)
        cur_f = F(a)
        if prev_f * cur_f < 0.0:
            return _bisect(F, a, prev_a)
        prev_a, prev_f = a, cur_f
    raise RuntimeError(
        f"could not bracket alpha eigenvalue "
        f"(m_pu={m_pu}, m_heu={m_heu}, eta={eta})"
    )


def _critical_mass_normal(fraction_pu):
    """Critical M_total (kg) at eta=1 for a composite with given Pu mass
    fraction. The fundamental is the SMALLEST M giving criticality --
    found by scanning M upward and taking the first sign change."""
    if fraction_pu <= 0.0:
        return _HEU["M0"]
    if fraction_pu >= 1.0:
        return _PU["M0"]

    def F(M):
        m_p = fraction_pu * M
        m_h = (1.0 - fraction_pu) * M
        R1, R2 = _radii(m_p, m_h, 1.0)
        return _det(0.0, 1.0, R1, R2)

    M_lo = 0.01 * min(_PU["M0"], _HEU["M0"])
    M_hi = 2.0 * max(_PU["M0"], _HEU["M0"])

    n = 200
    prev_M, prev_f = M_lo, F(M_lo)
    for i in range(1, n + 1):
        M = M_lo + (M_hi - M_lo) * (i / n)
        cur_f = F(M)
        if prev_f * cur_f < 0.0:
            return _bisect(F, prev_M, M)
        prev_M, prev_f = M, cur_f

    # Fallback: extend upward if the fundamental is beyond the initial range
    for _ in range(20):
        M_hi *= 2.0
        cur_f = F(M_hi)
        if prev_f * cur_f < 0.0:
            return _bisect(F, prev_M, M_hi)
        prev_M, prev_f = M_hi, cur_f
    raise RuntimeError(f"could not find M_c0 at fraction_pu={fraction_pu}")


def _critical_eta(m_pu, m_heu):
    """Compression eta_c at which the composite is just critical (alpha=0).
    Uses the M_c ~ 1/rho^2 scaling: eta_c = sqrt(M_c0 / M_total) where
    M_c0 is the critical total mass at normal density."""
    if m_pu == 0 and m_heu == 0:
        return float("inf")
    if m_pu == 0:
        return math.sqrt(_HEU["M0"] / m_heu)
    if m_heu == 0:
        return math.sqrt(_PU["M0"] / m_pu)
    M_total = m_pu + m_heu
    M_c0 = _critical_mass_normal(m_pu / M_total)
    return math.sqrt(M_c0 / M_total)


# ---------------------------------------------------------------------------
# Public yield function
# ---------------------------------------------------------------------------

def yield_kt(
    m_pu_kg: float,
    m_heu_kg: float,
    eta: float,
    *,
    b_pu: float = _PU["b"],
    b_heu: float = _HEU["b"],
    correction_factor: float = 1.0,
) -> float:
    """Composite-core yield in kilotons.

    Parameters
    ----------
    m_pu_kg : float
        Inner Pu (delta-WGPu) core mass in kg. Zero for a pure-HEU device.
    m_heu_kg : float
        Outer HEU (WGU-93.5) shell mass in kg. Zero for a pure-Pu device.
    eta : float
        Uniform compression ratio rho/rho_0 (both regions).
    b_pu, b_heu : float, optional
        Per-material Serber/Cochran 'b' calibration coefficients
        (Cochran eq. 6.60 / Primer Sec. 13). Defaults are 1.0 / 0.5.
        Pass b_pu=1.0, b_heu=1.0 for the uncalibrated rigorous prediction.
    correction_factor : float, optional
        Additional empirical multiplier applied AFTER the b calibration
        (default 1.0). Reserve for data-driven fits.

    Returns
    -------
    float
        Yield in kilotons.

    Raises
    ------
    ValueError
        if masses are negative, eta is non-positive, or any calibration
        factor is non-positive.
    """
    if m_pu_kg < 0 or m_heu_kg < 0:
        raise ValueError("masses must be non-negative")
    if eta <= 0:
        raise ValueError("compression must be positive")
    if correction_factor <= 0 or b_pu <= 0 or b_heu <= 0:
        raise ValueError("calibration values must be positive")

    M_total = m_pu_kg + m_heu_kg
    if M_total == 0:
        return 0.0

    # Mass-weighted physics calibration. Pure-material limits collapse cleanly.
    if m_pu_kg == 0:
        b_eff = b_heu
    elif m_heu_kg == 0:
        b_eff = b_pu
    else:
        b_eff = (m_pu_kg * b_pu + m_heu_kg * b_heu) / M_total

    eta_c = _critical_eta(m_pu_kg, m_heu_kg)
    if eta <= eta_c:
        return 0.0

    alpha = _alpha_eig(m_pu_kg, m_heu_kg, eta)
    if alpha <= 0.0:
        return 0.0

    _, R2_init = _radii(m_pu_kg, m_heu_kg, eta)
    _, R2_crit = _radii(m_pu_kg, m_heu_kg, eta_c)
    delta_R = R2_crit - R2_init
    V_over_dV = 1.0 / (1.0 - (R2_init / R2_crit) ** 3)

    Y_raw = 0.9 * M_total * (delta_R * alpha) ** 2 * V_over_dV
    return correction_factor * b_eff * _KT_PER_RAW * Y_raw


# ---------------------------------------------------------------------------
# Smoke test -- run "python composite_yield_standalone.py" to verify the
# port produces values matching the full fissionyield package.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Expected values cross-checked against the full fissionyield package.
    cases = [
        # (m_pu, m_heu, eta, expected_kt, description)
        (6.1, 0.0,  2.5,    5.530, "Pure 6.1 kg Pu, eta=2.5 (Fat Man path)"),
        (6.1, 0.0,  3.0,   19.190, "Pure 6.1 kg Pu, eta=3.0"),
        (0.0, 60.0, 2.0,  104.403, "Pure HEU 60 kg, eta=2.0 (Serber halves)"),
        (4.2, 6.8,  5.0,  532.192, "RDS-4 config @ eta=5 (b_eff~0.69)"),
        (4.0, 8.0,  2.5,   34.464, "4 Pu + 8 HEU, eta=2.5"),
        (2.0, 0.0,  4.67,   3.002, "Buffalo Kite anchor (UK 1956)"),
    ]
    print(f"{'M_Pu':>5} {'M_HEU':>6} {'eta':>5} | "
          f"{'Y_kt':>10} {'expected':>10}  {'diff%':>7}  {'status':>6}  notes")
    print("-" * 95)
    all_ok = True
    for m_pu, m_heu, eta, expected, desc in cases:
        Y = yield_kt(m_pu, m_heu, eta)
        diff = (Y - expected) / expected * 100 if expected else 0.0
        ok = abs(diff) < 1.0
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(f"{m_pu:>5.2f} {m_heu:>6.2f} {eta:>5.2f} | "
              f"{Y:>10.3f} {expected:>10.3f}  {diff:>+6.2f}%  "
              f"{status:>6}  {desc}")
    print()
    if all_ok:
        print("All smoke tests passed.")
    else:
        print("WARNING: some smoke tests failed; check expected values.")
