"""Reaction Q-values, atomic masses, and per-mass energy densities.

Sources:
  - Barroso, *A Fisica dos Explosivos Nucleares*, Tables in Ch. 10
    and Eqs. (10.2)-(10.4), (10.14).
  - Standard nuclear data (Audi-Wapstra mass evaluation) for atomic
    masses; values rounded to 4 sig figs.
"""

from __future__ import annotations

from .constants import AVOGADRO, ERG_PER_KT, ERG_PER_MEV

# Atomic mass numbers (g/mol)
M_H = 1.008
M_D = 2.014
M_T = 3.016
M_HE3 = 3.016
M_HE4 = 4.003
M_LI6 = 6.015
M_LI7 = 7.016
M_U235 = 235.044
M_U238 = 238.051
M_PU239 = 239.052
M_PU240 = 240.054

# Reaction Q-values in MeV
Q_DT = 17.59         # D + T -> He4 (3.5) + n (14.1)
Q_DDN = 3.27         # D + D -> He3 (0.82) + n (2.45)       (branch 1)
Q_DDP = 4.03         # D + D -> T   (1.01) + p (3.02)       (branch 2)
Q_DD_AVG = 0.5 * (Q_DDN + Q_DDP)
Q_DHE3 = 18.35       # D + He3 -> He4 (3.6) + p (14.7)
Q_LI6_NT = 4.78      # n + Li6 -> alpha + T                  (Eq. 10.14)
Q_LI7_NTAN = -2.47   # n + Li7 -> n' + alpha + T  (threshold ~3.5 MeV)
Q_FISSION = 200.0    # mean fission energy (Cochran convention)

# Neutron energies from fusion reactions (MeV)
EN_DT = 14.1
EN_DDN = 2.45

# Mean fission neutrons per fission (book values for fast spectrum)
NU_U235_FAST = 2.6
NU_PU239_FAST = 3.0
NU_U238_FAST = 2.5

# Equimolar DT mass per atom (used in Eq. 9.6: A_DT = 2.5)
A_DT_EQUIMOLAR = 0.5 * (M_D + M_T)  # ~ 2.515 g/mol

# Solid densities (g/cm^3) -- book values
RHO_DT_SOLID = 0.20
RHO_LID_SOLID = 0.82       # ~natural Li; book uses 0.85 for LiDT
RHO_LI6D_SOLID = 0.78      # Li-6 D
RHO_LI_SOLID = 0.535
RHO_U_NAT = 18.9


def kt_per_gram(q_mev: float, atomic_mass_g_per_mol: float) -> float:
    """Energy density of a reaction in kt TNT per gram of consumed fuel.

    `q_mev` is the energy released per reaction; `atomic_mass_g_per_mol`
    is the mass of fuel consumed per reaction (e.g. M_D + M_T for DT,
    or M_LI6 + M_D for the Li6+D net reaction in a layer cake).
    """
    erg_per_mol = q_mev * ERG_PER_MEV * AVOGADRO
    return erg_per_mol / (atomic_mass_g_per_mol * ERG_PER_KT)


# Per-gram energy yields (kt/g) at 100 % burn.  Useful as quick
# upper-bound estimators.
KT_PER_G_DT = kt_per_gram(Q_DT, M_D + M_T)               # ~ 0.0813 kt/g of consumed DT
KT_PER_G_DD_AVG = kt_per_gram(Q_DD_AVG, 2 * M_D)         # ~ 0.0436 kt/g of consumed D2
KT_PER_G_DHE3 = kt_per_gram(Q_DHE3, M_D + M_HE3)         # ~ 0.0875 kt/g
KT_PER_G_LI6D = kt_per_gram(Q_LI6_NT + Q_DT,
                            M_LI6 + M_D)                  # ~ 0.0670 kt/g of fully burned Li6D
KT_PER_G_U238_FISSION = kt_per_gram(Q_FISSION, M_U238)    # ~ 0.0193 kt/g
KT_PER_G_U235_FISSION = kt_per_gram(Q_FISSION, M_U235)    # ~ 0.0196 kt/g
KT_PER_G_PU239_FISSION = kt_per_gram(Q_FISSION, M_PU239)  # ~ 0.0192 kt/g
