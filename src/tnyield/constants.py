"""Physical constants and unit conversions, in CGS.

Convention matches Barroso, *A Fisica dos Explosivos Nucleares* (2009): all
formulas in CGS unless noted, temperatures usually in keV (for cross
sections) or Kelvin (for ablation / equation of state).
"""

# Universal constants (CGS)
RADIATION_CONST_a = 7.5657e-15  # erg / (cm^3 K^4); a = 4 sigma_SB / c
BOLTZMANN_k = 1.380649e-16      # erg / K
SPEED_OF_LIGHT = 2.99792458e10  # cm / s
ELECTRON_MASS = 9.1093837e-28   # g
PROTON_MASS = 1.67262192e-24    # g
ATOMIC_MASS_UNIT = 1.66053907e-24  # g
AVOGADRO = 6.02214076e23        # /mol

# Energy conversions
ERG_PER_KT = 4.184e19           # 1 kt TNT = 4.184e12 J = 4.184e19 erg
ERG_PER_MEV = 1.602176634e-6
ERG_PER_KEV = 1.602176634e-9
ERG_PER_EV = 1.602176634e-12
KEV_PER_MEV = 1000.0

# Temperature conversions: T_keV = T_K * K_TO_KEV
K_TO_KEV = BOLTZMANN_k / ERG_PER_KEV  # ~ 8.617e-8 keV/K
KEV_TO_K = 1.0 / K_TO_KEV             # ~ 1.160e7 K/keV

# Time
SHAKE = 1e-8                    # 1 shake = 10 ns (in seconds)
