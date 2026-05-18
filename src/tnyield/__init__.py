"""tnyield: thermonuclear-effects estimators (boost, layer cake, secondary).

Closed-form back-of-envelope companion to the `fissionyield` Cochran
model.  All equations are from D.E.G. Barroso, *A Fisica dos Explosivos
Nucleares* (Rio de Janeiro: SBPC, 2009), with equation numbers in the
individual module docstrings.

Three configurations are covered:

  - `boost`         : DT-boosted fission primary (Ch. 9).
  - `layer_cake`    : Sloika fission-fusion-fission three-phase (Ch. 10.4).
  - `secondary`     : Teller-Ulam radiation-driven secondary (Ch. 10.4-10.7).

Supporting modules:

  - `constants`     : universal constants and unit conversions (CGS).
  - `reactions`     : Q-values, atomic masses, per-mass yields.
  - `cross_sections`: <sigma v> for DT, DD, D-He3 (Eqs. 10.40, 10.48, 10.49).
  - `plasma`        : mean free paths, rho-R thresholds, ablation
                      pressure (Section 10.4.2), Lindl burn fraction.
"""

from .boost import BoostResult, DT_burn_fraction, DT_neutron_yield, boosted_yield
from .cross_sections import (
    reaction_time,
    sigma_Li6_nT,
    sigma_v_DD,
    sigma_v_DHe3,
    sigma_v_DT,
)
from .layer_cake import LayerCakeResult, layer_cake_yield, lid_yield_per_kg
from .plasma import (
    ablation_pressure_from_energy,
    burn_fraction_lindl,
    keV_to_kelvin,
    kelvin_to_keV,
    lambda_n_DT,
    lambda_n_LiDT,
    lambda_p_DT,
    lambda_p_LiDT,
    plasma_energy_density,
    plasma_pressure,
    temperature_for_energy,
)
from .secondary import SecondaryResult, secondary_yield

__all__ = [
    # boost
    "BoostResult",
    "DT_burn_fraction",
    "DT_neutron_yield",
    "boosted_yield",
    # cross sections
    "sigma_v_DT",
    "sigma_v_DD",
    "sigma_v_DHe3",
    "sigma_Li6_nT",
    "reaction_time",
    # layer cake
    "LayerCakeResult",
    "layer_cake_yield",
    "lid_yield_per_kg",
    # plasma
    "lambda_n_DT",
    "lambda_p_DT",
    "lambda_n_LiDT",
    "lambda_p_LiDT",
    "burn_fraction_lindl",
    "plasma_energy_density",
    "plasma_pressure",
    "temperature_for_energy",
    "ablation_pressure_from_energy",
    "keV_to_kelvin",
    "kelvin_to_keV",
    # secondary
    "SecondaryResult",
    "secondary_yield",
]
