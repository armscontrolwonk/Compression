# fissionyield

Python implementation of Thomas B. Cochran's one-group fast-fission yield
model (*Bare Homogeneous Fast Fission Device Using One-Group Diffusion
Theory*, 1994 / rev. 2007). The model relates three quantities for a
single bare spherical fissile core:

| symbol | meaning |
|--------|---------|
| `M`    | mass of fissile material (kg) |
| `η`    | compression `ρ / ρ₀` at the moment of disassembly |
| `Y`    | total fission yield (kt) |

Given any two of `{M, η, Y}` it solves for the third, and it can plot `Y`
as a function of either `M` (at fixed `η`) or `η` (at fixed `M`).
Material constants are taken from Cochran's Table 3.1.

This is the math behind Figures 1 and 2 of the NRDC report *The Amount
of Plutonium and Highly-Enriched Uranium Needed for Pure Fission
Nuclear Weapons* (Cochran & Paine, 1995).

## The model

```
κ₀ = M / M₀            (number of bare crits at normal density)
η  = ρᵢ / ρ₀           (compression at disassembly)
κ  = κ₀ · η²           (effective crits when compressed)
```

Simplified form (eq. 6.57, the default):

```
Y ≈ 0.55 · M · (R₀·α∞,₀)² · (κ^(1/3) − 1)³          (raw)
Y(kt) = 0.24 · 0.55 · M · (R₀·α∞,₀)² · (κ^(1/3) − 1)³
```

Rigorous form (eq. 6.49):

```
Y = (9/10) · M · (R₀·α∞,₀)² · κ^(2/3) · (κ^(1/6) − 1)² · (1 − κ^(−2/3))² / (1 − κ^(−1/2))
```

`M` is in kg, `R₀` in cm, `α∞,₀` in 1/shake (1 shake = 10 ns). The 0.24
prefactor converts the raw expression to kilotons (Cochran, footnote 9).

Cochran himself notes that "the accuracy of the model does not warrant"
the extra precision of 6.49 over 6.57, so 6.57 is the default. Both
yield equations are available via the `--model` flag.

Back-solves under 6.57 are closed-form; back-solves under 6.49 use
Brent's method.

## Install

```
pip install -e .
pip install -e ".[test]"   # also install pytest
```

## CLI

List known materials and constants:

```
fissionyield list
```

Solve for yield given mass + compression:

```
fissionyield solve --material delta-WGPu --mass 6.1 --compression 2.5
```

Solve for compression given mass + yield:

```
fissionyield solve --material delta-WGPu --mass 6.1 --yield 20
```

Solve for mass given compression + yield:

```
fissionyield solve --material WGU --compression 2.5 --yield 1
```

Use the rigorous eq. 6.49:

```
fissionyield solve --material WGU --mass 8 --compression 2 --model 6.49
```

Plot Y vs M for a few compression levels (NRDC Figure 1 style):

```
fissionyield plot --material delta-WGPu --vs mass \
    --mass-range 0.5 8 --fixed-compression 1.5 2.5 4.0 \
    -o pu_yield.png
```

Plot Y vs η for several fixed masses:

```
fissionyield plot --material delta-WGPu --vs compression \
    --compression-range 1 5 --fixed-mass 3 5 8
```

Compare two materials on one plot:

```
fissionyield plot --material delta-WGPu WGU --vs mass \
    --mass-range 1 20 --fixed-compression 2.5
```

## Library

```python
from fissionyield import yield_kt, mass_kg, compression, get_material

mat = get_material("delta-WGPu")
print(yield_kt(mass_kg=6.1, eta=2.5, material=mat))            # forward
print(compression(mass_kg=6.1, Y_kt=20.0, material=mat))       # eta for Y
print(mass_kg(eta=2.5, Y_kt=1.0, material=mat))                # mass for Y
```

## Composite cores (Pu inside + HEU outside, or reversed)

For two-region cores — an inner sphere of one fissile material surrounded by
an outer shell of another — the bare-sphere closed form does not apply.
`fissionyield.composite` extends Cochran's one-group derivation to two
regions by solving the transcendental criticality determinant for the
fundamental α directly, then plugging into Cochran's hydrodynamic yield
formula (his eq. 6.36 in its general pre-simplification form):

```
Y = (9/10) · M_total · (ΔR · α)² / (1 − (R_init/R_crit)³)
```

where α is the eigenvalue at compression η and ΔR is the expansion of the
outer surface from η to the second-criticality compression. The composite
formula reduces *exactly* to Cochran's eq. 6.49 (the rigorous bare-sphere
form) when one mass goes to zero, so the single-material limit is
apples-to-apples with the rest of the package.

Assumptions:
- One-group diffusion theory in both regions
- Both regions compressed uniformly by the same η (bulk-modulus differences
  ignored — a deliberate simplification)
- Zero flux at the outer surface (Cochran's B = π/R convention, no
  extrapolation distance)
- No tamper / reflector (bare two-region sphere)

Material constants are taken from Cochran Table 3.1; the per-region
diffusion coefficient is derived from `α∞,o` and `R₀` via eq. 5.8.
No new fitted parameters are introduced.

```python
from fissionyield import yield_kt_composite, alpha_eigenvalue
from fissionyield import critical_mass_composite, critical_compression

# 4 kg Pu inside, 8 kg HEU outside, compressed 2.5×
Y = yield_kt_composite(4.0, 8.0, 2.5, "delta-WGPu", "WGU")

# Bare critical mass at a given Pu mass fraction
M_c = critical_mass_composite(fraction_inner=0.5, eta=1.0,
                              inner_mat="delta-WGPu", outer_mat="WGU")

# Compression at which a given inner/outer mass split goes critical
eta_c = critical_compression(4.0, 8.0, "delta-WGPu", "WGU")

# Effective neutron-multiplication rate (eigenvalue) at compression η
alpha = alpha_eigenvalue(4.0, 8.0, 2.5, "delta-WGPu", "WGU")
```

Geometry matters: putting Pu inside (where the radial flux is highest) gives
a lower M_c than putting it outside. For a 1:1 mass split at η=1 the model
predicts roughly 18 kg (Pu inside / HEU outside) vs 44 kg (HEU inside / Pu
outside).

**Apples-to-apples note**: the composite formula generalizes Cochran's
*rigorous* eq. 6.49, not the simplified 6.57 that NRDC Figures 1 and 2 are
drawn with. When comparing composite yield curves to NRDC, use
`--model 6.49` on the single-material side as well. Cochran himself notes
that 6.57 and 6.49 agree to within ~30%.

### Inferring effective compression from observed yields

`compression_composite()` inverts the forward model: given observed yield
and the inner/outer masses, it returns the η consistent with the model.
This is the "Path 1" interpretation — the model's *relative* predictions
(yield vs. mass, geometry, compression) are trusted; the η that fits the
historical record is taken as the effective compression actually achieved.

Three historical composite-core tests give a coherent picture under this
interpretation:

| Test          | Pu (kg) | HEU (kg) | Yield (kt) | Fit η  |
|---------------|--------:|---------:|-----------:|-------:|
| RDS-4 (1952)  |     4.2 |      6.8 |         28 |  2.3   |
| Low Tony (1960) |   0.9 |      1.4 |          1 |  4.0   |
| CHIC-12 (PRC) |     2.0 |      0.5 |         15 |  5.4*  |

*CHIC-12 may have been boosted; the fit-η is an upper bound on the
pure-fission compression.

The fit-η rises monotonically with test date, mirroring the historical
progression of implosion sophistication. Bare-sphere one-group theory is
known to overestimate absolute yields by a factor of ~2–3 at moderate κ
and worse at high κ (Cochran himself applies an empirical `b` factor of
0.5 to HEU in eq. 6.60 to compensate); under "Path 1" that error is
absorbed into the effective η rather than carried as a separate
calibration. Use `compression_composite` for inversion, and `correction_factor`
on `yield_kt_composite` if you'd rather keep η at a nominal value and damp
the yield directly.

```python
from fissionyield import compression_composite, yield_kt_composite

# Fit effective compression to RDS-4 (1952 Soviet composite core)
eta_fit = compression_composite(
    mass_inner_kg=4.2, mass_outer_kg=6.8, Y_kt=28.0,
    inner_mat="delta-WGPu", outer_mat="WGU",
)  # ~2.34

# Or apply a Path-2 damping factor and recompute
Y_damped = yield_kt_composite(
    4.2, 6.8, 5.0, "delta-WGPu", "WGU", correction_factor=0.04
)  # ~28 kt at user-supplied eta=5
```

## Materials

From Cochran's Table 3.1. Use the `key` column (or any alias) with
`--material` / `get_material()`.

| key            | name                  | M₀ (kg) | R₀ (cm) | α∞,₀ (1/shake) |
|----------------|-----------------------|--------:|--------:|---------------:|
| `alpha-WGPu`   | α-WGPu (5% Pu-240)    |  10.50  |  5.05   |  2.73          |
| `delta-WGPu`   | δ-WGPu (5% Pu-240)    |  16.29  |  6.285  |  2.19          |
| `WGU-93.5`     | WGU (93.5% U-235)     |  52.42  |  8.74   |  1.10          |
| `WGU-95`       | WGU (95% U-235)       |  51.35  |  8.71   |  1.10          |
| `WGU-93.86`    | WGU (93.86% U-235)    |  51.94  |  8.70   |  1.10          |
| `U-233`        | U-233 (99.4%)         |  16.20  |  5.965  |  2.09          |
| `U-233-98`     | U-233 (98.11%)        |  16.53  |  5.984  |  2.09          |

Aliases include `WGPu` → `delta-WGPu`, `HEU` → `WGU-93.5`, etc. Run
`fissionyield list` for the full table.

## Tests

```
pytest
```

## References

- T.B. Cochran, *Bare Homogeneous Fast Fission Device Using One-Group
  Diffusion Theory*, DRAFT, August 1, 1994, rev. March 17, 2007.
- T.B. Cochran and C.E. Paine, *The Amount of Plutonium and
  Highly-Enriched Uranium Needed for Pure Fission Nuclear Weapons*,
  NRDC, revised 13 April 1995.
- R. Serber, *The Los Alamos Primer* (1992), pp. 38–43.
