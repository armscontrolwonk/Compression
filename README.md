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

### Physics and validation

The composite-core extension builds directly on Cochran's one-group
bare-sphere derivation (*Bare Homogeneous Fast Fission Device Using
One-Group Diffusion Theory*, 1994/rev. 2007). Where Cochran solves a single
Helmholtz equation for the radial flux, φ ∝ sin(πr/R)/r with the closed
form α = α∞·[1 − (R_c/R)²], the composite generalization replaces that
with a two-region eigenvalue problem: separate Helmholtz solutions in the
inner Pu core (φ₁ = A sin(B₁r)/r) and outer HEU shell (φ₂ = [C sin(B₂r) +
D cos(B₂r)]/r), joined by continuity of flux and current at the interface
and zero flux at the outer surface — exactly the boundary-condition
framework Serber sketches in §11 of *The Los Alamos Primer* for the
non-fissile-tamper case, here adapted by retaining the fission source
term in both regions (the textbook treatment of which is in Lamarsh,
ch. 6). The fundamental α is the root of the transcendental criticality
determinant from these conditions and feeds Cochran's hydrodynamic yield
expression Y = (9/10)·M·(ΔR·α)²·V/ΔV (his eq. 6.36, the general
pre-simplification form of eq. 6.49). Per-region diffusion coefficients
are derived from Cochran's Table 3.1 via R_c² = π²D/α∞,o — no new fitted
parameters are introduced. Serber observes in §13 of the *Primer* that
his rigorous yield coefficient K ≈ 1.1 overstates observed yields and
"the true value is probably K ≈ ¼ to ½"; Cochran's eq. 6.60 codifies this
as per-material `b` values (1.0 for WGPu, 0.8 for U-233, 0.5 for HEU), and
the composite code applies these as a mass-weighted multiplier across the
two regions by default, with an independent empirical scalar layered on
top for any future data-driven fit. Reed (Am. J. Phys. 86, 105, 2018; 88,
108, 2020) treats the same composite-core problem with a
geometric-progression chain-reaction model and confirms the bare-sphere
framework's well-documented factor-of-2 overestimate of historical
yields. Structural validation is exact: with the foundational calibration
disabled, the single-material limit equals Cochran's eq. 6.49 to
bit-precision, and the same-material composite (Pu/Pu, HEU/HEU)
reproduces single-material yields within ~0.5% (residual due to rounding
in Cochran's tabulated constants). Empirical validation uses a Path-1
interpretation — back-fit the effective compression η from the observed
yield rather than chasing absolute yield numbers — against five
historical anchors: Fat Man (1945, 6.1 kg Pu → 20 kt; pure-Pu reference),
SANDSTONE X-Ray (1948, 2.38 kg Pu + 4.77 kg HEU → 37 kt, masses derived
by Hansen in *Swords of Armageddon* Vol. II from declassified efficiency
figures), RDS-4 (1953, 4.2 kg Pu + 6.8 kg HEU → 28 kt), Low Tony (1960,
0.9 kg Pu + 5.6 kg HEU → 1 kt, masses from declassified UK MoD/AWRE
Controllable Document A/1171 dated 31 December 1959), and CHIC-12 (2 kg
Pu + 0.5 kg HEU → 15 kt, gas-boosted and therefore flagged as an upper
bound on the pure-fission compression).

### Inferring effective compression from observed yields

`compression_composite()` inverts the forward model: given observed yield
and the inner/outer masses, it returns the η consistent with the model.
This is the "Path 1" interpretation — the model's *relative* predictions
(yield vs. mass, geometry, compression) are trusted; the η that fits the
historical record is taken as the effective compression actually achieved.

Five historical composite-core tests give a coherent picture under this
interpretation (fit-η under the default Serber-b calibration; rigorous
values in parentheses):

| Test              | Pu (kg) | HEU (kg) | Yield (kt) | Fit η (Serber / rigorous) |
|-------------------|--------:|---------:|-----------:|---------------------------|
| Fat Man (1945)    |     6.1 |      0.0 |         20 |  3.02 / 3.02 (pure Pu)    |
| SANDSTONE X-Ray (1948) | 2.38 | 4.77    |         37 |  3.63 / 3.35              |
| RDS-4 (1953)      |     4.2 |      6.8 |         28 |  2.50 / 2.34              |
| Low Tony (1960)   |     0.9 |      5.6 |          1 |  2.68 / 2.55†             |
| CHIC-12 (PRC)     |     2.0 |      0.5 |         15 |  5.47 / 5.35*             |

*CHIC-12 may have been boosted; the fit-η is an upper bound on the
pure-fission compression.
†Low Tony masses are from declassified UK MoD/AWRE Controllable
Document A/1171 (31 December 1959).

Use `fissionyield historical` to list the table, and
`fissionyield plot-historical` to render fit-η versus test year
(markers colored by country, sized by yield, diamond markers for
boosted designs). The default plot uses the Serber-b calibration;
pass `--rigorous` for the uncorrected view. CHIC-12 has `year=0`
in the library (test date not recorded); it is filtered out by
default with a notice — pass `--include-undated` to place it at a
placeholder year (1975) for visualization.

```
fissionyield historical
fissionyield plot-historical -o trajectory.png
fissionyield plot-historical --include-undated -o full.png
fissionyield plot-historical --rigorous -o rigorous.png
```

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

# Fit effective compression to RDS-4 (1953 Soviet composite core).
# Defaults: Serber-b calibration applied (see "Two layered corrections" below).
eta_fit = compression_composite(
    mass_inner_kg=4.2, mass_outer_kg=6.8, Y_kt=28.0,
    inner_mat="delta-WGPu", outer_mat="WGU",
)  # ~2.50 (Serber-default); ~2.34 with calibration=1.0
```

### Two layered corrections: `calibration` and `correction_factor`

The composite yield is computed as

```
Y = correction_factor · (mass-weighted calibration) · Y_rigorous
```

— two conceptually distinct corrections, applied in that order:

**`calibration`** — *foundational physics calibration* of the bare-sphere
one-group model. Default is `fissionyield.SERBER_B`, the per-material
`b` coefficient from Cochran eq. 6.60 / Serber *Primer* Sec. 13:

- **WGPu**: b = 1.0 (no correction; matches Serber's "rough" K ≈ 1.1)
- **U-233**: b = 0.8
- **HEU**: b = 0.5 (Serber: "true K probably ¼ to ½")

For composite cores the multiplier is the mass-weighted average of the
two regions' b values. Accepts either a `dict` (per-material) or a
scalar (uniform across both regions). Pass `calibration=1.0` to recover
the uncorrected rigorous one-group prediction.

**`correction_factor`** — *additional empirical multiplier* applied on
top of the physics calibration. Always a scalar; default 1.0. Reserve
this for data-driven fits to observed test yields (e.g., a global
damping derived from the historical anchor library); it must not be
used as a substitute for the foundational physics correction.

```python
from fissionyield import SERBER_B, yield_kt_composite, fit_eta, get_test

# Default (Serber-b, no empirical damping)
Y = yield_kt_composite(4.2, 6.8, 5.0, "delta-WGPu", "WGU")  # ~532 kt

# Rigorous (no calibration, no empirical damping)
Y_rig = yield_kt_composite(4.2, 6.8, 5.0, "delta-WGPu", "WGU",
                            calibration=1.0)  # ~770 kt

# Custom per-material physics calibration
Y_custom = yield_kt_composite(4.2, 6.8, 5.0, "delta-WGPu", "WGU",
                               calibration={"delta-WGPu": 0.9, "WGU-93.5": 0.4})

# Default Serber-b plus an empirical 0.3x damping on top
Y_damped = yield_kt_composite(4.2, 6.8, 5.0, "delta-WGPu", "WGU",
                               correction_factor=0.3)  # ~160 kt

# Fit-eta defaults to Serber-b. Pass calibration=1.0 for the rigorous fit:
fit_eta(get_test("RDS-4"))                        # 2.495 (Serber default)
fit_eta(get_test("RDS-4"), calibration=1.0)       # 2.343 (rigorous)
fit_eta(get_test("Fat Man"))                      # 3.021 (pure Pu, unaffected)
```

In the CLI, the calibration layer is controlled by `--rigorous` (off by
default → Serber-b applied) and the empirical layer by
`--correction-factor FLOAT` (off by default → 1.0). Both are
independent: you can combine `--rigorous --correction-factor 0.3` to
disable Serber-b *and* apply a 0.3× empirical multiplier.

```
fissionyield historical                                 # Serber-b default
fissionyield historical --rigorous                      # uncorrected
fissionyield solve-composite --pu-mass 4.2 --shell-mass 6.8 --yield 28
fissionyield plot-composite --pu-range 0.1 5 --fixed-shell 0 4 \
    --fixed-compression 3 5
fissionyield plot-composite ... --rigorous --correction-factor 0.5
```

### CLI for composite pits

Two subcommands, parallel to the single-material `solve` / `plot`. The Pu
material always sits inside; the shell material always outside.

`solve-composite` takes three of {Pu mass, shell mass, compression, yield}
and computes the fourth. Back-solving for either mass given the other inputs
is not implemented (the composite has two mass degrees of freedom; use
`solve` for that on the single-material side).

```
# Forward: RDS-4 configuration at eta=5
fissionyield solve-composite --pu-mass 4.2 --shell-mass 6.8 --compression 5

# Inverse: what eta does the model assign to the observed RDS-4 yield?
fissionyield solve-composite --pu-mass 4.2 --shell-mass 6.8 --yield 28
```

`plot-composite` sweeps one of {Pu mass, shell mass, compression} as the
x-axis, with the remaining parameters either fixed (single value) or drawing
multiple curves (list of values).

```
# Yield vs Pu mass at eta=4 for several HEU shell masses
fissionyield plot-composite \
    --pu-range 0.1 5 \
    --fixed-shell 0 2 4 8 \
    --fixed-compression 4 \
    -o composite_shells.png

# Yield vs Pu mass with a 4 kg HEU shell across compression levels
fissionyield plot-composite \
    --pu-range 0.1 5 \
    --fixed-shell 4 \
    --fixed-compression 2 3 4 5 \
    -o composite_etas.png

# Yield vs compression for a few core/shell choices
fissionyield plot-composite --vs compression \
    --compression-range 1.5 6 \
    --fixed-pu 1 2 4 \
    --fixed-shell 4 \
    -o composite_eta_sweep.png

# Path-2 damping for ad-hoc calibration
fissionyield plot-composite \
    --pu-range 0.1 5 --fixed-shell 4 --fixed-compression 5 \
    --correction-factor 0.3
```

Default materials are `delta-WGPu` for the core and `WGU` (93.5% U-235) for
the shell; override with `--pu-material` / `--shell-material` if you want a
different fissile in either region.

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
- R. Serber, *The Los Alamos Primer* (1992), LA-1, pp. 22–28 (Sec. 10
  bare critical radius, Sec. 11 tamper, Sec. 13 efficiency).
- B.C. Reed, *A toy model for the yield of a tamped fission bomb*,
  Am. J. Phys. 86(2), 105–109 (2018).
- B.C. Reed, *Composite cores and tamper yield*, Am. J. Phys. 88(2),
  108–114 (2020).
