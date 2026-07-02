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

## Composite cores and the compression-sensitivity problem

For a two-region inner-Pu / outer-HEU core, `yield_kt_composite` solves the
Cochran/Serber two-region one-group eigenvalue problem (Serber-*b* calibrated:
*b*=1.0 for Pu, 0.5 for HEU, mass-weighted). Single-material limits route
through Cochran eq. 6.49 and match his published numbers.

The important caveat: for a **barely-supercritical** device the yield is
hypersensitive to the compression η. The elasticity d(ln Y)/d(ln η) runs
~5–11 across this whole design class, so a point yield estimate is false
precision — and a single multiplicative "correction factor" fit to one device
misgeneralizes, because the residual is really an *input* (η) uncertainty
amplified by that elasticity, not a systematic bias. Two honest tools follow
from this:

```python
from fissionyield import effective_compression, yield_band

# INVERSE: back out the effective compression a known device achieved,
# instead of a fudge factor. (Low Tony: 0.9 kg Pu + 5.6 kg HEU, ~1.0 kt.)
fit = effective_compression(0.9, 5.6, 1.0)
# fit.eta_eff=2.68, eta_c=1.96, crits=1.87, elasticity=10.6

# FORWARD as a band, not a point: sweep eta and see how knife-edge it is.
band = yield_band(0.9, 5.6, eta_nominal=3.0, frac=0.15)
# 0.56 .. [2.81] .. 7.64 kt  -- a 13.6x span over +/-15% compression
```

CLI:

```
# forward point
fissionyield composite --mass-pu 0.9 --mass-heu 5.6 -c 3.0
# forward band (+/-15% eta)
fissionyield composite --mass-pu 0.9 --mass-heu 5.6 -c 3.0 --band 0.15
# inverse: known yield -> effective compression
fissionyield composite --mass-pu 0.9 --mass-heu 5.6 --yield 1.0
```

`crits` is (η_eff/η_c)² — how many critical masses above the normal-density
critical the device sits, i.e. how far off the knife-edge. Devices only a
crit or two above critical (Low Tony) are compression-dominated; the band
narrows as `crits` grows.

## Historical anchors (calibration ladder)

`fissionyield.historical` holds a small library of publicly-known tests
(mass split, observed yield, citation). Inverting each one to its effective
compression *is* the honest calibration artifact for this model — there is no
single fudge factor, so the per-device η is what you compare.

```
fissionyield anchors                 # Serber-b, sorted by year
fissionyield anchors --rigorous      # b=1 (uncorrected one-group)
fissionyield anchors --sort crits    # order by distance off the knife-edge
```

```
test                     yr   cc    Pu   HEU   Y_kt  eta_eff  eta_c  crits  elast
Fat Man                1945  USA  6.10  0.00  20.00    3.021  1.634   3.42    5.9
SANDSTONE X-Ray        1948  USA  2.38  4.77  37.00    3.633  1.652   4.84    4.8
RDS-4                  1953 USSR  4.20  6.80  28.00    2.495  1.311   3.62    5.7
Buffalo Kite           1956   UK  2.00  0.00   3.00    4.670  2.854   2.68    7.0
Low Tony               1960   UK  0.90  5.60   1.00    2.681  1.963   1.87   10.6
Kazakh effects device  1991 USSR  1.00  0.00   0.30    5.476  4.036   1.84   10.7
CHIC-12                   ?  PRC  2.00  0.50  15.00    5.466  2.589   4.46    5.0 *boost
```

The empirical read: effective η clusters ~2.5–5.5, but the high-η entries
(Buffalo Kite, Kazakh, CHIC-12) are all small-mass / few-crit devices where
high compression is *required* to reach the yield — not evidence of more
sophisticated implosion. Boosted entries (CHIC-12) are an upper bound on η
because the fusion contribution isn't modeled; tampers aren't modeled either.

```python
from fissionyield import historical
for f in historical.calibration_table():
    print(f.test.name, round(f.eta_eff, 2), round(f.crits, 2))
```

Each entry carries a `source` string; see the module for citations. Add a
test by appending a `HistoricalTest(...)` with its provenance.

### Design curve (mass vs yield)

`fissionyield design-curve` plots the anchors at (total fissile mass, observed
yield) against pure-Pu iso-compression contours — an NRDC-Figure-2-style view
for placing a new design against the historical cloud. Pure-Pu anchors sit on
the contour matching their fitted η; composites sit *below* the same-η pure-Pu
contour because HEU is less efficient (Serber *b*=0.5). Each anchor is
annotated with its own fitted η.

```
fissionyield design-curve -o design_curve.png
fissionyield design-curve --eta 2 2.5 3 4 --mass-range 0.5 80
```

### Worked example: UK A/1171 warheads as bands

`examples/uk_a1171_bands.py` estimates the seven designs from the declassified
UK MoD/AWRE Table I (Controllable Document A/1171, 1959) as **yield bands**
rather than points, sweeping each design's dominant latent inputs (primary η
for all; plus boost multiplier, LiD layer burn, or secondary LiD burn). The
result shows which designs are pin-down-able and which are not:

```
design             Pu  HEU   type              low  nominal    high   span
Wee Gwen          1.6  2.4   pure fission     0.29    1.03     2.41     8x
Low Tony          0.9  5.6   pure fission     1.08    2.81     5.70     5x
Mk. 47            2.5 60.0   Teller-Ulam     92.0   146.1    357.8     4x
```

The barely-critical pure-fission designs span ~8x on compression alone; even
the big Teller-Ulam devices are only good to a factor of several. Run
`python examples/uk_a1171_bands.py` for the full table; see the module
docstring for the mass interpretation and caveats.

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

# tnyield — thermonuclear-effects companion

Closed-form back-of-envelope estimators for **DT-boosted fission**,
**layer-cake / Sloika**, and **Teller-Ulam secondary** yields.  All
equations are from D.E.G. Barroso, *A Física dos Explosivos Nucleares*
(SBPC, 2009).  Equation numbers in the source code refer to that book.

The package follows the same idioms as `fissionyield`: plug numbers into
one or two equations, get a yield back.  No PDE solvers, no
3-temperature hydrodynamics — those are well outside what a
back-of-envelope tool can deliver.

## Modules

| module             | purpose |
|--------------------|---------|
| `constants`        | universal constants and unit conversions (CGS) |
| `reactions`        | Q-values, atomic masses, per-mass energy yields |
| `cross_sections`   | `<σv>` for DT (10.40), DD (10.48), D-He³ (10.49); Li-6(n,T) |
| `plasma`           | mean free paths (10.10-10.13), ρR thresholds, blackbody EOS for the Section 10.4.2 ablation calculation, Lindl burn fraction |
| `boost`            | DT-boosted fission primary (Ch. 9, Eqs. 9.1-9.3) |
| `layer_cake`       | Sloika fission-fusion-fission decomposition |
| `secondary`        | Teller-Ulam radiation-driven secondary (Ch. 10.4-10.7) |

## CLI

The `tnyield` command has five subcommands.

Tabulate Maxwellian `<σv>` for DT, DD, D-He³ at several temperatures:

```
tnyield xsec --T-keV 1 5 10 20 50 100 200
```

Estimate a boosted primary (Table 9-2 R = 5.2 cm Pu sphere example —
4 g DT, baseline 0.3 kt):

```
tnyield boost --mDT 8.48 --rho-DT 0.6 --T 4 --tau 1e-7 --Y0 0.3
```

Compose a Sloika layer-cake yield (primary + LiD + U-238 outer shell):

```
tnyield layercake --Yp 100 --mLiD 50 --mU 300 --li6 0.4
```

W-87-like Teller-Ulam secondary with parameters tuned to match
Table 10-11:

```
tnyield secondary --Yp 122 --mLiD 50 --mU 1320 --mSP 10 \
                  -c 80 --LiD-burn 0.035 --U238-burn 0.0018
```

Section 10.4.2 ablation calculation (5 kt deposited in Li diffuser →
~21 MK, ~1070 Mbar):

```
tnyield plasma ablation --E-kt 5 --V-cm3 9.14e4 --N-cm3 5e22 --Z 3
```

Mean free paths for fusion neutrons and Planck-averaged radiation:

```
tnyield plasma mfp --rho 0.85 --T-keV 10
```

Lindl burn fraction `f = ρR / (ρR + H_B)`:

```
tnyield plasma burnfrac --rhoR 7 --H-B 7
```

## Library

```python
from tnyield import (
    sigma_v_DT, sigma_v_DD, sigma_v_DHe3,
    boosted_yield, layer_cake_yield, secondary_yield,
    ablation_pressure_from_energy,
)

# Boost a 0.3 kt primary with 8.48 g of DT
r = boosted_yield(
    mass_DT_g=8.48, rho_DT_g_per_cm3=0.6, T_keV=4.0,
    tau_s=1e-7, Y_baseline_kt=0.3,
)
print(r)
```

## Pipeline: fissionyield → tnyield

The two packages compose: use `fissionyield` to estimate the primary
fission yield, then feed it to `tnyield` for the thermonuclear stage.

```
fissionyield solve --material delta-WGPu --mass 6 --compression 2.5
# Yield Y : 4.95 kt

tnyield boost --mDT 4 --rho-DT 0.6 --T 4 --tau 1e-7 --Y0 4.95
# Boosted total yield : 6.17 kt  (boost ratio: 1.25 x)
```

## Multi-stage estimator (yield_kt_total)

A single call that composes all five energy channels — primary
(`fissionyield.yield_kt_composite`), boost, layer cake, secondary, and
tertiary FFF jacket — plus an inverse solver that fixes everything
except one mass and bisects to a target yield.

```python
from tnyield import yield_kt_total, solve_mass

# Joe-4 / Sloika style
r = yield_kt_total(
    m_pu_kg=4.0, m_heu_kg=6.0, eta=2.8,
    m_li6d_layer_kg=50.0, li6_enrichment_layer=0.40,
    m_u238_jacket_kg=300.0,
)
print(r)
# Total yield                :    425.581 kt

# Inverse: how much Li-6 D to hit 500 kt?
m_li6d = solve_mass(
    target_Y_kt=500.0,
    unknown="m_li6d_layer_kg",
    m_pu_kg=4.0, m_heu_kg=6.0, eta=2.8,
    li6_enrichment_layer=0.40, m_u238_jacket_kg=300.0,
)
# ~ 61.3 kg
```

Inputs (all default to zero — only supply the masses you actually have):

| arg                  | meaning |
|----------------------|---------|
| `m_pu_kg`, `m_heu_kg`, `eta` | Pu and HEU in the primary, compression |
| `m_dt_g`             | DT boost gas (grams) |
| `m_li6d_layer_kg`    | LiD layer mass (Sloika fusion layer) |
| `m_u238_layer_kg`    | U-238 immediately around the LiD layer |
| `m_lid_secondary_kg` | LiD fuel inside the Teller-Ulam secondary |
| `m_spark_plug_kg`    | Spark-plug fissile mass (HEU or Pu) |
| `m_u238_tamper_kg`   | U-238 tamper around the secondary LiD |
| `m_u238_jacket_kg`   | Outermost FFF jacket (third F) |

Any of those nine masses can be the `unknown` for `solve_mass`. CLI
equivalents:

```
tnyield total --m-pu 4 --m-heu 6 --eta 2.8 --m-li6d-layer 50 \
              --li6-layer 0.40 --m-u238-jacket 300

tnyield solve --Y-kt 500 --solve-for m_li6d_layer_kg \
              --m-pu 4 --m-heu 6 --eta 2.8 --li6-layer 0.40 \
              --m-u238-jacket 300
```

## Caveats

These are estimators, not simulations.  In particular:

- The DT-boost "extra fissions per fusion neutron" multiplier is
  empirical; the book does not give a closed-form expression for it.
  Default `--mult 8` matches the R = 5.2 cm Pu row of Table 9-2 to
  within a factor of ~4.  Values from 5 to 20 are plausible.
- The Lindl burn-fraction formula `f = ρR / (ρR + H_B)` assumes ideal
  hot-spot ignition with full burn propagation; real Teller-Ulam
  secondaries achieve a few per cent of that, because of finite
  Marshak-wave propagation time, non-uniform compression, and rapid
  disassembly.  Override the auto-computed value with `--LiD-burn` if
  you want realistic results.
- The U-238 tamper burn fraction is mass-based; reasonable defaults
  are ~10 % for a thin Sloika layer and ~1 % for a thick Teller-Ulam
  tamper.

## References

- D.E.G. Barroso, *A Física dos Explosivos Nucleares*, 2ª edição, Rio
  de Janeiro: SBPC (2009).  [PDF](https://github.com/armscontrolwonk/nuclear-stuff/blob/main/A-fisica-dos-explosivos-nucleares.pdf)
- T.B. Cochran, *Bare Homogeneous Fast Fission Device Using One-Group
  Diffusion Theory*, DRAFT, August 1, 1994, rev. March 17, 2007.
- T.B. Cochran and C.E. Paine, *The Amount of Plutonium and
  Highly-Enriched Uranium Needed for Pure Fission Nuclear Weapons*,
  NRDC, revised 13 April 1995.
- R. Serber, *The Los Alamos Primer* (1992), pp. 38–43.
- J.D. Lindl, *Inertial Confinement Fusion* (1998), for the
  `f = ρR / (ρR + H_B)` burn-fraction formula.
