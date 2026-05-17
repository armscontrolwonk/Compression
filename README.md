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

## GUI

A lite PySide6 desktop GUI with the solver on the left and an embedded
plot on the right:

```
pip install -e ".[gui]"
fissionyield-gui
```

Pick a material and model at the top; on the left choose which of
`{M, η, Y}` to solve for and enter the other two; on the right toggle
between *vs Mass* / *vs Compression*, set the sweep range and the
fixed-value list, then **Update plot**.

## Library

```python
from fissionyield import yield_kt, mass_kg, compression, get_material

mat = get_material("delta-WGPu")
print(yield_kt(mass_kg=6.1, eta=2.5, material=mat))            # forward
print(compression(mass_kg=6.1, Y_kt=20.0, material=mat))       # eta for Y
print(mass_kg(eta=2.5, Y_kt=1.0, material=mat))                # mass for Y
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
