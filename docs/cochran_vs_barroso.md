# Cochran vs. Barroso — the two models this repo implements

This repository is built on two independent primary sources, and it is worth
being explicit about which code implements which, because they treat the
central quantity — the compression η — in opposite ways.

| package / module | primary source | what it does |
|---|---|---|
| `fissionyield` (`model.py`, `composite.py`) | **T.B. Cochran**, *Bare Homogeneous Fast Fission Device Using One-Group Diffusion Theory* (DRAFT 1994, rev. 17 Mar 2007) | closed-form fission yield **given** a compression η |
| `fissionyield.implosion` | Barroso, Ch. 6 & 8 | crude device → η front-end (calibrated to Barroso's hydro) |
| `tnyield` (`boost`, `layer_cake`, `secondary`, `total`) | **D.E.G. Barroso**, *A Física dos Explosivos Nucleares* (SBPC, 2009), Ch. 9–10 | boost, Sloika, Teller-Ulam secondary on top of the primary |

## The one-line distinction

**Cochran takes compression as an input; Barroso computes it.** Cochran's model
answers "given a compression η, what yield?" — the implosion is a black box that
hands you η. Barroso's answers "given this HE + tamper + pit, what compression
and criticality do you actually *achieve*?" — the implosion is the thing being
simulated. They are complementary, not competing: Barroso is the hydro
simulation that would tell you what η to feed Cochran.

## Cochran's model (what `fissionyield` implements)

His derivation "parallels that of Serber but makes fewer approximations." From
his own text:

- **Compression is a defined input variable**: `η ≡ ρᵢ/ρ₀`, "the degree of
  compression." He defines `κ₀ = M/M₀` (critical masses at normal density) and
  `κ = M/Mᵢ = κ₀·η²` (critical masses when fully compressed, the "crits").
- **The implosion is explicitly outside the model**: "the fission energy is
  added instantaneously to the system *before* spherically-symmetric
  homogeneous isentropic expansion begins." The derivation starts *after* the
  pit is compressed to η.
- **Disassembly**: the core expands from maximum supercriticality back to
  criticality (α = 0) "during one neutron generation" (Δt = 1/α); ~half the
  energy is released while α ≥ 0, half as α falls from 0 to −1.
- **Homogeneous isentropic expansion**: density spatially uniform (time-
  dependent only), mass-shell velocity ∝ radius, average pressure
  time-independent during the expansion phase.
- **Bare sphere, one-group monoenergetic** diffusion (best-fit v = 1.51×10⁹
  cm/s ≈ 1.2 MeV), which he flags as approximate — the far-from-boundary
  diffusion assumption "is not the case for small nuclear weapon assemblies."

Yield, his eq. (6.49), rigorous:

```
Y = (9/10)·M·(R₀·α∞,₀)²·κ^(2/3)·(κ^(1/6) − 1)²·(1 − κ^(−2/3))²·(1 − κ^(−1/2))⁻¹
  = (9/10)·M·(R₀·α∞,₀)²·f(κ),   f(κ) = g(κ)·h(κ)
```

with `g(κ) = (κ^(1/3) − 1)³` — the simplified form, eq. (6.57) — and
`h(κ) ≈ 0.59–0.67`, slowly varying. Hence the ~`0.55` coefficient in the
simplified expression (0.9 × ~0.6). `M` in kg, `R₀` in cm, `α∞,₀` in 1/shake;
the 0.24 factor converts to kilotons. These are implemented verbatim in
`fissionyield/model.py` (single-material) and `composite.py` (two-region).

## Barroso's model (what `tnyield` draws on)

A full **Lagrangian radiation-hydrodynamics simulation** (his LUI1 / RIC1
codes) coupled to the neutron transport, modelling the whole chain from the
chemical detonation to the yield:

- Chemical detonation → **convergent shock** through the tamper into the pit;
  Rankine-Hugoniot shock relations, real equations of state (Grüneisen /
  Altshuler / three-term, fitted to uranium Hugoniot data).
- **Shock focusing and reflection at the centre** → central density/pressure
  spike; density, pressure, radii and k_eff tracked point-by-point in time.
- Physics Cochran's homogeneous-isentropic assumption smooths away: only ~5% of
  the chemical energy reaches the pit in spherical implosion; rarefaction can
  decompress the outer fuel; **allotropic phase transitions** (Pu α at 19.7 vs
  δ at 15.9 g/cm³); porous / hollow / isentropic multi-shock paths.
- Output is **criticality insertion vs time**, then yield — e.g. his Table 8-2
  implodes 6 kg hollow α-Pu by variable Composite B and reports k_eff ≈
  1.23–1.43 and 3–30 kt (the calibration behind `fissionyield.implosion`).

## Side by side

| aspect | Cochran (`fissionyield`) | Barroso (`tnyield` source) |
|---|---|---|
| Compression η | **input** (defined variable ρᵢ/ρ₀) | **computed** from HE + tamper + pit |
| When it starts | after compression, energy added instantaneously | at HE detonation; simulates the implosion |
| Method | closed-form analytic efficiency (eqs. 6.49/6.57) | Lagrangian rad-hydro + coupled transport |
| Expansion | homogeneous, isentropic, u ∝ r, ⟨p⟩ constant, one-generation disassembly | resolved ρ/p profiles, convergent-shock focusing + reflection, evolving internal/kinetic split |
| Neutronics | one-group monoenergetic (~1.2 MeV) | multigroup (Hansen-Roach) time-dependent |
| Tamper / HE | absent — folded into effective M₀, R₀, α∞ | explicit (Composite B, U-nat, EOS, shocks) |
| Pu metallurgy | none | α/δ allotropy, porous/hollow, multi-shock |
| Output | yield given η | criticality/compression vs time → yield |
| Role | back-of-envelope estimator | the simulation that *justifies* an η |

## Where they agree, and where they drift

Cochran's disassembly and Barroso's hydro give the *same* η → yield map only in
the regime where "homogeneous isentropic expansion" holds — compact,
near-critical primaries. For large, strongly-shocked assemblies (where the
convergent shock, rarefaction, and phase changes matter) they drift, and
Barroso is the better description.

This is the tension that runs through the whole codebase: `fissionyield` is
Cochran (η in, yield out), the compression-sensitivity problem is *why* η
dominates the uncertainty (yield is ~10× elastic in η), and
`fissionyield.implosion` is a crude first stab at Barroso's answer to it. See
also `fissionyield.effective_compression` (invert a known yield to the effective
η a device achieved) and `yield_band` (report yield over an η sweep).

## References

- T.B. Cochran, *Bare Homogeneous Fast Fission Device Using One-Group Diffusion
  Theory*, DRAFT 1 Aug 1994, rev. 17 Mar 2007.
- D.E.G. Barroso, *A Física dos Explosivos Nucleares*, 2ª ed., Rio de Janeiro:
  SBPC (2009).
- R. Serber, *The Los Alamos Primer* (1992) — the efficiency derivation both
  build on.
