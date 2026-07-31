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

## Why two models at all — the historical argument

The split between a simplified analytic model and a full simulation is not an
accident of this repo; it mirrors how the physics was actually done, and the
memoirs make the division sharper than "they're complementary."

**Simplified models falsify and scope; simulations design.** In 1950 the
classical "Super" was the design everyone believed in. It was killed not by a
supercomputer but by *deliberately simplified hand calculation* — Ulam with
C.J. Everett, and Fermi — showing the fuel would not stay lit. In Teller's own
words (quoted by Ulam), that work "indicated that we were on the wrong track,
that the hydrogen bomb design we thought would work best would not work at
all." That is the Cochran end of the spectrum: ruthless simplification, used to
be *decisive against intuition*. But the design that did work — radiation
implosion of a compressed secondary — could not be settled in closed form, and
Ulam went on to invent both Monte Carlo (for neutron transport) and the
particle-ensemble method for compressible flow precisely because the implosion
had to be *simulated*. That is the Barroso end. So the record says: reach for
the simplified model to falsify and to bound; reach for the simulation to
design. `fissionyield` is the first job; `tnyield` + a real hydrocode would be
the second.

**A cautionary note on regime of validity.** Before 1951 Teller argued that
compression could not help the Super: compressing the fuel raises the reaction
rate, but raises the radiation-loss rate by the same factor, so no net gain.
The argument is a correct scaling — *in the runaway regime it assumes*. Once the
fuel and radiation reach thermal equilibrium it is simply false (in equilibrium
matter is not losing energy to radiation, only exchanging it), and that
realization is the Teller-Ulam "first insight." A right scaling argument applied
in the wrong regime blocked the H-bomb for years. Every burn fraction and
compression scaling in this codebase carries the same hazard, which is why the
grounded parameters (e.g. `plasma.lid_burn_fraction`, `boost.dt_*`) state their
regime in the signature rather than in prose. The radiation term that Teller
mis-scaled — radiation, in equilibrium, occupies volume and so "soaks up less of
the total energy" when compressed — is literally the `a·T⁴·V` term in
`plasma.plasma_energy_density()` (Barroso's Section 10.4.2 EOS). The single
equation we lifted for the ablation calculation is the mathematical statement of
the invention.

## Design provenance (what the `tnyield` channels are)

The memoirs also date and attribute the architectures `tnyield` models:

- **Layer cake / Sloika** (`tnyield.layer_cake`) — alternating layers of fission
  and fusion fuel. Named the "alarm clock" by Teller and Richtmyer (1946) and,
  independently, the "Sloika" (layer cake) by Sakharov (1948). Originally
  U-235 + *deuterium*; the lithium-6-deuteride substitution is Ginzburg's
  "second idea" (1949; Teller had proposed LiD in the US in 1947). Our LiD path
  is that later variant.
- **Boosting** (`tnyield.boost`) — a separate, contemporaneous idea: "a small
  container of thermonuclear fuel at the center of a fission bomb," modeled here
  as its own channel, not a secondary.
- **Teller-Ulam secondary** (`tnyield.secondary`) — radiation implosion. The
  radiation is the *agent* of compression, but (per Ford, citing Sublette) most
  of the "push" is **ablation** of the outside of the fuel container — exactly
  the mechanism our `secondary` docstring invokes.

One soft empirical cross-check falls out of Ford: he states Joe-4 / Sloika
(Aug 1953) drew "about 15 to 20 percent" of its energy from thermonuclear
reactions. Our `layer_cake` default `LiD_burn_fraction = 0.20` gives ~33% on a
Joe-4-like configuration; ~0.10 reproduces Ford's figure. This is weak evidence
that the default is high, but the mass split it depends on is guesswork, so it
is recorded as a *tension*, not acted on — see the note by
`DEFAULT_LID_BURN_FRACTION` in `layer_cake.py`.

## References

- T.B. Cochran, *Bare Homogeneous Fast Fission Device Using One-Group Diffusion
  Theory*, DRAFT 1 Aug 1994, rev. 17 Mar 2007.
- D.E.G. Barroso, *A Física dos Explosivos Nucleares*, 2ª ed., Rio de Janeiro:
  SBPC (2009).
- R. Serber, *The Los Alamos Primer* (1992) — the efficiency derivation both
  build on.
- S. Ulam, *Adventures of a Mathematician* (1976) — the classical-Super
  falsification and the origins of Monte Carlo / particle-ensemble hydro.
- K.W. Ford, *Building the H Bomb: A Personal History* (2015) — the Teller-Ulam
  insights, radiation implosion / ablation, and the alarm-clock/Sloika
  provenance and Joe-4 thermonuclear fraction.
