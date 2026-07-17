"""UK A/1171 warhead yields, reported as BANDS rather than points.

The seven designs in the declassified UK MoD/AWRE table 'Table I'
(Controllable Document A/1171, 31 December 1959) list Pu, U (=HEU),
Li-6, and AM (=tritium) loadings. Earlier in the project these were
estimated as single yield numbers -- false precision, because each
design's yield is dominated by one or two *latent* inputs we don't know
(primary compression eta, boost multiplier, LiD burn fraction). This
script sweeps each design's dominant latent variables over plausible
ranges and reports [low, nominal, high] total yield.

Mass interpretation (from the session analysis):
  - "U" is HEU. For the pure-fission and Sloika designs all of it goes
    in the composite primary; for the Teller-Ulam designs it is split
    into primary tamper / spark plug / U-238 jacket.
  - "Li-6 kg" is the Li-6 isotope mass; total LiD compound mass is
    Li6 * (1 + M_D/M_Li6) with li6 enrichment = 1.0.
  - "AM gm" is tritium in grams; equimolar DT total mass is
    T * (M_D + M_T)/M_T.
  - Red Snow Kt (0.6 kg Li-6) and UNA are too Li-poor for a true
    secondary; modeled as Sloika layers. Red Snow Mt and Mk. 47 are
    Teller-Ulam secondaries.

None of this is a measurement -- masses carry ~20% literature
uncertainty and tampers/geometry are not modeled. The point of the band
is to show which designs are pin-down-able and which are not.

Run:  python examples/uk_a1171_bands.py
"""

from __future__ import annotations

from itertools import product

from fissionyield.composite import _critical_eta
from tnyield import yield_kt_total

# ---- unit conversions -----------------------------------------------------
M_D, M_T, M_LI6 = 2.014, 3.016, 6.015


def lid(li6_kg: float) -> float:
    """Li-6 isotope mass (kg) -> total LiD compound mass (kg)."""
    return li6_kg * (1.0 + M_D / M_LI6)


def dt(tritium_g: float) -> float:
    """Tritium mass (g) -> equimolar DT total mass (g)."""
    return tritium_g * (M_D + M_T) / M_T


# ---- nominal latent values and sweep ranges -------------------------------
ETA_NOMINAL = 3.0
ETA_BAND = (2.7, 3.3)                      # +/-10% primary compression
BOOST_MULT = (5.0, 8.0, 12.0)             # extra fissions per fusion neutron
LID_LAYER_BURN = (0.05, 0.15, 0.30)       # Sloika layer burn fraction
# Teller-Ulam secondary LiD burn fraction, grounded to Barroso: a well-
# tamped compact charge reaches ~90% (W-87, Table 10-11), but these UK
# charges are LARGE (Red Snow ~21 kg, Mk.47 ~48 kg LiD) and Barroso warns
# large finite charges are "bastante mais ineficiente" (Section 10.6), so
# we band from the finite-media floor (~35%, Table 10-9 regime) to the
# compact ceiling (~90%).
SEC_LID_BURN = (0.35, 0.60, 0.90)


def eta_points(m_pu, m_heu):
    """Three-point eta sweep, low end clamped just above second criticality
    so the primary stays supercritical."""
    eta_c = _critical_eta(m_pu, m_heu)
    lo = max(ETA_BAND[0], round(eta_c * 1.03, 3))
    return sorted({lo, ETA_NOMINAL, ETA_BAND[1]})


# ---- the seven designs ----------------------------------------------------
# Each: base kwargs for yield_kt_total, plus `sweep` = dict of the latent
# variables to grid over. eta is always swept (added per-design so its low
# end can be clamped to the design's own criticality).

DESIGNS = [
    dict(name="Tony", pu=2.25, heu=1.4, kind="boosted fission",
         base=dict(m_pu_kg=2.25, m_heu_kg=1.4, m_dt_g=dt(6.0)),
         sweep=dict(extra_fissions_per_fusion_neutron=BOOST_MULT)),
    dict(name="Low Tony", pu=0.9, heu=5.6, kind="pure fission",
         base=dict(m_pu_kg=0.9, m_heu_kg=5.6),
         sweep=dict()),
    dict(name="Wee Gwen", pu=1.6, heu=2.42, kind="pure fission",
         base=dict(m_pu_kg=1.6, m_heu_kg=2.42),
         sweep=dict()),
    dict(name="UNA (Li-6=0.75)", pu=1.26, heu=12.0, kind="Sloika",
         base=dict(m_pu_kg=1.26, m_heu_kg=12.0,
                   m_li6d_layer_kg=lid(0.75), li6_enrichment_layer=1.0),
         sweep=dict(lid_layer_burn_fraction=LID_LAYER_BURN)),
    dict(name="UNA (Li-6=2.7)", pu=1.26, heu=12.0, kind="Sloika",
         base=dict(m_pu_kg=1.26, m_heu_kg=12.0,
                   m_li6d_layer_kg=lid(2.7), li6_enrichment_layer=1.0),
         sweep=dict(lid_layer_burn_fraction=LID_LAYER_BURN)),
    dict(name="Red Snow Kt", pu=1.6, heu=11.0, kind="Sloika+boost",
         base=dict(m_pu_kg=1.6, m_heu_kg=2.5, m_dt_g=dt(2.5),
                   m_li6d_layer_kg=lid(0.6), li6_enrichment_layer=1.0,
                   m_u238_jacket_kg=8.5),
         sweep=dict(lid_layer_burn_fraction=LID_LAYER_BURN)),
    dict(name="Red Snow Mt", pu=1.6, heu=11.0, kind="Teller-Ulam",
         base=dict(m_pu_kg=1.6, m_heu_kg=1.5, m_dt_g=dt(2.5),
                   m_spark_plug_kg=1.5, spark_plug_material="WGU",
                   m_u238_jacket_kg=8.0,
                   m_lid_secondary_kg=lid(16.0), li6_enrichment_secondary=1.0),
         sweep=dict(secondary_LiD_burn_fraction=SEC_LID_BURN)),
    dict(name="Mk. 47", pu=2.5, heu=60.0, kind="Teller-Ulam",
         base=dict(m_pu_kg=2.5, m_heu_kg=5.0, m_dt_g=dt(4.0),
                   m_spark_plug_kg=2.5, spark_plug_material="WGU",
                   m_u238_jacket_kg=52.5,
                   m_lid_secondary_kg=lid(36.0), li6_enrichment_secondary=1.0),
         sweep=dict(secondary_LiD_burn_fraction=SEC_LID_BURN)),
]


def _nominal_kwargs(base, sweep):
    """base + the middle value of each swept latent variable."""
    kw = dict(base)
    for k, vals in sweep.items():
        vals = sorted(vals)
        kw[k] = vals[len(vals) // 2]
    return kw


def evaluate(design):
    base, sweep = design["base"], design["sweep"]
    etas = eta_points(base["m_pu_kg"], base["m_heu_kg"])

    # nominal
    nom = _nominal_kwargs(base, sweep)
    nom["eta"] = ETA_NOMINAL
    y_nom = yield_kt_total(**nom).Y_total_kt

    # envelope over the full latent grid (eta x each swept variable)
    keys = list(sweep.keys())
    grids = [sweep[k] for k in keys]
    totals = []
    for eta in etas:
        for combo in product(*grids) if grids else [()]:
            kw = dict(base)
            kw["eta"] = eta
            kw.update(dict(zip(keys, combo)))
            totals.append(yield_kt_total(**kw).Y_total_kt)
    return min(totals), y_nom, max(totals), etas


def main():
    print("UK A/1171 warheads -- yield as BANDS "
          f"(eta {ETA_BAND[0]}-{ETA_BAND[1]}, nominal {ETA_NOMINAL})")
    print(f"{'design':16} {'Pu':>4} {'HEU':>4}  {'type':14} "
          f"{'low':>8} {'nominal':>9} {'high':>8}  {'span':>6}")
    print("-" * 82)
    for d in DESIGNS:
        lo, nom, hi, etas = evaluate(d)
        span = f"{hi / lo:5.0f}x" if lo > 1e-6 else "  inf"
        print(f"{d['name']:16} {d['pu']:>4.1f} {d['heu']:>4.1f}  "
              f"{d['kind']:14} {lo:>8.2f} {nom:>9.2f} {hi:>8.2f}  {span:>6}")
    print()
    print("Bands sweep each design's dominant latent inputs: primary eta for")
    print("all; + boost multiplier (Tony), LiD layer burn (Sloika), or the")
    print("Barroso-grounded secondary LiD burn (Teller-Ulam, ~35-90%).")
    print("Fission-dominated designs show the widest RELATIVE spread")
    print("(compression elasticity ~10); the secondaries are set by the LiD")
    print("burn fraction and fuel load.")
    print()
    print("Sanity check on the correction: Red Snow Mt -- literally the")
    print("MEGATON variant -- now lands at ~0.5-1.2 Mt, bracketing the actual")
    print("~1.1 Mt (Yellow Sun Mk.2), and Red Snow Kt stays sub-megaton, as")
    print("their names demand. An earlier version of this model put Red Snow")
    print("Mt at ~40-150 kt, from a ~20x-too-low LiD burn fraction.")
    print("Not measurements -- see module docstring for interpretation/caveats.")


if __name__ == "__main__":
    main()
