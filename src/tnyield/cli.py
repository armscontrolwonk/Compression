"""Command-line interface for the tnyield package."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .boost import DEFAULT_EXTRA_FISSIONS_PER_FUSION_NEUTRON, boosted_yield
from .cross_sections import sigma_v_DD, sigma_v_DHe3, sigma_v_DT
from .layer_cake import (
    DEFAULT_LID_BURN_FRACTION,
    DEFAULT_U238_BURN_FRACTION,
    layer_cake_yield,
)
from .plasma import (
    ablation_pressure_from_energy,
    burn_fraction_lindl,
    lambda_n_DT,
    lambda_n_LiDT,
    lambda_p_DT,
    lambda_p_LiDT,
)
from .secondary import (
    DEFAULT_H_B_LID,
    secondary_yield,
)
from .secondary import (
    DEFAULT_U238_BURN_FRACTION as SEC_DEFAULT_U238_BURN,
)


# ---------- subcommands --------------------------------------------------


def _add_boost(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "boost",
        help="DT-boosted fission primary (Barroso Ch. 9).",
        description=(
            "Estimate the boosted yield of a fission primary with DT gas at "
            "the centre. Solves Eqs. (9.1)-(9.3) for the DT burn fraction "
            "and the number of fast fusion neutrons, then adds the direct "
            "fusion energy plus the extra fission energy those neutrons "
            "induce."
        ),
    )
    p.add_argument("--mass-DT", "--mDT", type=float, required=True,
                   metavar="GRAMS", help="DT mass in grams (book example: 8.48)")
    p.add_argument("--rho-DT", type=float, default=0.6,
                   metavar="G_PER_CM3",
                   help="DT density at compression (default: 0.6 = 3x solid)")
    p.add_argument("--T", "--T-keV", type=float, default=4.0,
                   metavar="KEV",
                   help="Peak DT temperature in keV (default: 4 keV ~ 50e6 K)")
    p.add_argument("--tau", type=float, default=1e-7,
                   metavar="SECONDS",
                   help="Confinement time in seconds (default: 1e-7)")
    p.add_argument("--Y-baseline", "--Y0", type=float, required=True,
                   metavar="KT",
                   help="Unboosted fission yield in kt (compute with fissionyield)")
    p.add_argument(
        "--mult",
        type=float,
        default=DEFAULT_EXTRA_FISSIONS_PER_FUSION_NEUTRON,
        metavar="N",
        help=(
            f"Extra fissions induced per fusion neutron "
            f"(default: {DEFAULT_EXTRA_FISSIONS_PER_FUSION_NEUTRON}; "
            f"5-20 plausible range)"
        ),
    )


def _add_layer_cake(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "layercake",
        help="Sloika-style fission-fusion-fission yield (Barroso Ch. 10.4).",
        description=(
            "Estimate the total yield of a layered Sloika-type device: "
            "fission primary + LiD layer + U-238 outer shell."
        ),
    )
    p.add_argument("--Y-primary", "--Yp", type=float, required=True,
                   metavar="KT", help="Primary fission yield in kt")
    p.add_argument("--mass-LiD", "--mLiD", type=float, required=True,
                   metavar="KG", help="LiD mass in kg")
    p.add_argument("--mass-U238", "--mU", type=float, required=True,
                   metavar="KG", help="U-238 tamper mass in kg")
    p.add_argument("--li6", type=float, default=0.95, metavar="FRAC",
                   help="Li-6 atomic fraction (default: 0.95)")
    p.add_argument("--LiD-burn", type=float, default=DEFAULT_LID_BURN_FRACTION,
                   metavar="FRAC",
                   help=f"LiD burn fraction (default: {DEFAULT_LID_BURN_FRACTION})")
    p.add_argument("--U238-burn", type=float, default=DEFAULT_U238_BURN_FRACTION,
                   metavar="FRAC",
                   help=f"U-238 burn fraction (default: {DEFAULT_U238_BURN_FRACTION})")


def _add_secondary(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "secondary",
        help="Teller-Ulam radiation-driven secondary (Barroso Ch. 10.4-10.7).",
        description=(
            "Estimate the total yield of a Teller-Ulam secondary stage. "
            "Uses the Lindl burn-fraction parameterization "
            "f = rho*R / (rho*R + H_B) applied to the compressed LiD."
        ),
    )
    p.add_argument("--Y-primary", "--Yp", type=float, required=True,
                   metavar="KT")
    p.add_argument("--mass-LiD", "--mLiD", type=float, required=True,
                   metavar="KG")
    p.add_argument("--mass-U238", "--mU", type=float, required=True,
                   metavar="KG", help="Tamper U-238 mass in kg")
    p.add_argument("--mass-spark", "--mSP", type=float, default=0.0,
                   metavar="KG", help="Spark-plug mass in kg (default: 0)")
    p.add_argument("--li6", type=float, default=0.95, metavar="FRAC")
    p.add_argument("--rho-LiD-solid", type=float, default=0.82, metavar="G_PER_CM3")
    p.add_argument("--compression", "-c", type=float, default=100.0, metavar="ETA",
                   help="Peak LiD compression (default: 100x solid)")
    p.add_argument(
        "--LiD-burn", type=float, default=None, metavar="FRAC",
        help=(
            "Override LiD burn fraction (default: compute from Lindl formula). "
            "Real Teller-Ulam values: 0.03 - 0.15."
        ),
    )
    p.add_argument("--U238-burn", type=float, default=SEC_DEFAULT_U238_BURN,
                   metavar="FRAC",
                   help=f"U-238 fast-fission burn fraction (default: {SEC_DEFAULT_U238_BURN})")
    p.add_argument("--spark-burn", type=float, default=0.20, metavar="FRAC")
    p.add_argument("--H-B", type=float, default=DEFAULT_H_B_LID, metavar="G_PER_CM2",
                   help=f"Lindl burn-fraction H_B (default: {DEFAULT_H_B_LID})")


def _add_xsec(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "xsec",
        help="Print thermonuclear <sigma v> values vs temperature.",
        description=(
            "Tabulate <sigma v> in cm^3/s for DT, DD (sum of branches), "
            "and D-He3 at the given temperatures. Eqs. (10.40), (10.48), (10.49)."
        ),
    )
    p.add_argument("--T-keV", type=float, nargs="+",
                   default=[1, 2, 5, 10, 20, 50, 100, 200],
                   help="Temperatures to evaluate at (default: 1..200 keV)")
    p.add_argument(
        "--reaction",
        choices=["DT", "DD", "DHe3", "all"],
        default="all",
        help="Which reaction(s) to print (default: all)",
    )


def _add_plasma(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plasma",
        help="Plasma helpers (mean free paths, ablation, burn fraction).",
    )
    pp = p.add_subparsers(dest="plasma_cmd", required=True)

    pa = pp.add_parser("ablation",
                       help="Section 10.4.2 ablation T and P from deposited energy.")
    pa.add_argument("--E-kt", type=float, required=True)
    pa.add_argument("--V-cm3", type=float, required=True)
    pa.add_argument("--N-cm3", type=float, required=True,
                    help="Ion number density in /cm^3 (e.g. 5e22 for solid Li)")
    pa.add_argument("--Z", type=int, required=True, help="Ion charge state")

    pm = pp.add_parser("mfp",
                       help="Mean free paths (DT and LiDT, neutron and radiation).")
    pm.add_argument("--rho", type=float, required=True, metavar="G_PER_CM3")
    pm.add_argument("--T-keV", type=float, default=10.0,
                    help="Plasma temperature in keV (default: 10)")

    pb = pp.add_parser("burnfrac",
                       help="Lindl burn fraction f = rho*R / (rho*R + H_B).")
    pb.add_argument("--rhoR", type=float, required=True, metavar="G_PER_CM2")
    pb.add_argument("--H-B", type=float, default=7.0, metavar="G_PER_CM2",
                    help="H_B (default 7 for DT; ~20 for LiD)")


# ---------- handlers -----------------------------------------------------


def _do_boost(args: argparse.Namespace) -> int:
    r = boosted_yield(
        mass_DT_g=args.mass_DT,
        rho_DT_g_per_cm3=args.rho_DT,
        T_keV=args.T,
        tau_s=args.tau,
        Y_baseline_kt=args.Y_baseline,
        extra_fissions_per_fusion_neutron=args.mult,
    )
    print("Boosted fission (Barroso Ch. 9):")
    print(f"  DT mass         : {args.mass_DT} g")
    print(f"  DT density      : {args.rho_DT} g/cm^3")
    print(f"  Temperature     : {args.T} keV")
    print(f"  Confinement     : {args.tau} s")
    print(f"  Baseline yield  : {args.Y_baseline} kt")
    print(f"  Extra fissions/n: {args.mult}")
    print()
    print(r)
    return 0


def _do_layer_cake(args: argparse.Namespace) -> int:
    r = layer_cake_yield(
        Y_primary_kt=args.Y_primary,
        mass_LiD_kg=args.mass_LiD,
        mass_U238_kg=args.mass_U238,
        li6_enrichment=args.li6,
        LiD_burn_fraction=args.LiD_burn,
        U238_burn_fraction=args.U238_burn,
    )
    print("Layer cake / Sloika (Barroso Ch. 10.4):")
    print(f"  Primary yield   : {args.Y_primary} kt")
    print(f"  LiD mass        : {args.mass_LiD} kg ({args.li6*100:.0f} % Li-6)")
    print(f"  U-238 mass      : {args.mass_U238} kg")
    print()
    print(r)
    return 0


def _do_secondary(args: argparse.Namespace) -> int:
    r = secondary_yield(
        Y_primary_kt=args.Y_primary,
        mass_LiD_kg=args.mass_LiD,
        mass_U238_tamper_kg=args.mass_U238,
        mass_spark_plug_kg=args.mass_spark,
        li6_enrichment=args.li6,
        rho_LiD_solid=args.rho_LiD_solid,
        compression=args.compression,
        LiD_burn_fraction=args.LiD_burn,
        U238_burn_fraction=args.U238_burn,
        spark_plug_burn_fraction=args.spark_burn,
        H_B_LiD=args.H_B,
    )
    print("Teller-Ulam secondary (Barroso Ch. 10.4-10.7):")
    print(f"  Primary yield   : {args.Y_primary} kt")
    print(f"  LiD mass        : {args.mass_LiD} kg ({args.li6*100:.0f} % Li-6)")
    print(f"  U-238 tamper    : {args.mass_U238} kg")
    print(f"  Spark plug      : {args.mass_spark} kg")
    print(f"  Compression eta : {args.compression}")
    print()
    print(r)
    return 0


def _do_xsec(args: argparse.Namespace) -> int:
    cols = []
    if args.reaction in ("DT", "all"):
        cols.append(("<sv>_DT", sigma_v_DT))
    if args.reaction in ("DD", "all"):
        cols.append(("<sv>_DD", sigma_v_DD))
    if args.reaction in ("DHe3", "all"):
        cols.append(("<sv>_DHe3", sigma_v_DHe3))
    header = f"{'T (keV)':>9}  " + "  ".join(f"{name:>14}" for name, _ in cols)
    print(header)
    print("-" * len(header))
    for T in args.T_keV:
        vals = "  ".join(f"{fn(T):14.4e}" for _, fn in cols)
        print(f"{T:9.3f}  {vals}")
    print("\nUnits: cm^3 / s.  Eqs. 10.40, 10.48, 10.49 from Barroso.")
    return 0


def _do_plasma(args: argparse.Namespace) -> int:
    if args.plasma_cmd == "ablation":
        T_K, P_Mbar = ablation_pressure_from_energy(
            args.E_kt, args.V_cm3, args.N_cm3, args.Z
        )
        from .plasma import kelvin_to_keV
        print("Section 10.4.2 ablation calculation:")
        print(f"  Deposited energy : {args.E_kt:g} kt")
        print(f"  Volume           : {args.V_cm3:g} cm^3")
        print(f"  Ion density N    : {args.N_cm3:g} /cm^3")
        print(f"  Charge state Z   : {args.Z}")
        print(f"  -> Temperature   : {T_K:.3e} K  ({kelvin_to_keV(T_K)*1000:.2f} eV)")
        print(f"  -> Pressure      : {P_Mbar:.2f} Mbar")
        return 0
    if args.plasma_cmd == "mfp":
        print("Mean free paths (Eqs. 10.10-10.13):")
        print(f"  rho        : {args.rho} g/cm^3")
        print(f"  T_e        : {args.T_keV} keV")
        print()
        print(f"  DT:   neutron MFP (14.1 MeV) : {lambda_n_DT(args.rho):.4g} cm")
        print(f"  DT:   Planck radiation MFP   : {lambda_p_DT(args.T_keV, args.rho):.4g} cm")
        print(f"  LiDT: neutron MFP            : {lambda_n_LiDT(args.rho):.4g} cm")
        print(f"  LiDT: Planck radiation MFP   : {lambda_p_LiDT(args.T_keV, args.rho):.4g} cm")
        return 0
    if args.plasma_cmd == "burnfrac":
        f = burn_fraction_lindl(args.rhoR, args.H_B)
        print("Lindl burn fraction f = rho*R / (rho*R + H_B):")
        print(f"  rho*R = {args.rhoR} g/cm^2")
        print(f"  H_B   = {args.H_B} g/cm^2")
        print(f"  -> f  = {f:.4f}  ({f*100:.2f} %)")
        return 0
    return 2


# ---------- top-level ----------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tnyield",
        description=(
            "Thermonuclear yield estimators (boosting, layer cake, "
            "Teller-Ulam secondary).  All equations from Barroso, "
            "A Fisica dos Explosivos Nucleares."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_boost(sub)
    _add_layer_cake(sub)
    _add_secondary(sub)
    _add_xsec(sub)
    _add_plasma(sub)

    args = parser.parse_args(argv)
    if args.cmd == "boost":
        return _do_boost(args)
    if args.cmd == "layercake":
        return _do_layer_cake(args)
    if args.cmd == "secondary":
        return _do_secondary(args)
    if args.cmd == "xsec":
        return _do_xsec(args)
    if args.cmd == "plasma":
        return _do_plasma(args)
    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
