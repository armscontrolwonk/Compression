"""Command-line interface for the fissionyield package."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import historical
from .composite import effective_compression, yield_band, yield_kt_composite
from .materials import MATERIALS, get_material
from .model import MODELS, compression, kappa, mass_kg, yield_kt


def _add_solve(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "solve",
        help="Compute one of {mass, compression, yield} from the other two.",
    )
    p.add_argument("--material", "-m", required=True, help="Material key or alias")
    p.add_argument("--mass", type=float, help="Mass in kg")
    p.add_argument("--compression", "-c", type=float, help="Compression eta = rho/rho_0")
    p.add_argument("--yield", dest="Y", type=float, help="Yield in kt")
    p.add_argument(
        "--model",
        choices=MODELS,
        default="6.57",
        help="Cochran equation to use (default: 6.57, simplified)",
    )


def _add_plot(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("plot", help="Plot yield curves.")
    p.add_argument(
        "--material",
        "-m",
        required=True,
        nargs="+",
        help="One or more material keys/aliases",
    )
    p.add_argument(
        "--vs",
        choices=["mass", "compression"],
        default="mass",
        help="Independent variable on the x-axis",
    )
    p.add_argument(
        "--mass-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[0.5, 20.0],
        help="Mass sweep range in kg (when --vs=mass) (default: 0.5 20)",
    )
    p.add_argument(
        "--compression-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[1.0, 5.0],
        help="Compression sweep range (when --vs=compression) (default: 1.0 5.0)",
    )
    p.add_argument(
        "--fixed-compression",
        type=float,
        nargs="+",
        default=[1.5, 2.5, 4.0],
        help="One or more eta values to draw curves at (when --vs=mass)",
    )
    p.add_argument(
        "--fixed-mass",
        type=float,
        nargs="+",
        help="One or more mass values in kg (when --vs=compression)",
    )
    p.add_argument(
        "--model",
        choices=MODELS,
        default="6.57",
        help="Cochran equation to use (default: 6.57, simplified)",
    )
    p.add_argument(
        "--ylog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use log y-axis (default: yes)",
    )
    p.add_argument(
        "--xlog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use log x-axis (default: no)",
    )
    p.add_argument(
        "--ymin",
        type=float,
        default=0.01,
        help="Lower y-axis clip in kt for log scale (default: 0.01)",
    )
    p.add_argument(
        "-n",
        "--num-points",
        type=int,
        default=400,
        help="Number of points along the sweep (default: 400)",
    )
    p.add_argument(
        "-o",
        "--output",
        help="Save figure to file instead of opening a window",
    )
    p.add_argument(
        "--title",
        help="Override plot title",
    )


def _add_list(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("list", help="List known materials and constants.")


def _do_list() -> int:
    print(f"{'KEY':14}  {'NAME':28}  {'M0 (kg)':>8}  {'R0 (cm)':>8}  "
          f"{'alpha_inf':>9}  {'(R0*a)^2':>9}  {'rho':>6}")
    for mat in MATERIALS.values():
        print(
            f"{mat.key:14}  {mat.name:28}  {mat.M0:8.3f}  {mat.R0:8.3f}  "
            f"{mat.alpha_inf:9.3f}  {mat.R0_alpha_sq:9.2f}  {mat.density:6.3f}"
        )
        if mat.aliases:
            print(f"    aliases: {', '.join(mat.aliases)}")
    return 0


def _do_solve(args: argparse.Namespace) -> int:
    given = [args.mass is not None, args.compression is not None, args.Y is not None]
    if sum(given) != 2:
        print(
            "error: provide exactly two of --mass, --compression, --yield",
            file=sys.stderr,
        )
        return 2

    mat = get_material(args.material)

    if args.mass is not None and args.compression is not None:
        Y = yield_kt(args.mass, args.compression, mat, model=args.model)
        unknown = ("yield", f"{Y:.4g} kt")
        M, eta = args.mass, args.compression
    elif args.mass is not None and args.Y is not None:
        eta = compression(args.mass, args.Y, mat, model=args.model)
        unknown = ("compression", f"eta = {eta:.4f}")
        M, Y = args.mass, args.Y
    else:
        M = mass_kg(args.compression, args.Y, mat, model=args.model)
        unknown = ("mass", f"{M:.4f} kg")
        eta, Y = args.compression, args.Y

    k0 = M / mat.M0
    k = kappa(M, eta, mat)
    print(f"Material       : {mat.name}")
    print(f"Model          : eq. {args.model}")
    print(f"Mass M         : {M:.4f} kg")
    print(f"Compression eta: {eta:.4f}")
    print(f"Yield Y        : {Y:.4g} kt")
    print(f"  -- solved for: {unknown[0]} = {unknown[1]}")
    print(f"kappa_0 = M/M0           : {k0:.4f}")
    print(f"kappa   = kappa_0 * eta^2: {k:.4f}")
    if k <= 1.0:
        print("note: kappa <= 1 (sub-critical or just critical) -- no yield.")
    return 0


def _do_plot(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    materials = [get_material(m) for m in args.material]
    fig, ax = plt.subplots(figsize=(8, 5.5))

    visibility_floor = args.ymin if args.ylog else 0.0
    skipped: list[str] = []
    if args.vs == "mass":
        xs = np.linspace(args.mass_range[0], args.mass_range[1], args.num_points)
        for mat in materials:
            for eta in args.fixed_compression:
                ys = np.array([_safe_yield(x, eta, mat, args.model) for x in xs])
                label = (
                    f"{mat.key}, eta={eta:g}"
                    if len(materials) > 1
                    else f"eta = {eta:g}"
                )
                if not np.any(ys > visibility_floor):
                    skipped.append(f"{label} (no yield above {visibility_floor} kt in this range)")
                    continue
                ax.plot(xs, ys, label=label)
        ax.set_xlabel("Mass M (kg)")
        title_default = "Yield vs Mass"
    else:
        xs = np.linspace(args.compression_range[0], args.compression_range[1], args.num_points)
        fixed_masses = args.fixed_mass or [round(materials[0].M0 / 4, 2)]
        for mat in materials:
            for m in fixed_masses:
                ys = np.array([_safe_yield(m, x, mat, args.model) for x in xs])
                label = (
                    f"{mat.key}, M={m:g} kg"
                    if len(materials) > 1
                    else f"M = {m:g} kg"
                )
                if not np.any(ys > visibility_floor):
                    skipped.append(f"{label} (no yield above {visibility_floor} kt in this range)")
                    continue
                ax.plot(xs, ys, label=label)
        ax.set_xlabel(r"Compression $\eta = \rho/\rho_0$")
        title_default = "Yield vs Compression"

    for note in skipped:
        print(f"note: skipped {note}", file=sys.stderr)

    ax.set_ylabel("Yield Y (kt)")
    if args.ylog:
        ax.set_yscale("log")
        ax.set_ylim(bottom=args.ymin)
    if args.xlog:
        ax.set_xscale("log")
    if len(materials) == 1:
        suffix = f" -- {materials[0].name} (eq. {args.model})"
    else:
        suffix = f" (eq. {args.model})"
    ax.set_title(args.title or (title_default + suffix))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=140)
        print(f"wrote {args.output}")
    else:
        plt.show()
    return 0


def _safe_yield(m: float, eta: float, mat, model: str) -> float:
    if m <= 0 or eta <= 0:
        return float("nan")
    return yield_kt(m, eta, mat, model=model)


def _add_composite(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "composite",
        help="Two-region Pu+HEU core: forward yield (point or band) or "
             "invert a known yield to effective compression.",
        description=(
            "Composite-core (inner Pu + outer HEU) yield tool. Give a "
            "compression to get a forward yield; add --band to report a "
            "range over an eta sweep with the local elasticity (knife-edge "
            "signal). Give --yield instead to back out the effective "
            "compression a device of that yield achieved."
        ),
    )
    p.add_argument("--mass-pu", type=float, default=0.0, metavar="KG",
                   help="Inner Pu (delta-WGPu) mass in kg (0 for pure HEU)")
    p.add_argument("--mass-heu", type=float, default=0.0, metavar="KG",
                   help="Outer HEU (WGU-93.5) mass in kg (0 for pure Pu)")
    p.add_argument("--compression", "-c", type=float, metavar="ETA",
                   help="Compression rho/rho_0 (forward mode)")
    p.add_argument("--yield", dest="Y", type=float, metavar="KT",
                   help="Known yield in kt (inverse mode: solve for eta)")
    p.add_argument("--band", type=float, nargs="?", const=0.10, default=None,
                   metavar="FRAC",
                   help="Report a yield band over eta_nominal*(1-/+FRAC) "
                        "(default FRAC=0.10). Forward mode only.")
    p.add_argument("--eta-low", type=float, default=None, metavar="ETA",
                   help="Explicit low end of the band sweep (implies --band)")
    p.add_argument("--eta-high", type=float, default=None, metavar="ETA",
                   help="Explicit high end of the band sweep (implies --band)")


def _do_composite(args: argparse.Namespace) -> int:
    if args.mass_pu <= 0 and args.mass_heu <= 0:
        print("error: give a positive --mass-pu and/or --mass-heu",
              file=sys.stderr)
        return 2

    label = f"{args.mass_pu:g} kg Pu + {args.mass_heu:g} kg HEU"

    # Inverse mode: known yield -> effective compression.
    if args.Y is not None:
        if args.compression is not None:
            print("note: --compression ignored in inverse (--yield) mode",
                  file=sys.stderr)
        fit = effective_compression(args.mass_pu, args.mass_heu, args.Y)
        print(f"Composite core   : {label}")
        print(f"Known yield      : {args.Y:g} kt")
        print(f"  effective eta  : {fit.eta_eff:.3f}")
        print(f"  2nd-crit eta_c : {fit.eta_c:.3f}  (yield onset)")
        print(f"  crits above    : {fit.crits:.2f}  "
              f"((eta_eff/eta_c)^2 -- how far off the knife-edge)")
        print(f"  elasticity     : {fit.elasticity:.1f}  "
              f"(d ln Y / d ln eta; >>1 => compression-dominated)")
        return 0

    # Forward mode requires a compression.
    if args.compression is None:
        print("error: give --compression (forward) or --yield (inverse)",
              file=sys.stderr)
        return 2

    want_band = (args.band is not None
                 or args.eta_low is not None
                 or args.eta_high is not None)
    if want_band:
        frac = args.band if args.band is not None else 0.10
        b = yield_band(args.mass_pu, args.mass_heu, args.compression,
                       eta_low=args.eta_low, eta_high=args.eta_high, frac=frac)
        print(f"Composite core   : {label}")
        print(f"Nominal eta      : {b.eta_nominal:g}")
        print(f"Yield band       : {b.y_low:.3g} .. [{b.y_nominal:.3g}] .. "
              f"{b.y_high:.3g} kt")
        print(f"  over eta        : {b.eta_low:.3f} .. {b.eta_high:.3f}")
        print(f"  elasticity      : {b.elasticity:.1f}  "
              f"(d ln Y / d ln eta at nominal)")
        span = b.y_high / b.y_low if b.y_low > 0 else float("inf")
        print(f"  band spans       : {span:.1f}x  "
              f"(high/low -- a point estimate hides this)")
        return 0

    Y = yield_kt_composite(args.mass_pu, args.mass_heu, args.compression)
    print(f"Composite core   : {label}")
    print(f"Compression eta  : {args.compression:g}")
    print(f"Yield            : {Y:.4g} kt")
    return 0


def _add_anchors(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "anchors",
        help="Invert the historical-test library to effective compression "
             "(the calibration ladder).",
        description=(
            "For each publicly-known test in the anchor library, invert its "
            "observed yield to the effective compression eta the composite "
            "model assigns, with knife-edge diagnostics (crits above "
            "critical, elasticity). This is the honest calibration artifact: "
            "per-device effective eta, not a single fudge factor."
        ),
    )
    p.add_argument(
        "--rigorous", action="store_true",
        help="Disable Serber-b (use b_pu=b_heu=1.0, the uncorrected "
             "one-group model) when inverting.",
    )
    p.add_argument(
        "--sort", choices=["year", "eta", "crits"], default="year",
        help="Sort order (default: year -- the calibration ladder).",
    )


def _do_anchors(args: argparse.Namespace) -> int:
    kw = {"b_pu": 1.0, "b_heu": 1.0} if args.rigorous else {}
    table = historical.calibration_table(**kw)
    if args.sort == "eta":
        table = sorted(table, key=lambda f: f.eta_eff)
    elif args.sort == "crits":
        table = sorted(table, key=lambda f: f.crits)

    cal = "rigorous (b=1)" if args.rigorous else "Serber-b (1.0/0.5)"
    print(f"Historical anchors -- effective compression  [calibration: {cal}]")
    print(f"{'test':22} {'yr':>4} {'cc':>4} {'Pu':>5} {'HEU':>5} "
          f"{'Y_kt':>6}  {'eta_eff':>7} {'eta_c':>6} {'crits':>6} {'elast':>6}")
    print("-" * 84)
    for f in table:
        t = f.test
        flag = " *boost" if t.boosted else ""
        yr = t.year if t.year else "?"
        print(f"{t.name:22} {yr:>4} {t.country:>4} {t.pu_kg:>5.2f} "
              f"{t.heu_kg:>5.2f} {t.yield_kt:>6.2f}  {f.eta_eff:>7.3f} "
              f"{f.eta_c:>6.3f} {f.crits:>6.2f} {f.elasticity:>6.1f}{flag}")
    print()
    print("eta_eff = compression reproducing the observed yield; "
          "eta_c = yield onset;")
    print("crits = (eta_eff/eta_c)^2 above critical; "
          "elast = d ln Y / d ln eta (>>1 => knife-edge).")
    print("* boosted: eta_eff is an UPPER bound (fusion not modeled). "
          "Tampers not modeled.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fissionyield",
        description=(
            "Cochran one-group fast-fission yield model. Solve for any one "
            "of {mass, compression, yield} given the other two, or plot "
            "yield curves."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_solve(sub)
    _add_plot(sub)
    _add_list(sub)
    _add_composite(sub)
    _add_anchors(sub)

    args = parser.parse_args(argv)
    if args.cmd == "list":
        return _do_list()
    if args.cmd == "solve":
        return _do_solve(args)
    if args.cmd == "plot":
        return _do_plot(args)
    if args.cmd == "composite":
        return _do_composite(args)
    if args.cmd == "anchors":
        return _do_anchors(args)
    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
