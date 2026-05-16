"""Command-line interface for the fissionyield package."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

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

    args = parser.parse_args(argv)
    if args.cmd == "list":
        return _do_list()
    if args.cmd == "solve":
        return _do_solve(args)
    if args.cmd == "plot":
        return _do_plot(args)
    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
