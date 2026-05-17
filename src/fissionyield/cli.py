"""Command-line interface for the fissionyield package."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .composite import (
    compression_composite,
    critical_compression,
    yield_kt_composite,
)
from .historical import anchors, fit_eta as _fit_eta
from .materials import MATERIALS, SERBER_B, get_material
from .model import MODELS, compression, kappa, mass_kg, yield_kt


def _resolve_calibration(args: argparse.Namespace):
    """Foundational physics calibration: SERBER_B by default, 1.0 if --rigorous."""
    return 1.0 if getattr(args, "rigorous", False) else SERBER_B


def _resolve_correction_factor_scalar(args: argparse.Namespace) -> float:
    """Empirical correction multiplier applied on top of the physics calibration."""
    return float(getattr(args, "correction_factor", 1.0))


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


def _add_plot_composite(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-composite",
        help="Plot composite-pit yield curves (Pu inside, fissile shell outside).",
    )
    p.add_argument(
        "--pu-material",
        default="delta-WGPu",
        help="Inner (Pu) material (default: delta-WGPu)",
    )
    p.add_argument(
        "--shell-material",
        default="WGU",
        help="Outer shell material (default: WGU)",
    )
    p.add_argument(
        "--vs",
        choices=["pu-mass", "shell-mass", "compression"],
        default="pu-mass",
        help="X-axis variable (default: pu-mass)",
    )
    p.add_argument(
        "--pu-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[0.1, 5.0],
        help="Pu mass sweep range (when --vs=pu-mass), in kg (default: 0.1 5)",
    )
    p.add_argument(
        "--shell-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[0.0, 20.0],
        help="Shell mass sweep range (when --vs=shell-mass), in kg",
    )
    p.add_argument(
        "--compression-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[1.0, 5.0],
        help="Compression sweep range (when --vs=compression)",
    )
    p.add_argument(
        "--fixed-pu",
        type=float,
        nargs="+",
        help="Fixed Pu mass(es) in kg",
    )
    p.add_argument(
        "--fixed-shell",
        type=float,
        nargs="+",
        help="Fixed shell mass(es) in kg",
    )
    p.add_argument(
        "--fixed-compression",
        type=float,
        nargs="+",
        help="Fixed compression(s) eta = rho/rho_0",
    )
    p.add_argument(
        "--correction-factor",
        type=float,
        default=1.0,
        help="Path-2 yield damping (default 1.0, i.e., rigorous one-group)",
    )
    p.add_argument(
        "--rigorous",
        action="store_true",
        help=(
            "Use the rigorous one-group yield (no calibration). Default is "
            "the Serber/Cochran per-material b calibration (b=1.0 WGPu, "
            "0.8 U-233, 0.5 HEU; mass-weighted for composites). Mutually "
            "exclusive with --correction-factor."
        ),
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
        help="Lower y-axis clip in kt (default: 0.01)",
    )
    p.add_argument(
        "-n",
        "--num-points",
        type=int,
        default=200,
        help="Number of points along the sweep (default: 200)",
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


def _add_solve_composite(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "solve-composite",
        help=(
            "Compute one of {Pu mass, shell mass, compression, yield} for a "
            "composite pit (Pu inside, fissile shell outside) given the others."
        ),
    )
    p.add_argument(
        "--pu-material",
        default="delta-WGPu",
        help="Inner (Pu) material (default: delta-WGPu)",
    )
    p.add_argument(
        "--shell-material",
        default="WGU",
        help="Outer shell material (default: WGU)",
    )
    p.add_argument("--pu-mass", type=float, help="Pu mass in kg")
    p.add_argument("--shell-mass", type=float, help="Shell mass in kg")
    p.add_argument(
        "--compression",
        "-c",
        type=float,
        help="Compression eta = rho/rho_0",
    )
    p.add_argument("--yield", dest="Y", type=float, help="Yield in kt")
    p.add_argument(
        "--correction-factor",
        type=float,
        default=1.0,
        help="Path-2 yield damping (default 1.0)",
    )
    p.add_argument(
        "--rigorous",
        action="store_true",
        help=(
            "Use the rigorous one-group yield (no calibration). Default is "
            "the Serber/Cochran per-material b calibration (b=1.0 WGPu, "
            "0.8 U-233, 0.5 HEU; mass-weighted for composites). Mutually "
            "exclusive with --correction-factor."
        ),
    )


def _add_list(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("list", help="List known materials and constants.")


def _add_historical(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "historical",
        help="List registered historical tests and their fit-eta values.",
    )
    p.add_argument(
        "--pu-material",
        default="delta-WGPu",
        help="Inner (Pu) material assumed in the fit (default: delta-WGPu)",
    )
    p.add_argument(
        "--shell-material",
        default="WGU",
        help="Outer shell material assumed in the fit (default: WGU)",
    )
    p.add_argument(
        "--correction-factor",
        type=float,
        default=1.0,
        help="Path-2 damping applied during the fit (default 1.0)",
    )
    p.add_argument(
        "--rigorous",
        action="store_true",
        help=(
            "Use the rigorous one-group yield (no calibration) for the fit. "
            "Default applies the Serber/Cochran per-material b calibration. "
            "Mutually exclusive with --correction-factor."
        ),
    )
    p.add_argument(
        "--composite-only",
        action="store_true",
        help="Restrict the listing to composite-core tests.",
    )
    p.add_argument(
        "--exclude-boosted",
        action="store_true",
        help="Exclude tests flagged as fusion-boosted.",
    )


def _add_plot_historical(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-historical",
        help=(
            "Plot fit-eta versus year for the registered historical anchors. "
            "Markers colored by country, sized by yield."
        ),
    )
    p.add_argument(
        "--pu-material",
        default="delta-WGPu",
        help="Inner (Pu) material assumed in the fit (default: delta-WGPu)",
    )
    p.add_argument(
        "--shell-material",
        default="WGU",
        help="Outer shell material assumed in the fit (default: WGU)",
    )
    p.add_argument(
        "--correction-factor",
        type=float,
        default=1.0,
        help="Path-2 empirical damping (default 1.0)",
    )
    p.add_argument(
        "--rigorous",
        action="store_true",
        help=(
            "Use the rigorous one-group yield (no calibration). Default "
            "applies the Serber/Cochran per-material b calibration."
        ),
    )
    p.add_argument(
        "--composite-only",
        action="store_true",
        help="Plot only composite-core tests.",
    )
    p.add_argument(
        "--exclude-boosted",
        action="store_true",
        help="Exclude tests flagged as fusion-boosted.",
    )
    p.add_argument(
        "--include-undated",
        action="store_true",
        help=(
            "Include tests with year=0 (unknown). By default they are "
            "filtered with a notice."
        ),
    )
    p.add_argument(
        "--no-labels",
        action="store_true",
        help="Suppress per-marker test-name labels.",
    )
    p.add_argument(
        "-o",
        "--output",
        help="Save figure to file instead of opening a window.",
    )
    p.add_argument("--title", help="Override plot title.")


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


def _safe_yield_composite(
    m_pu: float, m_shell: float, eta: float, pu_mat, shell_mat,
    calibration, correction_factor: float
) -> float:
    if m_pu < 0 or m_shell < 0 or eta <= 0:
        return float("nan")
    if m_pu == 0 and m_shell == 0:
        return 0.0
    return yield_kt_composite(
        m_pu, m_shell, eta, pu_mat, shell_mat,
        calibration=calibration,
        correction_factor=correction_factor,
    )


def _do_plot_composite(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    calibration = _resolve_calibration(args)
    correction = _resolve_correction_factor_scalar(args)
    pu_mat = get_material(args.pu_material)
    shell_mat = get_material(args.shell_material)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    visibility_floor = args.ymin if args.ylog else 0.0
    skipped: list[str] = []

    def plot_curve(xs, ys, label, color=None, linestyle=None):
        if not np.any(ys > visibility_floor):
            skipped.append(f"{label} (no yield above {visibility_floor} kt)")
            return
        kwargs: dict = {}
        if color is not None:
            kwargs["color"] = color
        if linestyle is not None:
            kwargs["linestyle"] = linestyle
        ax.plot(xs, ys, label=label, **kwargs)

    # Cycle colors and line styles in a way that makes the bare-vs-composite
    # pairings visually obvious: color encodes the "tech-level" dimension
    # (compression), linestyle encodes the design variant (shell mass for
    # vs=pu-mass; Pu mass for vs=shell-mass). Bare curves (shell=0) are
    # dashed by convention; with-shell curves are solid.
    cmap = plt.get_cmap("tab10")

    def style_for_shell(shells_sorted, m_shell):
        if m_shell == 0:
            return "--"
        # Multiple non-zero shells: cycle through solid variants.
        nonzero = [s for s in shells_sorted if s != 0]
        if len(nonzero) <= 1:
            return "-"
        idx = nonzero.index(m_shell)
        return ["-", "-.", ":"][idx % 3]

    def style_for_pu(pus_sorted, m_pu):
        # Mirror of style_for_shell. By convention a "missing Pu" core
        # is not a thing here, so just cycle linestyles in order.
        if len(pus_sorted) <= 1:
            return "-"
        styles = ["-", "--", "-.", ":"]
        return styles[pus_sorted.index(m_pu) % len(styles)]

    if args.vs == "pu-mass":
        shells = args.fixed_shell if args.fixed_shell else [0.0]
        etas = args.fixed_compression if args.fixed_compression else [2.5, 3.5, 5.0]
        pair_mode = len(shells) > 1 and len(etas) > 1
        eta_colors = {e: cmap(i % 10) for i, e in enumerate(etas)}
        shells_sorted = sorted(shells)
        xs = np.linspace(args.pu_range[0], args.pu_range[1], args.num_points)
        for m_shell in shells:
            for eta in etas:
                ys = np.array([
                    _safe_yield_composite(
                        x, m_shell, eta, pu_mat, shell_mat,
                        calibration, correction,
                    )
                    for x in xs
                ])
                shell_label = (
                    f"no shell" if m_shell == 0 else f"{m_shell:g} kg {shell_mat.key}"
                )
                if len(shells) > 1 and len(etas) > 1:
                    label = f"{shell_label}, eta={eta:g}"
                elif len(shells) > 1:
                    label = shell_label
                else:
                    label = f"eta = {eta:g}"
                if pair_mode:
                    plot_curve(
                        xs, ys, label,
                        color=eta_colors[eta],
                        linestyle=style_for_shell(shells_sorted, m_shell),
                    )
                else:
                    plot_curve(xs, ys, label)
        ax.set_xlabel(f"Inner Pu mass ({pu_mat.key}) (kg)")
        title_default = "Composite pit: yield vs Pu mass"
        if len(shells) == 1 and shells[0] > 0:
            title_default += f" ({shells[0]:g} kg {shell_mat.key} shell)"
        elif len(shells) == 1:
            title_default += " (no shell)"

    elif args.vs == "shell-mass":
        pus = args.fixed_pu if args.fixed_pu else [1.0]
        etas = args.fixed_compression if args.fixed_compression else [2.5, 3.5, 5.0]
        pair_mode = len(pus) > 1 and len(etas) > 1
        eta_colors = {e: cmap(i % 10) for i, e in enumerate(etas)}
        pus_sorted = sorted(pus)
        xs = np.linspace(args.shell_range[0], args.shell_range[1], args.num_points)
        for m_pu in pus:
            for eta in etas:
                ys = np.array([
                    _safe_yield_composite(
                        m_pu, x, eta, pu_mat, shell_mat,
                        calibration, correction,
                    )
                    for x in xs
                ])
                if len(pus) > 1 and len(etas) > 1:
                    label = f"M_Pu={m_pu:g} kg, eta={eta:g}"
                elif len(pus) > 1:
                    label = f"M_Pu={m_pu:g} kg"
                else:
                    label = f"eta = {eta:g}"
                if pair_mode:
                    plot_curve(
                        xs, ys, label,
                        color=eta_colors[eta],
                        linestyle=style_for_pu(pus_sorted, m_pu),
                    )
                else:
                    plot_curve(xs, ys, label)
        ax.set_xlabel(f"Outer shell mass ({shell_mat.key}) (kg)")
        title_default = "Composite pit: yield vs shell mass"
        if len(pus) == 1:
            title_default += f" ({pus[0]:g} kg {pu_mat.key} core)"

    else:  # vs == compression
        pus = args.fixed_pu if args.fixed_pu else [1.0]
        shells = args.fixed_shell if args.fixed_shell else [4.0]
        pair_mode = len(pus) > 1 and len(shells) > 1
        pu_colors = {m: cmap(i % 10) for i, m in enumerate(pus)}
        shells_sorted = sorted(shells)
        xs = np.linspace(
            args.compression_range[0], args.compression_range[1], args.num_points
        )
        for m_pu in pus:
            for m_shell in shells:
                ys = np.array([
                    _safe_yield_composite(
                        m_pu, m_shell, x, pu_mat, shell_mat,
                        calibration, correction,
                    )
                    for x in xs
                ])
                if len(pus) > 1 and len(shells) > 1:
                    label = f"M_Pu={m_pu:g}, M_shell={m_shell:g}"
                elif len(pus) > 1:
                    label = f"M_Pu={m_pu:g} kg"
                else:
                    label = (
                        f"{m_shell:g} kg {shell_mat.key} shell"
                        if m_shell > 0
                        else "no shell"
                    )
                if pair_mode:
                    plot_curve(
                        xs, ys, label,
                        color=pu_colors[m_pu],
                        linestyle=style_for_shell(shells_sorted, m_shell),
                    )
                else:
                    plot_curve(xs, ys, label)
        ax.set_xlabel(r"Compression $\eta = \rho/\rho_0$")
        title_default = "Composite pit: yield vs compression"

    for note in skipped:
        print(f"note: skipped {note}", file=sys.stderr)

    ax.set_ylabel("Yield Y (kt)")
    if args.ylog:
        ax.set_yscale("log")
        ax.set_ylim(bottom=args.ymin)
    if args.xlog:
        ax.set_xscale("log")
    annotations = []
    if not isinstance(calibration, dict) and calibration == 1.0:
        annotations.append("rigorous")
    if correction != 1.0:
        annotations.append(f"correction x{correction:g}")
    suffix = f"  [{', '.join(annotations)}]" if annotations else ""
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


def _do_solve_composite(args: argparse.Namespace) -> int:
    given = [
        args.pu_mass is not None,
        args.shell_mass is not None,
        args.compression is not None,
        args.Y is not None,
    ]
    if sum(given) != 3:
        print(
            "error: provide exactly three of --pu-mass, --shell-mass, "
            "--compression, --yield",
            file=sys.stderr,
        )
        return 2

    pu_mat = get_material(args.pu_material)
    shell_mat = get_material(args.shell_material)
    calibration = _resolve_calibration(args)
    correction = _resolve_correction_factor_scalar(args)

    if args.Y is None:
        Y = yield_kt_composite(
            args.pu_mass, args.shell_mass, args.compression,
            pu_mat, shell_mat,
            calibration=calibration, correction_factor=correction,
        )
        unknown = ("yield", f"{Y:.4g} kt")
        m_pu, m_shell, eta = args.pu_mass, args.shell_mass, args.compression
    elif args.compression is None:
        eta = compression_composite(
            args.pu_mass, args.shell_mass, args.Y,
            pu_mat, shell_mat,
            calibration=calibration, correction_factor=correction,
        )
        unknown = ("compression", f"eta = {eta:.4f}")
        m_pu, m_shell, Y = args.pu_mass, args.shell_mass, args.Y
    else:
        print(
            "error: back-solving for one of the masses given the other inputs "
            "is not implemented (the composite has two mass degrees of "
            "freedom; use solve to back-solve a single-material design)",
            file=sys.stderr,
        )
        return 2

    eta_c = critical_compression(m_pu, m_shell, pu_mat, shell_mat)
    print(f"Pu (inner)       : {pu_mat.name}")
    print(f"Shell (outer)    : {shell_mat.name}")
    print(f"Pu mass          : {m_pu:.4f} kg")
    print(f"Shell mass       : {m_shell:.4f} kg")
    print(f"Total mass       : {m_pu + m_shell:.4f} kg")
    print(f"Compression eta  : {eta:.4f}")
    print(f"Critical eta_c   : {eta_c:.4f}")
    print(f"Yield Y          : {Y:.4g} kt")
    print(f"  -- solved for: {unknown[0]} = {unknown[1]}")
    if not isinstance(calibration, dict) and calibration == 1.0:
        print("calibration      : rigorous one-group (no Serber b correction)")
    if correction != 1.0:
        print(f"correction factor: {correction:g}")
    if eta <= eta_c:
        print("note: eta <= eta_c (sub-critical or just critical) -- no yield.")
    return 0


def _do_historical(args: argparse.Namespace) -> int:
    calibration = _resolve_calibration(args)
    correction = _resolve_correction_factor_scalar(args)
    tests = anchors()
    if args.composite_only:
        tests = tuple(t for t in tests if t.is_composite)
    if args.exclude_boosted:
        tests = tuple(t for t in tests if not t.boosted)

    print(
        f"{'NAME':14}  {'YEAR':>4}  {'CC':2}  {'M_Pu':>6}  {'M_HEU':>6}  "
        f"{'Y (kt)':>8}  {'boost':>5}  {'fit eta':>8}"
    )
    notes = []
    if not isinstance(calibration, dict) and calibration == 1.0:
        notes.append("rigorous one-group (no Serber b)")
    if correction != 1.0:
        notes.append(f"correction_factor = {correction:g}")
    if notes:
        print(f"(fit eta computed with: {', '.join(notes)})")
    print("-" * 70)
    for t in tests:
        try:
            eta = _fit_eta(
                t,
                inner_mat=args.pu_material,
                outer_mat=args.shell_material,
                calibration=calibration,
                correction_factor=correction,
            )
            eta_str = f"{eta:8.3f}"
        except Exception as exc:  # noqa: BLE001 - surface bracket failures cleanly
            eta_str = f"   ERR ({exc})"
        boost_str = "  yes" if t.boosted else "   no"
        year_str = f"{t.year:>4}" if t.year > 0 else " ----"
        print(
            f"{t.name:14}  {year_str}  {t.country:2}  "
            f"{t.pu_kg:>6.2f}  {t.heu_kg:>6.2f}  "
            f"{t.yield_kt:>8.2f}  {boost_str}  {eta_str}"
        )
        if t.notes:
            print(f"    notes: {t.notes}")
        if t.source:
            print(f"    source: {t.source}")
    return 0


# Country color palette for plot-historical. Conventional flag-ish hues
# but deliberately desaturated so labels stay readable on top of markers.
_COUNTRY_COLORS = {
    "USA":  "#1f4e79",  # navy blue
    "USSR": "#c0504d",  # red
    "UK":   "#7030a0",  # purple
    "PRC":  "#e6b800",  # yellow
    "FR":   "#76608a",  # lavender
}
_DEFAULT_COLOR = "#666666"


def _do_plot_historical(args: argparse.Namespace) -> int:
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    calibration = _resolve_calibration(args)
    correction = _resolve_correction_factor_scalar(args)

    tests = anchors()
    if args.composite_only:
        tests = tuple(t for t in tests if t.is_composite)
    if args.exclude_boosted:
        tests = tuple(t for t in tests if not t.boosted)

    dated = [t for t in tests if t.year > 0]
    undated = [t for t in tests if t.year <= 0]
    if undated and not args.include_undated:
        for t in undated:
            print(
                f"note: {t.name} has no year recorded; excluded from plot "
                f"(use --include-undated to override)",
                file=sys.stderr,
            )
        tests = tuple(dated)

    if not tests:
        print("error: no tests to plot", file=sys.stderr)
        return 2

    rows = []
    for t in tests:
        try:
            eta = _fit_eta(
                t,
                inner_mat=args.pu_material,
                outer_mat=args.shell_material,
                calibration=calibration,
                correction_factor=correction,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"warn: skipping {t.name} ({exc})", file=sys.stderr)
            continue
        # Use a fallback year for undated entries when --include-undated.
        year = t.year if t.year > 0 else 1975
        rows.append((t, year, eta))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Marker size: scale by log10(yield) so 1-kt and 100-kt are
    # visibly different but neither dominates.
    def size_for(Y):
        return 60.0 + 200.0 * math.log10(max(Y, 0.1) + 1.0)

    seen_countries = []
    for t, year, eta in rows:
        color = _COUNTRY_COLORS.get(t.country, _DEFAULT_COLOR)
        if t.country not in seen_countries:
            seen_countries.append(t.country)
        marker = "o" if not t.boosted else "D"  # diamond = boosted
        edge = "black" if t.country == "PRC" else color
        ax.scatter(
            year, eta,
            s=size_for(t.yield_kt),
            c=color, edgecolors=edge, linewidths=1.0,
            marker=marker, zorder=3,
        )

    if not args.no_labels:
        # Hand-tune label placement to avoid overlap on the tight cluster.
        offsets = {
            "Fat Man": (4, 8),
            "SANDSTONE X-Ray": (4, 8),
            "RDS-4": (4, -14),
            "Low Tony": (4, 8),
            "CHIC-12": (-10, -18),
        }
        for t, year, eta in rows:
            dx, dy = offsets.get(t.name, (5, 5))
            yr_label = f"{t.year}" if t.year > 0 else "year?"
            ax.annotate(
                f"{t.name} ({yr_label})",
                (year, eta),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.5,
                color=_COUNTRY_COLORS.get(t.country, _DEFAULT_COLOR),
            )

    # Legend: one entry per country actually shown
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="none",
            markerfacecolor=_COUNTRY_COLORS.get(c, _DEFAULT_COLOR),
            markeredgecolor=_COUNTRY_COLORS.get(c, _DEFAULT_COLOR),
            label=c,
        )
        for c in seen_countries
    ]
    # Add boosted-flag legend entry if any boosted tests are plotted
    if any(t.boosted for t, _, _ in rows):
        handles.append(
            plt.Line2D(
                [], [], marker="D", linestyle="none",
                markerfacecolor="#888888", markeredgecolor="black",
                label="boosted (fit-η is upper bound)",
            )
        )
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=9)

    ax.set_xlabel("Test year")
    ax.set_ylabel(r"Fit $\eta$ (effective compression)")

    if args.title is None:
        if isinstance(calibration, dict):
            cal_str = "Serber-b default"
        elif calibration == 1.0:
            cal_str = "rigorous one-group"
        else:
            cal_str = f"calibration={calibration:g}"
        title = f"Historical anchors: fit-η vs year ({cal_str})"
        if correction != 1.0:
            title += f"  [×{correction:g}]"
    else:
        title = args.title
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    # X-axis padding
    years = [year for _, year, _ in rows]
    ax.set_xlim(min(years) - 3, max(years) + 5)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=140)
        print(f"wrote {args.output}")
    else:
        plt.show()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fissionyield",
        description=(
            "Cochran one-group fast-fission yield model. Solve for any one "
            "of {mass, compression, yield} given the other two, or plot "
            "yield curves. Includes a composite-pit extension (Pu core + "
            "fissile shell) and a historical-anchor library."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_solve(sub)
    _add_plot(sub)
    _add_solve_composite(sub)
    _add_plot_composite(sub)
    _add_historical(sub)
    _add_plot_historical(sub)
    _add_list(sub)

    args = parser.parse_args(argv)
    if args.cmd == "list":
        return _do_list()
    if args.cmd == "solve":
        return _do_solve(args)
    if args.cmd == "plot":
        return _do_plot(args)
    if args.cmd == "solve-composite":
        return _do_solve_composite(args)
    if args.cmd == "plot-composite":
        return _do_plot_composite(args)
    if args.cmd == "historical":
        return _do_historical(args)
    if args.cmd == "plot-historical":
        return _do_plot_historical(args)
    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
