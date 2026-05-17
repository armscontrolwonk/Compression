"""Smoke tests for the CLI commands."""

import sys
from io import StringIO

import pytest

from fissionyield.cli import main


def run(argv, capsys):
    rc = main(argv)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def test_list_runs(capsys):
    rc, out, err = run(["list"], capsys)
    assert rc == 0
    assert "delta-WGPu" in out
    assert "WGU" in out


def test_solve_single_material(capsys):
    rc, out, err = run(
        ["solve", "--material", "delta-WGPu", "--mass", "6.1", "--compression", "2.5"],
        capsys,
    )
    assert rc == 0
    assert "Yield" in out


def test_solve_composite_forward(capsys):
    rc, out, err = run(
        [
            "solve-composite",
            "--pu-mass", "4.2",
            "--shell-mass", "6.8",
            "--compression", "5.0",
        ],
        capsys,
    )
    assert rc == 0
    assert "Pu mass" in out and "Shell mass" in out
    assert "Yield" in out


def test_solve_composite_inverse_compression(capsys):
    """Back-solve eta from observed yield: RDS-4 anchor under default
    (Serber-b) calibration."""
    rc, out, err = run(
        [
            "solve-composite",
            "--pu-mass", "4.2",
            "--shell-mass", "6.8",
            "--yield", "28",
        ],
        capsys,
    )
    assert rc == 0
    assert "2.49" in out  # fit eta ~ 2.495 with Serber-b default


def test_solve_composite_rigorous_lowers_eta(capsys):
    """--rigorous disables Serber-b -> for HEU-containing composites the
    fit-eta is LOWER than the default."""
    rc, out, err = run(
        [
            "solve-composite",
            "--pu-mass", "4.2",
            "--shell-mass", "6.8",
            "--yield", "28",
            "--rigorous",
        ],
        capsys,
    )
    assert rc == 0
    assert "2.34" in out  # rigorous fit eta ~ 2.343


def test_solve_composite_requires_three_of_four(capsys):
    rc, _, err = run(
        ["solve-composite", "--pu-mass", "4.2", "--shell-mass", "6.8"],
        capsys,
    )
    assert rc != 0
    assert "exactly three" in err.lower() or "three" in err


def test_plot_composite_writes_file(tmp_path, capsys):
    out_png = tmp_path / "composite.png"
    rc, _, _ = run(
        [
            "plot-composite",
            "--pu-range", "0.1", "3",
            "--fixed-shell", "0", "4",
            "--fixed-compression", "3", "5",
            "-n", "30",
            "-o", str(out_png),
        ],
        capsys,
    )
    assert rc == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 1000  # non-trivial PNG


def test_plot_composite_vs_shell_mass(tmp_path, capsys):
    out_png = tmp_path / "shell_sweep.png"
    rc, _, _ = run(
        [
            "plot-composite",
            "--vs", "shell-mass",
            "--shell-range", "0", "10",
            "--fixed-pu", "1.0",
            "--fixed-compression", "3", "5",
            "-n", "30",
            "-o", str(out_png),
        ],
        capsys,
    )
    assert rc == 0
    assert out_png.exists()


def test_plot_composite_vs_compression(tmp_path, capsys):
    out_png = tmp_path / "eta_sweep.png"
    rc, _, _ = run(
        [
            "plot-composite",
            "--vs", "compression",
            "--compression-range", "1.5", "5",
            "--fixed-pu", "1.0", "2.0",
            "--fixed-shell", "4.0",
            "-n", "30",
            "-o", str(out_png),
        ],
        capsys,
    )
    assert rc == 0
    assert out_png.exists()


def test_plot_composite_correction_factor(tmp_path, capsys):
    out_png = tmp_path / "damped.png"
    rc, _, _ = run(
        [
            "plot-composite",
            "--pu-range", "0.1", "3",
            "--fixed-shell", "4",
            "--fixed-compression", "5",
            "--correction-factor", "0.3",
            "-n", "30",
            "-o", str(out_png),
        ],
        capsys,
    )
    assert rc == 0
    assert out_png.exists()


def test_rigorous_and_correction_factor_stack_independently(capsys):
    """--rigorous and --correction-factor are independent layers, not
    mutually exclusive. Both can be set; both apply."""
    rc, out, _ = run(
        [
            "solve-composite",
            "--pu-mass", "4.2",
            "--shell-mass", "6.8",
            "--compression", "5",
            "--rigorous",
            "--correction-factor", "0.5",
        ],
        capsys,
    )
    assert rc == 0
    assert "rigorous" in out
    assert "correction factor: 0.5" in out


def test_historical_default_uses_serber(capsys):
    """The default `historical` listing uses Serber-b; RDS-4 fit-eta is
    the Serber-calibrated value, not the rigorous one."""
    rc, out, _ = run(["historical", "--exclude-boosted"], capsys)
    assert rc == 0
    assert "2.495" in out or "2.49" in out  # Serber-default RDS-4 fit-eta
    assert "3.021" in out  # Fat Man unchanged


def test_historical_rigorous_lowers_eta(capsys):
    """--rigorous in the historical listing recovers the lower fit-eta
    for HEU-containing composites."""
    rc, out, _ = run(["historical", "--exclude-boosted", "--rigorous"], capsys)
    assert rc == 0
    assert "rigorous" in out
    assert "2.343" in out or "2.34" in out  # rigorous RDS-4 fit-eta


def test_plot_historical_default(tmp_path, capsys):
    out_png = tmp_path / "hist.png"
    rc, _, err = run(
        ["plot-historical", "-o", str(out_png)],
        capsys,
    )
    assert rc == 0
    assert out_png.exists()
    # CHIC-12 has year=0 in the seed library; it should be filtered by default.
    assert "CHIC-12" in err and "no year recorded" in err


def test_plot_historical_include_undated(tmp_path, capsys):
    out_png = tmp_path / "hist_all.png"
    rc, _, err = run(
        ["plot-historical", "--include-undated", "-o", str(out_png)],
        capsys,
    )
    assert rc == 0
    assert out_png.exists()
    # No filter notice when including undated entries
    assert "no year recorded" not in err


def test_plot_historical_rigorous(tmp_path, capsys):
    out_png = tmp_path / "hist_rig.png"
    rc, _, _ = run(
        ["plot-historical", "--rigorous", "-o", str(out_png)],
        capsys,
    )
    assert rc == 0
    assert out_png.exists()


def test_plot_historical_composite_only(tmp_path, capsys):
    out_png = tmp_path / "hist_comp.png"
    rc, _, _ = run(
        ["plot-historical", "--composite-only", "-o", str(out_png)],
        capsys,
    )
    assert rc == 0
    assert out_png.exists()


def test_plot_composite_rigorous_flag(tmp_path, capsys):
    out_png = tmp_path / "rigorous.png"
    rc, _, _ = run(
        [
            "plot-composite",
            "--pu-range", "0.1", "3",
            "--fixed-shell", "4",
            "--fixed-compression", "3", "5",
            "--rigorous",
            "-n", "30",
            "-o", str(out_png),
        ],
        capsys,
    )
    assert rc == 0
    assert out_png.exists()
