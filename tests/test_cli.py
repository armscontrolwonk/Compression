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
    """Back-solve eta from observed yield: RDS-4 anchor."""
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
    assert "2.34" in out  # fit eta ~ 2.343


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
