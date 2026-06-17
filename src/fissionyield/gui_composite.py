"""PySide6 tabbed GUI for the composite-core extension of fissionyield.

A new program with multiple tabs covering the composite-core and
historical-anchor functionality. The legacy single-material GUI
(``fissionyield.gui``) is preserved unchanged for users on the older
workflow.

Tabs:
- **Composite Pit** -- Pu + HEU + compression solver with embedded plot
  (Y vs Pu mass / shell mass / compression).
- **Historical Anchors** -- browse the seeded anchor library, view
  per-anchor notes and citations, and see the fit-eta-vs-year trajectory
  plot.

The Calibration toggle in the top bar (Serber-b default vs Rigorous)
affects every tab.
"""

from __future__ import annotations

import sys
from typing import Optional

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .composite import (
    alpha_eigenvalue,
    compression_composite,
    critical_compression,
    yield_kt_composite,
)
from .historical import anchors as historical_anchors
from .historical import fit_eta as historical_fit_eta
from .materials import MATERIALS, SERBER_B, Material, get_material


# Country palette mirrors the CLI plot-historical convention.
_COUNTRY_COLORS = {
    "USA":  "#1f4e79",
    "USSR": "#c0504d",
    "UK":   "#7030a0",
    "PRC":  "#e6b800",
    "FR":   "#76608a",
}
_DEFAULT_COLOR = "#666666"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parse_floats(text: str) -> list[float]:
    parts = [p for p in text.replace(",", " ").split() if p]
    return [float(p) for p in parts]


def _wrap(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


def _make_spin(lo: float, hi: float, decimals: int, default: float,
               step: float = 0.1) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setValue(default)
    sb.setSingleStep(step)
    return sb


def _safe_yield(m_pu: float, m_heu: float, eta: float, pu: Material,
                shell: Material, calib) -> float:
    if m_pu < 0 or m_heu < 0 or eta <= 0:
        return float("nan")
    if m_pu == 0 and m_heu == 0:
        return 0.0
    try:
        return yield_kt_composite(
            m_pu, m_heu, eta, pu, shell, calibration=calib
        )
    except (ValueError, RuntimeError):
        return float("nan")


# ===========================================================================
# Composite Pit tab
# ===========================================================================


class CompositeSolverPanel(QWidget):
    """Composite solver: pick masses + compression, solve for yield or eta."""

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__(parent)
        self._main = parent

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Materials box
        mat_box = QGroupBox("Materials")
        mat_form = QFormLayout(mat_box)
        self._pu_combo = QComboBox()
        for mat in MATERIALS.values():
            self._pu_combo.addItem(mat.name, mat.key)
        idx = self._pu_combo.findData("delta-WGPu")
        if idx >= 0:
            self._pu_combo.setCurrentIndex(idx)
        mat_form.addRow("Inner (Pu):", self._pu_combo)

        self._shell_combo = QComboBox()
        for mat in MATERIALS.values():
            self._shell_combo.addItem(mat.name, mat.key)
        idx = self._shell_combo.findData("WGU-93.5")
        if idx >= 0:
            self._shell_combo.setCurrentIndex(idx)
        mat_form.addRow("Outer (Shell):", self._shell_combo)
        layout.addWidget(mat_box)

        # Solve-for radio
        target_box = QGroupBox("Solve for")
        tlayout = QHBoxLayout(target_box)
        self._target_group = QButtonGroup(self)
        self._t_yield = QRadioButton("Yield (kt)")
        self._t_eta = QRadioButton("Compression η")
        self._t_yield.setChecked(True)
        self._target_group.addButton(self._t_yield, 0)
        self._target_group.addButton(self._t_eta, 1)
        self._target_group.idClicked.connect(self._on_target_changed)
        tlayout.addWidget(self._t_yield)
        tlayout.addWidget(self._t_eta)
        tlayout.addStretch(1)
        layout.addWidget(target_box)

        # Inputs
        in_box = QGroupBox("Inputs")
        form = QFormLayout(in_box)
        self._pu_mass = _make_spin(0.0, 1e6, 3, 4.0)
        self._heu_mass = _make_spin(0.0, 1e6, 3, 8.0)
        self._eta = _make_spin(0.001, 1e3, 4, 2.5, step=0.1)
        self._yield = _make_spin(0.0, 1e9, 4, 20.0, step=1.0)
        form.addRow("Pu mass M_Pu (kg):", self._pu_mass)
        form.addRow("HEU mass M_HEU (kg):", self._heu_mass)
        form.addRow("Compression η:", self._eta)
        form.addRow("Yield Y (kt):", self._yield)
        layout.addWidget(in_box)

        # Solve button
        self._solve_btn = QPushButton("Solve")
        self._solve_btn.clicked.connect(self._on_solve)
        layout.addWidget(self._solve_btn)

        # Result
        result_box = QGroupBox("Result")
        rl = QVBoxLayout(result_box)
        self._result = QTextEdit()
        self._result.setReadOnly(True)
        self._result.setMinimumHeight(200)
        rl.addWidget(self._result)
        layout.addWidget(result_box, stretch=1)

        self._on_target_changed(0)

    def pu_material(self) -> Material:
        return get_material(self._pu_combo.currentData())

    def shell_material(self) -> Material:
        return get_material(self._shell_combo.currentData())

    def _on_target_changed(self, target_id: int) -> None:
        # 0 = solve yield -> eta is input, yield is output
        # 1 = solve eta -> yield is input, eta is output
        self._eta.setEnabled(target_id != 1)
        self._yield.setEnabled(target_id != 0)

    def _on_solve(self) -> None:
        pu = self.pu_material()
        shell = self.shell_material()
        calib = self._main.calibration()
        target = self._target_group.checkedId()
        m_pu = self._pu_mass.value()
        m_heu = self._heu_mass.value()

        try:
            if target == 0:
                eta = self._eta.value()
                Y = yield_kt_composite(
                    m_pu, m_heu, eta, pu, shell, calibration=calib,
                )
                self._yield.setValue(Y)
                solved = ("yield", f"{Y:.4g} kt")
            else:
                Y = self._yield.value()
                eta = compression_composite(
                    m_pu, m_heu, Y, pu, shell, calibration=calib,
                )
                self._eta.setValue(eta)
                solved = ("compression", f"η = {eta:.4f}")
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Solve failed", str(exc))
            return

        try:
            eta_c = critical_compression(m_pu, m_heu, pu, shell)
        except Exception:
            eta_c = float("nan")
        try:
            alpha = alpha_eigenvalue(m_pu, m_heu, eta, pu, shell)
        except Exception:
            alpha = float("nan")

        cal_label = "Serber-b (default)" if isinstance(calib, dict) else "rigorous (no calibration)"
        M_total = m_pu + m_heu
        lines = [
            f"Pu (inner)        : {pu.name}",
            f"Shell (outer)     : {shell.name}",
            f"Pu mass M_Pu      : {m_pu:.4f} kg",
            f"Shell mass M_HEU  : {m_heu:.4f} kg",
            f"Total mass        : {M_total:.4f} kg",
            f"Compression η     : {eta:.4f}",
            f"Critical η_c      : {eta_c:.4f}",
            f"Yield Y           : {Y:.4g} kt",
            f"  solved for      : {solved[0]} = {solved[1]}",
            f"Calibration       : {cal_label}",
            f"α eigenvalue      : {alpha:.4f} /shake",
        ]
        if eta <= eta_c:
            lines.append("note: η ≤ η_c (sub-critical or just critical) — no yield.")
        self._result.setPlainText("\n".join(lines))


class CompositePlotPanel(QWidget):
    """Composite plot: Y vs Pu mass / shell mass / compression."""

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__(parent)
        self._main = parent

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        ctl_box = QGroupBox("Plot controls")
        cl = QFormLayout(ctl_box)

        # X-axis radio
        axis_row = QHBoxLayout()
        self._vs_group = QButtonGroup(self)
        self._vs_pu = QRadioButton("Pu mass")
        self._vs_shell = QRadioButton("Shell mass")
        self._vs_eta = QRadioButton("Compression")
        self._vs_pu.setChecked(True)
        for btn, idx in [(self._vs_pu, 0), (self._vs_shell, 1), (self._vs_eta, 2)]:
            self._vs_group.addButton(btn, idx)
            axis_row.addWidget(btn)
        axis_row.addStretch(1)
        self._vs_group.idClicked.connect(self._on_axis_changed)
        cl.addRow("X axis:", _wrap(axis_row))

        # Range
        range_row = QHBoxLayout()
        self._xmin = _make_spin(0.0, 1e6, 3, 0.1)
        self._xmax = _make_spin(0.0, 1e6, 3, 5.0)
        range_row.addWidget(self._xmin)
        range_row.addWidget(QLabel(" to "))
        range_row.addWidget(self._xmax)
        range_row.addStretch(1)
        cl.addRow("X range:", _wrap(range_row))

        # Fixed dimensions
        self._fixed1_label = QLabel("Shell masses (kg):")
        self._fixed1_edit = QLineEdit("0, 4")
        cl.addRow(self._fixed1_label, self._fixed1_edit)

        self._fixed2_label = QLabel("η values:")
        self._fixed2_edit = QLineEdit("2.5, 3.5, 5.0")
        cl.addRow(self._fixed2_label, self._fixed2_edit)

        # Log / floor
        opt_row = QHBoxLayout()
        self._logy = QCheckBox("Log Y")
        self._logy.setChecked(True)
        self._logx = QCheckBox("Log X")
        opt_row.addWidget(self._logy)
        opt_row.addWidget(self._logx)
        opt_row.addStretch(1)
        cl.addRow("Scale:", _wrap(opt_row))

        self._ymin = _make_spin(1e-6, 1e6, 4, 0.01)
        cl.addRow("Y floor (log):", self._ymin)

        # Buttons
        btn_row = QHBoxLayout()
        self._update_btn = QPushButton("Update plot")
        self._update_btn.clicked.connect(self.replot)
        self._save_btn = QPushButton("Save plot…")
        self._save_btn.clicked.connect(self._save_plot)
        btn_row.addWidget(self._update_btn)
        btn_row.addWidget(self._save_btn)
        cl.addRow(_wrap(btn_row))

        layout.addWidget(ctl_box)

        self._figure = Figure(figsize=(6, 4.5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas, stretch=1)

        # The first replot is deferred to MainWindow.__init__ after every
        # tab has been assigned, because replot() reads materials via
        # the MainWindow -> CompositeTab delegation chain.

    def _on_axis_changed(self, axis_id: int) -> None:
        if axis_id == 0:  # vs Pu mass; fixed: shell mass, eta
            self._xmin.setValue(0.1)
            self._xmax.setValue(5.0)
            self._fixed1_label.setText("Shell masses (kg):")
            self._fixed1_edit.setText("0, 4")
            self._fixed2_label.setText("η values:")
            self._fixed2_edit.setText("2.5, 3.5, 5.0")
        elif axis_id == 1:  # vs shell mass; fixed: Pu mass, eta
            self._xmin.setValue(0.0)
            self._xmax.setValue(20.0)
            self._fixed1_label.setText("Pu masses (kg):")
            self._fixed1_edit.setText("1.0")
            self._fixed2_label.setText("η values:")
            self._fixed2_edit.setText("2.5, 3.5, 5.0")
        else:  # vs compression; fixed: Pu mass, shell mass
            self._xmin.setValue(1.5)
            self._xmax.setValue(6.0)
            self._fixed1_label.setText("Pu masses (kg):")
            self._fixed1_edit.setText("0.5, 1.0")
            self._fixed2_label.setText("Shell masses (kg):")
            self._fixed2_edit.setText("4.0")

    def replot(self) -> None:
        pu = self._main.pu_material()
        shell = self._main.shell_material()
        calib = self._main.calibration()
        axis_id = self._vs_group.checkedId()

        try:
            f1 = _parse_floats(self._fixed1_edit.text())
            f2 = _parse_floats(self._fixed2_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Bad input", "Could not parse fixed values.")
            return
        if not f1 or not f2:
            QMessageBox.warning(
                self, "Bad input", "Need at least one value in each fixed list."
            )
            return

        xmin, xmax = self._xmin.value(), self._xmax.value()
        if xmax <= xmin:
            QMessageBox.warning(self, "Bad range", "X max must exceed X min.")
            return

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        xs = np.linspace(xmin, xmax, 200)
        floor = self._ymin.value() if self._logy.isChecked() else 0.0
        skipped: list[str] = []

        for v1 in f1:
            for v2 in f2:
                if axis_id == 0:    # x = Pu mass; v1 = shell mass; v2 = eta
                    ys = np.array([_safe_yield(x, v1, v2, pu, shell, calib) for x in xs])
                    label = f"shell={v1:g} kg, η={v2:g}"
                elif axis_id == 1:  # x = shell mass; v1 = Pu mass; v2 = eta
                    ys = np.array([_safe_yield(v1, x, v2, pu, shell, calib) for x in xs])
                    label = f"Pu={v1:g} kg, η={v2:g}"
                else:               # x = eta; v1 = Pu mass; v2 = shell mass
                    ys = np.array([_safe_yield(v1, v2, x, pu, shell, calib) for x in xs])
                    label = f"Pu={v1:g} kg, shell={v2:g} kg"
                if not np.any(ys > floor):
                    skipped.append(label)
                    continue
                ax.plot(xs, ys, label=label)

        if axis_id == 0:
            ax.set_xlabel(f"Pu mass ({pu.key}) (kg)")
            title = "Composite yield vs Pu mass"
        elif axis_id == 1:
            ax.set_xlabel(f"Shell mass ({shell.key}) (kg)")
            title = "Composite yield vs shell mass"
        else:
            ax.set_xlabel(r"Compression $\eta = \rho/\rho_0$")
            title = "Composite yield vs compression"
        ax.set_ylabel("Yield Y (kt)")
        if self._logy.isChecked():
            ax.set_yscale("log")
            ax.set_ylim(bottom=self._ymin.value())
        if self._logx.isChecked():
            ax.set_xscale("log")
        if not isinstance(calib, dict):
            title += "  [rigorous]"
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        if ax.has_data():
            ax.legend(loc="best", fontsize=8)
        if skipped:
            ax.text(
                0.02, 0.02,
                "skipped: " + ", ".join(skipped),
                transform=ax.transAxes, fontsize=7, alpha=0.6,
            )
        self._canvas.draw_idle()

    def _save_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", "composite_yield.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)",
        )
        if path:
            try:
                self._figure.savefig(path, dpi=140)
            except Exception as exc:
                QMessageBox.warning(self, "Save failed", str(exc))


class CompositeTab(QWidget):
    """Container: solver left, plot right, split."""

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__(parent)
        self._main = parent
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        self._solver = CompositeSolverPanel(parent)
        self._plot = CompositePlotPanel(parent)
        # Material changes in the solver should retrigger the plot.
        self._solver._pu_combo.currentIndexChanged.connect(lambda _i: self._plot.replot())
        self._solver._shell_combo.currentIndexChanged.connect(lambda _i: self._plot.replot())
        splitter.addWidget(self._solver)
        splitter.addWidget(self._plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 820])
        layout.addWidget(splitter)

    def replot(self) -> None:
        self._plot.replot()

    def pu_material(self) -> Material:
        return self._solver.pu_material()

    def shell_material(self) -> Material:
        return self._solver.shell_material()


# ===========================================================================
# Historical Anchors tab
# ===========================================================================


_ANCHOR_COLS = [
    "Name", "Year", "Country", "M_Pu (kg)", "M_HEU (kg)",
    "Yield (kt)", "Boosted", "Fit η",
]


class HistoricalTab(QWidget):
    """Anchor library browser + fit-η trajectory plot."""

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__(parent)
        self._main = parent
        self._rows: list[tuple] = []  # (test, year_for_plot, eta, undated)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # Left: filters + table + detail
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        filt_box = QGroupBox("Filters")
        fl = QVBoxLayout(filt_box)
        self._composite_only = QCheckBox("Composite cores only")
        self._exclude_boosted = QCheckBox("Exclude boosted")
        self._include_undated = QCheckBox("Include undated entries in plot")
        for cb in (self._composite_only, self._exclude_boosted, self._include_undated):
            cb.stateChanged.connect(self.refresh)
            fl.addWidget(cb)
        ll.addWidget(filt_box)

        self._table = QTableWidget()
        self._table.setColumnCount(len(_ANCHOR_COLS))
        self._table.setHorizontalHeaderLabels(_ANCHOR_COLS)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection)
        ll.addWidget(self._table, stretch=1)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setMaximumHeight(180)
        self._detail.setPlaceholderText("Select a row to see notes and citation.")
        ll.addWidget(self._detail)
        splitter.addWidget(left)

        # Right: trajectory plot + save
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save plot…")
        self._save_btn.clicked.connect(self._save_plot)
        btn_row.addStretch(1)
        btn_row.addWidget(self._save_btn)
        rl.addLayout(btn_row)

        self._figure = Figure(figsize=(7, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        rl.addWidget(self._canvas, stretch=1)
        splitter.addWidget(right)

        splitter.setSizes([560, 720])
        layout.addWidget(splitter)

        self.refresh()

    def refresh(self) -> None:
        calib = self._main.calibration()
        tests = list(historical_anchors())
        if self._composite_only.isChecked():
            tests = [t for t in tests if t.is_composite]
        if self._exclude_boosted.isChecked():
            tests = [t for t in tests if not t.boosted]

        self._rows.clear()
        for t in tests:
            try:
                eta = historical_fit_eta(t, calibration=calib)
            except Exception:
                eta = float("nan")
            self._rows.append((t, eta))

        self._table.setRowCount(len(self._rows))
        for i, (t, eta) in enumerate(self._rows):
            year_str = str(t.year) if t.year > 0 else "—"
            eta_str = f"{eta:.3f}" if eta == eta else "err"
            self._table.setItem(i, 0, QTableWidgetItem(t.name))
            self._table.setItem(i, 1, QTableWidgetItem(year_str))
            self._table.setItem(i, 2, QTableWidgetItem(t.country))
            self._table.setItem(i, 3, QTableWidgetItem(f"{t.pu_kg:.2f}"))
            self._table.setItem(i, 4, QTableWidgetItem(f"{t.heu_kg:.2f}"))
            self._table.setItem(i, 5, QTableWidgetItem(f"{t.yield_kt:.2f}"))
            self._table.setItem(i, 6, QTableWidgetItem("yes" if t.boosted else ""))
            self._table.setItem(i, 7, QTableWidgetItem(eta_str))

        self._replot_trajectory()

    def _on_selection(self) -> None:
        row = self._table.currentRow()
        if 0 <= row < len(self._rows):
            t, eta = self._rows[row]
            text_lines = [
                f"{t.name} ({t.year if t.year > 0 else 'year unknown'}, {t.country})",
                f"M_Pu = {t.pu_kg:.2f} kg, M_HEU = {t.heu_kg:.2f} kg, "
                f"Y = {t.yield_kt:g} kt, fit η = {eta:.3f}",
                "",
            ]
            if t.notes:
                text_lines.extend(["Notes:", t.notes, ""])
            if t.source:
                text_lines.extend(["Source:", t.source])
            self._detail.setPlainText("\n".join(text_lines))
        else:
            self._detail.clear()

    def _replot_trajectory(self) -> None:
        calib = self._main.calibration()
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        plotted = []
        for t, eta in self._rows:
            if eta != eta:
                continue
            if t.year > 0:
                plotted.append((t, t.year, eta, False))
            elif self._include_undated.isChecked():
                plotted.append((t, 1975, eta, True))

        seen_countries: list[str] = []
        any_boosted = False
        for t, year, eta, undated in plotted:
            color = _COUNTRY_COLORS.get(t.country, _DEFAULT_COLOR)
            if t.country not in seen_countries:
                seen_countries.append(t.country)
            marker = "D" if t.boosted else "o"
            if t.boosted:
                any_boosted = True
            size = 60 + 200 * np.log10(max(t.yield_kt, 0.1) + 1.0)
            ax.scatter(
                year, eta, s=size, c=color, edgecolors=color, linewidths=1.0,
                marker=marker, zorder=3, alpha=0.45 if undated else 0.9,
            )
            year_label = str(t.year) if t.year > 0 else "year?"
            ax.annotate(
                f"{t.name} ({year_label})", (year, eta),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=color,
            )

        handles = [
            Line2D(
                [], [], marker="o", linestyle="none",
                markerfacecolor=_COUNTRY_COLORS.get(c, _DEFAULT_COLOR),
                markeredgecolor=_COUNTRY_COLORS.get(c, _DEFAULT_COLOR),
                label=c,
            )
            for c in seen_countries
        ]
        if any_boosted:
            handles.append(Line2D(
                [], [], marker="D", linestyle="none",
                markerfacecolor="#888888", markeredgecolor="black",
                label="boosted (upper bound)",
            ))
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=8)

        cal_label = "Serber-b default" if isinstance(calib, dict) else "rigorous one-group"
        ax.set_xlabel("Test year")
        ax.set_ylabel(r"Fit $\eta$ (effective compression)")
        ax.set_title(f"Historical anchors: fit-η vs year ({cal_label})")
        ax.grid(True, alpha=0.3)
        if plotted:
            years = [year for _, year, _, _ in plotted]
            ax.set_xlim(min(years) - 3, max(years) + 5)
        self._canvas.draw_idle()

    def _save_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", "historical_trajectory.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)",
        )
        if path:
            try:
                self._figure.savefig(path, dpi=140)
            except Exception as exc:
                QMessageBox.warning(self, "Save failed", str(exc))


# ===========================================================================
# Main window
# ===========================================================================


class MainWindow(QMainWindow):
    """Tabbed top-level window. The Calibration toggle in the header
    propagates to every tab on change."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "fissionyield-composite — composite cores & historical anchors"
        )
        self.resize(1300, 800)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        # Top bar: calibration toggle
        top = QHBoxLayout()
        top.addWidget(QLabel("Calibration:"))
        self._calib_group = QButtonGroup(self)
        self._cal_serber = QRadioButton("Serber-b (default)")
        self._cal_rigorous = QRadioButton("Rigorous (no calibration)")
        self._cal_serber.setChecked(True)
        self._calib_group.addButton(self._cal_serber, 0)
        self._calib_group.addButton(self._cal_rigorous, 1)
        self._calib_group.idClicked.connect(self._on_calibration_changed)
        top.addWidget(self._cal_serber)
        top.addWidget(self._cal_rigorous)
        top.addStretch(1)
        outer.addLayout(top)

        # Tabs
        self._tabs = QTabWidget()
        self._composite_tab = CompositeTab(self)
        self._historical_tab = HistoricalTab(self)
        self._tabs.addTab(self._composite_tab, "Composite Pit")
        self._tabs.addTab(self._historical_tab, "Historical Anchors")
        outer.addWidget(self._tabs, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Composite cores + historical anchor browser. "
            "Default calibration: Serber-b (Cochran eq. 6.60)."
        )

        # First render -- safe to do now that every tab is assigned.
        self._composite_tab.replot()

    def calibration(self):
        return SERBER_B if self._calib_group.checkedId() == 0 else 1.0

    def pu_material(self) -> Material:
        return self._composite_tab.pu_material()

    def shell_material(self) -> Material:
        return self._composite_tab.shell_material()

    def _on_calibration_changed(self, _id: int) -> None:
        self._composite_tab.replot()
        self._historical_tab.refresh()


def main(argv: Optional[list[str]] = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
