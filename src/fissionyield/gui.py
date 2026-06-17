"""PySide6 GUI for the fissionyield package.

A lite single-window GUI: solver on the left, plot on the right.
"""

from __future__ import annotations

import sys
from typing import Optional

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .materials import MATERIALS, Material, get_material
from .model import MODELS, compression, kappa, mass_kg, yield_kt


SOLVE_TARGETS = ("Yield (kt)", "Mass (kg)", "Compression η")


def _parse_floats(text: str) -> list[float]:
    """Parse a free-form list of floats separated by commas or whitespace."""
    parts = [p for p in text.replace(",", " ").split() if p]
    return [float(p) for p in parts]


class SolverPanel(QWidget):
    """Left-hand solver: choose two of {M, η, Y}, compute the third."""

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__(parent)
        self._main = parent

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        target_box = QGroupBox("Solve for")
        target_layout = QVBoxLayout(target_box)
        self._target_group = QButtonGroup(self)
        self._target_buttons: list[QRadioButton] = []
        for i, label in enumerate(SOLVE_TARGETS):
            btn = QRadioButton(label)
            self._target_group.addButton(btn, i)
            target_layout.addWidget(btn)
            self._target_buttons.append(btn)
        self._target_buttons[0].setChecked(True)
        self._target_group.idClicked.connect(self._on_target_changed)
        layout.addWidget(target_box)

        inputs_box = QGroupBox("Inputs")
        form = QFormLayout(inputs_box)
        self._mass_edit = self._make_spin(0.0, 1e6, 3, 6.1)
        self._eta_edit = self._make_spin(0.0, 1e3, 4, 2.5)
        self._yield_edit = self._make_spin(0.0, 1e6, 4, 20.0)
        form.addRow("Mass M (kg):", self._mass_edit)
        form.addRow("Compression η:", self._eta_edit)
        form.addRow("Yield Y (kt):", self._yield_edit)
        layout.addWidget(inputs_box)

        self._solve_btn = QPushButton("Solve")
        self._solve_btn.clicked.connect(self._on_solve)
        layout.addWidget(self._solve_btn)

        results_box = QGroupBox("Result")
        rlayout = QVBoxLayout(results_box)
        self._result = QTextEdit()
        self._result.setReadOnly(True)
        self._result.setMinimumHeight(180)
        rlayout.addWidget(self._result)
        layout.addWidget(results_box, stretch=1)

        self._on_target_changed(0)

    @staticmethod
    def _make_spin(lo: float, hi: float, decimals: int, default: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(decimals)
        sb.setValue(default)
        sb.setSingleStep(0.1)
        return sb

    def _on_target_changed(self, target_id: int) -> None:
        self._mass_edit.setEnabled(target_id != 1)
        self._eta_edit.setEnabled(target_id != 2)
        self._yield_edit.setEnabled(target_id != 0)

    def _on_solve(self) -> None:
        mat = self._main.current_material()
        model = self._main.current_model()
        target = self._target_group.checkedId()
        try:
            if target == 0:
                M = self._mass_edit.value()
                eta = self._eta_edit.value()
                Y = yield_kt(M, eta, mat, model=model)
                self._yield_edit.setValue(Y)
                solved = ("yield", f"{Y:.4g} kt")
            elif target == 1:
                eta = self._eta_edit.value()
                Y = self._yield_edit.value()
                M = mass_kg(eta, Y, mat, model=model)
                self._mass_edit.setValue(M)
                solved = ("mass", f"{M:.4f} kg")
            else:
                M = self._mass_edit.value()
                Y = self._yield_edit.value()
                eta = compression(M, Y, mat, model=model)
                self._eta_edit.setValue(eta)
                solved = ("compression", f"η = {eta:.4f}")
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Solve failed", str(exc))
            return

        k0 = M / mat.M0
        k = kappa(M, eta, mat)
        lines = [
            f"Material         : {mat.name}",
            f"Model            : eq. {model}",
            f"Mass M           : {M:.4f} kg",
            f"Compression η    : {eta:.4f}",
            f"Yield Y          : {Y:.4g} kt",
            f"  solved for     : {solved[0]} = {solved[1]}",
            f"κ₀ = M / M₀      : {k0:.4f}",
            f"κ  = κ₀ · η²     : {k:.4f}",
        ]
        if k <= 1.0:
            lines.append("note: κ ≤ 1 (sub-critical) — no yield.")
        self._result.setPlainText("\n".join(lines))


class PlotPanel(QWidget):
    """Right-hand plot panel: yield curves vs mass or vs compression."""

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__(parent)
        self._main = parent

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        controls = QGroupBox("Plot controls")
        clayout = QFormLayout(controls)

        axis_row = QHBoxLayout()
        self._vs_mass = QRadioButton("vs Mass")
        self._vs_eta = QRadioButton("vs Compression")
        self._vs_mass.setChecked(True)
        self._vs_group = QButtonGroup(self)
        self._vs_group.addButton(self._vs_mass, 0)
        self._vs_group.addButton(self._vs_eta, 1)
        self._vs_group.idClicked.connect(self._on_axis_changed)
        axis_row.addWidget(self._vs_mass)
        axis_row.addWidget(self._vs_eta)
        axis_row.addStretch(1)
        clayout.addRow("X-axis:", _wrap(axis_row))

        range_row = QHBoxLayout()
        self._xmin = QDoubleSpinBox()
        self._xmin.setRange(0.0, 1e6)
        self._xmin.setDecimals(3)
        self._xmin.setValue(0.5)
        self._xmax = QDoubleSpinBox()
        self._xmax.setRange(0.0, 1e6)
        self._xmax.setDecimals(3)
        self._xmax.setValue(20.0)
        range_row.addWidget(self._xmin)
        range_row.addWidget(QLabel(" to "))
        range_row.addWidget(self._xmax)
        range_row.addStretch(1)
        clayout.addRow("X range:", _wrap(range_row))

        self._fixed_edit = QLineEdit("1.5, 2.5, 4.0")
        self._fixed_edit.setPlaceholderText("comma- or space-separated floats")
        self._fixed_label = QLabel("Fixed η values:")
        clayout.addRow(self._fixed_label, self._fixed_edit)

        opts_row = QHBoxLayout()
        self._logy = QCheckBox("Log Y")
        self._logy.setChecked(True)
        self._logx = QCheckBox("Log X")
        opts_row.addWidget(self._logy)
        opts_row.addWidget(self._logx)
        opts_row.addStretch(1)
        clayout.addRow("Scale:", _wrap(opts_row))

        self._ymin = QDoubleSpinBox()
        self._ymin.setRange(1e-9, 1e6)
        self._ymin.setDecimals(4)
        self._ymin.setValue(0.01)
        clayout.addRow("Y floor (log):", self._ymin)

        btn_row = QHBoxLayout()
        self._update_btn = QPushButton("Update plot")
        self._update_btn.clicked.connect(self.replot)
        self._save_btn = QPushButton("Save plot…")
        self._save_btn.clicked.connect(self._save_plot)
        btn_row.addWidget(self._update_btn)
        btn_row.addWidget(self._save_btn)
        clayout.addRow(_wrap(btn_row))

        layout.addWidget(controls)

        self._figure = Figure(figsize=(6, 4.5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas, stretch=1)

        self.replot()

    def _on_axis_changed(self, axis_id: int) -> None:
        if axis_id == 0:
            self._xmin.setValue(0.5)
            self._xmax.setValue(20.0)
            self._fixed_edit.setText("1.5, 2.5, 4.0")
            self._fixed_label.setText("Fixed η values:")
        else:
            self._xmin.setValue(1.0)
            self._xmax.setValue(5.0)
            mat = self._main.current_material()
            self._fixed_edit.setText(f"{mat.M0 / 4:.2f}")
            self._fixed_label.setText("Fixed M values (kg):")

    def replot(self) -> None:
        mat = self._main.current_material()
        model = self._main.current_model()
        vs_mass = self._vs_group.checkedId() == 0
        try:
            fixed = _parse_floats(self._fixed_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Bad input", "Could not parse fixed values.")
            return
        if not fixed:
            QMessageBox.warning(self, "Bad input", "Provide at least one fixed value.")
            return

        xmin, xmax = self._xmin.value(), self._xmax.value()
        if xmax <= xmin:
            QMessageBox.warning(self, "Bad range", "X max must be greater than X min.")
            return

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        xs = np.linspace(xmin, xmax, 400)
        floor = self._ymin.value() if self._logy.isChecked() else 0.0
        skipped: list[str] = []

        if vs_mass:
            for eta in fixed:
                ys = np.array([_safe_yield(x, eta, mat, model) for x in xs])
                label = f"η = {eta:g}"
                if not np.any(ys > floor):
                    skipped.append(label)
                    continue
                ax.plot(xs, ys, label=label)
            ax.set_xlabel("Mass M (kg)")
            title = f"Yield vs Mass — {mat.name} (eq. {model})"
        else:
            for m in fixed:
                ys = np.array([_safe_yield(m, x, mat, model) for x in xs])
                label = f"M = {m:g} kg"
                if not np.any(ys > floor):
                    skipped.append(label)
                    continue
                ax.plot(xs, ys, label=label)
            ax.set_xlabel(r"Compression $\eta = \rho/\rho_0$")
            title = f"Yield vs Compression — {mat.name} (eq. {model})"

        ax.set_ylabel("Yield Y (kt)")
        if self._logy.isChecked():
            ax.set_yscale("log")
            ax.set_ylim(bottom=self._ymin.value())
        if self._logx.isChecked():
            ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        if ax.has_data():
            ax.legend(loc="best")
        if skipped:
            ax.text(
                0.02, 0.02,
                "skipped (no yield above floor): " + ", ".join(skipped),
                transform=ax.transAxes, fontsize=8, alpha=0.6,
            )

        self._canvas.draw_idle()

    def _save_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            "yield_plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)",
        )
        if path:
            try:
                self._figure.savefig(path, dpi=140)
            except Exception as exc:
                QMessageBox.warning(self, "Save failed", str(exc))


def _wrap(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


def _safe_yield(m: float, eta: float, mat: Material, model: str) -> float:
    if m <= 0 or eta <= 0:
        return float("nan")
    return yield_kt(m, eta, mat, model=model)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("fissionyield — Cochran one-group model")
        self.resize(1180, 720)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        top.addWidget(QLabel("Material:"))
        self._material_combo = QComboBox()
        for mat in MATERIALS.values():
            self._material_combo.addItem(mat.name, mat.key)
        idx = self._material_combo.findData("delta-WGPu")
        if idx >= 0:
            self._material_combo.setCurrentIndex(idx)
        self._material_combo.currentIndexChanged.connect(self._on_material_changed)
        top.addWidget(self._material_combo, stretch=1)

        top.addSpacing(20)
        top.addWidget(QLabel("Model:"))
        self._model_group = QButtonGroup(self)
        self._model_657 = QRadioButton("6.57 (simplified)")
        self._model_649 = QRadioButton("6.49 (rigorous)")
        self._model_657.setChecked(True)
        self._model_group.addButton(self._model_657, 0)
        self._model_group.addButton(self._model_649, 1)
        self._model_group.idClicked.connect(self._on_model_changed)
        top.addWidget(self._model_657)
        top.addWidget(self._model_649)
        top.addStretch(1)
        outer.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        self._solver = SolverPanel(self)
        self._plot = PlotPanel(self)
        splitter.addWidget(self._solver)
        splitter.addWidget(self._plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 820])
        outer.addWidget(splitter, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Pick two of {M, η, Y} and Solve, or set ranges and Update plot."
        )

    def current_material(self) -> Material:
        return get_material(self._material_combo.currentData())

    def current_model(self) -> str:
        return "6.57" if self._model_group.checkedId() == 0 else "6.49"

    def _on_material_changed(self, _idx: int) -> None:
        self._plot.replot()

    def _on_model_changed(self, _id: int) -> None:
        self._plot.replot()


def main(argv: Optional[list[str]] = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
