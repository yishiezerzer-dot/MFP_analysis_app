from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class DataStudioExportDialog(QDialog):
    def __init__(self, parent: QWidget, *, payload: Dict[str, Any]) -> None:
        super().__init__(parent)
        self._payload = dict(payload or {})
        self._series = [dict(s) for s in (self._payload.get("series") or [])]
        self._meta = dict(self._payload)

        self.setWindowTitle("Export Editor — Data Studio")
        self.resize(1200, 800)

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self._btn_save = QPushButton("Save As…")
        self._btn_save.clicked.connect(self._save_as)
        top.addWidget(self._btn_save)
        self._btn_export = QPushButton("Export Data…")
        self._btn_export.clicked.connect(self._export_data)
        top.addWidget(self._btn_export)
        top.addStretch(1)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.close)
        top.addWidget(b_close)
        layout.addLayout(top)

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)

        self._controls = QWidget()
        form = QFormLayout(self._controls)

        self._title_edit = QLineEdit(str(self._meta.get("title", "")))
        self._xlabel_edit = QLineEdit(str(self._meta.get("xlabel", "")))
        self._ylabel_edit = QLineEdit(str(self._meta.get("ylabel", "")))
        form.addRow("Title", self._title_edit)
        form.addRow("X label", self._xlabel_edit)
        form.addRow("Y label", self._ylabel_edit)

        self._legend_cb = QCheckBox("Show legend")
        self._legend_cb.setChecked(True)
        form.addRow(self._legend_cb)

        self._grid_cb = QCheckBox("Show grid")
        self._grid_cb.setChecked(True)
        form.addRow(self._grid_cb)

        self._reverse_x_cb = QCheckBox("Reverse X axis")
        self._reverse_x_cb.setChecked(False)
        form.addRow(self._reverse_x_cb)

        self._xmin_edit = QLineEdit("")
        self._xmax_edit = QLineEdit("")
        self._ymin_edit = QLineEdit("")
        self._ymax_edit = QLineEdit("")
        form.addRow("X min", self._xmin_edit)
        form.addRow("X max", self._xmax_edit)
        form.addRow("Y min", self._ymin_edit)
        form.addRow("Y max", self._ymax_edit)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._replot)
        form.addRow("", apply_btn)

        body.addWidget(self._controls)

        plot_host = QWidget()
        plot_layout = QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self._fig = Figure(figsize=(11.0, 7.0), dpi=110)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._canvas = FigureCanvasQTAgg(self._fig)
        plot_layout.addWidget(self._canvas, 1)
        body.addWidget(plot_host)

        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 3)
        layout.addWidget(body, 1)

        for w in (self._title_edit, self._xlabel_edit, self._ylabel_edit):
            w.textChanged.connect(self._replot)
        self._legend_cb.toggled.connect(self._replot)
        self._grid_cb.toggled.connect(self._replot)
        self._reverse_x_cb.toggled.connect(self._replot)

        self._replot()

    def _replot(self) -> None:
        self._ax.clear()
        plot_type = str(self._meta.get("plot_type", "Line"))

        if self._meta.get("heatmap") is not None:
            hm = self._meta.get("heatmap")
            if isinstance(hm, dict) and "values" in hm:
                im = self._ax.imshow(hm["values"], aspect="auto")
                self._ax.set_xticks(range(len(hm.get("cols", []))))
                self._ax.set_xticklabels([str(c) for c in hm.get("cols", [])], rotation=45, ha="right")
                self._ax.set_yticks(range(len(hm.get("rows", []))))
                self._ax.set_yticklabels([str(r) for r in hm.get("rows", [])])
                self._fig.colorbar(im, ax=self._ax, fraction=0.046, pad=0.04)
        elif plot_type in ("Box plot", "Violin plot"):
            data = [np.asarray(s.get("y", []), dtype=float) for s in self._series]
            labels = [str(s.get("label", "")) for s in self._series]
            if plot_type == "Box plot":
                self._ax.boxplot(data, labels=labels, showfliers=True)
            else:
                self._ax.violinplot(data, showmeans=True, showmedians=True)
                self._ax.set_xticks(range(1, len(labels) + 1))
                self._ax.set_xticklabels(labels, rotation=45, ha="right")
        elif plot_type == "Histogram":
            for s in self._series:
                y = np.asarray(s.get("y", []), dtype=float)
                self._ax.hist(y, bins=20, alpha=0.5, label=str(s.get("label", "")), color=s.get("color"))
        elif plot_type in ("Bar (grouped)", "Bar (stacked)"):
            xcats = self._meta.get("xcats", [])
            x = np.arange(len(xcats)) if xcats else np.arange(len(self._series[0].get("y", [])))
            width = 0.8 / max(1, len(self._series))
            bottoms = np.zeros_like(x, dtype=float)
            for i, s in enumerate(self._series):
                y = np.asarray(s.get("y", []), dtype=float)
                if plot_type == "Bar (stacked)":
                    self._ax.bar(x, y, bottom=bottoms, label=str(s.get("label", "")), color=s.get("color"))
                    bottoms = bottoms + y
                else:
                    self._ax.bar(x + i * width - (len(self._series) - 1) * width / 2, y, width=width, label=str(s.get("label", "")), color=s.get("color"))
            if xcats:
                self._ax.set_xticks(x)
                self._ax.set_xticklabels([str(c) for c in xcats], rotation=45, ha="right")
        elif plot_type == "Bubble":
            for s in self._series:
                x = np.asarray(s.get("x", []), dtype=float)
                y = np.asarray(s.get("y", []), dtype=float)
                size = np.asarray(s.get("size", []), dtype=float)
                if size.size == 0:
                    size = np.ones_like(y)
                size = 40 + 160 * (size - np.nanmin(size)) / (np.nanmax(size) - np.nanmin(size) + 1e-9)
                self._ax.scatter(x, y, s=size, label=str(s.get("label", "")), alpha=0.6, color=s.get("color"))
        else:
            for s in self._series:
                kind = s.get("kind")
                x = np.asarray(s.get("x", []), dtype=float)
                y = np.asarray(s.get("y", []), dtype=float)
                label = str(s.get("label", ""))
                color = s.get("color")
                if kind == "scatter":
                    self._ax.scatter(x, y, s=24, label=label, color=color)
                elif kind == "area":
                    self._ax.fill_between(x, 0, y, label=label, alpha=0.35, color=color)
                elif kind == "step":
                    self._ax.step(x, y, label=label, where="mid", color=color)
                elif kind == "stem":
                    markerline, stemlines, _baseline = self._ax.stem(x, y, label=label)
                    try:
                        if color:
                            markerline.set_color(color)
                            stemlines.set_color(color)
                    except Exception:
                        pass
                elif kind == "errorbar":
                    self._ax.errorbar(x, y, xerr=s.get("xerr"), yerr=s.get("yerr"), label=label, color=color, fmt="o")
                else:
                    self._ax.plot(x, y, label=label, color=color, marker=("o" if plot_type == "Line + markers" else None))

        self._ax.set_title(self._title_edit.text().strip())
        self._ax.set_xlabel(self._xlabel_edit.text().strip())
        self._ax.set_ylabel(self._ylabel_edit.text().strip())

        if self._legend_cb.isChecked() and len(self._series) > 1:
            self._ax.legend(loc="best")

        self._ax.grid(bool(self._grid_cb.isChecked()), alpha=0.25)

        if self._reverse_x_cb.isChecked():
            try:
                self._ax.invert_xaxis()
            except Exception:
                pass

        self._apply_limits()
        self._canvas.draw_idle()

    def _apply_limits(self) -> None:
        def _parse(v: str) -> Optional[float]:
            try:
                return float(str(v).strip())
            except Exception:
                return None

        xmin = _parse(self._xmin_edit.text())
        xmax = _parse(self._xmax_edit.text())
        ymin = _parse(self._ymin_edit.text())
        ymax = _parse(self._ymax_edit.text())
        if xmin is not None or xmax is not None:
            self._ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            self._ax.set_ylim(bottom=ymin, top=ymax)

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG (*.png);;JPG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;PDF (*.pdf);;All files (*.*)",
        )
        if not path:
            return
        try:
            self._fig.savefig(path)
        except Exception:
            return

    def _export_data(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            "",
            "CSV (*.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            df = self._build_export_dataframe()
            df.to_csv(path, index=False)
        except Exception:
            return

    def _build_export_dataframe(self) -> pd.DataFrame:
        heatmap = self._meta.get("heatmap")
        if isinstance(heatmap, dict) and "values" in heatmap:
            values = np.asarray(heatmap.get("values"), dtype=float)
            rows = [str(r) for r in heatmap.get("rows", [])]
            cols = [str(c) for c in heatmap.get("cols", [])]
            df = pd.DataFrame(values, index=rows, columns=cols)
            df.insert(0, "row", rows)
            return df

        rows: List[Dict[str, Any]] = []
        for s in self._series:
            label = str(s.get("label", ""))
            x = np.asarray(s.get("x", []), dtype=float)
            y = np.asarray(s.get("y", []), dtype=float)
            for i in range(min(len(x), len(y))):
                rows.append({"series": label, "x": float(x[i]), "y": float(y[i])})
        return pd.DataFrame(rows)
