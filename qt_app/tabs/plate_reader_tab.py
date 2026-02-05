from __future__ import annotations

import traceback
from typing import Any, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QColorDialog,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from lab_gui.plate_reader_io import preview_dataframe, read_plate_file
from lab_gui.plate_reader_model import PlateReaderDataset, build_mic_wizard_config_and_result
from qt_app.adapters import PlateReaderAdapter
from qt_app.services import DialogService, StatusService
from qt_app.services.worker import run_in_worker


class PlateReaderPreviewDialog(QDialog):
    def __init__(self, parent: QWidget, *, max_rows_default: int = 120) -> None:
        super().__init__(parent)
        self._max_rows = int(max_rows_default)
        self._df = None

        self.setWindowTitle("Plate Reader Preview")
        self.resize(920, 600)

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        layout.addLayout(top)

        top.addWidget(QLabel("Rows:"))
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(10, 2000)
        self._rows_spin.setValue(self._max_rows)
        self._rows_spin.valueChanged.connect(self._refresh)
        top.addWidget(self._rows_spin)

        self._info_label = QLabel("")
        top.addWidget(self._info_label, 1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        top.addWidget(close_btn)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self._table, 1)

    def set_df(self, df, *, title_suffix: str = "") -> None:
        self._df = df
        if title_suffix:
            self.setWindowTitle(f"Plate Reader Preview — {title_suffix}")
        else:
            self.setWindowTitle("Plate Reader Preview")
        self._refresh()

    def _refresh(self) -> None:
        try:
            self._max_rows = int(self._rows_spin.value())
        except Exception:
            self._max_rows = 120
        self._render(self._df)

    def _render(self, df) -> None:
        table = self._table
        table.clear()
        if df is None or df.empty:
            self._info_label.setText("(no data)")
            table.setRowCount(0)
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels(["(empty)"])
            return

        self._info_label.setText(f"{int(df.shape[0])} rows × {int(df.shape[1])} cols")
        cols, rows = preview_dataframe(df, max_rows=self._max_rows)
        table.setColumnCount(len(cols))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(cols)
        for r_idx, row in enumerate(rows):
            for c_idx, val in enumerate(row):
                table.setItem(r_idx, c_idx, QTableWidgetItem(str(val)))
        table.resizeColumnsToContents()


class PlateReaderRunWizardDialog(QDialog):
    def __init__(self, parent: QWidget, *, dataset: PlateReaderDataset, df, on_apply, dialogs: DialogService) -> None:
        super().__init__(parent)
        self._dataset = dataset
        self._df = df
        self._on_apply = on_apply
        self._dialogs = dialogs

        self.setWindowTitle("Run Analysis")
        self.resize(980, 680)

        layout = QVBoxLayout(self)
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        self._step_select = QWidget()
        self._step_mic = QWidget()
        self._stack.addWidget(self._step_select)
        self._stack.addWidget(self._step_mic)

        self._analysis_group = QButtonGroup(self)
        self._analysis_group.setExclusive(True)

        self._build_step_select()
        self._build_step_mic()
        self._show_step("select")

    def _show_step(self, which: str) -> None:
        if which == "mic":
            self._stack.setCurrentWidget(self._step_mic)
        else:
            self._stack.setCurrentWidget(self._step_select)

    def _build_step_select(self) -> None:
        f = self._step_select
        layout = QVBoxLayout(f)
        layout.addWidget(QLabel("Choose analysis"))
        layout.addWidget(QLabel("Only MIC is implemented right now."))

        box = QGroupBox("Analysis")
        box_layout = QVBoxLayout(box)
        rb_mic = QRadioButton("MIC")
        rb_mic.setChecked(True)
        self._analysis_group.addButton(rb_mic)
        box_layout.addWidget(rb_mic)

        for name in ["Fluorescence", "UV/Absorbance", "Kinetics", "Custom"]:
            rb = QRadioButton(name + " (coming soon)")
            rb.setEnabled(False)
            box_layout.addWidget(rb)
        layout.addWidget(box)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        cont_btn = QPushButton("Continue")
        cont_btn.clicked.connect(lambda: self._show_step("mic"))
        btns.addWidget(cont_btn)
        layout.addLayout(btns)

    def _build_step_mic(self) -> None:
        f = self._step_mic
        layout = QVBoxLayout(f)

        layout.addWidget(QLabel("MIC configuration"))
        layout.addWidget(
            QLabel(
                "Select sample/control replicate rows, select concentration columns, and (optionally) remap tick labels."
            )
        )

        top = QGridLayout()
        layout.addLayout(top, 1)

        self._use_header_cb = QCheckBox("First row is headers")
        self._use_header_cb.setChecked(True)
        self._use_header_cb.toggled.connect(self._reload_df)
        top.addWidget(self._use_header_cb, 0, 0, 1, 1)

        self._prev_table = QTableWidget()
        self._prev_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._prev_table.setSelectionMode(QAbstractItemView.NoSelection)
        prev_box = QGroupBox("Preview (first rows)")
        prev_layout = QVBoxLayout(prev_box)
        prev_layout.addWidget(self._prev_table)
        top.addWidget(prev_box, 1, 0)

        rows_box = QGroupBox("Rows")
        rows_layout = QGridLayout(rows_box)
        rows_layout.addWidget(QLabel("Sample rows (replicates)"), 0, 0)
        rows_layout.addWidget(QLabel("Control rows (optional)"), 0, 1)

        self._sample_rows = QListWidget()
        self._sample_rows.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._control_rows = QListWidget()
        self._control_rows.setSelectionMode(QAbstractItemView.ExtendedSelection)
        rows_layout.addWidget(self._sample_rows, 1, 0)
        rows_layout.addWidget(self._control_rows, 1, 1)
        top.addWidget(rows_box, 1, 1)

        cols_box = QGroupBox("Columns = concentrations")
        cols_layout = QVBoxLayout(cols_box)
        cols_layout.addWidget(QLabel("Select concentration columns (in sheet order)"))
        self._cols_list = QListWidget()
        self._cols_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._cols_list.itemSelectionChanged.connect(self._update_auto_ticks)
        cols_layout.addWidget(self._cols_list)
        top.addWidget(cols_box, 1, 2)

        bot = QGridLayout()
        layout.addLayout(bot)

        bot.addWidget(QLabel("Tick labels:"), 0, 0)
        self._ticks_entry = QLineEdit()
        bot.addWidget(self._ticks_entry, 0, 1)

        self._auto_ticks_cb = QCheckBox("Default 1024,512,…,0")
        self._auto_ticks_cb.setChecked(True)
        self._auto_ticks_cb.toggled.connect(self._on_auto_ticks_toggle)
        bot.addWidget(self._auto_ticks_cb, 0, 2)

        bot.addWidget(QLabel("Plot type:"), 0, 3)
        self._plot_type = QComboBox()
        self._plot_type.addItems(["bar", "line", "scatter"])
        bot.addWidget(self._plot_type, 0, 4)

        bot.addWidget(QLabel("Control style:"), 0, 5)
        self._control_style = QComboBox()
        self._control_style.addItems(["bars", "line"])
        bot.addWidget(self._control_style, 0, 6)

        titles = QGridLayout()
        layout.addLayout(titles)
        titles.addWidget(QLabel("Title:"), 0, 0)
        self._title_entry = QLineEdit("MIC")
        titles.addWidget(self._title_entry, 0, 1)
        titles.addWidget(QLabel("X label:"), 0, 2)
        self._xlabel_entry = QLineEdit("Concentration (µM)")
        titles.addWidget(self._xlabel_entry, 0, 3)
        titles.addWidget(QLabel("Y label:"), 0, 4)
        self._ylabel_entry = QLineEdit("OD 600nm")
        titles.addWidget(self._ylabel_entry, 0, 5)

        btns = QHBoxLayout()
        btns.addStretch(1)
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self._show_step("select"))
        btns.addWidget(back_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        btns.addWidget(apply_btn)
        layout.addLayout(btns)

        self._load_from_dataset()
        self._render_preview_table()
        self._on_auto_ticks_toggle()

    def _selected_rows(self, lb: QListWidget) -> List[int]:
        return [int(idx.row()) for idx in lb.selectedIndexes()]

    def _selected_columns(self) -> List[str]:
        idxs = [int(idx.row()) for idx in self._cols_list.selectedIndexes()]
        cols = [str(c) for c in self._df.columns]
        return [cols[i] for i in idxs if 0 <= i < len(cols)]

    def _load_from_dataset(self) -> None:
        cfg = getattr(self._dataset, "wizard_mic_config", None)
        if cfg is None:
            self._populate_row_and_col_lists()
            return

        try:
            self._use_header_cb.setChecked(bool(cfg.use_first_row_as_header))
        except Exception:
            pass
        try:
            self._ticks_entry.setText(",".join(cfg.tick_labels or []))
        except Exception:
            pass
        try:
            self._auto_ticks_cb.setChecked(bool(getattr(cfg, "auto_tick_labels_power2", True)))
        except Exception:
            pass
        self._plot_type.setCurrentText(cfg.plot_type or "bar")
        self._control_style.setCurrentText(cfg.control_style or "bars")
        self._title_entry.setText(cfg.title or "MIC")
        self._xlabel_entry.setText(cfg.x_label or "Concentration")
        self._ylabel_entry.setText(cfg.y_label or "OD 600nm")

        self._populate_row_and_col_lists()
        try:
            for i in cfg.sample_rows:
                self._sample_rows.item(int(i)).setSelected(True)
        except Exception:
            pass
        try:
            for i in cfg.control_rows:
                self._control_rows.item(int(i)).setSelected(True)
        except Exception:
            pass
        try:
            cols = [str(c) for c in self._df.columns]
            want = set([str(c) for c in (cfg.concentration_columns or [])])
            for idx, c in enumerate(cols):
                if c in want:
                    self._cols_list.item(idx).setSelected(True)
        except Exception:
            pass

    def _populate_row_and_col_lists(self) -> None:
        self._sample_rows.clear()
        self._control_rows.clear()
        try:
            n = int(self._df.shape[0])
        except Exception:
            n = 0
        for i in range(n):
            self._sample_rows.addItem(QListWidgetItem(f"Row {i+1}"))
            self._control_rows.addItem(QListWidgetItem(f"Row {i+1}"))

        self._cols_list.clear()
        for c in [str(c) for c in self._df.columns]:
            self._cols_list.addItem(QListWidgetItem(c))

    def _render_preview_table(self) -> None:
        table = self._prev_table
        table.clear()
        if self._df is None or self._df.empty:
            table.setRowCount(0)
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels(["(empty)"])
            return
        cols, rows = preview_dataframe(self._df, max_rows=12)
        table.setColumnCount(len(cols))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(cols)
        for r_idx, row in enumerate(rows):
            for c_idx, val in enumerate(row):
                table.setItem(r_idx, c_idx, QTableWidgetItem(str(val)))
        table.resizeColumnsToContents()

    def _reload_df(self) -> None:
        try:
            use_header = bool(self._use_header_cb.isChecked())
        except Exception:
            use_header = True

        try:
            if use_header:
                self._df = getattr(self._dataset, "df_header0", None)
            else:
                self._df = getattr(self._dataset, "df_header_none", None)
        except Exception:
            self._df = None

        if self._df is None:
            try:
                header = 0 if use_header else None
                self._df = read_plate_file(self._dataset.path, sheet_name=self._dataset.sheet_name, header_row=header)
            except Exception:
                msg = traceback.format_exc()
                self._dialogs.error("Run Analysis", "Failed to reload data.\n\n" + msg)
                return

        self._render_preview_table()
        self._populate_row_and_col_lists()

    def _on_auto_ticks_toggle(self) -> None:
        if bool(self._auto_ticks_cb.isChecked()):
            self._ticks_entry.setEnabled(False)
        else:
            self._ticks_entry.setEnabled(True)
        self._update_auto_ticks()

    def _update_auto_ticks(self) -> None:
        if not bool(self._auto_ticks_cb.isChecked()):
            return
        try:
            n = len(self._selected_columns())
        except Exception:
            n = 0
        if n <= 0:
            self._ticks_entry.setText("")
            return
        if n == 1:
            labels = ["0"]
        else:
            labels = [str(int(2 ** (n - 2 - i))) for i in range(n - 1)] + ["0"]
        self._ticks_entry.setText(",".join(labels))

    def _apply(self) -> None:
        sample_rows = self._selected_rows(self._sample_rows)
        control_rows = self._selected_rows(self._control_rows)
        conc_cols = self._selected_columns()

        try:
            cfg, result, sample_nan = build_mic_wizard_config_and_result(
                self._df,
                use_first_row_as_header=bool(self._use_header_cb.isChecked()),
                sample_rows=sample_rows,
                control_rows=control_rows,
                concentration_columns=conc_cols,
                tick_text=str(self._ticks_entry.text() or ""),
                auto_tick_labels_power2=bool(self._auto_ticks_cb.isChecked()),
                title=str(self._title_entry.text() or "MIC"),
                x_label=str(self._xlabel_entry.text() or "Concentration"),
                y_label=str(self._ylabel_entry.text() or "OD 600nm"),
                plot_type=str(self._plot_type.currentText() or "bar"),
                control_style=str(self._control_style.currentText() or "bars"),
                prev_cfg=self._dataset.wizard_mic_config,
            )
        except ValueError as exc:
            self._dialogs.error("MIC", str(exc))
            return

        if sample_nan > 0.35:
            self._dialogs.warn("MIC", f"Many selected sample cells are non-numeric (NaN ratio: {sample_nan:.0%}).")

        self._on_apply(cfg, result)
        self.accept()


class PlateReaderPlotEditorDialog(QDialog):
    def __init__(self, parent: QWidget, *, dataset: PlateReaderDataset, on_apply) -> None:
        super().__init__(parent)
        self._dataset = dataset
        self._on_apply = on_apply

        self.setWindowTitle("Plate Reader — Edit Plot")
        self.resize(520, 620)

        cfg = self._dataset.wizard_mic_config
        if cfg is None:
            raise RuntimeError("No wizard config to edit")

        layout = QVBoxLayout(self)
        body = QWidget()
        form = QFormLayout(body)

        self._title_txt = QLineEdit(str(getattr(cfg, "title", "MIC")))
        self._xlabel_txt = QLineEdit(str(getattr(cfg, "x_label", "Concentration")))
        self._ylabel_txt = QLineEdit(str(getattr(cfg, "y_label", "OD 600nm")))
        form.addRow("Title", self._title_txt)
        form.addRow("X label", self._xlabel_txt)
        form.addRow("Y label", self._ylabel_txt)

        self._invert_x = QCheckBox("Invert X axis")
        self._invert_x.setChecked(bool(getattr(cfg, "invert_x", False)))
        form.addRow(self._invert_x)

        self._sample_color = QLineEdit(str(getattr(cfg, "sample_color", "#1f77b4")))
        self._control_color = QLineEdit(str(getattr(cfg, "control_color", "#ff7f0e")))
        sample_btn = QPushButton("Sample color…")
        sample_btn.clicked.connect(self._pick_sample_color)
        control_btn = QPushButton("Control color…")
        control_btn.clicked.connect(self._pick_control_color)
        sample_row = QHBoxLayout()
        sample_row.addWidget(sample_btn)
        sample_row.addWidget(self._sample_color)
        control_row = QHBoxLayout()
        control_row.addWidget(control_btn)
        control_row.addWidget(self._control_color)
        form.addRow("Sample", sample_row)
        form.addRow("Control", control_row)

        self._line_w = QLineEdit(str(getattr(cfg, "line_width", 1.6)))
        self._marker_s = QLineEdit(str(getattr(cfg, "marker_size", 6.0)))
        self._bar_w = QLineEdit(str(getattr(cfg, "bar_width", 0.65)))
        self._cap = QLineEdit(str(getattr(cfg, "capsize", 3.0)))
        self._e_lw = QLineEdit(str(getattr(cfg, "errorbar_linewidth", 1.0)))
        form.addRow("Line width", self._line_w)
        form.addRow("Marker size", self._marker_s)
        form.addRow("Bar width", self._bar_w)
        form.addRow("Errorbar cap size", self._cap)
        form.addRow("Errorbar line width", self._e_lw)

        self._title_fs = QLineEdit(str(getattr(cfg, "title_fontsize", 12)))
        self._label_fs = QLineEdit(str(getattr(cfg, "label_fontsize", 10)))
        self._tick_fs = QLineEdit(str(getattr(cfg, "tick_fontsize", 9)))
        self._grid_on = QCheckBox("Grid")
        self._grid_on.setChecked(bool(getattr(cfg, "grid_on", False)))
        self._legend_on = QCheckBox("Legend")
        self._legend_on.setChecked(bool(getattr(cfg, "legend_on", True)))
        form.addRow("Title font size", self._title_fs)
        form.addRow("Label font size", self._label_fs)
        form.addRow("Tick font size", self._tick_fs)
        form.addRow(self._grid_on)
        form.addRow(self._legend_on)

        self._x_min = QLineEdit("" if getattr(cfg, "x_min", None) is None else str(getattr(cfg, "x_min")))
        self._x_max = QLineEdit("" if getattr(cfg, "x_max", None) is None else str(getattr(cfg, "x_max")))
        self._y_min = QLineEdit("" if getattr(cfg, "y_min", None) is None else str(getattr(cfg, "y_min")))
        self._y_max = QLineEdit("" if getattr(cfg, "y_max", None) is None else str(getattr(cfg, "y_max")))
        limits = QHBoxLayout()
        limits.addWidget(QLabel("X min"))
        limits.addWidget(self._x_min)
        limits.addWidget(QLabel("X max"))
        limits.addWidget(self._x_max)
        limits.addWidget(QLabel("Y min"))
        limits.addWidget(self._y_min)
        limits.addWidget(QLabel("Y max"))
        limits.addWidget(self._y_max)
        form.addRow("Axis limits", limits)

        layout.addWidget(body, 1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        btns.addWidget(apply_btn)
        layout.addLayout(btns)

    def _parse_opt_float(self, s: str) -> Optional[float]:
        s = (s or "").strip()
        if not s:
            return None
        return float(s)

    def _apply(self) -> None:
        cfg = self._dataset.wizard_mic_config
        if cfg is None:
            return

        try:
            cfg.title = str(self._title_txt.text() or "")
            cfg.x_label = str(self._xlabel_txt.text() or "")
            cfg.y_label = str(self._ylabel_txt.text() or "")
            cfg.invert_x = bool(self._invert_x.isChecked())
            cfg.sample_color = str(self._sample_color.text() or "#1f77b4")
            cfg.control_color = str(self._control_color.text() or "#ff7f0e")

            cfg.line_width = float(self._line_w.text())
            cfg.marker_size = float(self._marker_s.text())
            cfg.bar_width = float(self._bar_w.text())
            cfg.capsize = float(self._cap.text())
            cfg.errorbar_linewidth = float(self._e_lw.text())

            cfg.title_fontsize = int(float(self._title_fs.text()))
            cfg.label_fontsize = int(float(self._label_fs.text()))
            cfg.tick_fontsize = int(float(self._tick_fs.text()))
            cfg.grid_on = bool(self._grid_on.isChecked())
            cfg.legend_on = bool(self._legend_on.isChecked())

            cfg.x_min = self._parse_opt_float(self._x_min.text())
            cfg.x_max = self._parse_opt_float(self._x_max.text())
            cfg.y_min = self._parse_opt_float(self._y_min.text())
            cfg.y_max = self._parse_opt_float(self._y_max.text())
        except Exception as exc:
            QMessageBox.critical(self, "Edit Plot", f"Invalid value: {exc}")
            return

        try:
            self._on_apply()
        except Exception:
            pass
        self.accept()

    def _pick_sample_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._sample_color.text()), self, "Sample color")
        if c.isValid():
            self._sample_color.setText(c.name())

    def _pick_control_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._control_color.text()), self, "Control color")
        if c.isValid():
            self._control_color.setText(c.name())


class PlateReaderTab(QWidget):
    def __init__(self, status: StatusService, dialogs: DialogService, worker_runner=None, adapter: PlateReaderAdapter | None = None) -> None:
        super().__init__()
        self.status = status
        self.dialogs = dialogs
        self.worker_runner = worker_runner or run_in_worker
        self.adapter = adapter or PlateReaderAdapter(status=status, dialogs=dialogs)
        self._last_workspace_path: Optional[str] = None

        self._dataset: Optional[PlateReaderDataset] = None
        self._df = None
        self._preview_win: Optional[PlateReaderPreviewDialog] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._build_ui())

        self._render_empty_plot()
        self._update_buttons()
        self._refresh_workspace_list()

    def _build_ui(self) -> QWidget:
        root = QSplitter(Qt.Horizontal)
        root.setChildrenCollapsible(False)
        self._root_splitter = root

        left = self._build_workspace_panel()
        right = self._build_plot_panel()

        root.addWidget(left)
        root.addWidget(right)
        root.setStretchFactor(0, 1)
        root.setStretchFactor(1, 3)
        return root

    def reset_layout(self) -> None:
        try:
            if getattr(self, "_root_splitter", None) is not None:
                self._root_splitter.setSizes([360, 820])
        except Exception:
            pass

    def _build_workspace_panel(self) -> QWidget:
        ws = QGroupBox("Workspace")
        layout = QVBoxLayout(ws)

        btns = QHBoxLayout()
        add_btn = QPushButton("Add Files…")
        add_btn.clicked.connect(self._add_files)
        btns.addWidget(add_btn)

        load_btn = QPushButton("Load…")
        load_btn.clicked.connect(self._load_workspace)
        btns.addWidget(load_btn)

        save_btn = QPushButton("Save…")
        save_btn.clicked.connect(self._save_workspace)
        btns.addWidget(save_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected)
        btns.addWidget(remove_btn)

        rename_btn = QPushButton("Rename…")
        rename_btn.clicked.connect(self._rename_selected)
        btns.addWidget(rename_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_workspace)
        btns.addWidget(clear_btn)

        btns.addStretch(1)
        layout.addLayout(btns)

        self._ws_tree = QTreeWidget()
        self._ws_tree.setHeaderLabels(["Name", "Type", "Shape", "MIC"])
        self._ws_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._ws_tree.itemSelectionChanged.connect(self._on_tree_select)
        layout.addWidget(self._ws_tree, 1)

        return ws

    def _build_plot_panel(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)

        top = QHBoxLayout()
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._open_preview)
        top.addWidget(self._preview_btn)

        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._open_wizard)
        top.addWidget(self._run_btn)

        self._edit_plot_btn = QPushButton("Edit Plot…")
        self._edit_plot_btn.clicked.connect(self._open_plot_editor)
        top.addWidget(self._edit_plot_btn)

        self._active_file_label = QLabel("Active file: (none)")
        self._active_mic_label = QLabel("MIC: —")
        top.addWidget(self._active_file_label, 1)
        top.addWidget(self._active_mic_label)

        self._status_label = QLabel("Ready")
        top.addWidget(self._status_label)
        layout.addLayout(top)

        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self._fig = Figure(figsize=(9.0, 6.0), dpi=110)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._canvas = FigureCanvasQTAgg(self._fig)
        plot_layout.addWidget(self._canvas, 1)

        self._toolbar = NavigationToolbar(self._canvas, self)
        plot_layout.addWidget(self._toolbar)

        self._coord_label = QLabel("")
        plot_layout.addWidget(self._coord_label)

        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        layout.addWidget(plot, 1)

        return right

    def _on_mouse_move(self, event: Any) -> None:
        try:
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                self._coord_label.setText(f"x={event.xdata:.4g}  y={event.ydata:.4g}")
            else:
                self._coord_label.setText("")
        except Exception:
            self._coord_label.setText("")

    def _set_status(self, text: str) -> None:
        self._status_label.setText(str(text))
        self.status.set_status(str(text))

    def _update_buttons(self) -> None:
        has = self._df is not None and self._dataset is not None
        self._preview_btn.setEnabled(bool(has))
        self._run_btn.setEnabled(bool(has))
        can_edit = bool(
            self._dataset is not None
            and getattr(self._dataset, "wizard_mic_result", None) is not None
            and getattr(self._dataset, "wizard_mic_config", None) is not None
        )
        self._edit_plot_btn.setEnabled(bool(can_edit))

    def _refresh_workspace_list(self) -> None:
        tree = self._ws_tree
        tree.clear()

        dss = self.adapter.list_datasets()
        active_id = self.adapter.active_plate_reader_id
        for ds in dss:
            pid = str(getattr(ds, "id", ""))
            name = str(getattr(ds, "display_name", "") or getattr(ds, "name", "") or "")
            if not name:
                try:
                    name = str(ds.path.name)
                except Exception:
                    name = "(dataset)"

            suf = ""
            try:
                suf = str(ds.path.suffix).lower().lstrip(".")
            except Exception:
                suf = ""
            ftype = suf if suf else "file"

            shape = ""
            try:
                df0 = getattr(ds, "df_header0", None)
                if df0 is not None:
                    shape = f"{int(df0.shape[0])}×{int(df0.shape[1])}"
            except Exception:
                shape = ""

            mic = "✅" if getattr(ds, "wizard_mic_result", None) is not None else "❌"
            item = QTreeWidgetItem([name, ftype, shape, mic])
            item.setData(0, Qt.UserRole, pid)
            tree.addTopLevelItem(item)

            if active_id and str(active_id) == pid:
                tree.setCurrentItem(item)

    def _on_tree_select(self) -> None:
        items = self._ws_tree.selectedItems()
        if not items:
            return
        ds_id = str(items[0].data(0, Qt.UserRole) or "")
        self._set_active_dataset(ds_id)

    def _set_active_dataset(self, dataset_id: str) -> None:
        ds = next((d for d in self.adapter.list_datasets() if str(getattr(d, "id", "")) == str(dataset_id)), None)
        if ds is None:
            return
        self.adapter.set_active_dataset_id(ds.id)
        self._dataset = ds
        try:
            cached = ds.current_df()
        except Exception:
            cached = None
        if cached is not None:
            self._df = cached
            self._set_active_labels()
            self._render_from_dataset()
            self._set_status("Active changed")
            self._update_buttons()
            return

        def _work(_h):
            return self.adapter.get_dataset_df(ds)

        def _done(df):
            self._df = df
            self._set_active_labels()
            self._render_from_dataset()
            self._set_status("Active changed")
            self._update_buttons()

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Loading Plate Reader data",
            group="plate_reader_active",
            cancel_previous=True,
        )

    def _set_active_labels(self) -> None:
        ds = self._dataset
        if ds is None:
            self._active_file_label.setText("Active file: (none)")
            self._active_mic_label.setText("MIC: —")
            return
        nm = str(getattr(ds, "display_name", "") or getattr(ds, "name", "") or "")
        if not nm:
            try:
                nm = str(ds.path.name)
            except Exception:
                nm = "(dataset)"
        self._active_file_label.setText(f"Active file: {nm}")
        has_mic = bool(getattr(ds, "wizard_mic_result", None) is not None)
        self._active_mic_label.setText("MIC: configured" if has_mic else "MIC: not configured")

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Plate Reader files",
            "",
            "Excel (*.xlsx *.xlsm *.xls);;CSV (*.csv);;All files (*.*)",
        )
        if not paths:
            return

        def _work(_h):
            return self.adapter.load_files(list(paths))

        def _done(_res):
            self._restore_from_adapter()
            self._set_status("Files added")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Loading Plate Reader files",
            group="plate_reader_files",
            cancel_previous=True,
        )

    def _remove_selected(self) -> None:
        items = self._ws_tree.selectedItems()
        if not items:
            return
        rm_id = str(items[0].data(0, Qt.UserRole) or "")

        dss = [d for d in self.adapter.list_datasets() if str(getattr(d, "id", "")) != rm_id]
        self.adapter.plate_reader_datasets = dss
        if str(self.adapter.active_plate_reader_id or "") == rm_id:
            self.adapter.active_plate_reader_id = (str(getattr(dss[-1], "id", "")) if dss else None)

        self._restore_from_adapter()
        self._set_status("Removed")

    def _clear_workspace(self) -> None:
        if not self.adapter.list_datasets():
            return
        if not self.dialogs.confirm("Plate Reader", "Clear Plate Reader workspace?"):
            return
        self.adapter.plate_reader_datasets = []
        self.adapter.active_plate_reader_id = None
        self._dataset = None
        self._df = None
        self._set_active_labels()
        self._render_empty_plot()
        self._refresh_workspace_list()
        self._update_buttons()
        self._set_status("Cleared")

    def _rename_selected(self) -> None:
        ds = self._dataset
        if ds is None:
            return
        current = str(getattr(ds, "display_name", "") or getattr(ds, "name", "") or "")
        new, ok = QInputDialog.getText(self, "Rename", "Dataset name:", text=current)
        if not ok:
            return
        new = str(new).strip()
        if not new:
            return
        try:
            ds.display_name = new
        except Exception:
            pass
        self._set_active_labels()
        self._refresh_workspace_list()
        self._render_from_dataset()
        self._set_status("Renamed")

    def _open_preview(self) -> None:
        if self._df is None or self._dataset is None:
            self.dialogs.info("Plate Reader", "Load or select a file first.")
            return
        if self._preview_win is None or not self._preview_win.isVisible():
            self._preview_win = PlateReaderPreviewDialog(self)
        try:
            suffix = str(getattr(self._dataset, "display_name", "") or self._dataset.path.name)
            self._preview_win.set_df(self._df, title_suffix=suffix)
        except Exception:
            pass
        self._preview_win.show()
        self._preview_win.raise_()
        self._preview_win.activateWindow()

    def _open_wizard(self) -> None:
        if self._dataset is None or self._df is None:
            self.dialogs.info("Plate Reader", "Load or select a file first.")
            return

        def on_apply(cfg, result) -> None:
            self._dataset.wizard_last_analysis = "mic"
            self._dataset.wizard_mic_config = cfg
            self._dataset.wizard_mic_result = result
            self._dataset.header_row = 0 if cfg.use_first_row_as_header else None
            self._df = self._dataset.current_df()
            self._render_from_dataset()
            self._set_status("Applied")
            self._set_active_labels()
            self._refresh_workspace_list()
            self._update_buttons()

        dlg = PlateReaderRunWizardDialog(self, dataset=self._dataset, df=self._df, on_apply=on_apply, dialogs=self.dialogs)
        dlg.exec()

    def _open_plot_editor(self) -> None:
        if self._dataset is None or self._dataset.wizard_mic_config is None or self._dataset.wizard_mic_result is None:
            return

        dlg = PlateReaderPlotEditorDialog(self, dataset=self._dataset, on_apply=self._on_plot_style_applied)
        dlg.exec()

    def _on_plot_style_applied(self) -> None:
        self._render_from_dataset()
        self._set_status("Style updated")
        self._update_buttons()

    def _render_empty_plot(self) -> None:
        try:
            self._ax.clear()
            self._ax.set_title("")
            self._ax.set_xlabel("")
            self._ax.set_ylabel("")
            self._canvas.draw_idle()
        except Exception:
            pass

    def _render_from_dataset(self) -> None:
        if self._dataset is None:
            self._render_empty_plot()
            return
        try:
            self._dataset.render_current_plot(self._ax)
            self._canvas.draw_idle()
        except Exception:
            self._render_empty_plot()

    def _restore_from_adapter(self) -> None:
        dss = self.adapter.list_datasets()
        if not dss:
            self._dataset = None
            self._df = None
            self._set_active_labels()
            self._refresh_workspace_list()
            self._update_buttons()
            return

        ds = self.adapter.get_active_dataset()
        if ds is None:
            ds = dss[-1]
            self.adapter.active_plate_reader_id = ds.id

        self._dataset = ds
        self._df = self.adapter.get_dataset_df(ds)
        self._set_active_labels()
        self._render_from_dataset()
        self._refresh_workspace_list()
        self._update_buttons()

    def _save_workspace(self) -> Optional[str]:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plate Reader Workspace",
            "",
            "Plate Reader Workspace (*.plate_reader.workspace.json);;JSON (*.json);;All files (*.*)",
        )
        if not path:
            return None
        ok = self.adapter.save_workspace(path)
        if ok:
            self._last_workspace_path = str(path)
            self._set_status("Workspace saved")
            return self._last_workspace_path
        return None

    def open_workspace(self) -> Optional[str]:
        return self._load_workspace()

    def save_workspace(self) -> Optional[str]:
        return self._save_workspace()

    def _load_workspace(self) -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Plate Reader Workspace",
            "",
            "Plate Reader Workspace (*.plate_reader.workspace.json);;JSON (*.json);;All files (*.*)",
        )
        if not path:
            return None

        has_any = bool(self.adapter.list_datasets())
        if has_any:
            if not self.dialogs.confirm("Plate Reader", "Load workspace and replace current Plate Reader files?"):
                return None

        def _work(_h):
            return self.adapter.load_workspace(path)

        def _done(res):
            loaded, active_id, failures = res
            if not loaded:
                msg = "No datasets were loaded."
                if failures:
                    msg += "\n\n" + "\n".join(failures[:12])
                self.dialogs.error("Plate Reader", msg)
                return
            self.adapter.plate_reader_datasets = loaded
            self.adapter.active_plate_reader_id = (active_id if active_id is not None else str(getattr(loaded[-1], "id", "")))
            self._last_workspace_path = str(path)
            self._restore_from_adapter()
            if failures:
                self.dialogs.warn("Plate Reader", "Workspace loaded with some issues:\n\n" + "\n".join(failures[:12]))
            self._set_status("Workspace loaded")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Loading Plate Reader workspace",
            group="plate_reader_workspace",
            cancel_previous=True,
        )

    def open_workspace_path(self, path: str) -> None:
        if not path:
            return None
        self._load_workspace_path(str(path))

    def _load_workspace_path(self, path: str) -> None:
        if not path:
            return

        def _work(_h):
            return self.adapter.load_workspace(path)

        def _done(res):
            loaded, active_id, failures = res
            if not loaded:
                msg = "No datasets were loaded."
                if failures:
                    msg += "\n\n" + "\n".join(failures[:12])
                self.dialogs.error("Plate Reader", msg)
                return
            self.adapter.plate_reader_datasets = loaded
            self.adapter.active_plate_reader_id = (active_id if active_id is not None else str(getattr(loaded[-1], "id", "")))
            self._last_workspace_path = str(path)
            self._restore_from_adapter()
            if failures:
                self.dialogs.warn("Plate Reader", "Workspace loaded with some issues:\n\n" + "\n".join(failures[:12]))
            self._set_status("Workspace loaded")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Loading Plate Reader workspace",
            group="plate_reader_workspace",
            cancel_previous=True,
        )
        return str(path)

    def get_last_workspace_path(self) -> Optional[str]:
        return self._last_workspace_path
