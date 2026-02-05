from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from lab_gui.data_studio_io import column_type_map, get_sheet_names, load_table
from lab_gui.data_studio_model import DataStudioDataset, DataStudioPlotDef
from lab_gui.data_studio_workspace_io import decode_workspace, encode_workspace
from qt_app.adapters import DataStudioAdapter
from qt_app.widgets import DataStudioExportDialog
from qt_app.services import DialogService, StatusService
from qt_app.services.worker import run_in_worker


PLOT_TYPES = [
    "Line",
    "Scatter",
    "Line + markers",
    "Bar (grouped)",
    "Bar (stacked)",
    "Area",
    "Histogram",
    "Box plot",
    "Violin plot",
    "Heatmap",
    "Bubble",
    "Step",
    "Stem",
    "Errorbar",
]


class DataStudioPreviewDialog(QDialog):
    def __init__(self, parent: QWidget, *, path: Path, dataset: DataStudioDataset, df) -> None:
        super().__init__(parent)
        self._path = path
        self._dataset = dataset
        self._df = df

        self.setWindowTitle(f"Preview — {path.name}")
        self.resize(1000, 700)

        layout = QVBoxLayout(self)
        info = QHBoxLayout()
        self._rows_label = QLabel("")
        self._cols_label = QLabel("")
        info.addWidget(self._rows_label)
        info.addWidget(self._cols_label)
        info.addStretch(1)

        self._sheet_cb = QComboBox()
        sheets = get_sheet_names(path)
        if sheets:
            info.addWidget(QLabel("Sheet"))
            self._sheet_cb.addItems([str(s) for s in sheets])
            self._sheet_cb.setCurrentText(str(dataset.sheet_name or sheets[0]))
            self._sheet_cb.currentIndexChanged.connect(self._reload_sheet)
            info.addWidget(self._sheet_cb)

        layout.addLayout(info)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self._table, 1)

        self._render_table(df)

    def _render_table(self, df) -> None:
        self._rows_label.setText(f"Rows: {len(df)}")
        self._cols_label.setText(f"Columns: {len(df.columns)}")
        self._table.clear()
        self._table.setColumnCount(len(df.columns))
        self._table.setRowCount(min(500, len(df)))
        self._table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r_idx, (_, row) in enumerate(df.head(500).iterrows()):
            for c_idx, c in enumerate(df.columns):
                self._table.setItem(r_idx, c_idx, QTableWidgetItem(str(row.get(c, ""))))
        self._table.resizeColumnsToContents()

    def _reload_sheet(self) -> None:
        sheet = str(self._sheet_cb.currentText())
        df = load_table(self._path, sheet_name=sheet, header_row=self._dataset.header_row)
        self._dataset.sheet_name = sheet
        self._df = df
        self._render_table(df)


class DataStudioTab(QWidget):
    def __init__(self, status: StatusService, dialogs: DialogService, worker_runner=None, adapter: DataStudioAdapter | None = None) -> None:
        super().__init__()
        self.status = status
        self.dialogs = dialogs
        self.worker_runner = worker_runner or run_in_worker
        self.adapter = adapter or DataStudioAdapter()
        self._ws = self.adapter.ws
        self._plotted_ids: set[str] = set()
        self._last_payload: Optional[Dict[str, Any]] = None
        self._overlay_refresh_timer: Optional[QTimer] = None
        self._restoring_ui = False
        self._last_workspace_path: Optional[str] = None

        self._x_display_to_col: Dict[str, str] = {}
        self._x_col_to_display: Dict[str, str] = {}
        self._y_display_to_col: Dict[str, str] = {}
        self._y_col_to_display: Dict[str, str] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._build_ui())

        self._refresh_workspace()

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
                self._root_splitter.setSizes([340, 980])
        except Exception:
            pass

    def _build_workspace_panel(self) -> QWidget:
        ws = QGroupBox("Workspace")
        layout = QVBoxLayout(ws)

        btns = QHBoxLayout()
        add_btn = QPushButton("Add Files…")
        add_btn.clicked.connect(self._add_files)
        btns.addWidget(add_btn)
        rm_btn = QPushButton("Remove Selected")
        rm_btn.clicked.connect(self._remove_selected)
        btns.addWidget(rm_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_workspace)
        btns.addWidget(clear_btn)
        save_btn = QPushButton("Save…")
        save_btn.clicked.connect(self._save_workspace)
        btns.addWidget(save_btn)
        load_btn = QPushButton("Load…")
        load_btn.clicked.connect(self._load_workspace)
        btns.addWidget(load_btn)
        btns.addStretch(1)
        layout.addLayout(btns)

        self._ws_tree = QTreeWidget()
        self._ws_tree.setHeaderLabels(["Active", "File"])
        self._ws_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._ws_tree.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self._ws_tree, 1)

        set_active = QPushButton("Set Active")
        set_active.clicked.connect(self._set_active_from_selection)
        layout.addWidget(set_active)

        preview = QPushButton("Preview Table")
        preview.clicked.connect(self._preview_data)
        layout.addWidget(preview)

        defs = QGroupBox("Plot Definitions")
        defs_layout = QVBoxLayout(defs)
        self._plot_tree = QTreeWidget()
        self._plot_tree.setHeaderLabels(["Active", "Plot"])
        self._plot_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._plot_tree.itemSelectionChanged.connect(self._on_plot_select)
        defs_layout.addWidget(self._plot_tree)
        defs_btns = QHBoxLayout()
        new_btn = QPushButton("New Plot")
        new_btn.clicked.connect(self._new_plot_def)
        defs_btns.addWidget(new_btn)
        rm_plot = QPushButton("Remove Plot")
        rm_plot.clicked.connect(self._remove_plot_def)
        defs_btns.addWidget(rm_plot)
        set_plot = QPushButton("Set Active")
        set_plot.clicked.connect(self._set_active_plot_from_selection)
        defs_btns.addWidget(set_plot)
        defs_btns.addStretch(1)
        defs_layout.addLayout(defs_btns)
        layout.addWidget(defs)

        overlay = QGroupBox("Overlay")
        overlay_layout = QVBoxLayout(overlay)
        self._overlay_tree = QTreeWidget()
        self._overlay_tree.setHeaderLabels(["Overlay", "File"])
        self._overlay_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._overlay_tree.itemClicked.connect(self._on_overlay_click)
        overlay_layout.addWidget(self._overlay_tree)
        ov_btn = QPushButton("Overlay Selected")
        ov_btn.clicked.connect(self._apply_overlay)
        overlay_layout.addWidget(ov_btn)
        clear_ov = QPushButton("Clear Overlay")
        clear_ov.clicked.connect(self._clear_overlay)
        overlay_layout.addWidget(clear_ov)
        layout.addWidget(overlay)

        return ws

    def _build_plot_panel(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)

        top = QHBoxLayout()
        self._status_label = QLabel("Ready")
        self._dirty_label = QLabel("")
        self._dirty_label.setStyleSheet("color:#b00020")
        top.addWidget(self._status_label)
        top.addWidget(self._dirty_label)
        top.addStretch(1)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_plot)
        top.addWidget(apply_btn)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_plot_builder)
        top.addWidget(reset_btn)
        export_btn = QPushButton("Export…")
        export_btn.clicked.connect(self._export_plot)
        top.addWidget(export_btn)
        layout.addLayout(top)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("X"))
        self._x_cb = QComboBox()
        self._x_cb.currentIndexChanged.connect(self._on_x_changed)
        controls.addWidget(self._x_cb)

        controls.addWidget(QLabel("Plot"))
        self._plot_cb = QComboBox()
        self._plot_cb.addItems(PLOT_TYPES)
        self._plot_cb.currentIndexChanged.connect(self._on_plot_type_changed)
        controls.addWidget(self._plot_cb)

        y_btn = QPushButton("Y columns…")
        y_btn.clicked.connect(self._open_y_selector)
        controls.addWidget(y_btn)

        data_btn = QPushButton("Data options…")
        data_btn.clicked.connect(self._open_data_options)
        controls.addWidget(data_btn)

        adv_btn = QPushButton("Advanced…")
        adv_btn.clicked.connect(self._toggle_advanced_panel)
        controls.addWidget(adv_btn)

        controls.addWidget(QLabel("Offset"))
        self._overlay_mode_cb = QComboBox()
        self._overlay_mode_cb.addItems(["Normal", "Offset Y", "Offset X"])
        self._overlay_mode_cb.currentIndexChanged.connect(self._schedule_overlay_refresh)
        controls.addWidget(self._overlay_mode_cb)
        self._overlay_offset_edit = QLineEdit("0.0")
        self._overlay_offset_edit.textChanged.connect(self._schedule_overlay_refresh)
        controls.addWidget(self._overlay_offset_edit)

        self._y_summary_label = QLabel("Y: (none)")
        controls.addWidget(self._y_summary_label, 1)
        layout.addLayout(controls)

        self._banner_label = QLabel("")
        self._banner_label.setStyleSheet("color:#0b5394")
        layout.addWidget(self._banner_label)

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)

        self._builder_panel = QGroupBox("Advanced options")
        builder_layout = QVBoxLayout(self._builder_panel)
        builder_layout.addWidget(QLabel("Group / Series column"))
        self._group_cb = QComboBox()
        self._group_cb.currentIndexChanged.connect(lambda _i: self._on_group_changed())
        builder_layout.addWidget(self._group_cb)

        self._extra = QWidget()
        self._extra_layout = QVBoxLayout(self._extra)
        builder_layout.addWidget(self._extra)

        self._builder_panel.setVisible(False)
        body.addWidget(self._builder_panel)

        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(10.5, 7.5), dpi=110)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._canvas = FigureCanvasQTAgg(self._fig)
        plot_layout.addWidget(self._canvas, 1)
        self._toolbar = NavigationToolbar(self._canvas, self)
        plot_layout.addWidget(self._toolbar)
        self._coord_label = QLabel("")
        plot_layout.addWidget(self._coord_label)
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        body.addWidget(plot)

        layout.addWidget(body, 1)

        self._init_data_options()
        return right

    def _init_data_options(self) -> None:
        self._drop_na_var = True
        self._decimal_var = False
        self._autocast_var = True
        self._norm_var = "None"
        self._bins_var = 20
        self._roll_var = 5
        self._size_var = "(None)"
        self._heat_row_var = "(None)"
        self._heat_col_var = "(None)"
        self._heat_val_var = "(None)"
        self._heat_agg_var = "mean"
        self._xerr_var = "(None)"
        self._yerr_var = "(None)"

    def _on_mouse_move(self, event: Any) -> None:
        try:
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                self._coord_label.setText(f"x={event.xdata:.4g}, y={event.ydata:.4g}")
            else:
                self._coord_label.setText("")
        except Exception:
            self._coord_label.setText("")

    def _mark_dirty(self) -> None:
        self._dirty_label.setText("● Unsaved changes")

    def _clear_dirty(self) -> None:
        self._dirty_label.setText("")

    def _active_plot_def(self) -> Optional[DataStudioPlotDef]:
        pid = self._ws.active_plot_id
        if not pid:
            return None
        return self._ws.plot_defs.get(pid)

    def _plot_def_name(self, pd: DataStudioPlotDef) -> str:
        ds = self._ws.datasets.get(pd.dataset_id)
        base = str(ds.display_name) if ds is not None else str(pd.dataset_id)
        return f"{base} · {pd.plot_type or 'Plot'}"

    def _toggle_advanced_panel(self) -> None:
        self._builder_panel.setVisible(not self._builder_panel.isVisible())

    def _open_data_options(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Data options")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("These settings control how data is parsed before plotting."))

        self._drop_na_cb = QCheckBox("Drop NaNs")
        self._drop_na_cb.setChecked(bool(self._drop_na_var))
        self._decimal_cb = QCheckBox("Comma → dot numeric")
        self._decimal_cb.setChecked(bool(self._decimal_var))
        self._autocast_cb = QCheckBox("Auto-cast numeric")
        self._autocast_cb.setChecked(bool(self._autocast_var))
        self._norm_cb = QComboBox()
        self._norm_cb.addItems(["None", "Min-Max", "Z-score"])
        self._norm_cb.setCurrentText(str(self._norm_var))

        layout.addWidget(self._drop_na_cb)
        layout.addWidget(self._decimal_cb)
        layout.addWidget(self._autocast_cb)
        layout.addWidget(QLabel("Normalize Y"))
        layout.addWidget(self._norm_cb)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(lambda: self._apply_data_options_and_close(dlg))
        layout.addWidget(close_btn, alignment=Qt.AlignRight)
        dlg.exec()

    def _apply_data_options_and_close(self, dlg: QDialog) -> None:
        self._drop_na_var = bool(self._drop_na_cb.isChecked())
        self._decimal_var = bool(self._decimal_cb.isChecked())
        self._autocast_var = bool(self._autocast_cb.isChecked())
        self._norm_var = str(self._norm_cb.currentText())
        self._mark_dirty()
        self._store_current_config()
        dlg.accept()

    def _open_y_selector(self) -> None:
        pd = self._active_plot_def()
        if pd is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Y columns")
        dlg.resize(420, 420)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Pick one or more Y columns. Changes apply when you press Apply."))
        filter_edit = QLineEdit()
        layout.addWidget(filter_edit)

        listbox = QListWidget()
        listbox.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(listbox, 1)

        def _refresh() -> None:
            items = self._available_y_items(filter_edit.text())
            listbox.clear()
            for d in items:
                listbox.addItem(QListWidgetItem(d))
            selected = set(pd.y_cols or [])
            for i in range(listbox.count()):
                disp = listbox.item(i).text()
                col = self._y_display_to_col.get(disp, disp)
                if col in selected:
                    listbox.item(i).setSelected(True)

        def _apply() -> None:
            selected: List[str] = []
            for it in listbox.selectedItems():
                disp = it.text()
                selected.append(self._y_display_to_col.get(disp, disp))
            pd.y_cols = list(selected)
            self._ws.plot_defs[pd.plot_id] = pd
            self._update_y_summary()
            self._mark_dirty()
            self._store_current_config()
            dlg.accept()

        filter_edit.textChanged.connect(_refresh)
        _refresh()

        btns = QHBoxLayout()
        btns.addStretch(1)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(_apply)
        btns.addWidget(apply_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)
        dlg.exec()

    def _schedule_overlay_refresh(self) -> None:
        if self._overlay_refresh_timer is not None:
            self._overlay_refresh_timer.stop()
        self._overlay_refresh_timer = QTimer(self)
        self._overlay_refresh_timer.setSingleShot(True)
        self._overlay_refresh_timer.timeout.connect(self._on_overlay_mode_changed)
        self._overlay_refresh_timer.start(180)

    def _on_overlay_mode_changed(self) -> None:
        pd = self._active_plot_def()
        if pd is None or not pd.y_cols:
            return
        try:
            self._plot()
            self._update_y_summary()
        except Exception:
            pass

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add files",
            "",
            "Data (*.csv *.tsv *.xlsx *.xls);;All files (*.*)",
        )
        if not paths:
            return
        for p in paths:
            path = Path(p)
            sid = str(uuid.uuid4())
            name = path.name
            if name in [d.display_name for d in self._ws.datasets.values()]:
                base = path.stem
                idx = 2
                while f"{base} ({idx}){path.suffix}" in [d.display_name for d in self._ws.datasets.values()]:
                    idx += 1
                name = f"{base} ({idx}){path.suffix}"
            self._ws.datasets[sid] = DataStudioDataset(dataset_id=sid, path=path, display_name=name)
            self._ws.order.append(sid)
            self.adapter.ensure_plot_def_for_dataset(sid, plot_type=PLOT_TYPES[0])
            if self._ws.active_id is None:
                self._ws.active_id = sid
            self._infer_schema_async(sid)
        self._refresh_workspace()
        self._status_label.setText(f"Loaded {len(paths)} file(s)")

    def _infer_schema_async(self, dataset_id: str) -> None:
        ds = self._ws.datasets.get(dataset_id)
        if ds is None:
            return

        def _work(_h):
            try:
                cols, schema_hash = self.adapter.infer_schema(ds, decimal_comma=self._decimal_var, auto_cast=self._autocast_var)
            except Exception:
                cols, schema_hash = {}, ""
            return cols, schema_hash

        def _done(res):
            cols, schema_hash = res
            d = self._ws.datasets.get(dataset_id)
            if d is None:
                return
            d.columns = cols
            d.schema_hash = schema_hash
            self._ws.datasets[dataset_id] = d
            if self._ws.active_id == dataset_id:
                self._populate_columns()
                self._restore_config_for_active()

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Inferring Data Studio schema",
            group=f"data_studio_schema:{dataset_id}",
            cancel_previous=True,
        )

    def _set_active_dataset(self, dataset_id: str) -> None:
        if not dataset_id:
            return
        self._ws.active_id = dataset_id
        self.adapter.ensure_plot_def_for_dataset(dataset_id, plot_type=PLOT_TYPES[0])
        cur_pd = self._active_plot_def()
        if cur_pd is None or cur_pd.dataset_id != dataset_id:
            for pid, pd in self._ws.plot_defs.items():
                if pd.dataset_id == dataset_id:
                    self._ws.active_plot_id = pid
                    break
        self._refresh_workspace()
        self._populate_columns()
        self._restore_config_for_active()
        self._clear_dirty()

    def _remove_selected(self) -> None:
        sid = self._selected_id(self._ws_tree)
        if not sid:
            return
        self._ws.datasets.pop(sid, None)
        if sid in self._ws.order:
            self._ws.order.remove(sid)
        if sid in self.adapter.df_cache:
            self.adapter.df_cache.pop(sid, None)
        self.adapter.remove_plot_defs_for_dataset(sid)
        if sid in self._plotted_ids:
            self._plotted_ids.discard(sid)
        if self._ws.active_id == sid:
            self._ws.active_id = self._ws.order[0] if self._ws.order else None
        self._refresh_workspace()

    def _clear_workspace(self) -> None:
        self.adapter.reset()
        self._ws = self.adapter.ws
        self._plotted_ids = set()
        self._refresh_workspace()

    def _save_workspace(self) -> Optional[str]:
        if not self._ws.datasets:
            self.dialogs.info("Data Studio", "No datasets to save.")
            return None
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data Studio Workspace",
            "",
            "Data Studio Workspace (*.data_studio.workspace.json);;JSON (*.json);;All files (*.*)",
        )
        if not path:
            return None
        payload = encode_workspace(self._ws)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.dialogs.error("Data Studio", f"Failed to save workspace:\n\n{exc}")
            return None
        self._last_workspace_path = str(path)
        self._save_ui_state(path)
        self._status_label.setText("Workspace saved")
        return self._last_workspace_path

    def _load_workspace(self) -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Data Studio Workspace",
            "",
            "Data Studio Workspace (*.data_studio.workspace.json);;JSON (*.json);;All files (*.*)",
        )
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            self.dialogs.error("Data Studio", f"Failed to load workspace:\n\n{exc}")
            return None
        if not isinstance(payload, dict):
            self.dialogs.error("Data Studio", "Workspace JSON must be an object.")
            return None

        ws, errors = decode_workspace(payload)
        self._ws = ws
        self.adapter.ws = ws
        self.adapter.df_cache = {}
        self._plotted_ids = set()
        self._last_workspace_path = str(path)
        self._refresh_workspace()
        for sid in list(self._ws.order):
            ds = self._ws.datasets.get(sid)
            if ds is not None and not ds.columns:
                self._infer_schema_async(sid)
        self._load_ui_state(path)
        if errors:
            self.dialogs.warn("Data Studio", "Workspace loaded with some issues:\n\n" + "\n".join(errors[:10]))
        self._status_label.setText("Workspace loaded")
        return str(path)

    def _set_active_from_selection(self) -> None:
        if self._ws.active_id:
            self._store_current_config()
        sid = self._selected_id(self._ws_tree)
        if not sid:
            return
        self._set_active_dataset(sid)
        self._auto_plot_for_selection()

    def _on_select(self) -> None:
        if self._ws.active_id:
            self._store_current_config()
        sid = self._selected_id(self._ws_tree)
        if sid:
            self._set_active_dataset(sid)
            self._auto_plot_for_selection()

    def _on_plot_select(self) -> None:
        pid = self._selected_id(self._plot_tree)
        if not pid:
            return
        if pid in self._ws.plot_defs:
            self._ws.active_plot_id = pid
            pd = self._ws.plot_defs[pid]
            if pd.dataset_id:
                self._ws.active_id = pd.dataset_id
            self._refresh_workspace()
            self._populate_columns()
            self._restore_config_for_active()
            self._auto_plot_for_selection()

    def _set_active_plot_from_selection(self) -> None:
        self._on_plot_select()

    def _new_plot_def(self) -> None:
        ds_id = self._ws.active_id
        if not ds_id:
            return
        pid = str(uuid.uuid4())
        pd = DataStudioPlotDef(plot_id=pid, dataset_id=ds_id, plot_type=str(self._plot_cb.currentText()))
        self._ws.plot_defs[pid] = pd
        self._ws.active_plot_id = pid
        self._refresh_workspace()
        self._restore_config_for_active()
        self._mark_dirty()

    def _remove_plot_def(self) -> None:
        pid = self._selected_id(self._plot_tree)
        if not pid:
            return
        self._ws.plot_defs.pop(pid, None)
        if self._ws.active_plot_id == pid:
            self._ws.active_plot_id = None
        self._refresh_workspace()

    def _preview_data(self) -> None:
        sid = self._ws.active_id
        if not sid:
            self.dialogs.info("Preview", "No active dataset.")
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        def _work(_h):
            return self.adapter.load_df(ds, decimal_comma=self._decimal_var, auto_cast=self._autocast_var)

        def _done(df):
            dlg = DataStudioPreviewDialog(self, path=ds.path, dataset=ds, df=df)
            dlg.exec()

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Loading preview",
            group="data_studio_preview",
            cancel_previous=True,
        )

    def _apply_overlay(self) -> None:
        overlay_ids = [
            str(self._overlay_tree.topLevelItem(i).data(0, Qt.UserRole))
            for i in range(self._overlay_tree.topLevelItemCount())
            if self._overlay_tree.topLevelItem(i).text(0) == "✔"
        ]
        if not overlay_ids:
            self._ws.overlay_ids = []
            self._status_label.setText("Overlay: 0 dataset(s)")
            return

        missing = [sid for sid in overlay_ids if sid not in self._plotted_ids]
        if missing:
            self.dialogs.info("Overlay", "Plot each dataset first before overlay.")
            self._ws.overlay_ids = []
            self._refresh_workspace()
            return

        cfgs = [pd for pd in self._ws.plot_defs.values() if pd.dataset_id in overlay_ids]
        if len(cfgs) != len(overlay_ids):
            self.dialogs.info("Overlay", "Overlay requires plotting each dataset first.")
            self._ws.overlay_ids = []
            self._refresh_workspace()
            return

        plot_type = str(cfgs[0].plot_type)
        if any(str(c.plot_type) != plot_type for c in cfgs if c is not None):
            self.dialogs.info("Overlay", "Overlay requires the same plot type across datasets.")
            self._ws.overlay_ids = []
            self._refresh_workspace()
            return

        self._ws.overlay_ids = overlay_ids
        if self._ws.active_id not in self._ws.overlay_ids:
            self._ws.active_id = self._ws.overlay_ids[0]
        self._status_label.setText(f"Overlay: {len(self._ws.overlay_ids)} dataset(s)")
        self._plot()

    def _clear_overlay(self) -> None:
        self._ws.overlay_ids = []
        self._refresh_workspace()
        self._auto_plot_for_selection()

    def _on_overlay_click(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        cur = item.text(0)
        item.setText(0, "" if cur == "✔" else "✔")

    def _refresh_workspace(self) -> None:
        self._ws_tree.clear()
        self._overlay_tree.clear()
        self._plot_tree.clear()

        for sid in self._ws.order:
            ds = self._ws.datasets.get(sid)
            if ds is None:
                continue
            active = "●" if sid == self._ws.active_id else ""
            ws_item = QTreeWidgetItem([active, ds.display_name])
            ws_item.setData(0, Qt.UserRole, str(sid))
            self._ws_tree.addTopLevelItem(ws_item)

            ov = "✔" if sid in self._ws.overlay_ids else ""
            ov_item = QTreeWidgetItem([ov, ds.display_name])
            ov_item.setData(0, Qt.UserRole, str(sid))
            self._overlay_tree.addTopLevelItem(ov_item)

        for pid, pd in self._ws.plot_defs.items():
            name = f"{self._plot_def_name(pd)}"
            active = "●" if pid == self._ws.active_plot_id else ""
            plot_item = QTreeWidgetItem([active, name])
            plot_item.setData(0, Qt.UserRole, str(pid))
            self._plot_tree.addTopLevelItem(plot_item)

        self._populate_columns()
        self._restore_config_for_active()

    def _selected_id(self, tree: QTreeWidget) -> Optional[str]:
        items = tree.selectedItems()
        if not items:
            return None
        return str(items[0].data(0, Qt.UserRole) or "")

    def _populate_columns(self) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        self._restoring_ui = True
        cols_map = dict(ds.columns or {})
        if not cols_map:
            self._x_display_to_col = {"(Index)": "(Index)"}
            self._x_col_to_display = {"(Index)": "(Index)"}
            self._x_cb.clear()
            self._x_cb.addItems(["(Index)"])
            self._x_cb.setCurrentText("(Index)")

            self._y_display_to_col = {}
            self._y_col_to_display = {}
            self._refresh_y_list()

            self._group_cb.clear()
            self._group_cb.addItems(["(None)"])
            self._group_cb.setCurrentText("(None)")

            self._restoring_ui = False
            self._infer_schema_async(sid)
            return
        cols = [str(c) for c in cols_map.keys()]

        def _disp(name: str, dtype: str) -> str:
            return f"{name} ({dtype})" if dtype else str(name)

        self._x_display_to_col = {"(Index)": "(Index)"}
        self._x_col_to_display = {"(Index)": "(Index)"}
        x_values = ["(Index)"]
        for c in cols:
            d = _disp(c, cols_map.get(c, ""))
            x_values.append(d)
            self._x_display_to_col[d] = c
            self._x_col_to_display[c] = d
        self._x_cb.clear()
        self._x_cb.addItems(x_values)
        if self._x_cb.currentText() not in x_values:
            self._x_cb.setCurrentText("(Index)")

        self._y_display_to_col = {}
        self._y_col_to_display = {}
        for c in cols:
            d = _disp(c, cols_map.get(c, ""))
            self._y_display_to_col[d] = c
            self._y_col_to_display[c] = d
        self._refresh_y_list()

        group_vals = ["(None)"] + cols
        self._group_cb.clear()
        self._group_cb.addItems(group_vals)
        if self._group_cb.currentText() not in group_vals:
            self._group_cb.setCurrentText("(None)")

        self._size_var = "(None)"
        self._xerr_var = "(None)"
        self._yerr_var = "(None)"
        self._heat_row_var = "(None)"
        self._heat_col_var = "(None)"
        self._heat_val_var = "(None)"
        self._restoring_ui = False

    def _restore_config_for_active(self) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        pd = self._active_plot_def()
        if pd is None:
            return
        self._restoring_ui = True

        ds = self._ws.datasets.get(sid)
        cols_set = set((ds.columns or {}).keys()) if ds is not None else set()
        needs_fix = False
        if pd.x_col and pd.x_col not in cols_set:
            pd.x_col = None
            needs_fix = True
        if pd.y_cols:
            filtered = [y for y in pd.y_cols if y in cols_set]
            if len(filtered) != len(pd.y_cols):
                pd.y_cols = filtered
                needs_fix = True
        if needs_fix:
            x_def, y_def = self._pick_default_axes(sid)
            if not pd.x_col:
                pd.x_col = x_def
            if not pd.y_cols:
                pd.y_cols = list(y_def)
            self._banner_label.setText(
                f"Columns changed; auto-selected X={pd.x_col or 'Index'}, Y={', '.join(pd.y_cols or [])}"
            )

        if not pd.x_col or not pd.y_cols:
            x_def, y_def = self._pick_default_axes(sid)
            if not pd.x_col:
                pd.x_col = x_def
            if not pd.y_cols:
                pd.y_cols = list(y_def)

        try:
            disp = self._x_col_to_display.get(str(pd.x_col or ""), "(Index)") if pd.x_col else "(Index)"
            self._x_cb.setCurrentText(str(disp))
        except Exception:
            pass
        try:
            self._plot_cb.setCurrentText(str(pd.plot_type or PLOT_TYPES[0]))
        except Exception:
            pass
        self._toggle_extra_fields()

        opts = dict(pd.options or {})
        self._group_cb.setCurrentText(str(opts.get("group_col") or "(None)"))
        self._size_var = str(opts.get("size_col") or "(None)")
        self._xerr_var = str(opts.get("x_err_col") or "(None)")
        self._yerr_var = str(opts.get("y_err_col") or "(None)")
        self._heat_row_var = str(opts.get("heatmap_row") or "(None)")
        self._heat_col_var = str(opts.get("heatmap_col") or "(None)")
        self._heat_val_var = str(opts.get("heatmap_val") or "(None)")
        self._heat_agg_var = str(opts.get("heatmap_agg") or "mean")
        self._bins_var = int(opts.get("hist_bins") or 20)
        self._roll_var = int(opts.get("rolling_window") or 5)

        self._update_y_summary()

        try:
            self._drop_na_var = bool(opts.get("drop_na", True))
            self._decimal_var = bool(opts.get("decimal_comma", False))
            self._autocast_var = bool(opts.get("auto_cast", True))
            self._norm_var = str(opts.get("normalize") or "None")
        except Exception:
            pass
        self._restoring_ui = False

    def _auto_plot_for_selection(self) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        if self._ws.overlay_ids:
            try:
                self._plot()
            except Exception:
                return
            return
        if sid in self._plotted_ids:
            try:
                self._plot()
            except Exception:
                return

    def _pick_default_axes(self, dataset_id: str) -> Tuple[Optional[str], List[str]]:
        ds = self._ws.datasets.get(dataset_id)
        if ds is None:
            return None, []
        cols_map = dict(ds.columns or {})
        cols = list(cols_map.keys())
        low_cols = [c.lower() for c in cols]

        time_keys = ("time", "sec", "s", "min", "hour", "date", "datetime")
        idx_keys = ("index", "scan", "cycle", "frame")

        def _is_time(name: str) -> bool:
            return any(k in name for k in time_keys)

        def _is_idx(name: str) -> bool:
            return any(k in name for k in idx_keys)

        x_col = None
        for c, lc in zip(cols, low_cols):
            if _is_time(lc):
                x_col = c
                break
        if x_col is None:
            for c, lc in zip(cols, low_cols):
                if _is_idx(lc):
                    x_col = c
                    break

        numeric_cols: List[str] = []
        for name, dtype in cols_map.items():
            if "int" in str(dtype) or "float" in str(dtype) or "double" in str(dtype):
                numeric_cols.append(str(name))

        if x_col is None:
            x_col = numeric_cols[0] if numeric_cols else None

        y_cols: List[str] = []
        for c in numeric_cols:
            if c != x_col:
                y_cols.append(c)
                break

        if not y_cols and numeric_cols:
            y_cols = [numeric_cols[0]]

        pref = self._ws.preferred_axes_by_dataset.get(dataset_id)
        if pref:
            px, py = pref
            if px in cols:
                x_col = px
            if py and all(y in cols for y in py):
                y_cols = list(py)

        return x_col, y_cols

    def _toggle_extra_fields(self) -> None:
        for i in reversed(range(self._extra_layout.count())):
            w = self._extra_layout.itemAt(i).widget()
            if w is not None:
                w.deleteLater()

        kind = str(self._plot_cb.currentText())

        def _combo(title: str, values: List[str], current: str) -> QComboBox:
            self._extra_layout.addWidget(QLabel(title))
            cb = QComboBox()
            cb.addItems(values)
            cb.setCurrentText(current)
            self._extra_layout.addWidget(cb)
            return cb

        cols = ["(None)"]
        sid = self._ws.active_id
        if sid and sid in self._ws.datasets:
            df = self.adapter.load_df(self._ws.datasets[sid], decimal_comma=self._decimal_var, auto_cast=self._autocast_var)
            cols += [str(c) for c in df.columns]

        if kind == "Bubble":
            self._size_cb = _combo("Size column", cols, self._size_var)
            self._size_cb.currentTextChanged.connect(lambda v: self._set_extra_var("size", v))
        if kind == "Heatmap":
            self._heat_row_cb = _combo("Row", cols, self._heat_row_var)
            self._heat_row_cb.currentTextChanged.connect(lambda v: self._set_extra_var("heat_row", v))
            self._heat_col_cb = _combo("Col", cols, self._heat_col_var)
            self._heat_col_cb.currentTextChanged.connect(lambda v: self._set_extra_var("heat_col", v))
            self._heat_val_cb = _combo("Value", cols, self._heat_val_var)
            self._heat_val_cb.currentTextChanged.connect(lambda v: self._set_extra_var("heat_val", v))
            self._heat_agg_cb = _combo("Agg", ["mean", "sum", "median"], self._heat_agg_var)
            self._heat_agg_cb.currentTextChanged.connect(lambda v: self._set_extra_var("heat_agg", v))
        if kind == "Histogram":
            self._extra_layout.addWidget(QLabel("Bins"))
            self._bins_spin = QSpinBox()
            self._bins_spin.setRange(5, 200)
            self._bins_spin.setValue(int(self._bins_var))
            self._bins_spin.valueChanged.connect(lambda v: self._set_extra_var("bins", v))
            self._extra_layout.addWidget(self._bins_spin)
        if kind == "Rolling mean":
            self._extra_layout.addWidget(QLabel("Window"))
            self._roll_spin = QSpinBox()
            self._roll_spin.setRange(2, 200)
            self._roll_spin.setValue(int(self._roll_var))
            self._roll_spin.valueChanged.connect(lambda v: self._set_extra_var("roll", v))
            self._extra_layout.addWidget(self._roll_spin)
        if kind == "Errorbar":
            self._yerr_cb = _combo("Y error", cols, self._yerr_var)
            self._yerr_cb.currentTextChanged.connect(lambda v: self._set_extra_var("yerr", v))
            self._xerr_cb = _combo("X error", cols, self._xerr_var)
            self._xerr_cb.currentTextChanged.connect(lambda v: self._set_extra_var("xerr", v))

    def _set_extra_var(self, key: str, value: Any) -> None:
        if key == "size":
            self._size_var = str(value)
        elif key == "heat_row":
            self._heat_row_var = str(value)
        elif key == "heat_col":
            self._heat_col_var = str(value)
        elif key == "heat_val":
            self._heat_val_var = str(value)
        elif key == "heat_agg":
            self._heat_agg_var = str(value)
        elif key == "bins":
            self._bins_var = int(value)
        elif key == "roll":
            self._roll_var = int(value)
        elif key == "yerr":
            self._yerr_var = str(value)
        elif key == "xerr":
            self._xerr_var = str(value)
        self._mark_dirty()

    def _refresh_y_list(self) -> None:
        items = self._available_y_items("")
        pd = self._active_plot_def()
        if pd is not None and pd.y_cols:
            available_cols = {self._y_display_to_col.get(d, d) for d in items}
            if available_cols:
                pd.y_cols = [y for y in pd.y_cols if y in available_cols]
                self._ws.plot_defs[pd.plot_id] = pd
        self._update_y_summary()

    def _available_y_items(self, filter_text: str) -> List[str]:
        items = list(self._y_display_to_col.keys())
        filt = str(filter_text or "").strip().lower()
        if filt:
            items = [d for d in items if filt in d.lower()]

        numeric_only = str(self._plot_cb.currentText()) not in ("Heatmap",)
        if numeric_only:
            sid = self._ws.active_id
            ds = self._ws.datasets.get(sid) if sid else None
            cols_map = dict(ds.columns or {}) if ds else {}
            numeric = set()
            for name, dtype in cols_map.items():
                if "int" in str(dtype) or "float" in str(dtype) or "double" in str(dtype):
                    numeric.add(str(name))
            items = [d for d in items if self._y_display_to_col.get(d, d) in numeric or not numeric]

        return items

    def _update_y_summary(self) -> None:
        pd = self._active_plot_def()
        if pd is None or not pd.y_cols:
            self._y_summary_label.setText("Y: (none)")
            return
        disp = [self._y_col_to_display.get(str(y), str(y)) for y in pd.y_cols]
        if len(disp) <= 3:
            self._y_summary_label.setText("Y: " + ", ".join(disp))
        else:
            self._y_summary_label.setText(f"Y: {len(disp)} selected")

    def _on_plot_type_changed(self) -> None:
        if self._restoring_ui:
            return
        self._toggle_extra_fields()
        self._refresh_y_list()
        self._mark_dirty()
        self._store_current_config()
        if self._ws.overlay_ids:
            new_type = str(self._plot_cb.currentText())
            for pd in self._ws.plot_defs.values():
                if pd.dataset_id in self._ws.overlay_ids:
                    pd.plot_type = new_type
                    self._ws.plot_defs[pd.plot_id] = pd
            self._refresh_workspace()

    def _on_x_changed(self) -> None:
        if self._restoring_ui:
            return
        self._mark_dirty()
        self._store_current_config()

    def _on_group_changed(self) -> None:
        if self._restoring_ui:
            return
        self._mark_dirty()
        self._store_current_config()

    def _reset_plot_builder(self) -> None:
        self._plot_cb.setCurrentText(PLOT_TYPES[0])
        self._drop_na_var = True
        self._decimal_var = False
        self._autocast_var = True
        self._norm_var = "None"
        self._populate_columns()
        self._restore_config_for_active()
        self._clear_dirty()

    def _plot(self) -> None:
        pd = self._active_plot_def()
        if pd is None:
            self.dialogs.error("Plot", "No active plot.")
            return

        def _work(_h):
            return self.adapter.build_plot_series(
                active_plot_def=pd,
                overlay_ids=self._ws.overlay_ids,
                plot_defs=self._ws.plot_defs,
                drop_na_default=bool(self._drop_na_var),
                normalize_default=str(self._norm_var),
                decimal_comma=bool(self._decimal_var),
                auto_cast=bool(self._autocast_var),
            )

        def _done(res):
            base_series, meta = res
            self._render_plot(base_series, meta)

        def _err(msg: str) -> None:
            self.dialogs.error("Plot", str(msg))

        self.worker_runner(
            _work,
            on_result=_done,
            on_error=_err,
            status=self.status,
            description="Plotting Data Studio",
            group="data_studio_plot",
            cancel_previous=True,
        )

    def _render_plot(self, base_series: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        series = [dict(s) for s in (base_series or [])]
        self.adapter.apply_overlay_offset(
            series,
            overlay_ids=self._ws.overlay_ids,
            mode=str(self._overlay_mode_cb.currentText()),
            offset=self._safe_float(self._overlay_offset_edit.text()),
        )

        self._ax.clear()
        plot_type = str(meta.get("plot_type", "Line"))

        if plot_type in ("Box plot", "Violin plot"):
            data = [np.asarray(s.get("y", []), dtype=float) for s in series]
            labels = [str(s.get("label", "")) for s in series]
            if plot_type == "Box plot":
                self._ax.boxplot(data, labels=labels, showfliers=True)
            else:
                self._ax.violinplot(data, showmeans=True, showmedians=True)
                self._ax.set_xticks(range(1, len(labels) + 1))
                self._ax.set_xticklabels(labels, rotation=45, ha="right")
        elif plot_type == "Histogram":
            for s in series:
                y = np.asarray(s.get("y", []), dtype=float)
                self._ax.hist(y, bins=int(self._bins_var), alpha=0.5, label=str(s.get("label", "")), color=s.get("color"))
        elif plot_type in ("Bar (grouped)", "Bar (stacked)"):
            labels = [str(s.get("label", "")) for s in series]
            xcats = meta.get("xcats", [])
            x = np.arange(len(xcats)) if xcats else np.arange(len(series[0].get("y", [])))
            width = 0.8 / max(1, len(series))
            bottoms = np.zeros_like(x, dtype=float)
            for i, s in enumerate(series):
                y = np.asarray(s.get("y", []), dtype=float)
                if plot_type == "Bar (stacked)":
                    self._ax.bar(x, y, bottom=bottoms, label=labels[i], color=s.get("color"))
                    bottoms = bottoms + y
                else:
                    self._ax.bar(x + i * width - (len(series) - 1) * width / 2, y, width=width, label=labels[i], color=s.get("color"))
            if xcats:
                self._ax.set_xticks(x)
                self._ax.set_xticklabels([str(c) for c in xcats], rotation=45, ha="right")
        elif plot_type == "Bubble":
            for s in series:
                x = np.asarray(s.get("x", []), dtype=float)
                y = np.asarray(s.get("y", []), dtype=float)
                size = np.asarray(s.get("size", []), dtype=float)
                size = 40 + 160 * (size - np.nanmin(size)) / (np.nanmax(size) - np.nanmin(size) + 1e-9)
                self._ax.scatter(x, y, s=size, label=str(s.get("label", "")), alpha=0.6, color=s.get("color"))
        else:
            for s in series:
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

        if meta.get("heatmap") is not None:
            hm = meta.get("heatmap")
            self._ax.clear()
            im = self._ax.imshow(hm["values"], aspect="auto")
            self._ax.set_xticks(range(len(hm["cols"])))
            self._ax.set_xticklabels([str(c) for c in hm["cols"]], rotation=45, ha="right")
            self._ax.set_yticks(range(len(hm["rows"])))
            self._ax.set_yticklabels([str(r) for r in hm["rows"]])
            self._fig.colorbar(im, ax=self._ax, fraction=0.046, pad=0.04)

        self._ax.set_title(meta.get("title", ""))
        self._ax.set_xlabel(meta.get("xlabel", ""))
        self._ax.set_ylabel(meta.get("ylabel", ""))
        if len(series) > 1:
            self._ax.legend(loc="best")
        self._ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()
        self._last_payload = {
            "series": base_series,
            "overlay_mode": str(self._overlay_mode_cb.currentText()),
            "overlay_offset": self._safe_float(self._overlay_offset_edit.text()),
            **meta,
        }
        self._store_current_config()
        if not self._ws.overlay_ids and self._ws.active_id:
            self._plotted_ids.add(self._ws.active_id)

    def _apply_plot(self) -> None:
        self._store_current_config()
        pd = self._active_plot_def()
        sid = self._ws.active_id
        ds = self._ws.datasets.get(sid) if sid else None
        if pd is not None and ds is not None:
            cols = set((ds.columns or {}).keys())
            if ds.schema_hash and pd.last_validated_schema_hash != ds.schema_hash:
                msg = None
                if pd.x_col and pd.x_col not in cols:
                    pd.x_col = None
                if pd.y_cols:
                    pd.y_cols = [y for y in pd.y_cols if y in cols]
                if not pd.x_col or not pd.y_cols:
                    x_def, y_def = self._pick_default_axes(sid)
                    if not pd.x_col:
                        pd.x_col = x_def
                    if not pd.y_cols:
                        pd.y_cols = list(y_def)
                    msg = f"Columns changed; auto-selected X={pd.x_col or 'Index'}, Y={', '.join(pd.y_cols or [])}"
                if msg:
                    self._banner_label.setText(msg)
                    self._restore_config_for_active()
            pd.last_validated_schema_hash = str(ds.schema_hash or "")
            self._ws.plot_defs[pd.plot_id] = pd

        self._plot()
        self._clear_dirty()

    def _store_current_config(self) -> None:
        pd = self._active_plot_def()
        if pd is None:
            return
        x_disp = self._x_cb.currentText()
        x_col = None if x_disp == "(Index)" else self._x_display_to_col.get(x_disp, x_disp)
        pd.x_col = x_col
        pd.y_cols = list(pd.y_cols or [])
        pd.plot_type = str(self._plot_cb.currentText())
        pd.options = {
            "group_col": (None if self._group_cb.currentText() == "(None)" else self._group_cb.currentText()),
            "y_err_col": (None if self._yerr_var == "(None)" else self._yerr_var),
            "x_err_col": (None if self._xerr_var == "(None)" else self._xerr_var),
            "size_col": (None if self._size_var == "(None)" else self._size_var),
            "heatmap_row": (None if self._heat_row_var == "(None)" else self._heat_row_var),
            "heatmap_col": (None if self._heat_col_var == "(None)" else self._heat_col_var),
            "heatmap_val": (None if self._heat_val_var == "(None)" else self._heat_val_var),
            "heatmap_agg": str(self._heat_agg_var),
            "hist_bins": int(self._bins_var),
            "rolling_window": int(self._roll_var),
            "drop_na": bool(self._drop_na_var),
            "decimal_comma": bool(self._decimal_var),
            "auto_cast": bool(self._autocast_var),
            "normalize": str(self._norm_var),
        }
        self._ws.plot_defs[pd.plot_id] = pd
        if pd.dataset_id:
            self._ws.preferred_axes_by_dataset[pd.dataset_id] = (pd.x_col, list(pd.y_cols or []))

    def _export_plot(self) -> None:
        payload = getattr(self, "_last_payload", None)
        if not payload:
            self.dialogs.info("Export", "Plot something first.")
            return
        dlg = DataStudioExportDialog(self, payload=payload)
        dlg.exec()

    def open_workspace(self) -> Optional[str]:
        return self._load_workspace()

    def save_workspace(self) -> Optional[str]:
        return self._save_workspace()

    def open_workspace_path(self, path: str) -> None:
        if not path:
            return
        self._load_workspace_path(str(path))

    def _load_workspace_path(self, path: str) -> None:
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            self.dialogs.error("Data Studio", f"Failed to load workspace:\n\n{exc}")
            return
        if not isinstance(payload, dict):
            self.dialogs.error("Data Studio", "Workspace JSON must be an object.")
            return

        ws, errors = decode_workspace(payload)
        self._ws = ws
        self.adapter.ws = ws
        self.adapter.df_cache = {}
        self._plotted_ids = set()
        self._last_workspace_path = str(path)
        self._refresh_workspace()
        for sid in list(self._ws.order):
            ds = self._ws.datasets.get(sid)
            if ds is not None and not ds.columns:
                self._infer_schema_async(sid)
        self._load_ui_state(path)
        if errors:
            self.dialogs.warn("Data Studio", "Workspace loaded with some issues:\n\n" + "\n".join(errors[:10]))
        self._status_label.setText("Workspace loaded")

    def get_last_workspace_path(self) -> Optional[str]:
        return self._last_workspace_path

    def _ui_state_path(self, workspace_path: str) -> str:
        return str(workspace_path) + ".ui.json"

    def _save_ui_state(self, workspace_path: str) -> None:
        try:
            payload = {
                "overlay_mode": str(self._overlay_mode_cb.currentText()),
                "overlay_offset": self._overlay_offset_edit.text(),
            }
            with open(self._ui_state_path(workspace_path), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_ui_state(self, workspace_path: str) -> None:
        try:
            with open(self._ui_state_path(workspace_path), "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return
        try:
            mode = str(payload.get("overlay_mode") or "Normal")
            if mode in ("Normal", "Offset Y", "Offset X"):
                self._overlay_mode_cb.setCurrentText(mode)
            self._overlay_offset_edit.setText(str(payload.get("overlay_offset") or "0.0"))
        except Exception:
            pass

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(str(value).strip())
        except Exception:
            return 0.0
