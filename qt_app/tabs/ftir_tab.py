from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QColorDialog,
    QFileDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import cm, rcParams

from lab_gui.ftir_io import _parse_ftir_xy_numpy
from lab_gui.ftir_analysis import pick_peaks, format_peak_label
from lab_gui.ftir_model import FTIRBondAnnotation, FTIRDataset, FTIRWorkspace, OverlayGroup, StyleState, FTIRDatasetKey
from qt_app.services import DialogService, StatusService
from qt_app.services.worker import run_in_worker


class FTIRTab(QWidget):
    def __init__(self, status: StatusService, dialogs: DialogService, worker_runner=None) -> None:
        super().__init__()
        self.status = status
        self.dialogs = dialogs
        self.worker_runner = worker_runner or run_in_worker

        self._workspaces: Dict[str, FTIRWorkspace] = {}
        self._workspace_order: List[str] = []
        self._active_workspace_id: Optional[str] = None
        self._last_workspace_path: Optional[str] = None

        self._overlay_groups: Dict[str, OverlayGroup] = {}
        self._overlay_order: List[str] = []
        self._active_overlay_group_id: Optional[str] = None
        self._overlay_color_mode: str = "Auto (Tableau)"
        self._overlay_offset_mode: str = "Normal"
        self._overlay_offset_value: float = 0.0
        self._overlay_single_hue: str = "#1f77b4"

        self._reverse_x: bool = False
        self._show_peaks: bool = False
        self._show_peaks_all_overlay: bool = False
        self._peaks_top_n: int = 6
        self._peaks_min_prom: float = 0.02

        self._peak_texts: List[object] = []
        self._peak_markers: List[object] = []
        self._peak_artist_to_info: Dict[object, Tuple[str, str, str]] = {}
        self._peak_texts_by_key: Dict[Tuple[str, str], List[object]] = {}
        self._peak_summary_text: Optional[object] = None

        self._bond_texts: List[object] = []
        self._bond_vlines: List[object] = []
        self._bond_artist_to_info: Dict[object, Tuple[str, str, int]] = {}

        self._drag_peak_id: Optional[str] = None
        self._drag_peak_key: Optional[Tuple[str, str]] = None
        self._drag_peak_artist: Optional[object] = None
        self._drag_dx: float = 0.0
        self._drag_dy: float = 0.0

        self._drag_bond_idx: Optional[int] = None
        self._drag_bond_key: Optional[Tuple[str, str]] = None
        self._drag_bond_artist: Optional[object] = None
        self._drag_bond_dx: float = 0.0
        self._drag_bond_dy: float = 0.0

        self._bond_place_active: bool = False
        self._bond_place_opts: Dict[str, object] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._build_ui())

        self._ensure_default_workspace()
        self._refresh_workspace_combo()
        self._refresh_dataset_tree()
        self._refresh_overlay_group_list()
        self._refresh_overlay_members_list()
        self._render_plot()

    def _build_ui(self) -> QWidget:
        root = QSplitter(Qt.Horizontal)
        root.setChildrenCollapsible(False)
        self._root_splitter = root

        left = self._build_left_panel()
        right = self._build_right_panel()

        root.addWidget(left)
        root.addWidget(right)
        root.setStretchFactor(0, 1)
        root.setStretchFactor(1, 3)
        return root

    def reset_layout(self) -> None:
        try:
            if getattr(self, "_root_splitter", None) is not None:
                self._root_splitter.setSizes([320, 980])
        except Exception:
            pass
        try:
            self._render_plot()
        except Exception:
            pass

    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        body = QWidget()
        scroll.setWidget(body)

        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        layout.addWidget(self._build_workspace_block())
        layout.addWidget(self._build_overlay_blocks())
        layout.addStretch(1)

        return scroll

    def _build_workspace_block(self) -> QWidget:
        wsblk = QGroupBox("Workspaces")
        v = QVBoxLayout(wsblk)

        row = QHBoxLayout()
        row.addWidget(QLabel("Workspace"))
        self._ws_combo = QComboBox()
        self._ws_combo.currentIndexChanged.connect(self._on_workspace_selected)
        row.addWidget(self._ws_combo, 1)
        v.addLayout(row)

        btns = QHBoxLayout()
        b_new = QPushButton("New Workspace")
        b_new.clicked.connect(self._new_workspace)
        btns.addWidget(b_new)
        b_ren = QPushButton("Rename")
        b_ren.clicked.connect(self._rename_workspace)
        btns.addWidget(b_ren)
        b_dup = QPushButton("Duplicate")
        b_dup.clicked.connect(self._duplicate_workspace)
        btns.addWidget(b_dup)
        b_col = QPushButton("Graph Color…")
        b_col.clicked.connect(self._edit_workspace_graph_color)
        btns.addWidget(b_col)
        b_del = QPushButton("Delete")
        b_del.clicked.connect(self._delete_workspace)
        btns.addWidget(b_del)
        btns.addStretch(1)
        v.addLayout(btns)

        v.addWidget(QLabel("FTIR datasets (current workspace)"))
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["", "Name", "Points"])
        self._tree.itemSelectionChanged.connect(self._on_dataset_select)
        v.addWidget(self._tree)

        d_btns = QHBoxLayout()
        b_load = QPushButton("Load FTIR…")
        b_load.clicked.connect(self._load_ftir_dialog)
        d_btns.addWidget(b_load)
        b_remove = QPushButton("Remove")
        b_remove.clicked.connect(self._remove_selected)
        d_btns.addWidget(b_remove)
        b_clear = QPushButton("Clear All")
        b_clear.clicked.connect(self._clear_all)
        d_btns.addWidget(b_clear)
        d_btns.addStretch(1)
        v.addLayout(d_btns)

        return wsblk

    def _build_overlay_blocks(self) -> QWidget:
        outer = QWidget()
        v = QVBoxLayout(outer)
        v.setContentsMargins(0, 0, 0, 0)

        groups_blk = QGroupBox("Overlays")
        g = QVBoxLayout(groups_blk)

        g.addWidget(QLabel("Overlay Groups"))
        self._overlay_groups_tree = QTreeWidget()
        self._overlay_groups_tree.setHeaderLabels(["Name", "#", "WS"])
        self._overlay_groups_tree.itemSelectionChanged.connect(self._on_overlay_group_select)
        g.addWidget(self._overlay_groups_tree)

        g_btns = QHBoxLayout()
        b_new = QPushButton("New Overlay from Selection")
        b_new.clicked.connect(self._new_overlay_group_from_selection)
        g_btns.addWidget(b_new)
        b_act = QPushButton("Activate Overlay")
        b_act.clicked.connect(self._activate_selected_overlay_group)
        g_btns.addWidget(b_act)
        b_ren = QPushButton("Rename…")
        b_ren.clicked.connect(self._rename_selected_overlay_group)
        g_btns.addWidget(b_ren)
        b_dup = QPushButton("Duplicate")
        b_dup.clicked.connect(self._duplicate_selected_overlay_group)
        g_btns.addWidget(b_dup)
        b_del = QPushButton("Delete")
        b_del.clicked.connect(self._delete_selected_overlay_group)
        g_btns.addWidget(b_del)
        b_clr = QPushButton("Clear Active Overlay")
        b_clr.clicked.connect(self._clear_active_overlay_group)
        g_btns.addWidget(b_clr)
        g_btns.addStretch(1)
        g.addLayout(g_btns)

        colors = QHBoxLayout()
        colors.addWidget(QLabel("Overlay colors"))
        self._overlay_colors = QComboBox()
        self._overlay_colors.addItems(["Auto (Tableau)", "Single hue", "Viridis", "Plasma", "Magma", "Cividis", "Turbo"])
        self._overlay_colors.currentIndexChanged.connect(self._on_overlay_color_changed)
        colors.addWidget(self._overlay_colors)
        b_pick = QPushButton("Pick hue…")
        b_pick.clicked.connect(self._pick_overlay_single_hue_color)
        colors.addWidget(b_pick)
        colors.addStretch(1)
        g.addLayout(colors)

        offset = QHBoxLayout()
        offset.addWidget(QLabel("Overlay offset"))
        self._overlay_offset_mode = QComboBox()
        self._overlay_offset_mode.addItems(["Normal", "Offset Y", "Offset X"])
        self._overlay_offset_mode.currentIndexChanged.connect(self._on_overlay_offset_changed)
        offset.addWidget(self._overlay_offset_mode)
        offset.addWidget(QLabel("Value"))
        self._overlay_offset_val = QLineEdit()
        self._overlay_offset_val.textChanged.connect(self._on_overlay_offset_changed)
        offset.addWidget(self._overlay_offset_val)
        offset.addStretch(1)
        g.addLayout(offset)

        g.addWidget(QLabel("Members (selected group)"))
        self._overlay_members_tree = QTreeWidget()
        self._overlay_members_tree.setHeaderLabels(["Workspace :: Dataset"])
        self._overlay_members_tree.itemSelectionChanged.connect(self._on_overlay_member_select)
        g.addWidget(self._overlay_members_tree)
        m_btns = QHBoxLayout()
        b_set_active = QPushButton("Set Active = selected member")
        b_set_active.clicked.connect(self._set_active_overlay_member_from_selected)
        m_btns.addWidget(b_set_active)
        m_btns.addStretch(1)
        g.addLayout(m_btns)

        v.addWidget(groups_blk)

        select_blk = QGroupBox("Selection (for new overlay group)")
        s = QVBoxLayout(select_blk)
        s.addWidget(QLabel("Filter"))
        self._overlay_filter = QLineEdit()
        self._overlay_filter.textChanged.connect(self._rebuild_overlay_selection_list)
        s.addWidget(self._overlay_filter)
        self._overlay_selection_tree = QTreeWidget()
        self._overlay_selection_tree.setHeaderLabels(["Workspace :: Dataset"])
        s.addWidget(self._overlay_selection_tree)
        v.addWidget(select_blk)

        return outer

    def _build_right_panel(self) -> QWidget:
        right = QWidget()
        v = QVBoxLayout(right)
        v.setContentsMargins(6, 6, 6, 6)

        top = QHBoxLayout()
        b_save = QPushButton("Save FTIR Plot…")
        b_save.clicked.connect(self._save_plot_dialog)
        top.addWidget(b_save)
        b_peaks = QPushButton("Peaks…")
        b_peaks.clicked.connect(self._open_peaks_dialog)
        top.addWidget(b_peaks)
        b_export = QPushButton("Export Peaks…")
        b_export.clicked.connect(self._export_peaks)
        top.addWidget(b_export)
        b_bond = QPushButton("Add Bond Label…")
        b_bond.clicked.connect(self._open_add_bond_label_dialog)
        top.addWidget(b_bond)
        cb_rev = QCheckBox("Reverse x-axis (common FTIR)")
        cb_rev.toggled.connect(self._on_toggle_reverse)
        top.addWidget(cb_rev)
        cb_peaks_all = QCheckBox("Show peaks for all overlayed spectra")
        cb_peaks_all.setChecked(bool(self._show_peaks_all_overlay))
        cb_peaks_all.toggled.connect(self._on_toggle_peaks_all_overlay)
        top.addWidget(cb_peaks_all)
        self._peaks_all_cb = cb_peaks_all
        top.addStretch(1)
        v.addLayout(top)

        self._status_label = QLabel("")
        v.addWidget(self._status_label)

        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(10.5, 8.5), dpi=100)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_title("FTIR")
        self._ax.set_xlabel("Wavenumber")
        self._ax.set_ylabel("Absorbance")

        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        plot_layout.addWidget(self._canvas, 1)
        plot_layout.addWidget(self._toolbar)

        v.addWidget(plot, 1)

        self._canvas.mpl_connect("draw_event", self._on_mpl_draw_event)
        self._canvas.mpl_connect("pick_event", self._on_peak_pick_event)
        self._canvas.mpl_connect("button_press_event", self._on_peak_drag_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_peak_drag_motion)
        self._canvas.mpl_connect("button_release_event", self._on_peak_drag_release)
        self._canvas.mpl_connect("key_press_event", self._on_ftir_keypress)

        return right

    def _utc_now_iso(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _ensure_default_workspace(self) -> None:
        if self._workspaces:
            return
        wid = str(uuid.uuid4())
        ws = FTIRWorkspace(id=wid, name="Default")
        self._workspaces[wid] = ws
        self._workspace_order.append(wid)
        self._active_workspace_id = wid

    def _current_workspace(self) -> Optional[FTIRWorkspace]:
        if not self._active_workspace_id:
            return None
        return self._workspaces.get(self._active_workspace_id)

    def _refresh_workspace_combo(self) -> None:
        self._ws_combo.blockSignals(True)
        self._ws_combo.clear()
        for wid in self._workspace_order:
            ws = self._workspaces.get(wid)
            if ws is None:
                continue
            self._ws_combo.addItem(ws.name, wid)
        idx = 0
        if self._active_workspace_id:
            for i in range(self._ws_combo.count()):
                if str(self._ws_combo.itemData(i)) == str(self._active_workspace_id):
                    idx = i
                    break
        self._ws_combo.setCurrentIndex(idx)
        self._ws_combo.blockSignals(False)

    def _refresh_dataset_tree(self) -> None:
        self._tree.clear()
        ws = self._current_workspace()
        if ws is None:
            return
        for ds in ws.datasets:
            pts = len(ds.x_disp) if ds.x_disp is not None else 0
            active = "●" if ws.active_dataset_id == ds.id else ""
            item = QTreeWidgetItem([active, ds.name, str(pts)])
            item.setData(0, Qt.UserRole, ds.id)
            self._tree.addTopLevelItem(item)
        self._rebuild_overlay_selection_list()

    def _on_workspace_selected(self, _index: int = 0) -> None:
        wid = self._ws_combo.currentData()
        if wid and str(wid) in self._workspaces:
            self._active_workspace_id = str(wid)
            self._refresh_dataset_tree()
            self._render_plot()

    def _new_workspace(self) -> None:
        name = f"Workspace {len(self._workspace_order) + 1}"
        wid = str(uuid.uuid4())
        self._workspaces[wid] = FTIRWorkspace(id=wid, name=name)
        self._workspace_order.append(wid)
        self._active_workspace_id = wid
        self._refresh_workspace_combo()
        self._refresh_dataset_tree()
        self._render_plot()

    def _rename_workspace(self) -> None:
        ws = self._current_workspace()
        if ws is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Workspace", "Name", text=ws.name)
        if not ok:
            return
        ws.name = str(new_name or ws.name)
        self._workspaces[ws.id] = ws
        self._refresh_workspace_combo()

    def _duplicate_workspace(self) -> None:
        ws = self._current_workspace()
        if ws is None:
            return
        wid = str(uuid.uuid4())
        clone = FTIRWorkspace(id=wid, name=f"{ws.name} (copy)")
        clone.datasets = list(ws.datasets)
        clone.active_dataset_id = ws.active_dataset_id
        clone.line_color = ws.line_color
        self._workspaces[wid] = clone
        self._workspace_order.append(wid)
        self._active_workspace_id = wid
        self._refresh_workspace_combo()
        self._refresh_dataset_tree()
        self._render_plot()

    def _delete_workspace(self) -> None:
        if len(self._workspace_order) <= 1:
            self.dialogs.info("FTIR", "At least one workspace is required.")
            return
        ws = self._current_workspace()
        if ws is None:
            return
        self._workspaces.pop(ws.id, None)
        if ws.id in self._workspace_order:
            self._workspace_order.remove(ws.id)
        self._active_workspace_id = (self._workspace_order[0] if self._workspace_order else None)
        self._refresh_workspace_combo()
        self._refresh_dataset_tree()
        self._render_plot()

    def _edit_workspace_graph_color(self) -> None:
        ws = self._current_workspace()
        if ws is None:
            return
        color = QColorDialog.getColor(parent=self, title="Pick FTIR graph color")
        if not color.isValid():
            return
        ws.line_color = color.name()
        self._workspaces[ws.id] = ws
        self._render_plot()

    def _on_dataset_select(self) -> None:
        ws = self._current_workspace()
        if ws is None:
            return
        items = self._tree.selectedItems()
        if not items:
            return
        ds_id = str(items[0].data(0, Qt.UserRole) or "")
        if not ds_id:
            return
        ws.active_dataset_id = ds_id
        self._workspaces[ws.id] = ws
        self._refresh_dataset_tree()
        self._render_plot()

    def _load_ftir_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Load FTIR", "", "Text/CSV (*.txt *.csv);;All files (*.*)")
        if not paths:
            return

        def _work(_h):
            loaded: List[FTIRDataset] = []
            for p in paths:
                path = Path(p)
                x, y, meta = _parse_ftir_xy_numpy(str(path))
                ds = FTIRDataset(
                    id=str(uuid.uuid4()),
                    name=path.name,
                    path=path,
                    x_full=np.asarray(x, dtype=float),
                    y_full=np.asarray(y, dtype=float),
                    x_disp=np.asarray(x, dtype=float),
                    y_disp=np.asarray(y, dtype=float),
                    x_units=meta.get("XUNITS"),
                    y_units=meta.get("YUNITS"),
                    loaded_at_utc=self._utc_now_iso(),
                )
                loaded.append(ds)
            return loaded

        def _done(loaded: List[FTIRDataset]) -> None:
            ws = self._current_workspace()
            if ws is None:
                return
            if not loaded:
                self.dialogs.info("FTIR", "No FTIR datasets loaded.")
                return
            ws.datasets.extend(loaded)
            ws.active_dataset_id = loaded[-1].id
            self._workspaces[ws.id] = ws
            self._refresh_dataset_tree()
            self._render_plot()
            self.status.set_status(f"Loaded {len(loaded)} FTIR file(s)")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Loading FTIR",
            group="ftir_load",
            cancel_previous=True,
        )

    def _remove_selected(self) -> None:
        ws = self._current_workspace()
        if ws is None:
            return
        items = self._tree.selectedItems()
        if not items:
            return
        ds_id = str(items[0].data(0, Qt.UserRole) or "")
        ws.datasets = [d for d in ws.datasets if str(d.id) != ds_id]
        if ws.active_dataset_id == ds_id:
            ws.active_dataset_id = (ws.datasets[-1].id if ws.datasets else None)
        self._workspaces[ws.id] = ws
        self._refresh_dataset_tree()
        self._render_plot()

    def _clear_all(self) -> None:
        ws = self._current_workspace()
        if ws is None:
            return
        ws.datasets = []
        ws.active_dataset_id = None
        self._workspaces[ws.id] = ws
        self._refresh_dataset_tree()
        self._render_plot()

    def _on_overlay_color_changed(self, _index: int = 0) -> None:
        self._overlay_color_mode = str(self._overlay_colors.currentText() or "Auto (Tableau)")
        self._render_plot()

    def _pick_overlay_single_hue_color(self) -> None:
        color = QColorDialog.getColor(parent=self, title="Pick overlay hue")
        if not color.isValid():
            return
        self._overlay_single_hue = color.name()
        self._overlay_color_mode = "Single hue"
        try:
            self._overlay_colors.setCurrentText("Single hue")
        except Exception:
            pass
        self._render_plot()

    def _on_overlay_offset_changed(self) -> None:
        self._overlay_offset_mode = str(self._overlay_offset_mode.currentText() or "Normal")
        try:
            self._overlay_offset_value = float(self._overlay_offset_val.text() or 0.0)
        except Exception:
            self._overlay_offset_value = 0.0
        self._render_plot()

    def _rebuild_overlay_selection_list(self) -> None:
        text = str(self._overlay_filter.text() or "").strip().lower()
        self._overlay_selection_tree.clear()
        for wid in self._workspace_order:
            ws = self._workspaces.get(wid)
            if ws is None:
                continue
            for ds in ws.datasets:
                label = f"{ws.name} :: {ds.name}"
                if text and text not in label.lower():
                    continue
                item = QTreeWidgetItem([label])
                item.setData(0, Qt.UserRole, (ws.id, ds.id))
                item.setCheckState(0, Qt.Unchecked)
                self._overlay_selection_tree.addTopLevelItem(item)

    def _new_overlay_group_from_selection(self) -> None:
        members: List[FTIRDatasetKey] = []
        for i in range(self._overlay_selection_tree.topLevelItemCount()):
            item = self._overlay_selection_tree.topLevelItem(i)
            if item.checkState(0) != Qt.Checked:
                continue
            data = item.data(0, Qt.UserRole)
            if isinstance(data, tuple) and len(data) == 2:
                members.append((str(data[0]), str(data[1])))
        if not members:
            self.dialogs.info("FTIR", "Select at least one dataset for the overlay group.")
            return
        gid = str(uuid.uuid4())
        group = OverlayGroup(group_id=gid, name=f"Overlay {len(self._overlay_order) + 1}")
        group.members = list(members)
        group.active_member = members[0]
        group.per_member_style = {m: StyleState(linewidth=1.2) for m in members}
        self._overlay_groups[gid] = group
        self._overlay_order.append(gid)
        self._active_overlay_group_id = gid
        self._refresh_overlay_group_list()
        self._refresh_overlay_members_list()
        self._render_plot()

    def _activate_selected_overlay_group(self) -> None:
        items = self._overlay_groups_tree.selectedItems()
        if not items:
            return
        gid = str(items[0].data(0, Qt.UserRole) or "")
        if not gid:
            return
        if gid in self._overlay_groups:
            self._active_overlay_group_id = gid
            self._refresh_overlay_group_list()
            self._refresh_overlay_members_list()
            self._render_plot()

    def _rename_selected_overlay_group(self) -> None:
        items = self._overlay_groups_tree.selectedItems()
        if not items:
            return
        gid = str(items[0].data(0, Qt.UserRole) or "")
        group = self._overlay_groups.get(gid)
        if group is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Overlay", "Name", text=group.name)
        if not ok:
            return
        group.name = str(new_name or group.name)
        self._overlay_groups[gid] = group
        self._refresh_overlay_group_list()

    def _duplicate_selected_overlay_group(self) -> None:
        items = self._overlay_groups_tree.selectedItems()
        if not items:
            return
        gid = str(items[0].data(0, Qt.UserRole) or "")
        group = self._overlay_groups.get(gid)
        if group is None:
            return
        ngid = str(uuid.uuid4())
        ng = OverlayGroup(group_id=ngid, name=f"{group.name} (copy)")
        ng.members = list(group.members)
        ng.active_member = group.active_member
        ng.per_member_style = dict(group.per_member_style)
        self._overlay_groups[ngid] = ng
        self._overlay_order.append(ngid)
        self._active_overlay_group_id = ngid
        self._refresh_overlay_group_list()
        self._refresh_overlay_members_list()
        self._render_plot()

    def _delete_selected_overlay_group(self) -> None:
        items = self._overlay_groups_tree.selectedItems()
        if not items:
            return
        gid = str(items[0].data(0, Qt.UserRole) or "")
        self._overlay_groups.pop(gid, None)
        if gid in self._overlay_order:
            self._overlay_order.remove(gid)
        if self._active_overlay_group_id == gid:
            self._active_overlay_group_id = None
        self._refresh_overlay_group_list()
        self._refresh_overlay_members_list()
        self._render_plot()

    def _clear_active_overlay_group(self) -> None:
        self._active_overlay_group_id = None
        self._refresh_overlay_group_list()
        self._refresh_overlay_members_list()
        self._render_plot()

    def _on_overlay_group_select(self) -> None:
        self._refresh_overlay_members_list()

    def _on_overlay_member_select(self) -> None:
        pass

    def _set_active_overlay_member_from_selected(self) -> None:
        items = self._overlay_members_tree.selectedItems()
        if not items:
            return
        data = items[0].data(0, Qt.UserRole)
        if not (isinstance(data, tuple) and len(data) == 2):
            return
        gid = self._active_overlay_group_id
        group = self._overlay_groups.get(gid) if gid else None
        if group is None:
            return
        group.active_member = (str(data[0]), str(data[1]))
        self._overlay_groups[group.group_id] = group
        self._render_plot()

    def _refresh_overlay_group_list(self) -> None:
        self._overlay_groups_tree.clear()
        for gid in self._overlay_order:
            g = self._overlay_groups.get(gid)
            if g is None:
                continue
            ws_names = {self._workspaces.get(wid).name for (wid, _did) in g.members if self._workspaces.get(wid)}
            ws_text = ", ".join(sorted([w for w in ws_names if w]))
            item = QTreeWidgetItem([g.name, str(len(g.members)), ws_text])
            item.setData(0, Qt.UserRole, gid)
            if gid == self._active_overlay_group_id:
                item.setText(0, f"● {g.name}")
            self._overlay_groups_tree.addTopLevelItem(item)

    def _refresh_overlay_members_list(self) -> None:
        self._overlay_members_tree.clear()
        gid = self._active_overlay_group_id
        group = self._overlay_groups.get(gid) if gid else None
        if group is None:
            return
        for wid, did in group.members:
            ws = self._workspaces.get(wid)
            ds = self._get_dataset_by_key((wid, did))
            if ws is None or ds is None:
                continue
            label = f"{ws.name} :: {ds.name}"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.UserRole, (wid, did))
            self._overlay_members_tree.addTopLevelItem(item)

    def _get_dataset_by_key(self, key: FTIRDatasetKey) -> Optional[FTIRDataset]:
        ws = self._workspaces.get(str(key[0]))
        if ws is None:
            return None
        for ds in ws.datasets:
            if str(ds.id) == str(key[1]):
                return ds
        return None

    def _overlay_colors_for_count(self, count: int) -> List[str]:
        if count <= 0:
            return []
        mode = str(self._overlay_color_mode or "Auto (Tableau)")
        if mode == "Single hue":
            return [self._overlay_single_hue for _ in range(count)]
        if mode and mode.lower() in ("viridis", "plasma", "magma", "cividis", "turbo"):
            cmap = cm.get_cmap(mode.lower())
            return [cmap(i / max(1, count - 1)) for i in range(count)]
        cycle = rcParams.get("axes.prop_cycle")
        colors = cycle.by_key().get("color", ["#1f77b4"])
        out = []
        for i in range(count):
            out.append(colors[i % len(colors)])
        return out

    def _render_plot(self) -> None:
        self._ax.clear()
        self._clear_peak_artists()
        self._clear_bond_artists()
        self._ax.set_title("FTIR")
        self._ax.set_xlabel("Wavenumber")
        self._ax.set_ylabel("Absorbance")
        group = self._overlay_groups.get(self._active_overlay_group_id or "")
        if group and group.members:
            colors = self._overlay_colors_for_count(len(group.members))
            offset_mode = str(self._overlay_offset_mode or "Normal")
            offset = float(self._overlay_offset_value or 0.0)
            active_key = group.active_member or (group.members[0] if group.members else None)
            for idx, key in enumerate(group.members):
                ds = self._get_dataset_by_key(key)
                if ds is None:
                    continue
                x = np.asarray(ds.x_disp, dtype=float)
                y = np.asarray(ds.y_disp, dtype=float)
                if offset_mode == "Offset Y":
                    y = y + offset * idx
                elif offset_mode == "Offset X":
                    x = x + offset * idx
                lw = 2.0 if group.active_member == key else 1.2
                color = colors[idx] if idx < len(colors) else "#1f77b4"
                self._ax.plot(x, y, color=color, linewidth=lw, label=f"{key[0]}::{key[1]}")
                if self._show_peaks and (self._show_peaks_all_overlay or active_key == key):
                    try:
                        self._render_peaks_for_dataset(ds, dataset_key=(str(key[0]), str(key[1])), peak_color=color, pickable=(active_key == key))
                    except Exception:
                        pass
            try:
                self._ax.legend(loc="best", fontsize=8)
            except Exception:
                pass
            if active_key is not None:
                ads = self._get_dataset_by_key(active_key)
                if ads is not None:
                    self._render_bond_labels(ads)
            if self._reverse_x:
                try:
                    self._ax.invert_xaxis()
                except Exception:
                    pass
            self._canvas.draw_idle()
            return

        ws = self._current_workspace()
        if ws is None or not ws.datasets:
            self._canvas.draw_idle()
            return
        ds = next((d for d in ws.datasets if d.id == ws.active_dataset_id), None)
        if ds is None:
            ds = ws.datasets[-1]
            ws.active_dataset_id = ds.id
            self._workspaces[ws.id] = ws
        color = ws.line_color or "#1f77b4"
        self._ax.plot(ds.x_disp, ds.y_disp, color=color, linewidth=1.2)
        if self._show_peaks:
            self._render_peaks(ds)
        self._render_bond_labels(ds)
        if self._reverse_x:
            try:
                self._ax.invert_xaxis()
            except Exception:
                pass
        self._canvas.draw_idle()

    def _save_plot_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FTIR Plot",
            "",
            "PNG (*.png);;JPG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;PDF (*.pdf);;All files (*.*)",
        )
        if not path:
            return
        try:
            self._figure.savefig(path)
            self.status.set_status("Plot saved")
        except Exception as exc:
            self.dialogs.error("FTIR", f"Failed to save plot:\n\n{exc}")

    def _on_toggle_reverse(self, checked: bool) -> None:
        self._reverse_x = bool(checked)
        self._render_plot()

    def _on_toggle_peaks_all_overlay(self, checked: bool) -> None:
        self._show_peaks_all_overlay = bool(checked)
        self._render_plot()

    def _open_peaks_dialog(self) -> None:
        top_n, ok = QInputDialog.getInt(self, "Peaks", "Top N peaks", value=int(self._peaks_top_n), min=0, max=50)
        if not ok:
            return
        prom, ok2 = QInputDialog.getDouble(
            self,
            "Peaks",
            "Minimum prominence",
            value=float(self._peaks_min_prom),
            min=0.0,
            max=1e6,
            decimals=4,
        )
        if not ok2:
            return
        self._peaks_top_n = int(top_n)
        self._peaks_min_prom = float(prom)
        self._show_peaks = True
        ws = self._current_workspace()
        if ws is not None:
            ds = next((d for d in ws.datasets if d.id == ws.active_dataset_id), None)
            if ds is None and ws.datasets:
                ds = ws.datasets[-1]
            if ds is not None:
                ds.peak_settings = {
                    "enabled": True,
                    "min_prominence": float(self._peaks_min_prom),
                    "top_n": int(self._peaks_top_n),
                    "label_fmt": "{wn:.0f}",
                }
        self._render_plot()

    def _clear_peak_artists(self) -> None:
        for m in list(self._peak_markers):
            try:
                m.remove()
            except Exception:
                pass
        for t in list(self._peak_texts):
            try:
                t.remove()
            except Exception:
                pass
        if self._peak_summary_text is not None:
            try:
                self._peak_summary_text.remove()
            except Exception:
                pass
        self._peak_markers = []
        self._peak_texts = []
        self._peak_artist_to_info = {}
        self._peak_texts_by_key = {}
        self._peak_summary_text = None

    def _clear_bond_artists(self) -> None:
        for t in list(self._bond_texts):
            try:
                t.remove()
            except Exception:
                pass
        for ln in list(self._bond_vlines):
            try:
                ln.remove()
            except Exception:
                pass
        self._bond_texts = []
        self._bond_vlines = []
        self._bond_artist_to_info = {}

    def _ensure_peaks_for_dataset(self, ds: FTIRDataset) -> None:
        try:
            enabled = bool(ds.peak_settings.get("enabled", self._show_peaks))
        except Exception:
            enabled = bool(self._show_peaks)
        if not enabled:
            return
        try:
            prom = float(ds.peak_settings.get("min_prominence", self._peaks_min_prom))
        except Exception:
            prom = float(self._peaks_min_prom)
        try:
            top_n = int(ds.peak_settings.get("top_n", self._peaks_top_n))
        except Exception:
            top_n = int(self._peaks_top_n)

        try:
            peaks = pick_peaks(
                ds.x_disp,
                ds.y_disp,
                mode=str(ds.y_mode or "absorbance"),
                min_prominence=float(prom),
                top_n=int(top_n),
            )
        except Exception:
            peaks = []

        out: List[Dict[str, object]] = []
        for p in peaks:
            try:
                pid = f"{float(p.wn):.3f}"
                out.append(
                    {
                        "id": pid,
                        "wn": float(p.wn),
                        "y": float(p.y),
                        "y_display": float(p.y),
                        "prominence": float(p.prominence),
                    }
                )
            except Exception:
                continue
        ds.peaks = out

    def _render_peaks_for_dataset(
        self,
        ds: FTIRDataset,
        *,
        dataset_key: Optional[Tuple[str, str]] = None,
        peak_color: Optional[str] = None,
        pickable: bool = True,
        max_peaks: int = 0,
    ) -> None:
        self._ensure_peaks_for_dataset(ds)

        peak_color = str(peak_color or "#d32f2f")
        peaks = list(getattr(ds, "peaks", []) or [])
        suppressed = set(getattr(ds, "peak_suppressed", set()) or set())
        overrides = dict(getattr(ds, "peak_label_overrides", {}) or {})
        positions = dict(getattr(ds, "peak_label_positions", {}) or {})
        fmt = str(getattr(ds, "peak_settings", {}).get("label_fmt") or "{wn:.0f}")

        shown = [p for p in peaks if isinstance(p, dict) and str(p.get("id") or "") and (str(p.get("id")) not in suppressed)]
        if int(max_peaks or 0) > 0 and len(shown) > int(max_peaks or 0):
            try:
                shown.sort(key=lambda p: float(p.get("prominence", 0.0) or 0.0), reverse=True)
            except Exception:
                pass
            shown = shown[: int(max_peaks or 0)]

        xs: List[float] = []
        ys: List[float] = []
        for p in shown:
            try:
                xs.append(float(p.get("wn")))
                ys.append(float(p.get("y_display", p.get("y", 0.0))))
            except Exception:
                continue
        if xs and ys:
            try:
                (mline,) = self._ax.plot(
                    xs,
                    ys,
                    linestyle="none",
                    marker="o",
                    markersize=4,
                    color=peak_color,
                    markerfacecolor=peak_color,
                    markeredgecolor=peak_color,
                )
                self._peak_markers.append(mline)
            except Exception:
                pass

        for p in shown:
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue
            try:
                peak_wn = float(p.get("wn"))
                peak_y = float(p.get("y_display", p.get("y", 0.0)))
                prom0 = float(p.get("prominence", 0.0))
            except Exception:
                continue

            pos_x = float(peak_wn)
            pos_y = float(peak_y)
            try:
                if pid in positions:
                    pos = positions.get(pid)
                    if isinstance(pos, tuple) and len(pos) == 2:
                        pos_x = float(pos[0])
                        pos_y = float(pos[1])
            except Exception:
                pass

            label = str(overrides.get(pid, "") or "").strip()
            if not label:
                try:
                    label = format_peak_label(
                        peak=type("_P", (), {"wn": peak_wn, "y": peak_y, "prominence": prom0})(),
                        fmt=fmt,
                    )
                except Exception:
                    label = f"{peak_wn:.0f}"

            try:
                txt = self._ax.annotate(
                    str(label),
                    xy=(float(peak_wn), float(peak_y)),
                    xytext=(float(pos_x), float(pos_y)),
                    textcoords="data",
                    xycoords="data",
                    va="bottom",
                    ha="left",
                    fontsize=8,
                    color=peak_color,
                    clip_on=True,
                    arrowprops={"arrowstyle": "-", "color": peak_color, "lw": 0.8, "shrinkA": 0.0, "shrinkB": 0.0},
                )
                try:
                    txt.set_picker(bool(pickable))
                except Exception:
                    pass
                self._peak_texts.append(txt)
                if dataset_key is not None:
                    self._peak_texts_by_key.setdefault((str(dataset_key[0]), str(dataset_key[1])), []).append(txt)
                    if pickable:
                        self._peak_artist_to_info[txt] = (str(dataset_key[0]), str(dataset_key[1]), str(pid))
            except Exception:
                continue

        if shown:
            try:
                n_total = len([p for p in peaks if isinstance(p, dict)])
                n_show = len(shown)
                summary = f"Peaks: {n_show}/{n_total}"
                self._peak_summary_text = self._ax.text(0.01, 0.99, summary, transform=self._ax.transAxes, va="top", ha="left", fontsize=9)
            except Exception:
                pass

    def _render_peaks(self, ds: FTIRDataset) -> None:
        ws = self._current_workspace()
        key = None
        if ws is not None:
            key = (ws.id, ds.id)
        self._render_peaks_for_dataset(ds, dataset_key=key, peak_color="#d32f2f", pickable=True)

    def _export_peaks(self) -> None:
        ws = self._current_workspace()
        if ws is None or not ws.datasets:
            self.dialogs.info("FTIR", "No datasets to export.")
            return
        ds = next((d for d in ws.datasets if d.id == ws.active_dataset_id), None)
        if ds is None:
            ds = ws.datasets[-1]
        self._ensure_peaks_for_dataset(ds)
        peaks = list(getattr(ds, "peaks", []) or [])
        if not peaks:
            self.dialogs.info("FTIR", "No peaks detected.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Peaks",
            "",
            "CSV (*.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            import csv

            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["wavenumber", "y", "prominence"])
                for p in peaks:
                    if not isinstance(p, dict):
                        continue
                    writer.writerow([float(p.get("wn", 0.0)), float(p.get("y", 0.0)), float(p.get("prominence", 0.0))])
            self.status.set_status("Peaks exported")
        except Exception as exc:
            self.dialogs.error("FTIR", f"Failed to export peaks:\n\n{exc}")

    def _open_add_bond_label_dialog(self) -> None:
        ws = self._current_workspace()
        if ws is None or not ws.datasets:
            self.dialogs.info("FTIR", "No dataset selected.")
            return
        ds = next((d for d in ws.datasets if d.id == ws.active_dataset_id), None)
        if ds is None:
            ds = ws.datasets[-1]

        text, ok = QInputDialog.getText(self, "Add Bond Label", "Label text")
        if not ok or not str(text).strip():
            return
        mode, ok2 = QInputDialog.getItem(
            self,
            "Add Bond Label",
            "Placement mode",
            ["Click to place", "Auto place (nearest peak)", "Manual value"],
            0,
            False,
        )
        if not ok2:
            return

        show_vline = True
        try:
            vline_choice, okv = QInputDialog.getItem(self, "Add Bond Label", "Show vertical guide line?", ["Yes", "No"], 0, False)
            if okv:
                show_vline = (str(vline_choice).lower() == "yes")
        except Exception:
            show_vline = True

        opts = {
            "text": str(text).strip(),
            "show_vline": bool(show_vline),
            "line_color": "#444444",
            "text_color": "#111111",
            "fontsize": 9,
            "rotation": 0,
            "target_dataset_id": ds.id,
        }

        if str(mode) == "Manual value":
            xval, ok3 = QInputDialog.getDouble(
                self,
                "Add Bond Label",
                "Wavenumber (cm⁻¹)",
                value=float(ds.x_disp[0]) if ds.x_disp.size else 0.0,
                min=-1e9,
                max=1e9,
                decimals=2,
            )
            if not ok3:
                return
            yval = self._interp_y_at(ds, float(xval))
            self._bond_add_annotation(opts, x_cm1=float(xval), y_value=float(yval), xytext=(float(xval), float(yval)))
            self._render_plot()
            return

        if str(mode) == "Auto place (nearest peak)":
            self._bond_autoplace(ds, opts)
            self._render_plot()
            return

        self._bond_begin_placement(opts)

    def _render_bond_labels(self, ds: FTIRDataset) -> None:
        for idx, ann in enumerate(list(getattr(ds, "bond_annotations", []) or [])):
            try:
                txt = self._ax.text(
                    float(ann.x_cm1),
                    float(ann.y_value),
                    str(ann.text),
                    color=str(ann.text_color or "#111111"),
                    fontsize=int(getattr(ann, "fontsize", 9) or 9),
                    rotation=int(getattr(ann, "rotation", 0) or 0),
                    va="bottom",
                    ha="center",
                )
                self._bond_texts.append(txt)
                self._bond_artist_to_info[txt] = (str(getattr(ds, "id", "")), str(getattr(ds, "id", "")), int(idx))
                if getattr(ann, "show_vline", False):
                    ln = self._ax.axvline(float(ann.x_cm1), color=str(ann.line_color or "#444444"), linewidth=1.0, alpha=0.6)
                    self._bond_vlines.append(ln)
                    self._bond_artist_to_info[ln] = (str(getattr(ds, "id", "")), str(getattr(ds, "id", "")), int(idx))
            except Exception:
                continue

    def _interp_y_at(self, ds: FTIRDataset, xval: float) -> float:
        try:
            xp = np.asarray(ds.x_disp, dtype=float)
            yp = np.asarray(ds.y_disp, dtype=float)
            mask = np.isfinite(xp) & np.isfinite(yp)
            xp = xp[mask]
            yp = yp[mask]
            if xp.size >= 2:
                order = np.argsort(xp)
                xp = xp[order]
                yp = yp[order]
                return float(np.interp(float(xval), xp, yp))
        except Exception:
            return 0.0
        return 0.0

    def _bond_begin_placement(self, opts: Dict[str, object]) -> None:
        self._bond_place_active = True
        self._bond_place_opts = dict(opts or {})
        try:
            self._status_label.setText("Click on the plot to place the bond label. Esc to cancel.")
        except Exception:
            pass

    def _bond_add_annotation(self, opts: Dict[str, object], *, x_cm1: float, y_value: float, xytext: Tuple[float, float]) -> None:
        ws = self._current_workspace()
        if ws is None:
            return
        ds = next((d for d in ws.datasets if d.id == ws.active_dataset_id), None)
        if ds is None and ws.datasets:
            ds = ws.datasets[-1]
        if ds is None:
            return
        ann = FTIRBondAnnotation(
            dataset_id=str(opts.get("target_dataset_id") or ds.id),
            text=str(opts.get("text") or ""),
            x_cm1=float(x_cm1),
            y_value=float(y_value),
            xytext=(float(xytext[0]), float(xytext[1])),
            show_vline=bool(opts.get("show_vline", True)),
            line_color=str(opts.get("line_color") or "#444444"),
            text_color=str(opts.get("text_color") or "#111111"),
            fontsize=int(opts.get("fontsize") or 9),
            rotation=int(opts.get("rotation") or 0),
            preset_id=(None if not opts.get("preset_id") else str(opts.get("preset_id"))),
        )
        ds.bond_annotations.append(ann)

    def _bond_autoplace(self, ds: FTIRDataset, opts: Dict[str, object]) -> None:
        x_peak = None
        y_peak = None
        try:
            peaks = list(getattr(ds, "peaks", []) or [])
            if peaks:
                best = None
                best_prom = -float("inf")
                for p in peaks:
                    try:
                        prom = float(p.get("prominence", 0.0) or 0.0)
                    except Exception:
                        prom = 0.0
                    if prom > best_prom:
                        best_prom = prom
                        best = p
                if isinstance(best, dict):
                    x_peak = float(best.get("wn"))
                    y_peak = float(best.get("y_display", best.get("y", 0.0)))
        except Exception:
            pass

        if x_peak is None or y_peak is None:
            try:
                x = np.asarray(ds.x_disp, dtype=float)
                y = np.asarray(ds.y_disp, dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]
                if x.size >= 2:
                    order = np.argsort(x)
                    x = x[order]
                    y = y[order]
                    idx = int(np.nanargmax(y))
                    x_peak = float(x[idx])
                    y_peak = float(y[idx])
            except Exception:
                x_peak = 0.0
                y_peak = 0.0

        self._bond_add_annotation(opts, x_cm1=float(x_peak or 0.0), y_value=float(y_peak or 0.0), xytext=(float(x_peak or 0.0), float(y_peak or 0.0)))

    def _on_mpl_draw_event(self, _evt) -> None:
        return

    def _on_peak_pick_event(self, evt) -> None:
        try:
            artist = evt.artist
            if artist not in self._peak_artist_to_info:
                return
            info = self._peak_artist_to_info.get(artist)
            if not info:
                return
        except Exception:
            return

        button = None
        try:
            button = int(getattr(evt.mouseevent, "button", 1) or 1)
        except Exception:
            button = 1

        ws_id, ds_id, pid = info
        ds = self._get_dataset_by_key((str(ws_id), str(ds_id)))
        if ds is None:
            return

        if button == 3:
            try:
                suppressed = set(getattr(ds, "peak_suppressed", set()) or set())
                if pid in suppressed:
                    suppressed.remove(pid)
                else:
                    suppressed.add(pid)
                ds.peak_suppressed = suppressed
            except Exception:
                pass
            self._render_plot()
            return

        current = str((getattr(ds, "peak_label_overrides", {}) or {}).get(pid, ""))
        label, ok = QInputDialog.getText(self, "Peak Label", "Label (empty to reset):", text=current)
        if not ok:
            return
        label = str(label or "").strip()
        overrides = dict(getattr(ds, "peak_label_overrides", {}) or {})
        if not label:
            overrides.pop(pid, None)
        else:
            overrides[pid] = label
        ds.peak_label_overrides = overrides
        self._render_plot()

    def _on_peak_drag_press(self, evt) -> None:
        if self._bond_place_active:
            try:
                if evt is None or int(getattr(evt, "button", 0) or 0) != 1:
                    return
                if getattr(evt, "inaxes", None) is None:
                    return
                if evt.xdata is None:
                    return
            except Exception:
                return
            ws = self._current_workspace()
            if ws is None or not ws.datasets:
                return
            ds = next((d for d in ws.datasets if d.id == ws.active_dataset_id), None)
            if ds is None:
                ds = ws.datasets[-1]
            x = float(evt.xdata)
            y = self._interp_y_at(ds, x)
            opts = dict(self._bond_place_opts or {})
            self._bond_add_annotation(opts, x_cm1=float(x), y_value=float(y), xytext=(float(x), float(y)))
            self._bond_place_active = False
            self._bond_place_opts = {}
            try:
                self._status_label.setText("Bond label placed.")
            except Exception:
                pass
            self._render_plot()
            return

        try:
            if evt is None or int(getattr(evt, "button", 0) or 0) != 1:
                return
            if getattr(evt, "inaxes", None) is None:
                return
            if evt.xdata is None or evt.ydata is None:
                return
        except Exception:
            return

        for t in list(self._bond_texts):
            try:
                contains, _ = t.contains(evt)
                if not contains:
                    continue
                info = self._bond_artist_to_info.get(t)
                if not info:
                    continue
                ds_id, _target, idx = str(info[0]), str(info[1]), int(info[2])
                x0, y0 = t.get_position()
                self._drag_bond_key = (ds_id, "bond")
                self._drag_bond_idx = int(idx)
                self._drag_bond_artist = t
                self._drag_bond_dx = float(x0) - float(evt.xdata)
                self._drag_bond_dy = float(y0) - float(evt.ydata)
                return
            except Exception:
                continue

        hit_artist = None
        hit_info: Optional[Tuple[str, str, str]] = None
        for t in list(self._peak_texts):
            try:
                contains, _ = t.contains(evt)
                if contains:
                    hit_artist = t
                    hit_info = self._peak_artist_to_info.get(t)
                    break
            except Exception:
                continue

        if not hit_artist or not hit_info:
            return
        try:
            x0, y0 = hit_artist.get_position()
            self._drag_peak_key = (str(hit_info[0]), str(hit_info[1]))
            self._drag_peak_id = str(hit_info[2])
            self._drag_peak_artist = hit_artist
            self._drag_dx = float(x0) - float(evt.xdata)
            self._drag_dy = float(y0) - float(evt.ydata)
        except Exception:
            self._drag_peak_id = None
            self._drag_peak_key = None
            self._drag_peak_artist = None
            self._drag_dx = 0.0
            self._drag_dy = 0.0

    def _on_peak_drag_motion(self, evt) -> None:
        if self._drag_bond_idx is not None and self._drag_bond_artist is not None:
            try:
                if evt is None or getattr(evt, "inaxes", None) is None:
                    return
                if evt.xdata is None or evt.ydata is None:
                    return
            except Exception:
                return
            try:
                new_x = float(evt.xdata) + float(self._drag_bond_dx)
                new_y = float(evt.ydata) + float(self._drag_bond_dy)
                self._drag_bond_artist.set_position((new_x, new_y))
            except Exception:
                return
            try:
                ws = self._current_workspace()
                if ws is None:
                    return
                ds = next((d for d in ws.datasets if d.id == ws.active_dataset_id), None)
                if ds is None and ws.datasets:
                    ds = ws.datasets[-1]
                if ds is None:
                    return
                idx = int(self._drag_bond_idx)
                anns = list(getattr(ds, "bond_annotations", []) or [])
                if 0 <= idx < len(anns):
                    ann = anns[idx]
                    ann.x_cm1 = float(new_x)
                    ann.y_value = float(new_y)
                    ann.xytext = (float(new_x), float(new_y))
                    for ln in list(self._bond_vlines):
                        try:
                            inf = self._bond_artist_to_info.get(ln)
                            if not inf:
                                continue
                            if int(inf[2]) != int(idx):
                                continue
                            ln.set_xdata([float(new_x), float(new_x)])
                        except Exception:
                            continue
            except Exception:
                pass
            try:
                self._canvas.draw_idle()
            except Exception:
                pass
            return

        if not self._drag_peak_id or self._drag_peak_artist is None or self._drag_peak_key is None:
            return
        try:
            if evt is None or getattr(evt, "inaxes", None) is None:
                return
            if evt.xdata is None or evt.ydata is None:
                return
        except Exception:
            return

        try:
            new_x = float(evt.xdata) + float(self._drag_dx)
            new_y = float(evt.ydata) + float(self._drag_dy)
            self._drag_peak_artist.set_position((new_x, new_y))
        except Exception:
            return

        try:
            d = self._get_dataset_by_key((str(self._drag_peak_key[0]), str(self._drag_peak_key[1])))
            if d is not None:
                d.peak_label_positions[str(self._drag_peak_id)] = (float(new_x), float(new_y))
        except Exception:
            pass

        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _on_peak_drag_release(self, _evt) -> None:
        self._drag_bond_key = None
        self._drag_bond_idx = None
        self._drag_bond_artist = None
        self._drag_bond_dx = 0.0
        self._drag_bond_dy = 0.0

        self._drag_peak_id = None
        self._drag_peak_key = None
        self._drag_peak_artist = None
        self._drag_dx = 0.0
        self._drag_dy = 0.0

    def _on_ftir_keypress(self, evt) -> None:
        key = str(getattr(evt, "key", "") or "").lower()
        if key == "escape" and self._bond_place_active:
            self._bond_place_active = False
            self._bond_place_opts = {}
            try:
                self._status_label.setText("Bond placement cancelled.")
            except Exception:
                pass

    def open_workspace(self) -> Optional[str]:
        return self._load_workspace()

    def save_workspace(self) -> Optional[str]:
        return self._save_workspace()

    def open_workspace_path(self, path: str) -> None:
        if not path:
            return None
        self._load_workspace_path(str(path))

    def get_last_workspace_path(self) -> Optional[str]:
        return self._last_workspace_path

    def _encode_workspace(self) -> Dict[str, object]:
        ftir_files: List[Dict[str, object]] = []
        seen: Dict[str, FTIRDataset] = {}
        for ws in self._workspaces.values():
            for ds in ws.datasets:
                if ds.id in seen:
                    continue
                seen[ds.id] = ds
                ftir_files.append(
                    {
                        "id": ds.id,
                        "path": (str(ds.path) if ds.path is not None else ""),
                        "name": str(ds.name),
                        "y_mode": str(getattr(ds, "y_mode", "absorbance")),
                        "x_units": (None if ds.x_units is None else str(ds.x_units)),
                        "y_units": (None if ds.y_units is None else str(ds.y_units)),
                        "peak_settings": dict(getattr(ds, "peak_settings", {}) or {}),
                        "peaks": list(getattr(ds, "peaks", []) or []),
                        "peak_label_overrides": dict(getattr(ds, "peak_label_overrides", {}) or {}),
                        "peak_suppressed": sorted(list(getattr(ds, "peak_suppressed", set()) or set())),
                        "peak_label_positions": {
                            str(k): [float(v[0]), float(v[1])]
                            for k, v in (getattr(ds, "peak_label_positions", {}) or {}).items()
                            if isinstance(v, (list, tuple)) and len(v) == 2
                        },
                        "bond_annotations": [
                            {
                                "dataset_id": str(a.dataset_id),
                                "text": str(a.text),
                                "x_cm1": float(a.x_cm1),
                                "y_value": float(a.y_value),
                                "xytext": [float(a.xytext[0]), float(a.xytext[1])],
                                "show_vline": bool(a.show_vline),
                                "line_color": str(a.line_color),
                                "text_color": str(a.text_color),
                                "fontsize": int(a.fontsize),
                                "rotation": int(a.rotation),
                                "preset_id": (None if a.preset_id is None else str(a.preset_id)),
                            }
                            for a in (getattr(ds, "bond_annotations", []) or [])
                        ],
                    }
                )

        ftir_workspaces: List[Dict[str, object]] = []
        for wid in self._workspace_order:
            ws = self._workspaces.get(wid)
            if ws is None:
                continue
            ftir_workspaces.append(
                {
                    "id": ws.id,
                    "name": ws.name,
                    "dataset_ids": [str(d.id) for d in ws.datasets],
                    "active_dataset_id": (None if not ws.active_dataset_id else str(ws.active_dataset_id)),
                    "line_color": (None if ws.line_color is None else str(ws.line_color)),
                    "bond_annotations": [],
                }
            )

        overlay_rows: List[Dict[str, object]] = []
        for gid in self._overlay_order:
            g = self._overlay_groups.get(gid)
            if g is None:
                continue
            per_style: Dict[str, object] = {}
            for k, st in (g.per_member_style or {}).items():
                key = f"{k[0]}::{k[1]}"
                per_style[key] = {"linewidth": float(getattr(st, "linewidth", 1.2) or 1.2)}
            overlay_rows.append(
                {
                    "group_id": g.group_id,
                    "name": g.name,
                    "created_at": float(getattr(g, "created_at", 0.0) or 0.0),
                    "members": [[a, b] for (a, b) in (g.members or [])],
                    "active_member": ([str(g.active_member[0]), str(g.active_member[1])] if g.active_member else None),
                    "per_member_style": per_style,
                }
            )

        return {
            "schema_version": 1,
            "app": "MFP lab analysis tool",
            "created_utc": self._utc_now_iso(),
            "ftir": {
                "ftir_files": ftir_files,
                "ftir_workspaces": ftir_workspaces,
                "active_ftir_workspace_id": self._active_workspace_id,
                "ftir_overlay_groups": overlay_rows,
                "active_ftir_overlay_group_id": self._active_overlay_group_id,
                "active_ftir_id": None,
                "ui": {
                    "reverse_pref_by_id": {},
                    "show_peaks_all_overlay": bool(self._show_peaks_all_overlay),
                },
            },
        }

    def _save_workspace(self) -> Optional[str]:
        if not self._workspaces or not any(ws.datasets for ws in self._workspaces.values()):
            self.dialogs.info("FTIR", "No FTIR datasets to save.")
            return None
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FTIR Workspace",
            "",
            "FTIR Workspace JSON (*.json);;All files (*.*)",
        )
        if not path:
            return None
        payload = self._encode_workspace()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.dialogs.error("FTIR", f"Failed to save workspace:\n\n{exc}")
            return None
        self._last_workspace_path = str(path)
        self.status.set_status("FTIR workspace saved")
        return str(path)

    def _load_workspace(self) -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load FTIR Workspace",
            "",
            "FTIR Workspace JSON (*.json);;All files (*.*)",
        )
        if not path:
            return None
        return self._load_workspace_path(str(path))

    def _load_workspace_path(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            self.dialogs.error("FTIR", f"Failed to read workspace JSON:\n\n{exc}")
            return None

        if not isinstance(payload, dict):
            self.dialogs.error("FTIR", "Workspace JSON must be an object.")
            return None

        ft = payload.get("ftir") if isinstance(payload.get("ftir"), dict) else payload
        if not isinstance(ft, dict):
            self.dialogs.error("FTIR", "Workspace JSON missing FTIR data.")
            return None

        ftir_files = ft.get("ftir_files") if isinstance(ft.get("ftir_files"), list) else []
        ftir_workspaces = ft.get("ftir_workspaces") if isinstance(ft.get("ftir_workspaces"), list) else []
        active_ws_id = ft.get("active_ftir_workspace_id")
        overlay_groups = ft.get("ftir_overlay_groups") if isinstance(ft.get("ftir_overlay_groups"), list) else []
        active_overlay_id = ft.get("active_ftir_overlay_group_id")
        ui_state = ft.get("ui") if isinstance(ft.get("ui"), dict) else {}

        def _work(_h):
            loaded: Dict[str, FTIRDataset] = {}
            missing: List[str] = []
            for row in ftir_files:
                if not isinstance(row, dict):
                    continue
                p = str(row.get("path") or "")
                if not p:
                    continue
                path_obj = Path(p)
                if not path_obj.exists():
                    missing.append(p)
                    continue
                x, y, meta = _parse_ftir_xy_numpy(str(path_obj))
                ds = FTIRDataset(
                    id=str(row.get("id") or uuid.uuid4()),
                    name=str(row.get("name") or path_obj.name),
                    path=path_obj,
                    x_full=np.asarray(x, dtype=float),
                    y_full=np.asarray(y, dtype=float),
                    x_disp=np.asarray(x, dtype=float),
                    y_disp=np.asarray(y, dtype=float),
                    x_units=(row.get("x_units") or meta.get("XUNITS")),
                    y_units=(row.get("y_units") or meta.get("YUNITS")),
                    loaded_at_utc=self._utc_now_iso(),
                )
                ds.peak_settings = dict(row.get("peak_settings") or {}) if isinstance(row.get("peak_settings"), dict) else {}
                ds.peaks = list(row.get("peaks") or []) if isinstance(row.get("peaks"), list) else []
                ds.peak_label_overrides = dict(row.get("peak_label_overrides") or {}) if isinstance(row.get("peak_label_overrides"), dict) else {}
                try:
                    ds.peak_suppressed = set(row.get("peak_suppressed") or [])
                except Exception:
                    ds.peak_suppressed = set()
                pos = row.get("peak_label_positions") if isinstance(row.get("peak_label_positions"), dict) else {}
                ds.peak_label_positions = {}
                for k, v in pos.items():
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        try:
                            ds.peak_label_positions[str(k)] = (float(v[0]), float(v[1]))
                        except Exception:
                            continue
                bond_rows = row.get("bond_annotations") if isinstance(row.get("bond_annotations"), list) else []
                for br in bond_rows:
                    if not isinstance(br, dict):
                        continue
                    try:
                        xy = br.get("xytext")
                        if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                            xy = (float(br.get("x_cm1")), float(br.get("y_value")))
                        ds.bond_annotations.append(
                            FTIRBondAnnotation(
                                dataset_id=str(br.get("dataset_id") or ds.id),
                                text=str(br.get("text") or ""),
                                x_cm1=float(br.get("x_cm1")),
                                y_value=float(br.get("y_value")),
                                xytext=(float(xy[0]), float(xy[1])),
                                show_vline=bool(br.get("show_vline", False)),
                                line_color=str(br.get("line_color") or "#444444"),
                                text_color=str(br.get("text_color") or "#111111"),
                                fontsize=int(br.get("fontsize", 9) or 9),
                                rotation=int(br.get("rotation", 0) or 0),
                                preset_id=(None if not br.get("preset_id") else str(br.get("preset_id"))),
                            )
                        )
                    except Exception:
                        continue
                loaded[ds.id] = ds
            return loaded, missing

        def _done(res) -> None:
            loaded, missing = res
            self._workspaces = {}
            self._workspace_order = []
            self._overlay_groups = {}
            self._overlay_order = []
            self._active_overlay_group_id = None

            for ws_row in ftir_workspaces:
                if not isinstance(ws_row, dict):
                    continue
                wid = str(ws_row.get("id") or uuid.uuid4())
                ws = FTIRWorkspace(id=wid, name=str(ws_row.get("name") or "Workspace"))
                ds_ids = ws_row.get("dataset_ids") if isinstance(ws_row.get("dataset_ids"), list) else []
                for did in ds_ids:
                    if str(did) in loaded:
                        ws.datasets.append(loaded[str(did)])
                ws.active_dataset_id = (None if not ws_row.get("active_dataset_id") else str(ws_row.get("active_dataset_id")))
                ws.line_color = (None if ws_row.get("line_color") is None else str(ws_row.get("line_color")))
                self._workspaces[wid] = ws
                self._workspace_order.append(wid)

            if not self._workspaces and loaded:
                wid = str(uuid.uuid4())
                ws = FTIRWorkspace(id=wid, name="Default")
                ws.datasets = list(loaded.values())
                ws.active_dataset_id = ws.datasets[-1].id if ws.datasets else None
                self._workspaces[wid] = ws
                self._workspace_order.append(wid)

            if active_ws_id and str(active_ws_id) in self._workspaces:
                self._active_workspace_id = str(active_ws_id)
            elif self._workspace_order:
                self._active_workspace_id = self._workspace_order[0]

            for row in overlay_groups:
                if not isinstance(row, dict):
                    continue
                gid = str(row.get("group_id") or uuid.uuid4())
                g = OverlayGroup(group_id=gid, name=str(row.get("name") or "Overlay"))
                members = row.get("members") if isinstance(row.get("members"), list) else []
                g.members = [(str(a), str(b)) for (a, b) in members if isinstance(a, str) and isinstance(b, str)]
                am = row.get("active_member")
                if isinstance(am, (list, tuple)) and len(am) == 2:
                    g.active_member = (str(am[0]), str(am[1]))
                per_style = row.get("per_member_style") if isinstance(row.get("per_member_style"), dict) else {}
                for k, v in per_style.items():
                    if "::" in str(k):
                        ws_id, ds_id = str(k).split("::", 1)
                        try:
                            lw = float(v.get("linewidth") if isinstance(v, dict) else 1.2)
                        except Exception:
                            lw = 1.2
                        g.per_member_style[(ws_id, ds_id)] = StyleState(linewidth=lw)
                self._overlay_groups[gid] = g
                self._overlay_order.append(gid)

            if active_overlay_id and str(active_overlay_id) in self._overlay_groups:
                self._active_overlay_group_id = str(active_overlay_id)
            elif self._overlay_order:
                self._active_overlay_group_id = self._overlay_order[0]

            try:
                self._show_peaks_all_overlay = bool(ui_state.get("show_peaks_all_overlay", self._show_peaks_all_overlay))
            except Exception:
                self._show_peaks_all_overlay = bool(self._show_peaks_all_overlay)
            try:
                ws = self._workspaces.get(self._active_workspace_id or "")
                if ws is not None:
                    ds = next((d for d in ws.datasets if d.id == ws.active_dataset_id), None)
                    if ds is not None:
                        self._show_peaks = bool(ds.peak_settings.get("enabled", self._show_peaks))
            except Exception:
                pass
            try:
                if getattr(self, "_peaks_all_cb", None) is not None:
                    self._peaks_all_cb.blockSignals(True)
                    self._peaks_all_cb.setChecked(bool(self._show_peaks_all_overlay))
                    self._peaks_all_cb.blockSignals(False)
            except Exception:
                pass

            self._refresh_workspace_combo()
            self._refresh_dataset_tree()
            self._refresh_overlay_group_list()
            self._refresh_overlay_members_list()
            self._render_plot()
            self._last_workspace_path = str(path)
            if missing:
                self.dialogs.warn("FTIR", "Some files were missing and skipped:\n\n" + "\n".join(missing[:12]))
            self.status.set_status("FTIR workspace loaded")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Loading FTIR workspace",
            group="ftir_workspace",
            cancel_previous=True,
        )
        return str(path)
