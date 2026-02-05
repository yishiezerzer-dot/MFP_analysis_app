from __future__ import annotations

import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyteomics import mzml
import pyqtgraph as pg

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QColorDialog,
    QFileDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lab_gui.lcms_io import MzMLTICIndex, infer_uv_columns, parse_uv_arrays
from lab_gui.lcms_model import MzMLSession, UVSession
import lab_gui.lcms_polymer_match as poly_match
from qt_app.services import DialogService, StatusService
from qt_app.services.worker import run_in_worker
from qt_app.widgets import PlotPanel


class LCMSTab(QWidget):
    def __init__(self, status: StatusService, dialogs: DialogService, worker_runner=None) -> None:
        super().__init__()
        self.status = status
        self.dialogs = dialogs
        self.worker_runner = worker_runner or run_in_worker
        self._sessions: Dict[str, MzMLSession] = {}
        self._session_order: List[str] = []
        self._active_session_id: Optional[str] = None

        self._uv_sessions: Dict[str, UVSession] = {}
        self._uv_order: List[str] = []
        self._active_uv_id: Optional[str] = None

        self._current_spectrum_id: Optional[str] = None
        self._current_rt_min: Optional[float] = None
        self._last_workspace_path: Optional[str] = None

        self._overlay_session_ids: List[str] = []
        self._overlay_mode: str = "Stacked"
        self._overlay_scheme: str = "Auto (Tableau)"
        self._overlay_single_hue: str = "#1f77b4"
        self._overlay_show_uv: bool = True
        self._overlay_stack_spectra: bool = False
        self._show_tic: bool = True
        self._show_uv: bool = True
        self._show_spec: bool = True
        self._polarity_filter: str = "all"
        self._uv_rt_offset_min: float = 0.0
        self._annotate_enabled: bool = False
        self._annotate_top_n: int = 6
        self._annotate_min_rel: float = 0.05
        self._uv_align_enabled: bool = False
        self._tic_region_select_enabled: bool = False
        self._tic_region: Optional[Tuple[float, float]] = None
        self._tic_region_cb: Optional[QCheckBox] = None
        self._plot_grid_enabled: bool = True
        self._plot_grid_alpha: float = 0.2
        self._tic_line_width: float = 1.2
        self._uv_line_width: float = 1.2
        self._spec_line_width: float = 1.2

        self._annotate_cb: Optional[QCheckBox] = None
        self._annotate_topn_edit: Optional[QLineEdit] = None
        self._annotate_minrel_edit: Optional[QLineEdit] = None

        self._poly_enabled: bool = False
        self._poly_monomers_text: str = ""
        self._poly_bond_delta: float = -18.010565
        self._poly_extra_delta: float = 0.0
        self._poly_adduct_mass: float = 1.007276
        self._poly_cluster_adduct_mass: float = -1.007276
        self._poly_adduct_na: bool = False
        self._poly_adduct_k: bool = False
        self._poly_adduct_cl: bool = False
        self._poly_adduct_formate: bool = False
        self._poly_adduct_acetate: bool = False
        self._poly_charges_text: str = "1"
        self._poly_decarb_enabled: bool = False
        self._poly_oxid_enabled: bool = False
        self._poly_cluster_enabled: bool = False
        self._poly_max_dp: int = 12
        self._poly_tol_value: float = 0.02
        self._poly_tol_unit: str = "Da"
        self._poly_min_rel_int: float = 0.01

        self._poly_enable_cb: Optional[QCheckBox] = None

        self._rt_jump_edit = QLineEdit()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._build_ui())

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
        try:
            root.setSizes([320, 980])
        except Exception:
            pass
        return root

    def reset_layout(self) -> None:
        try:
            if getattr(self, "_root_splitter", None) is not None:
                self._root_splitter.setSizes([320, 980])
        except Exception:
            pass
        try:
            self._reset_view_all()
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
        layout.addWidget(self._build_quick_actions())
        layout.addWidget(self._build_advanced_block())
        layout.addStretch(1)

        return scroll

    def _build_workspace_block(self) -> QWidget:
        ws = QGroupBox("Workspace")
        v = QVBoxLayout(ws)

        self._ws_tree = QTreeWidget()
        self._ws_tree.setHeaderLabels(["Overlay", "", "Color", "mzML", "MS1", "Pol"])
        self._ws_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._ws_tree.itemSelectionChanged.connect(self._on_ws_select)
        self._ws_tree.itemChanged.connect(self._on_ws_item_changed)
        v.addWidget(self._ws_tree)

        btns = QHBoxLayout()
        b_add = QPushButton("Add mzML…")
        b_add.clicked.connect(self._open_mzml_many)
        btns.addWidget(b_add)
        b_remove = QPushButton("Remove mzML")
        b_remove.clicked.connect(self._remove_selected_mzml)
        btns.addWidget(b_remove)
        btns.addStretch(1)
        v.addLayout(btns)

        self._uv_tree = QTreeWidget()
        self._uv_tree.setHeaderLabels(["Linked to", "UV CSV", "RT range", "Signal"])
        self._uv_tree.itemSelectionChanged.connect(self._on_uv_select)
        v.addWidget(self._uv_tree)

        uv_btns = QHBoxLayout()
        b_uv_add = QPushButton("Add UV CSV…")
        b_uv_add.clicked.connect(self._open_uv_many)
        uv_btns.addWidget(b_uv_add)
        b_uv_remove = QPushButton("Remove UV")
        b_uv_remove.clicked.connect(self._remove_selected_uv)
        uv_btns.addWidget(b_uv_remove)
        b_uv_link = QPushButton("Link UV to mzML")
        b_uv_link.clicked.connect(self._link_selected_uv_to_selected_mzml)
        uv_btns.addWidget(b_uv_link)
        b_uv_auto = QPushButton("Auto-link by name")
        b_uv_auto.clicked.connect(self._auto_link_uv_by_name)
        uv_btns.addWidget(b_uv_auto)
        uv_btns.addStretch(1)
        v.addLayout(uv_btns)

        return ws

    def _build_quick_actions(self) -> QWidget:
        quick = QGroupBox("Quick Actions")
        v = QVBoxLayout(quick)
        b_eic = QPushButton("EIC (new chromatogram)…")
        b_eic.clicked.connect(self._open_sim_dialog)
        v.addWidget(b_eic)
        b_jump = QPushButton("Jump to m/z…")
        b_jump.clicked.connect(self._open_jump_to_mz_dialog)
        v.addWidget(b_jump)
        b_export = QPushButton("Export labels (all scans)…")
        b_export.clicked.connect(self._export_all_labels_xlsx)
        v.addWidget(b_export)
        return quick

    def _build_advanced_block(self) -> QWidget:
        adv = QGroupBox("Advanced")
        v = QVBoxLayout(adv)

        tabs = QTabWidget()
        tabs.addTab(self._build_nav_tab(), "Navigate")
        tabs.addTab(self._build_view_tab(), "View")
        tabs.addTab(self._build_annotate_tab(), "Annotate")
        tabs.addTab(self._build_polymer_tab(), "Polymer")
        v.addWidget(tabs)

        return adv

    def _build_nav_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        nav = QGroupBox("Spectrum")
        nav_grid = QHBoxLayout(nav)
        b_prev = QPushButton("◀ Prev")
        b_prev.clicked.connect(lambda: self._step_spectrum(-1))
        nav_grid.addWidget(b_prev)
        b_next = QPushButton("Next ▶")
        b_next.clicked.connect(lambda: self._step_spectrum(1))
        nav_grid.addWidget(b_next)
        b_first = QPushButton("First")
        b_first.clicked.connect(lambda: self._go_to_index(0))
        nav_grid.addWidget(b_first)
        b_last = QPushButton("Last")
        b_last.clicked.connect(self._go_last)
        nav_grid.addWidget(b_last)
        v.addWidget(nav)

        b_find = QPushButton("Find m/z…")
        b_find.clicked.connect(self._open_find_mz_dialog)
        v.addWidget(b_find)

        b_align = QPushButton("Auto-align UV↔MS")
        b_align.clicked.connect(self._auto_align_uv_ms)
        v.addWidget(b_align)

        b_diag = QPushButton("Alignment Diagnostics…")
        b_diag.clicked.connect(self._open_alignment_diagnostics)
        v.addWidget(b_diag)

        b_sim = QPushButton("EIC…")
        b_sim.clicked.connect(self._open_sim_dialog)
        v.addWidget(b_sim)

        jump = QGroupBox("Jump")
        form = QFormLayout(jump)
        self._rt_jump_edit.setPlaceholderText("RT (min)")
        form.addRow("RT (min)", self._rt_jump_edit)
        b_go = QPushButton("Go")
        b_go.clicked.connect(self._jump_to_rt)
        form.addRow("", b_go)
        v.addWidget(jump)

        return w

    def _build_view_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        filters = QGroupBox("Filters")
        f_form = QFormLayout(filters)
        rt_unit = QComboBox()
        rt_unit.addItems(["minutes", "seconds"])
        f_form.addRow("RT unit", rt_unit)
        v.addWidget(filters)

        pol = QGroupBox("Polarity")
        pol_row = QHBoxLayout(pol)
        self._pol_all_cb = QCheckBox("All")
        self._pol_pos_cb = QCheckBox("Positive")
        self._pol_neg_cb = QCheckBox("Negative")
        self._pol_all_cb.setChecked(True)
        self._pol_all_cb.toggled.connect(lambda v: self._set_polarity_filter("all") if v else None)
        self._pol_pos_cb.toggled.connect(lambda v: self._set_polarity_filter("positive") if v else None)
        self._pol_neg_cb.toggled.connect(lambda v: self._set_polarity_filter("negative") if v else None)
        pol_row.addWidget(self._pol_all_cb)
        pol_row.addWidget(self._pol_pos_cb)
        pol_row.addWidget(self._pol_neg_cb)
        v.addWidget(pol)

        off = QGroupBox("UV↔MS alignment")
        off_form = QFormLayout(off)
        self._uv_offset_edit = QLineEdit()
        self._uv_offset_edit.setPlaceholderText("0.0")
        off_form.addRow("Offset (min)", self._uv_offset_edit)
        b_apply = QPushButton("Apply")
        b_apply.clicked.connect(self._apply_uv_ms_offset)
        off_form.addRow("", b_apply)
        cb_auto = QCheckBox("Enable auto-align")
        cb_auto.toggled.connect(self._on_uv_ms_align_enabled_changed)
        off_form.addRow("", cb_auto)
        v.addWidget(off)

        more = QHBoxLayout()
        b_graph = QPushButton("Graph Settings…")
        b_graph.clicked.connect(self._open_graph_settings)
        more.addWidget(b_graph)
        more.addStretch(1)
        v.addLayout(more)

        panels = QGroupBox("Panels")
        p_row = QVBoxLayout(panels)
        self._show_tic_cb = QCheckBox("Show TIC")
        self._show_uv_cb = QCheckBox("Show UV")
        self._show_spec_cb = QCheckBox("Show Spectrum")
        self._show_tic_cb.setChecked(True)
        self._show_uv_cb.setChecked(True)
        self._show_spec_cb.setChecked(True)
        self._show_tic_cb.toggled.connect(self._on_panels_changed)
        self._show_uv_cb.toggled.connect(self._on_panels_changed)
        self._show_spec_cb.toggled.connect(self._on_panels_changed)
        p_row.addWidget(self._show_tic_cb)
        p_row.addWidget(self._show_spec_cb)
        p_row.addWidget(self._show_uv_cb)
        v.addWidget(panels)

        region = QGroupBox("TIC region")
        r_row = QVBoxLayout(region)
        cb_region = QCheckBox("Region Select (drag on TIC)")
        cb_region.toggled.connect(self._on_tic_region_select_changed)
        r_row.addWidget(cb_region)
        self._tic_region_cb = cb_region
        b_clear = QPushButton("Clear Region")
        b_clear.clicked.connect(self._clear_tic_region_selection)
        r_row.addWidget(b_clear)
        v.addWidget(region)

        v.addStretch(1)
        return w

    def _build_annotate_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        spec = QGroupBox("Spectrum labels")
        s_form = QFormLayout(spec)
        self._annotate_cb = QCheckBox("Annotate spectrum peaks with m/z")
        self._annotate_cb.toggled.connect(self._on_annotate_settings_changed)
        s_form.addRow(self._annotate_cb)
        self._annotate_topn_edit = QLineEdit()
        self._annotate_topn_edit.setPlaceholderText("6")
        self._annotate_topn_edit.textChanged.connect(self._on_annotate_settings_changed)
        s_form.addRow("Top N", self._annotate_topn_edit)
        self._annotate_minrel_edit = QLineEdit()
        self._annotate_minrel_edit.setPlaceholderText("0.05")
        self._annotate_minrel_edit.textChanged.connect(self._on_annotate_settings_changed)
        s_form.addRow("Min rel", self._annotate_minrel_edit)
        cb_drag = QCheckBox("Enable dragging labels with mouse")
        s_form.addRow(cb_drag)
        v.addWidget(spec)

        uv = QGroupBox("UV labels")
        u_form = QFormLayout(uv)
        cb_uv = QCheckBox("Transfer top MS peaks to UV labels at selected RT")
        cb_uv.toggled.connect(lambda _v: self._apply_quick_annotate_settings())
        u_form.addRow(cb_uv)
        u_form.addRow("How many peaks", QComboBox())
        v.addWidget(uv)

        btns = QHBoxLayout()
        b_ann = QPushButton("Annotate Peaks…")
        b_ann.clicked.connect(self._open_annotation_settings)
        btns.addWidget(b_ann)
        b_custom = QPushButton("Custom Labels…")
        b_custom.clicked.connect(self._open_custom_labels)
        btns.addWidget(b_custom)
        btns.addStretch(1)
        v.addLayout(btns)

        overlay = QGroupBox("Overlay labels")
        o_row = QVBoxLayout(overlay)
        cb_all = QCheckBox("Show labels for all overlayed spectra")
        cb_all.toggled.connect(lambda _v: self._refresh_overlay_view())
        o_row.addWidget(cb_all)
        cb_multi = QCheckBox("Multi-drag labels across overlay")
        o_row.addWidget(cb_multi)
        v.addWidget(overlay)

        v.addStretch(1)
        return w

    def _build_polymer_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self._poly_enable_cb = QCheckBox("Enable polymer/reaction matching")
        self._poly_enable_cb.setChecked(bool(self._poly_enabled))
        self._poly_enable_cb.toggled.connect(self._on_poly_enabled_changed)
        v.addWidget(self._poly_enable_cb)
        v.addWidget(QLabel("Use Polymer Match… for full settings"))
        b_poly = QPushButton("Polymer Match…")
        b_poly.clicked.connect(self._open_polymer_match)
        v.addWidget(b_poly)
        v.addStretch(1)
        return w

    def _build_right_panel(self) -> QWidget:
        right = QWidget()
        v = QVBoxLayout(right)
        v.setContentsMargins(6, 6, 6, 6)

        toolbar = self._build_toolbar()
        v.addWidget(toolbar)

        self._overlay_legend = QTreeWidget()
        self._overlay_legend.setHeaderLabels(["Color", "mzML", "MS1", "Pol", "Status"])
        v.addWidget(self._overlay_legend)

        self._tic_plot = PlotPanel(
            on_status=self.status.set_status,
            on_click=lambda x, y, b, d: self._on_plot_click(axis="tic", x=x, y=y, button=b, dblclick=d),
            on_move=lambda x, y: self._on_plot_move(axis="tic", x=x, y=y),
            on_release=lambda x, y: self._on_plot_release(axis="tic", x=x, y=y),
        )
        self._tic_plot.set_title("TIC")
        self._tic_plot.set_labels("RT (min)", "TIC")

        self._uv_plot = PlotPanel(
            on_status=self.status.set_status,
            on_click=lambda x, y, b, d: self._on_plot_click(axis="uv", x=x, y=y, button=b, dblclick=d),
            on_move=lambda x, y: self._on_plot_move(axis="uv", x=x, y=y),
            on_release=lambda x, y: self._on_plot_release(axis="uv", x=x, y=y),
        )
        self._uv_plot.set_title("UV")
        self._uv_plot.set_labels("RT (min)", "Signal")

        self._spec_plot = PlotPanel(
            on_status=self.status.set_status,
            on_click=lambda x, y, b, d: self._on_plot_click(axis="spec", x=x, y=y, button=b, dblclick=d),
            on_move=lambda x, y: self._on_plot_move(axis="spec", x=x, y=y),
            on_release=lambda x, y: self._on_plot_release(axis="spec", x=x, y=y),
        )
        self._spec_plot.set_title("Spectrum")
        self._spec_plot.set_labels("m/z", "Intensity")

        v.addWidget(self._tic_plot, 1)
        v.addWidget(self._uv_plot, 1)
        v.addWidget(self._spec_plot, 2)

        self._apply_plot_style()

        return right

    def _build_toolbar(self) -> QWidget:
        bar = QGroupBox()
        h = QHBoxLayout(bar)

        b_open = QPushButton("Open mzML…")
        b_open.clicked.connect(self._open_mzml_single)
        h.addWidget(b_open)

        b_uv = QPushButton("Add UV…")
        b_uv.clicked.connect(self._open_uv_single)
        h.addWidget(b_uv)

        b_export = QToolButton()
        b_export.setText("Export…")
        b_export.setPopupMode(QToolButton.MenuButtonPopup)
        export_menu = QMenu(b_export)
        export_menu.addAction("Export Spectrum CSV…", self._export_current_spectrum)
        export_menu.addAction("Export Overlay TIC CSV…", self._export_overlay_tic_csv)
        export_menu.addAction("Export Overlay Spectra CSV…", self._export_overlay_spectra_csv)
        export_menu.addSeparator()
        export_menu.addAction("Save TIC Plot Image…", self._save_tic_plot)
        export_menu.addAction("Save UV Plot Image…", self._save_uv_plot)
        export_menu.addAction("Save Spectrum Plot Image…", self._save_spectrum_plot)
        export_menu.addSeparator()
        export_menu.addAction("Open TIC Window…", self._open_tic_window)
        export_menu.addAction("Open UV Window…", self._open_uv_window)
        export_menu.addAction("Open Spectrum Window…", self._open_spectrum_window)
        b_export.setMenu(export_menu)
        b_export.clicked.connect(self._export_current_spectrum)
        h.addWidget(b_export)

        b_reset = QPushButton("Reset View")
        b_reset.clicked.connect(self._reset_view_all)
        h.addWidget(b_reset)

        h.addSpacing(12)
        b_overlay = QPushButton("Overlay Selected")
        b_overlay.clicked.connect(self._start_overlay_selected)
        h.addWidget(b_overlay)
        b_clear = QPushButton("Clear Overlay")
        b_clear.clicked.connect(self._clear_overlay)
        h.addWidget(b_clear)

        h.addWidget(QLabel("Mode"))
        self._overlay_mode_cb = QComboBox()
        self._overlay_mode_cb.addItems(["Stacked", "Normalized", "Offset", "Percent of max"])
        self._overlay_mode_cb.currentIndexChanged.connect(self._on_overlay_mode_changed)
        h.addWidget(self._overlay_mode_cb)

        h.addWidget(QLabel("Colors"))
        self._overlay_colors_cb = QComboBox()
        self._overlay_colors_cb.addItems(self._overlay_scheme_options())
        self._overlay_colors_cb.currentIndexChanged.connect(self._on_overlay_scheme_changed)
        h.addWidget(self._overlay_colors_cb)

        b_pick = QPushButton("Pick hue…")
        b_pick.clicked.connect(self._pick_overlay_single_hue_color)
        h.addWidget(b_pick)

        self._overlay_show_uv_cb = QCheckBox("Show UV overlays")
        self._overlay_show_uv_cb.setChecked(True)
        self._overlay_show_uv_cb.toggled.connect(self._on_overlay_show_uv_changed)
        h.addWidget(self._overlay_show_uv_cb)
        self._overlay_stack_cb = QCheckBox("Stack spectra")
        self._overlay_stack_cb.toggled.connect(self._on_overlay_stack_changed)
        h.addWidget(self._overlay_stack_cb)
        self._overlay_persist_cb = QCheckBox("Persist overlay")
        h.addWidget(self._overlay_persist_cb)

        h.addStretch(1)
        h.addWidget(QLabel("Tip: Click TIC/UV to select RT • Right-click label to edit"))
        return bar

    def _on_double_click(self, _event) -> None:
        self._reset_view_all()

    def _open_mzml_single(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open mzML", "", "mzML (*.mzML *.mzml);;All files (*.*)")
        if not path:
            return
        self._load_mzml_paths([path])

    def _open_mzml_many(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open mzML", "", "mzML (*.mzML *.mzml);;All files (*.*)")
        if not paths:
            return
        self._load_mzml_paths(list(paths))

    def _load_mzml_paths(self, paths: List[str]) -> None:
        def _work(_h):
            loaded: List[MzMLSession] = []
            base_order = len(self._session_order)
            for i, p in enumerate(paths):
                mzml_path = Path(p)
                index = MzMLTICIndex(mzml_path, rt_unit="minutes")
                index.build()
                ms1 = list(index.ms1 or [])
                if not ms1:
                    continue
                pols = {m.polarity for m in ms1 if m.polarity}
                if not pols:
                    pol_summary = "unknown"
                elif len(pols) == 1:
                    pol_summary = str(next(iter(pols)))
                else:
                    pol_summary = "mixed"
                rt_vals = [float(m.rt_min) for m in ms1]
                rt_min = min(rt_vals) if rt_vals else None
                rt_max = max(rt_vals) if rt_vals else None
                display = mzml_path.name
                existing = {self._sessions[sid].display_name for sid in self._session_order if sid in self._sessions}
                if display in existing:
                    stem = mzml_path.stem
                    suffix = mzml_path.suffix
                    idx = 2
                    while f"{stem} ({idx}){suffix}" in existing:
                        idx += 1
                    display = f"{stem} ({idx}){suffix}"
                session_id = str(uuid.uuid4())
                loaded.append(
                    MzMLSession(
                        session_id=session_id,
                        path=mzml_path,
                        index=index,
                        load_order=base_order + i,
                        display_name=display,
                        custom_labels_by_spectrum={},
                        spec_label_overrides={},
                        ms1_count=int(len(ms1)),
                        rt_min=rt_min,
                        rt_max=rt_max,
                        polarity_summary=pol_summary,
                    )
                )
            return loaded

        def _done(loaded: List[MzMLSession]) -> None:
            if not loaded:
                self.dialogs.info("LCMS", "No usable MS1 spectra found in selected mzML files.")
                return
            for sess in loaded:
                self._sessions[sess.session_id] = sess
                self._session_order.append(sess.session_id)
            self._active_session_id = loaded[-1].session_id
            self._refresh_workspace_tree()
            self._render_active_plots()
            self.status.set_status(f"Loaded {len(loaded)} mzML file(s)")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Loading mzML",
            group="lcms_mzml",
            cancel_previous=True,
        )

    def _open_uv_single(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open UV CSV", "", "CSV (*.csv);;All files (*.*)")
        if not path:
            return
        self._load_uv_paths([path])

    def _open_uv_many(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open UV CSV", "", "CSV (*.csv);;All files (*.*)")
        if not paths:
            return
        self._load_uv_paths(list(paths))

    def _load_uv_paths(self, paths: List[str]) -> None:
        def _work(_h):
            loaded: List[UVSession] = []
            base_order = len(self._uv_order)
            for i, p in enumerate(paths):
                uv_path = Path(p)
                try:
                    df = pd.read_csv(uv_path, sep=None, engine="python")
                except Exception:
                    df = pd.read_table(uv_path)
                info = infer_uv_columns(df)
                rt_min, signal, rt_range, warnings = parse_uv_arrays(
                    df,
                    xcol=str(info.get("xcol")),
                    ycol=str(info.get("ycol")),
                    unit_guess=str(info.get("unit_guess")),
                )
                uv_id = str(uuid.uuid4())
                loaded.append(
                    UVSession(
                        uv_id=uv_id,
                        path=uv_path,
                        rt_min=rt_min,
                        signal=signal,
                        xcol=str(info.get("xcol")),
                        ycol=str(info.get("ycol")),
                        n_points=int(len(rt_min)),
                        rt_min_range=rt_range,
                        load_order=base_order + i,
                        import_warnings=list(warnings or []),
                    )
                )
            return loaded

        def _done(loaded: List[UVSession]) -> None:
            if not loaded:
                self.dialogs.info("LCMS", "No UV CSV data was loaded.")
                return
            for uv in loaded:
                self._uv_sessions[uv.uv_id] = uv
                self._uv_order.append(uv.uv_id)
            self._active_uv_id = loaded[-1].uv_id
            self._refresh_uv_tree()
            self._render_uv_plot()
            self.status.set_status(f"Loaded {len(loaded)} UV file(s)")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Loading UV CSV",
            group="lcms_uv",
            cancel_previous=True,
        )

    def _remove_selected_mzml(self) -> None:
        items = self._ws_tree.selectedItems()
        if not items:
            return
        sid = str(items[0].data(0, Qt.UserRole) or "")
        if not sid:
            return
        self._sessions.pop(sid, None)
        if sid in self._session_order:
            self._session_order.remove(sid)
        if self._active_session_id == sid:
            self._active_session_id = (self._session_order[-1] if self._session_order else None)
        self._refresh_workspace_tree()
        self._render_active_plots()

    def _remove_selected_uv(self) -> None:
        items = self._uv_tree.selectedItems()
        if not items:
            return
        uid = str(items[0].data(0, Qt.UserRole) or "")
        if not uid:
            return
        self._uv_sessions.pop(uid, None)
        if uid in self._uv_order:
            self._uv_order.remove(uid)
        for sess in self._sessions.values():
            if sess.linked_uv_id == uid:
                sess.linked_uv_id = None
        if self._active_uv_id == uid:
            self._active_uv_id = (self._uv_order[-1] if self._uv_order else None)
        self._refresh_uv_tree()
        self._render_uv_plot()

    def _link_selected_uv_to_selected_mzml(self) -> None:
        ws_items = self._ws_tree.selectedItems()
        uv_items = self._uv_tree.selectedItems()
        if not ws_items or not uv_items:
            return
        sid = str(ws_items[0].data(0, Qt.UserRole) or "")
        uid = str(uv_items[0].data(0, Qt.UserRole) or "")
        sess = self._sessions.get(sid)
        if sess is None or uid not in self._uv_sessions:
            return
        sess.linked_uv_id = uid
        self._active_session_id = sid
        self._active_uv_id = uid
        self._refresh_uv_tree()
        self._render_uv_plot()

    def _auto_link_uv_by_name(self) -> None:
        if not self._sessions or not self._uv_sessions:
            return
        uv_by_stem = {Path(uv.path).stem.lower(): uv.uv_id for uv in self._uv_sessions.values()}
        for sid in self._session_order:
            sess = self._sessions.get(sid)
            if sess is None:
                continue
            stem = sess.path.stem.lower()
            if stem in uv_by_stem:
                sess.linked_uv_id = uv_by_stem[stem]
        self._refresh_uv_tree()
        self._render_uv_plot()

    def _on_ws_select(self) -> None:
        items = self._ws_tree.selectedItems()
        if not items:
            return
        sid = str(items[0].data(0, Qt.UserRole) or "")
        if sid:
            self._set_active_session(sid)

    def _on_uv_select(self) -> None:
        items = self._uv_tree.selectedItems()
        if not items:
            return
        uid = str(items[0].data(0, Qt.UserRole) or "")
        if uid:
            self._active_uv_id = uid
            self._render_uv_plot()

    def _set_active_session(self, session_id: str) -> None:
        if session_id == self._active_session_id:
            return
        if session_id not in self._sessions:
            return
        self._active_session_id = session_id
        self._current_spectrum_id = None
        self._current_rt_min = None
        self._refresh_workspace_tree()
        self._render_active_plots()

    def _refresh_workspace_tree(self) -> None:
        self._ws_tree.clear()
        for sid in self._session_order:
            sess = self._sessions.get(sid)
            if sess is None:
                continue
            active = "●" if sid == self._active_session_id else ""
            color = sess.overlay_color or ""
            item = QTreeWidgetItem(["", active, color, sess.display_name, str(sess.ms1_count), str(sess.polarity_summary)])
            item.setData(0, Qt.UserRole, str(sid))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if sid in self._overlay_session_ids else Qt.Unchecked)
            self._ws_tree.addTopLevelItem(item)

    def _refresh_uv_tree(self) -> None:
        self._uv_tree.clear()
        for uid in self._uv_order:
            uv = self._uv_sessions.get(uid)
            if uv is None:
                continue
            linked_to = ""
            for sess in self._sessions.values():
                if sess.linked_uv_id == uid:
                    linked_to = sess.display_name
                    break
            rt_range = f"{uv.rt_min_range[0]:.3g}–{uv.rt_min_range[1]:.3g}" if uv.rt_min_range else ""
            item = QTreeWidgetItem([linked_to, uv.path.name, rt_range, str(uv.n_points)])
            item.setData(0, Qt.UserRole, str(uid))
            self._uv_tree.addTopLevelItem(item)

    def _render_active_plots(self) -> None:
        self._refresh_overlay_legend()
        self._render_tic_plot()
        self._render_uv_plot()
        self._render_spectrum_plot()
        self._tic_plot.setVisible(bool(self._show_tic))
        self._uv_plot.setVisible(bool(self._show_uv))
        self._spec_plot.setVisible(bool(self._show_spec))

    def _render_tic_plot(self) -> None:
        self._tic_plot.clear()
        sessions = self._overlay_session_ids or ([self._active_session_id] if self._active_session_id else [])
        if not sessions:
            return
        colors = self._overlay_colors_for_count(len(sessions))
        for idx, sid in enumerate(sessions):
            sess = self._sessions.get(sid or "")
            if sess is None:
                continue
            ms1 = list(sess.index.ms1 or [])
            if self._polarity_filter in ("positive", "negative"):
                ms1 = [m for m in ms1 if str(m.polarity or "") == self._polarity_filter]
            if not ms1:
                continue
            rt = np.asarray([m.rt_min for m in ms1], dtype=float)
            tic = np.asarray([m.tic for m in ms1], dtype=float)
            pen = pg.mkPen(color=colors[idx] if idx < len(colors) else None, width=float(self._tic_line_width))
            self._tic_plot.plot_line(rt, tic, name=sess.display_name, pen=pen)

    def _render_uv_plot(self) -> None:
        self._uv_plot.clear()
        if self._overlay_session_ids and not self._overlay_show_uv:
            return
        sessions = self._overlay_session_ids or ([self._active_session_id] if self._active_session_id else [])
        if not sessions:
            return
        colors = self._overlay_colors_for_count(len(sessions))
        plotted = False
        for idx, sid in enumerate(sessions):
            sess = self._sessions.get(sid or "")
            if sess is None or not sess.linked_uv_id:
                continue
            uv = self._uv_sessions.get(sess.linked_uv_id)
            if uv is None:
                continue
            pen = pg.mkPen(color=colors[idx] if idx < len(colors) else None, width=float(self._uv_line_width))
            x = np.asarray(uv.rt_min, dtype=float) + float(self._uv_rt_offset_min or 0.0)
            self._uv_plot.plot_line(x, uv.signal, name=uv.path.name, pen=pen)
            plotted = True
        if plotted:
            return
        if self._active_uv_id:
            uv = self._uv_sessions.get(self._active_uv_id)
            if uv is not None:
                x = np.asarray(uv.rt_min, dtype=float) + float(self._uv_rt_offset_min or 0.0)
                pen = pg.mkPen(width=float(self._uv_line_width))
                self._uv_plot.plot_line(x, uv.signal, name=uv.path.name, pen=pen)

    def _render_spectrum_plot(self) -> None:
        self._spec_plot.clear()
        if self._overlay_session_ids:
            self._render_spectrum_overlay_async()
            return
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None:
            return
        ms1 = list(sess.index.ms1 or [])
        if not ms1:
            return
        if not self._current_spectrum_id:
            self._current_spectrum_id = str(ms1[0].spectrum_id)
            self._current_rt_min = float(ms1[0].rt_min)
        mz, inten = self._load_spectrum_arrays(sess, self._current_spectrum_id)
        if mz is None or inten is None:
            return
        pen = pg.mkPen(width=float(self._spec_line_width))
        self._spec_plot.plot_line(mz, inten, name="Spectrum", pen=pen)
        current_meta = None
        try:
            current_meta = next((m for m in ms1 if str(m.spectrum_id) == str(self._current_spectrum_id)), None)
        except Exception:
            current_meta = None
        self._render_spectrum_annotations(mz, inten, polarity=(current_meta.polarity if current_meta else None))

    def _on_plot_click(self, *, axis: str, x: float, y: float, button: int, dblclick: bool) -> None:
        if axis in ("tic", "uv"):
            self._select_rt(float(x))
        elif axis == "spec" and dblclick:
            self._reset_view_all()

    def _on_plot_move(self, *, axis: str, x: float, y: float) -> None:
        if axis in ("tic", "uv"):
            self._current_rt_min = float(x)

    def _on_plot_release(self, *, axis: str, x: float, y: float) -> None:
        if axis in ("tic", "uv"):
            self._select_rt(float(x))

    def _select_rt(self, rt_min: float) -> None:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None:
            return
        ms1 = list(sess.index.ms1 or [])
        if not ms1:
            return
        target = min(ms1, key=lambda m: abs(float(m.rt_min) - float(rt_min)))
        self._current_spectrum_id = str(target.spectrum_id)
        self._current_rt_min = float(target.rt_min)
        self._render_spectrum_plot()
        self.status.set_status(f"RT={target.rt_min:.4g} min")

    def _step_spectrum(self, step: int) -> None:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None:
            return
        ms1 = list(sess.index.ms1 or [])
        if not ms1:
            return
        idx = 0
        if self._current_spectrum_id:
            for i, m in enumerate(ms1):
                if str(m.spectrum_id) == str(self._current_spectrum_id):
                    idx = i
                    break
        idx = max(0, min(len(ms1) - 1, idx + int(step)))
        target = ms1[idx]
        self._current_spectrum_id = str(target.spectrum_id)
        self._current_rt_min = float(target.rt_min)
        self._render_spectrum_plot()
        self.status.set_status(f"RT={target.rt_min:.4g} min")

    def _go_to_index(self, idx: int) -> None:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None:
            return
        ms1 = list(sess.index.ms1 or [])
        if not ms1:
            return
        idx = max(0, min(len(ms1) - 1, int(idx)))
        target = ms1[idx]
        self._current_spectrum_id = str(target.spectrum_id)
        self._current_rt_min = float(target.rt_min)
        self._render_spectrum_plot()
        self.status.set_status(f"RT={target.rt_min:.4g} min")

    def _go_last(self) -> None:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None:
            return
        ms1 = list(sess.index.ms1 or [])
        if not ms1:
            return
        target = ms1[-1]
        self._current_spectrum_id = str(target.spectrum_id)
        self._current_rt_min = float(target.rt_min)
        self._render_spectrum_plot()
        self.status.set_status(f"RT={target.rt_min:.4g} min")

    def _jump_to_rt(self) -> None:
        txt = self._rt_jump_edit.text().strip()
        if not txt:
            return
        try:
            rt = float(txt)
        except Exception:
            self.dialogs.error("LCMS", "Invalid RT value")
            return
        self._select_rt(rt)

    def _load_spectrum_arrays(self, sess: MzMLSession, spectrum_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            reader = mzml.MzML(str(sess.path))
        except Exception as exc:
            self.status.set_status(f"Failed to open mzML: {exc}")
            return None, None
        try:
            with reader:
                try:
                    spec = reader.get_by_id(spectrum_id)
                except Exception:
                    spec = None
                    for s in reader:
                        if str(s.get("id")) == str(spectrum_id):
                            spec = s
                            break
                if not spec:
                    return None, None
                mzs = np.asarray(spec.get("m/z array", []), dtype=float)
                intens = np.asarray(spec.get("intensity array", []), dtype=float)
                return mzs, intens
        except Exception as exc:
            self.status.set_status(f"Failed to read spectrum: {exc}")
            return None, None

    def _export_current_spectrum(self) -> None:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None or not self._current_spectrum_id:
            self.dialogs.info("LCMS", "No spectrum selected.")
            return
        mzs, intens = self._load_spectrum_arrays(sess, self._current_spectrum_id)
        if mzs is None or intens is None:
            self.dialogs.info("LCMS", "Failed to load spectrum.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Spectrum CSV", "", "CSV (*.csv);;All files (*.*)")
        if not path:
            return
        try:
            df = pd.DataFrame({"mz": mzs, "intensity": intens})
            df.to_csv(path, index=False)
            self.status.set_status("Spectrum exported")
        except Exception as exc:
            self.dialogs.error("LCMS", f"Failed to export spectrum:\n\n{exc}")

    def _overlay_sessions(self) -> List[str]:
        if self._overlay_session_ids:
            return list(self._overlay_session_ids)
        if self._active_session_id:
            return [self._active_session_id]
        return []

    def _export_overlay_tic_csv(self) -> None:
        sessions = self._overlay_sessions()
        if not sessions:
            self.dialogs.info("LCMS", "No sessions to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Overlay TIC CSV", "", "CSV (*.csv);;All files (*.*)")
        if not path:
            return

        def _work(_h):
            series = []
            for sid in sessions:
                sess = self._sessions.get(sid)
                if sess is None:
                    continue
                ms1 = list(sess.index.ms1 or [])
                if self._polarity_filter in ("positive", "negative"):
                    ms1 = [m for m in ms1 if str(m.polarity or "") == self._polarity_filter]
                if not ms1:
                    continue
                rt = np.asarray([m.rt_min for m in ms1], dtype=float)
                tic = np.asarray([m.tic for m in ms1], dtype=float)
                order = np.argsort(rt)
                rt = rt[order]
                tic = tic[order]
                series.append((sess.display_name, rt, tic))

            if not series:
                return None

            union = np.unique(np.concatenate([s[1] for s in series]))
            out = {"rt_min": union}
            for name, rt, tic in series:
                out[name] = np.interp(union, rt, tic, left=np.nan, right=np.nan)
            return pd.DataFrame(out)

        def _done(df) -> None:
            if df is None or df.empty:
                self.dialogs.info("LCMS", "No TIC data available for export.")
                return
            try:
                df.to_csv(path, index=False)
                self.status.set_status("Overlay TIC exported")
            except Exception as exc:
                self.dialogs.error("LCMS", f"Failed to export overlay TIC:\n\n{exc}")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Exporting overlay TIC",
            group="lcms_export_overlay_tic",
            cancel_previous=True,
        )

    def _export_overlay_spectra_csv(self) -> None:
        sessions = self._overlay_sessions()
        if not sessions:
            self.dialogs.info("LCMS", "No sessions to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Overlay Spectra CSV", "", "CSV (*.csv);;All files (*.*)")
        if not path:
            return

        rt_target = self._current_rt_min

        def _work(_h):
            spectra = []
            for sid in sessions:
                sess = self._sessions.get(sid)
                if sess is None or not sess.index.ms1:
                    continue
                ms1 = list(sess.index.ms1)
                target = ms1[0] if rt_target is None else min(ms1, key=lambda m: abs(float(m.rt_min) - float(rt_target)))
                mz, inten = self._load_spectrum_arrays(sess, str(target.spectrum_id))
                if mz is None or inten is None or mz.size == 0:
                    continue
                order = np.argsort(mz)
                spectra.append((sess.display_name, mz[order], inten[order]))

            if not spectra:
                return None

            union = np.unique(np.concatenate([s[1] for s in spectra]))
            out = {"mz": union}
            for name, mz, inten in spectra:
                out[name] = np.interp(union, mz, inten, left=0.0, right=0.0)
            return pd.DataFrame(out)

        def _done(df) -> None:
            if df is None or df.empty:
                self.dialogs.info("LCMS", "No spectra available for export.")
                return
            try:
                df.to_csv(path, index=False)
                self.status.set_status("Overlay spectra exported")
            except Exception as exc:
                self.dialogs.error("LCMS", f"Failed to export overlay spectra:\n\n{exc}")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Exporting overlay spectra",
            group="lcms_export_overlay_spec",
            cancel_previous=True,
        )

    def _save_tic_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save TIC Plot", "", "PNG (*.png);;JPG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;All files (*.*)")
        if not path:
            return
        self._tic_plot.export_image_to_path(path)

    def _save_uv_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save UV Plot", "", "PNG (*.png);;JPG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;All files (*.*)")
        if not path:
            return
        self._uv_plot.export_image_to_path(path)

    def _save_spectrum_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Spectrum Plot", "", "PNG (*.png);;JPG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;All files (*.*)")
        if not path:
            return
        self._spec_plot.export_image_to_path(path)

    def _open_tic_window(self) -> None:
        self._open_plot_window(kind="tic")

    def _open_uv_window(self) -> None:
        self._open_plot_window(kind="uv")

    def _open_spectrum_window(self) -> None:
        self._open_plot_window(kind="spec")

    def _open_plot_window(self, *, kind: str) -> None:
        dlg = QDialog(self)
        title = {"tic": "TIC", "uv": "UV", "spec": "Spectrum"}.get(kind, "Plot")
        dlg.setWindowTitle(f"LCMS {title}")
        layout = QVBoxLayout(dlg)
        plot = PlotPanel(on_status=self.status.set_status)
        plot.set_title(title)
        if kind == "tic":
            plot.set_labels("RT (min)", "TIC")
            series = self._build_tic_series()
        elif kind == "uv":
            plot.set_labels("RT (min)", "Signal")
            series = self._build_uv_series()
        else:
            plot.set_labels("m/z", "Intensity")
            series = self._build_spectrum_series()

        for name, x, y, color in series:
            pen = pg.mkPen(color=color, width=float(self._spec_line_width if kind == "spec" else self._tic_line_width))
            plot.plot_line(x, y, name=name, pen=pen)
        plot.set_grid(self._plot_grid_enabled, alpha=self._plot_grid_alpha)

        layout.addWidget(plot, 1)
        dlg.resize(780, 520)
        dlg.exec()

    def _build_tic_series(self) -> List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]]:
        series: List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]] = []
        sessions = self._overlay_sessions()
        colors = self._overlay_colors_for_count(len(sessions))
        for idx, sid in enumerate(sessions):
            sess = self._sessions.get(sid or "")
            if sess is None:
                continue
            ms1 = list(sess.index.ms1 or [])
            if self._polarity_filter in ("positive", "negative"):
                ms1 = [m for m in ms1 if str(m.polarity or "") == self._polarity_filter]
            if not ms1:
                continue
            rt = np.asarray([m.rt_min for m in ms1], dtype=float)
            tic = np.asarray([m.tic for m in ms1], dtype=float)
            series.append((sess.display_name, rt, tic, colors[idx] if idx < len(colors) else None))
        return series

    def _build_uv_series(self) -> List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]]:
        series: List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]] = []
        sessions = self._overlay_sessions()
        colors = self._overlay_colors_for_count(len(sessions))
        for idx, sid in enumerate(sessions):
            sess = self._sessions.get(sid or "")
            if sess is None or not sess.linked_uv_id:
                continue
            uv = self._uv_sessions.get(sess.linked_uv_id)
            if uv is None:
                continue
            x = np.asarray(uv.rt_min, dtype=float) + float(self._uv_rt_offset_min or 0.0)
            series.append((uv.path.name, x, np.asarray(uv.signal, dtype=float), colors[idx] if idx < len(colors) else None))
        if not series and self._active_uv_id:
            uv = self._uv_sessions.get(self._active_uv_id)
            if uv is not None:
                x = np.asarray(uv.rt_min, dtype=float) + float(self._uv_rt_offset_min or 0.0)
                series.append((uv.path.name, x, np.asarray(uv.signal, dtype=float), None))
        return series

    def _build_spectrum_series(self) -> List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]]:
        series: List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]] = []
        sessions = self._overlay_sessions()
        colors = self._overlay_colors_for_count(len(sessions))
        rt_target = self._current_rt_min
        for idx, sid in enumerate(sessions):
            sess = self._sessions.get(sid or "")
            if sess is None or not sess.index.ms1:
                continue
            ms1 = list(sess.index.ms1)
            target = ms1[0] if rt_target is None else min(ms1, key=lambda m: abs(float(m.rt_min) - float(rt_target)))
            mz, inten = self._load_spectrum_arrays(sess, str(target.spectrum_id))
            if mz is None or inten is None:
                continue
            series.append((sess.display_name, mz, inten, colors[idx] if idx < len(colors) else None))
        return series

    def _reset_view_all(self) -> None:
        try:
            self._tic_plot._reset_view()
            self._uv_plot._reset_view()
            self._spec_plot._reset_view()
        except Exception:
            pass

    def _on_ws_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        sid = str(item.data(0, Qt.UserRole) or "")
        if not sid:
            return
        checked = item.checkState(0) == Qt.Checked
        if checked and sid not in self._overlay_session_ids:
            self._overlay_session_ids.append(sid)
        if not checked and sid in self._overlay_session_ids:
            self._overlay_session_ids.remove(sid)
        self._render_active_plots()

    def _start_overlay_selected(self) -> None:
        items = self._ws_tree.selectedItems()
        selected = [str(it.data(0, Qt.UserRole) or "") for it in items if it is not None]
        selected = [sid for sid in selected if sid]
        if not selected:
            selected = list(self._overlay_session_ids)
        if not selected:
            self.dialogs.info("LCMS", "Select one or more mzML sessions to overlay.")
            return
        self._overlay_session_ids = selected
        self._refresh_workspace_tree()
        self._render_active_plots()

    def _clear_overlay(self) -> None:
        self._overlay_session_ids = []
        self._refresh_workspace_tree()
        self._render_active_plots()

    def _on_overlay_mode_changed(self, _idx: int) -> None:
        self._overlay_mode = str(self._overlay_mode_cb.currentText())
        self._render_active_plots()

    def _on_overlay_scheme_changed(self, _idx: int) -> None:
        self._overlay_scheme = str(self._overlay_colors_cb.currentText())
        self._render_active_plots()

    def _pick_overlay_single_hue_color(self) -> None:
        color = QColorDialog.getColor(parent=self, title="Pick overlay hue")
        if not color.isValid():
            return
        self._overlay_single_hue = color.name()
        self._render_active_plots()

    def _on_overlay_show_uv_changed(self, checked: bool) -> None:
        self._overlay_show_uv = bool(checked)
        self._render_active_plots()

    def _on_overlay_stack_changed(self, checked: bool) -> None:
        self._overlay_stack_spectra = bool(checked)
        self._render_active_plots()

    def _overlay_scheme_options(self) -> List[str]:
        return ["Auto (Tableau)", "Single hue…", "Viridis", "Plasma", "Magma", "Cividis", "Turbo"]

    def _overlay_colors_for_count(self, count: int) -> List[str]:
        if count <= 0:
            return []
        scheme = str(self._overlay_scheme or "Auto (Tableau)")
        if scheme.startswith("Single"):
            return [self._overlay_single_hue] * count
        if scheme in ("Viridis", "Plasma", "Magma", "Cividis", "Turbo"):
            try:
                cmap = pg.colormap.get(scheme.lower())
            except Exception:
                cmap = None
            if cmap is not None:
                return [cmap.map(i / max(1, count - 1), mode="qcolor").name() for i in range(count)]
        return [pg.intColor(i, hues=max(1, count)).name() for i in range(count)]

    def _set_polarity_filter(self, value: str) -> None:
        val = str(value or "all")
        if val not in ("all", "positive", "negative"):
            val = "all"
        self._polarity_filter = val
        self._pol_all_cb.blockSignals(True)
        self._pol_pos_cb.blockSignals(True)
        self._pol_neg_cb.blockSignals(True)
        self._pol_all_cb.setChecked(val == "all")
        self._pol_pos_cb.setChecked(val == "positive")
        self._pol_neg_cb.setChecked(val == "negative")
        self._pol_all_cb.blockSignals(False)
        self._pol_pos_cb.blockSignals(False)
        self._pol_neg_cb.blockSignals(False)
        self._render_active_plots()

    def _apply_uv_ms_offset(self) -> None:
        try:
            self._uv_rt_offset_min = float(self._uv_offset_edit.text().strip() or 0.0)
        except Exception:
            self._uv_rt_offset_min = 0.0
        self._render_active_plots()

    def _on_panels_changed(self, _checked: bool) -> None:
        self._show_tic = bool(self._show_tic_cb.isChecked())
        self._show_uv = bool(self._show_uv_cb.isChecked())
        self._show_spec = bool(self._show_spec_cb.isChecked())
        self._render_active_plots()

    def _on_annotate_settings_changed(self, *_args) -> None:
        self._annotate_enabled = bool(self._annotate_cb.isChecked())
        try:
            self._annotate_top_n = int(self._annotate_topn_edit.text().strip() or 6)
        except Exception:
            self._annotate_top_n = 6
        try:
            self._annotate_min_rel = float(self._annotate_minrel_edit.text().strip() or 0.05)
        except Exception:
            self._annotate_min_rel = 0.05
        self._render_active_plots()

    def _apply_quick_annotate_settings(self) -> None:
        self._on_annotate_settings_changed()

    def _refresh_overlay_view(self) -> None:
        self._render_active_plots()

    def _on_tic_region_select_changed(self, checked: bool) -> None:
        self._tic_region_select_enabled = bool(checked)
        if self._tic_region_select_enabled:
            self._tic_plot.enable_region_select(True, initial=self._tic_region, on_change=self._on_tic_region_updated)
            self.status.set_status("Drag on TIC plot to select a region")
            return
        self._tic_plot.enable_region_select(False)
        self._tic_region = None

    def _clear_tic_region_selection(self) -> None:
        self._tic_region_select_enabled = False
        self._tic_region = None
        self._tic_plot.enable_region_select(False)
        if self._tic_region_cb is not None:
            try:
                self._tic_region_cb.blockSignals(True)
                self._tic_region_cb.setChecked(False)
                self._tic_region_cb.blockSignals(False)
            except Exception:
                pass
        self.status.set_status("Cleared TIC region selection")

    def _on_tic_region_updated(self, region: Tuple[float, float]) -> None:
        lo, hi = region
        self._tic_region = (min(lo, hi), max(lo, hi))
        self.status.set_status(f"TIC region: {self._tic_region[0]:.4g}–{self._tic_region[1]:.4g} min")

    def _open_graph_settings(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("LCMS Graph Settings")
        layout = QVBoxLayout(dlg)
        form = QFormLayout()
        layout.addLayout(form)

        grid_cb = QCheckBox("Show grid")
        grid_cb.setChecked(bool(self._plot_grid_enabled))
        form.addRow(grid_cb)

        grid_alpha = QLineEdit(str(self._plot_grid_alpha))
        form.addRow("Grid alpha", grid_alpha)

        tic_w = QLineEdit(str(self._tic_line_width))
        uv_w = QLineEdit(str(self._uv_line_width))
        spec_w = QLineEdit(str(self._spec_line_width))
        form.addRow("TIC line width", tic_w)
        form.addRow("UV line width", uv_w)
        form.addRow("Spectrum line width", spec_w)

        btns = QHBoxLayout()
        btns.addStretch(1)
        b_apply = QPushButton("Apply")
        b_close = QPushButton("Close")
        btns.addWidget(b_apply)
        btns.addWidget(b_close)
        layout.addLayout(btns)

        def _apply() -> None:
            try:
                self._plot_grid_enabled = bool(grid_cb.isChecked())
                self._plot_grid_alpha = float(grid_alpha.text().strip() or 0.2)
                self._tic_line_width = float(tic_w.text().strip() or 1.2)
                self._uv_line_width = float(uv_w.text().strip() or 1.2)
                self._spec_line_width = float(spec_w.text().strip() or 1.2)
            except Exception:
                self.dialogs.error("LCMS", "Invalid graph settings values.")
                return
            self._apply_plot_style()
            self._render_active_plots()

        b_apply.clicked.connect(_apply)
        b_close.clicked.connect(dlg.close)

        dlg.resize(420, 260)
        dlg.exec()

    def _apply_plot_style(self) -> None:
        try:
            self._tic_plot.set_grid(self._plot_grid_enabled, alpha=self._plot_grid_alpha)
            self._uv_plot.set_grid(self._plot_grid_enabled, alpha=self._plot_grid_alpha)
            self._spec_plot.set_grid(self._plot_grid_enabled, alpha=self._plot_grid_alpha)
        except Exception:
            pass

    def _open_jump_to_mz_dialog(self) -> None:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None or not self._current_spectrum_id:
            self.dialogs.info("LCMS", "Select a spectrum first.")
            return
        mz_val, ok = QInputDialog.getDouble(self, "Jump to m/z", "m/z", value=0.0, min=0.0, max=1e9, decimals=4)
        if not ok:
            return
        mzs, intens = self._load_spectrum_arrays(sess, self._current_spectrum_id)
        if mzs is None or intens is None or mzs.size == 0:
            self.dialogs.info("LCMS", "Failed to load spectrum.")
            return
        idx = int(np.argmin(np.abs(mzs - float(mz_val))))
        mz_hit = float(mzs[idx])
        inten_hit = float(intens[idx])
        self._spec_plot.add_annotation(mz_hit, inten_hit, f"{mz_hit:.4g}", color="#7C3AED")
        self.status.set_status(f"Nearest m/z: {mz_hit:.4g} (I={inten_hit:.3g})")

    def _open_find_mz_dialog(self) -> None:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None:
            self.dialogs.info("LCMS", "Select an mzML session first.")
            return
        mz_val, ok = QInputDialog.getDouble(self, "Find m/z", "m/z", value=0.0, min=0.0, max=1e9, decimals=4)
        if not ok:
            return
        tol_val, ok2 = QInputDialog.getDouble(self, "Find m/z", "Tolerance (Da)", value=0.01, min=0.0, max=10.0, decimals=4)
        if not ok2:
            return

        def _work(_h):
            try:
                reader = mzml.MzML(str(sess.path))
            except Exception as exc:
                return exc
            best = None
            with reader:
                ms1 = list(sess.index.ms1 or [])
                for meta in ms1:
                    try:
                        spec = reader.get_by_id(str(meta.spectrum_id))
                    except Exception:
                        continue
                    mzs = np.asarray(spec.get("m/z array", []), dtype=float)
                    intens = np.asarray(spec.get("intensity array", []), dtype=float)
                    if mzs.size == 0 or intens.size == 0:
                        continue
                    lo = float(mz_val) - float(tol_val)
                    hi = float(mz_val) + float(tol_val)
                    mask = (mzs >= lo) & (mzs <= hi)
                    if not np.any(mask):
                        continue
                    idx = int(np.argmax(intens[mask]))
                    mz_hit = float(mzs[mask][idx])
                    inten_hit = float(intens[mask][idx])
                    if best is None or inten_hit > best[0]:
                        best = (inten_hit, str(meta.spectrum_id), float(meta.rt_min), mz_hit)
            return best

        def _done(res) -> None:
            if isinstance(res, Exception):
                self.dialogs.error("LCMS", f"Find m/z failed:\n\n{res}")
                return
            if not res:
                self.dialogs.info("LCMS", "No matching peaks found.")
                return
            inten_hit, spectrum_id, rt_min, mz_hit = res
            self._current_spectrum_id = str(spectrum_id)
            self._current_rt_min = float(rt_min)
            self._render_spectrum_plot()
            self.status.set_status(f"Found m/z {mz_hit:.4g} at RT={rt_min:.4g} min (I={inten_hit:.3g})")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Finding m/z",
            group="lcms_find_mz",
            cancel_previous=True,
        )

    def _open_sim_dialog(self) -> None:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None:
            self.dialogs.info("LCMS", "Select an mzML session first.")
            return
        mz_val, ok = QInputDialog.getDouble(self, "EIC", "m/z", value=0.0, min=0.0, max=1e9, decimals=4)
        if not ok:
            return
        tol_val, ok2 = QInputDialog.getDouble(self, "EIC", "Tolerance (Da)", value=0.01, min=0.0, max=10.0, decimals=4)
        if not ok2:
            return

        def _work(_h):
            try:
                reader = mzml.MzML(str(sess.path))
            except Exception as exc:
                return exc
            rt_vals = []
            eic_vals = []
            lo = float(mz_val) - float(tol_val)
            hi = float(mz_val) + float(tol_val)
            with reader:
                for meta in list(sess.index.ms1 or []):
                    try:
                        spec = reader.get_by_id(str(meta.spectrum_id))
                    except Exception:
                        continue
                    mzs = np.asarray(spec.get("m/z array", []), dtype=float)
                    intens = np.asarray(spec.get("intensity array", []), dtype=float)
                    if mzs.size == 0 or intens.size == 0:
                        continue
                    mask = (mzs >= lo) & (mzs <= hi)
                    if not np.any(mask):
                        val = 0.0
                    else:
                        val = float(np.sum(intens[mask]))
                    rt_vals.append(float(meta.rt_min))
                    eic_vals.append(val)
            return (np.asarray(rt_vals, dtype=float), np.asarray(eic_vals, dtype=float))

        def _done(res) -> None:
            if isinstance(res, Exception):
                self.dialogs.error("LCMS", f"EIC failed:\n\n{res}")
                return
            rt_vals, eic_vals = res
            if rt_vals.size == 0:
                self.dialogs.info("LCMS", "No data for EIC.")
                return
            dlg = QDialog(self)
            dlg.setWindowTitle(f"EIC m/z {mz_val:.4g} ± {tol_val:.4g}")
            layout = QVBoxLayout(dlg)
            plot = PlotPanel(on_status=self.status.set_status)
            plot.set_title("Extracted Ion Chromatogram")
            plot.set_labels("RT (min)", "Intensity")
            plot.plot_line(rt_vals, eic_vals, name="EIC")
            layout.addWidget(plot, 1)
            btns = QHBoxLayout()
            b_close = QPushButton("Close")
            b_close.clicked.connect(dlg.close)
            btns.addStretch(1)
            btns.addWidget(b_close)
            layout.addLayout(btns)
            dlg.resize(680, 420)
            dlg.exec()

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Computing EIC",
            group="lcms_eic",
            cancel_previous=True,
        )

    def _estimate_uv_ms_offset(self) -> Optional[Tuple[float, float]]:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None:
            return None
        uv = None
        if sess.linked_uv_id and sess.linked_uv_id in self._uv_sessions:
            uv = self._uv_sessions.get(sess.linked_uv_id)
        elif self._active_uv_id:
            uv = self._uv_sessions.get(self._active_uv_id)
        if uv is None:
            return None
        ms1 = list(sess.index.ms1 or [])
        if len(ms1) < 5 or uv.rt_min is None or uv.signal is None:
            return None
        rt_ms = np.asarray([m.rt_min for m in ms1], dtype=float)
        tic = np.asarray([m.tic for m in ms1], dtype=float)
        rt_uv = np.asarray(uv.rt_min, dtype=float)
        sig = np.asarray(uv.signal, dtype=float)
        if rt_ms.size == 0 or rt_uv.size == 0:
            return None
        tic = (tic - np.nanmean(tic)) / (np.nanstd(tic) or 1.0)
        sig = (sig - np.nanmean(sig)) / (np.nanstd(sig) or 1.0)
        rt_range = min(rt_ms.max() - rt_ms.min(), rt_uv.max() - rt_uv.min())
        max_offset = max(0.2, min(5.0, float(rt_range) / 3.0))
        offsets = np.linspace(-max_offset, max_offset, 81)
        best = None
        for off in offsets:
            uv_interp = np.interp(rt_ms, rt_uv + off, sig, left=np.nan, right=np.nan)
            mask = np.isfinite(uv_interp) & np.isfinite(tic)
            if mask.sum() < 5:
                continue
            corr = float(np.corrcoef(tic[mask], uv_interp[mask])[0, 1])
            if not np.isfinite(corr):
                continue
            if best is None or corr > best[1]:
                best = (float(off), corr)
        return best

    def _auto_align_uv_ms(self) -> None:
        def _work(_h):
            return self._estimate_uv_ms_offset()

        def _done(res) -> None:
            if not res:
                self.dialogs.info("LCMS", "Unable to estimate UV↔MS offset.")
                return
            offset, score = res
            self._uv_rt_offset_min = float(offset)
            self._uv_offset_edit.setText(f"{offset:.4g}")
            self._render_active_plots()
            self.status.set_status(f"Auto-align applied: {offset:.4g} min (corr={score:.3f})")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Auto-align UV↔MS",
            group="lcms_align",
            cancel_previous=True,
        )

    def _open_alignment_diagnostics(self) -> None:
        res = self._estimate_uv_ms_offset()
        if not res:
            self.dialogs.info("LCMS", "No alignment diagnostics available.")
            return
        offset, score = res
        self.dialogs.info("LCMS", f"Suggested offset: {offset:.4g} min\nCorrelation: {score:.3f}")

    def _on_uv_ms_align_enabled_changed(self, checked: bool) -> None:
        self._uv_align_enabled = bool(checked)

    def _render_spectrum_annotations(self, mz: np.ndarray, inten: np.ndarray, *, polarity: Optional[str]) -> None:
        self._spec_plot.clear_annotations()
        if mz.size == 0 or inten.size == 0:
            return
        try:
            max_i = float(np.nanmax(inten))
        except Exception:
            return
        if not np.isfinite(max_i) or max_i <= 0:
            return
        if self._annotate_enabled:
            threshold = max_i * float(self._annotate_min_rel or 0.0)
            try:
                idx = np.argsort(inten)[::-1]
            except Exception:
                return
            added = 0
            for i in idx:
                try:
                    if float(inten[i]) < threshold:
                        break
                    self._spec_plot.add_annotation(float(mz[i]), float(inten[i]), f"{float(mz[i]):.2f}")
                    added += 1
                    if added >= max(1, int(self._annotate_top_n)):
                        break
                except Exception:
                    continue

        sess = self._sessions.get(self._active_session_id or "")
        if sess is None or not self._current_spectrum_id:
            return
        custom = sess.custom_labels_by_spectrum.get(self._current_spectrum_id, []) if sess.custom_labels_by_spectrum else []
        for it in custom:
            try:
                mzv = float(it.get("mz"))
                label = str(it.get("text") or "")
                if not label:
                    label = f"{mzv:.2f}"
                yv = float(np.interp(mzv, mz, inten))
                self._spec_plot.add_annotation(mzv, yv, label, color="#7F1D1D")
            except Exception:
                continue

        if not self._poly_enabled:
            return

        monomers = self._parse_poly_monomers(self._poly_monomers_text)
        if not monomers:
            return
        charges = self._parse_poly_charges(self._poly_charges_text)
        if not charges:
            charges = [1]

        max_dp = max(1, min(200, int(self._poly_max_dp or 1)))
        bond_delta = float(self._poly_bond_delta)
        extra_delta = float(self._poly_extra_delta)
        adduct_mass = float(self._poly_adduct_mass)
        decarb_enabled = bool(self._poly_decarb_enabled)
        oxid_enabled = bool(self._poly_oxid_enabled)
        cluster_enabled = bool(self._poly_cluster_enabled)
        cluster_adduct_mass = float(self._poly_cluster_adduct_mass)

        pol = polarity
        if pol in ("positive", "negative"):
            h = 1.007276
            sign = 1.0 if pol == "positive" else -1.0
            if abs(abs(adduct_mass) - h) <= 0.01:
                adduct_mass = sign * abs(adduct_mass)
            if abs(abs(cluster_adduct_mass) - h) <= 0.01:
                cluster_adduct_mass = sign * abs(cluster_adduct_mass)

        min_rel = max(0.0, min(1.0, float(self._poly_min_rel_int)))

        order = np.argsort(mz)
        mz_s = mz[order]
        int_s = inten[order]
        compat = bool(str(os.environ.get("LAB_GUI_POLYMER_COMPAT", "")).strip())

        try:
            best_by_peak = poly_match.compute_polymer_best_by_peak_sorted(
                mz_s,
                int_s,
                monomer_names=[n for n, _m in monomers],
                monomer_masses=[m for _n, m in monomers],
                charges=list(charges),
                max_dp=int(max_dp),
                bond_delta=float(bond_delta),
                extra_delta=float(extra_delta),
                polarity=pol,
                base_adduct_mass=float(adduct_mass),
                enable_decarb=bool(decarb_enabled),
                enable_oxid=bool(oxid_enabled),
                enable_cluster=bool(cluster_enabled),
                cluster_adduct_mass=float(cluster_adduct_mass),
                enable_na=bool(self._poly_adduct_na),
                enable_k=bool(self._poly_adduct_k),
                enable_cl=bool(self._poly_adduct_cl),
                enable_formate=bool(self._poly_adduct_formate),
                enable_acetate=bool(self._poly_adduct_acetate),
                tol_value=float(self._poly_tol_value),
                tol_unit=str(self._poly_tol_unit or "Da"),
                min_rel_int=float(min_rel),
                allow_variant_combo=True,
                compatibility_mode=bool(compat),
            )
        except poly_match.PolymerSearchTooLarge as exc:
            self.dialogs.warn("Polymer Match", str(exc))
            return
        except Exception as exc:
            self.dialogs.error("Polymer Match", f"Polymer matching failed:\n\n{exc}")
            return

        if not best_by_peak:
            return

        y_offset = 0.10 * max_i
        kind_order = ["poly", "ox", "decarb", "oxdecarb", "2m"]
        for _peak_i, kinds in best_by_peak.items():
            items: List[Tuple[str, float, str, float, float]] = []
            for knd in kind_order:
                if knd in kinds:
                    _err, label, mz_act, inten_act = kinds[knd]
                    items.append((str(knd), float(_err), str(label), float(mz_act), float(inten_act)))
            if not items:
                for knd, (_err, label, mz_act, inten_act) in kinds.items():
                    items.append((str(knd), float(_err), str(label), float(mz_act), float(inten_act)))

            for j, (_knd, _err, label, mz_act, inten_act) in enumerate(items):
                yv = float(inten_act) + (j * y_offset)
                self._spec_plot.add_annotation(float(mz_act), yv, str(label), color="#0F766E")

    def _export_all_labels_xlsx(self) -> None:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None:
            self.dialogs.info("LCMS", "No active mzML session.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Labels (All Scans)",
            "",
            "Excel (*.xlsx);;CSV (*.csv);;All files (*.*)",
        )
        if not path:
            return

        top_n = max(1, int(self._annotate_top_n))
        min_rel = float(self._annotate_min_rel)

        def _work(_h):
            rows: List[Dict[str, object]] = []
            try:
                reader = mzml.MzML(str(sess.path))
            except Exception as exc:
                return exc
            with reader:
                for spec in reader:
                    try:
                        sid = str(spec.get("id") or spec.get("index"))
                        inten = np.asarray(spec.get("intensity array", []), dtype=float)
                        mzs = np.asarray(spec.get("m/z array", []), dtype=float)
                        if inten.size == 0 or mzs.size == 0:
                            continue
                        max_i = float(np.nanmax(inten)) if inten.size else 0.0
                        if max_i <= 0:
                            continue
                        threshold = max_i * min_rel
                        idx = np.argsort(inten)[::-1]
                        added = 0
                        for i in idx:
                            if float(inten[i]) < threshold:
                                break
                            rows.append(
                                {
                                    "spectrum_id": sid,
                                    "mz": float(mzs[i]),
                                    "intensity": float(inten[i]),
                                }
                            )
                            added += 1
                            if added >= top_n:
                                break
                    except Exception:
                        continue
            return rows

        def _done(res) -> None:
            if isinstance(res, Exception):
                self.dialogs.error("LCMS", f"Failed to export labels:\n\n{res}")
                return
            rows = res or []
            if not rows:
                self.dialogs.info("LCMS", "No labels to export.")
                return
            df = pd.DataFrame(rows)
            try:
                if path.lower().endswith(".csv"):
                    df.to_csv(path, index=False)
                else:
                    df.to_excel(path, index=False)
                self.status.set_status("Labels exported")
            except Exception as exc:
                self.dialogs.error("LCMS", f"Failed to write file:\n\n{exc}")

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Exporting labels",
            group="lcms_export_labels",
            cancel_previous=True,
        )

    def _open_annotation_settings(self) -> None:
        top_n, ok = QInputDialog.getInt(self, "Annotate Peaks", "Top N", value=int(self._annotate_top_n), min=1, max=100)
        if not ok:
            return
        min_rel, ok2 = QInputDialog.getDouble(
            self,
            "Annotate Peaks",
            "Min relative intensity",
            value=float(self._annotate_min_rel),
            min=0.0,
            max=1.0,
            decimals=3,
        )
        if not ok2:
            return
        self._annotate_top_n = int(top_n)
        self._annotate_min_rel = float(min_rel)
        self._annotate_enabled = True
        if self._annotate_cb is not None:
            self._annotate_cb.setChecked(True)
        if self._annotate_topn_edit is not None:
            self._annotate_topn_edit.setText(str(self._annotate_top_n))
        if self._annotate_minrel_edit is not None:
            self._annotate_minrel_edit.setText(str(self._annotate_min_rel))
        self._render_active_plots()

    def _open_custom_labels(self) -> None:
        sess = self._sessions.get(self._active_session_id or "")
        if sess is None or not self._current_spectrum_id:
            self.dialogs.info("LCMS", "Select a spectrum first.")
            return
        mz_val, ok = QInputDialog.getDouble(self, "Custom Label", "m/z", value=0.0, min=0.0, max=1e9, decimals=4)
        if not ok:
            return
        text, ok2 = QInputDialog.getText(self, "Custom Label", "Label text (optional)")
        if not ok2:
            return
        row = {"mz": float(mz_val), "text": str(text or "")}
        labels = sess.custom_labels_by_spectrum.get(self._current_spectrum_id, []) if sess.custom_labels_by_spectrum else []
        labels.append(row)
        sess.custom_labels_by_spectrum[self._current_spectrum_id] = labels
        self._render_active_plots()

    def _render_spectrum_overlay_async(self) -> None:
        session_ids = list(self._overlay_session_ids)
        if not session_ids:
            return
        rt_target = self._current_rt_min
        if rt_target is None and self._active_session_id:
            sess = self._sessions.get(self._active_session_id)
            if sess and sess.index.ms1:
                rt_target = float(sess.index.ms1[0].rt_min)

        def _work(_h):
            series = []
            for sid in session_ids:
                sess = self._sessions.get(sid)
                if sess is None or not sess.index.ms1:
                    continue
                ms1 = list(sess.index.ms1)
                target = ms1[0] if rt_target is None else min(ms1, key=lambda m: abs(float(m.rt_min) - float(rt_target)))
                mz, inten = self._load_spectrum_arrays(sess, str(target.spectrum_id))
                if mz is None or inten is None:
                    continue
                series.append((sess.display_name, mz, inten))
            return series

        def _done(series) -> None:
            self._spec_plot.clear()
            colors = self._overlay_colors_for_count(len(series))
            for idx, (name, mz, inten) in enumerate(series):
                pen = pg.mkPen(color=colors[idx] if idx < len(colors) else None, width=float(self._spec_line_width))
                self._spec_plot.plot_line(mz, inten, name=name, pen=pen)

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Rendering spectrum overlay",
            group="lcms_spec_overlay",
            cancel_previous=True,
        )

    def _refresh_overlay_legend(self) -> None:
        try:
            self._overlay_legend.clear()
        except Exception:
            return
        if not self._overlay_session_ids:
            return
        colors = self._overlay_colors_for_count(len(self._overlay_session_ids))
        for idx, sid in enumerate(self._overlay_session_ids):
            sess = self._sessions.get(sid)
            if sess is None:
                continue
            color = colors[idx] if idx < len(colors) else ""
            sess.overlay_color = color
            status = "active" if sid == self._active_session_id else ""
            item = QTreeWidgetItem([color, sess.display_name, str(sess.ms1_count), str(sess.polarity_summary), status])
            self._overlay_legend.addTopLevelItem(item)

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
        mzml_files: List[Dict[str, object]] = []
        for sid in self._session_order:
            sess = self._sessions.get(sid)
            if sess is None:
                continue
            mzml_files.append(
                {
                    "path": str(sess.path),
                    "polarity": str(sess.last_polarity_filter or sess.polarity_summary or "all"),
                    "rt_unit": "minutes",
                    "display_name": str(sess.display_name),
                    "last_scan_index": (None if sess.last_scan_index is None else int(sess.last_scan_index)),
                    "last_selected_rt_min": (None if sess.last_selected_rt_min is None else float(sess.last_selected_rt_min)),
                }
            )

        active_index = 0
        if self._active_session_id and self._active_session_id in self._session_order:
            active_index = int(self._session_order.index(self._active_session_id))

        linked_uv = None
        try:
            if self._active_session_id and self._active_session_id in self._sessions:
                sess = self._sessions[self._active_session_id]
                if sess.linked_uv_id and sess.linked_uv_id in self._uv_sessions:
                    uv = self._uv_sessions[sess.linked_uv_id]
                    linked_uv = {
                        "mzml_path": str(sess.path),
                        "uv_csv_path": str(uv.path),
                        "uv_ms_offset": 0.0,
                    }
        except Exception:
            linked_uv = None

        return {
            "schema": "LCMS_WORKSPACE",
            "version": 1,
            "saved_at": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "mzml_files": mzml_files,
            "active_mzml_index": active_index,
            "linked_uv": linked_uv,
            "annotations": {},
            "annotations_by_mzml": {},
            "current_scan_index": None,
            "tic_settings": {},
            "polymer_settings": self._encode_polymer_settings(),
        }

    def _encode_polymer_settings(self) -> Dict[str, object]:
        monomers = [ln.strip() for ln in str(self._poly_monomers_text or "").splitlines() if ln.strip()]
        common = {
            "bond_delta": float(self._poly_bond_delta),
            "extra_delta": float(self._poly_extra_delta),
            "adduct_mass": float(self._poly_adduct_mass),
            "cluster_adduct_mass": float(self._poly_cluster_adduct_mass),
            "adduct_na": bool(self._poly_adduct_na),
            "adduct_k": bool(self._poly_adduct_k),
            "adduct_cl": bool(self._poly_adduct_cl),
            "adduct_formate": bool(self._poly_adduct_formate),
            "adduct_acetate": bool(self._poly_adduct_acetate),
            "charges": str(self._poly_charges_text or "1"),
            "decarb": bool(self._poly_decarb_enabled),
            "oxid": bool(self._poly_oxid_enabled),
            "cluster": bool(self._poly_cluster_enabled),
            "min_rel_int": float(self._poly_min_rel_int),
        }
        return {
            "enabled": bool(self._poly_enabled),
            "monomers": list(monomers),
            "max_dp": int(self._poly_max_dp),
            "tolerance": float(self._poly_tol_value),
            "tolerance_unit": str(self._poly_tol_unit or "Da"),
            "positive_mode": dict(common),
            "negative_mode": dict(common),
        }

    def _save_workspace(self) -> Optional[str]:
        if not self._sessions:
            self.dialogs.info("LCMS", "No mzML files to save.")
            return None
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save LCMS Workspace",
            "",
            "LCMS Workspace (*.lcms_workspace.json);;JSON (*.json);;All files (*.*)",
        )
        if not path:
            return None
        payload = self._encode_workspace()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.dialogs.error("LCMS", f"Failed to save workspace:\n\n{exc}")
            return None
        self._last_workspace_path = str(path)
        self.status.set_status("LCMS workspace saved")
        return str(path)

    def _load_workspace(self) -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load LCMS Workspace",
            "",
            "LCMS Workspace (*.lcms_workspace.json);;JSON (*.json);;All files (*.*)",
        )
        if not path:
            return None
        return self._load_workspace_path(str(path))

    def _load_workspace_path(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            self.dialogs.error("LCMS", f"Failed to read workspace JSON:\n\n{exc}")
            return None
        if not isinstance(payload, dict):
            self.dialogs.error("LCMS", "Workspace JSON must be an object.")
            return None

        if payload.get("schema") not in ("LCMS_WORKSPACE", None):
            self.dialogs.error("LCMS", "Unsupported LCMS workspace format.")
            return None

        mzml_rows = payload.get("mzml_files") if isinstance(payload.get("mzml_files"), list) else []
        active_index = int(payload.get("active_mzml_index") or 0)
        linked_uv = payload.get("linked_uv") if isinstance(payload.get("linked_uv"), dict) else None
        poly_payload = payload.get("polymer_settings") if isinstance(payload.get("polymer_settings"), dict) else None

        def _work(_h):
            loaded_sessions: List[MzMLSession] = []
            missing: List[str] = []
            for i, row in enumerate(mzml_rows):
                if not isinstance(row, dict):
                    continue
                p = str(row.get("path") or "")
                if not p:
                    continue
                mzml_path = Path(p)
                if not mzml_path.exists():
                    missing.append(p)
                    continue
                index = MzMLTICIndex(mzml_path, rt_unit=str(row.get("rt_unit") or "minutes"))
                index.build()
                ms1 = list(index.ms1 or [])
                if not ms1:
                    continue
                pol = str(row.get("polarity") or "all")
                pol_summary = pol if pol in ("positive", "negative", "all") else "mixed"
                rt_vals = [float(m.rt_min) for m in ms1]
                rt_min = min(rt_vals) if rt_vals else None
                rt_max = max(rt_vals) if rt_vals else None
                display = str(row.get("display_name") or mzml_path.name)
                session_id = str(uuid.uuid4())
                loaded_sessions.append(
                    MzMLSession(
                        session_id=session_id,
                        path=mzml_path,
                        index=index,
                        load_order=int(i),
                        display_name=display,
                        custom_labels_by_spectrum={},
                        spec_label_overrides={},
                        ms1_count=int(len(ms1)),
                        rt_min=rt_min,
                        rt_max=rt_max,
                        polarity_summary=pol_summary,
                        last_selected_rt_min=row.get("last_selected_rt_min"),
                        last_scan_index=row.get("last_scan_index"),
                        last_polarity_filter=pol if pol in ("positive", "negative", "all") else None,
                    )
                )

            uv_session: Optional[UVSession] = None
            if linked_uv and linked_uv.get("uv_csv_path"):
                uv_path = Path(str(linked_uv.get("uv_csv_path")))
                if uv_path.exists():
                    try:
                        df = pd.read_csv(uv_path, sep=None, engine="python")
                    except Exception:
                        df = pd.read_table(uv_path)
                    info = infer_uv_columns(df)
                    rt_min, signal, rt_range, warnings = parse_uv_arrays(
                        df,
                        xcol=str(info.get("xcol")),
                        ycol=str(info.get("ycol")),
                        unit_guess=str(info.get("unit_guess")),
                    )
                    uv_session = UVSession(
                        uv_id=str(uuid.uuid4()),
                        path=uv_path,
                        rt_min=rt_min,
                        signal=signal,
                        xcol=str(info.get("xcol")),
                        ycol=str(info.get("ycol")),
                        n_points=int(len(rt_min)),
                        rt_min_range=rt_range,
                        load_order=0,
                        import_warnings=list(warnings or []),
                    )
            return loaded_sessions, uv_session, missing

        def _done(res) -> None:
            loaded_sessions, uv_session, missing = res
            self._sessions = {}
            self._session_order = []
            self._uv_sessions = {}
            self._uv_order = []
            self._active_session_id = None
            self._active_uv_id = None

            for sess in loaded_sessions:
                self._sessions[sess.session_id] = sess
                self._session_order.append(sess.session_id)

            if uv_session is not None:
                self._uv_sessions[uv_session.uv_id] = uv_session
                self._uv_order.append(uv_session.uv_id)
                self._active_uv_id = uv_session.uv_id

            if self._session_order:
                idx = active_index if 0 <= active_index < len(self._session_order) else len(self._session_order) - 1
                self._active_session_id = self._session_order[idx]

            if linked_uv and self._active_session_id and self._active_uv_id:
                mzml_path = str(linked_uv.get("mzml_path") or "")
                for sid, sess in self._sessions.items():
                    if str(sess.path) == mzml_path:
                        sess.linked_uv_id = self._active_uv_id
                        if self._active_session_id != sid:
                            self._active_session_id = sid
                        break

            self._refresh_workspace_tree()
            self._refresh_uv_tree()
            self._render_active_plots()
            if self._active_session_id:
                sess = self._sessions.get(self._active_session_id)
                if sess and sess.last_selected_rt_min is not None:
                    try:
                        self._select_rt(float(sess.last_selected_rt_min))
                    except Exception:
                        pass

            self._last_workspace_path = str(path)
            if missing:
                self.dialogs.warn("LCMS", "Some files were missing and skipped:\n\n" + "\n".join(missing[:12]))
            self.status.set_status("LCMS workspace loaded")

            self._apply_polymer_settings(poly_payload)
            self._render_active_plots()

        self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Loading LCMS workspace",
            group="lcms_workspace",
            cancel_previous=True,
        )
        return str(path)

    def _apply_polymer_settings(self, payload: Optional[Dict[str, object]]) -> None:
        if not isinstance(payload, dict):
            return
        self._poly_enabled = bool(payload.get("enabled", self._poly_enabled))
        monomers = payload.get("monomers") if isinstance(payload.get("monomers"), list) else []
        self._poly_monomers_text = "\n".join([str(m).strip() for m in monomers if str(m).strip()])
        self._poly_max_dp = int(payload.get("max_dp", self._poly_max_dp))
        self._poly_tol_value = float(payload.get("tolerance", self._poly_tol_value))
        self._poly_tol_unit = str(payload.get("tolerance_unit", self._poly_tol_unit or "Da"))

        mode = payload.get("positive_mode") if isinstance(payload.get("positive_mode"), dict) else {}
        self._poly_bond_delta = float(mode.get("bond_delta", self._poly_bond_delta))
        self._poly_extra_delta = float(mode.get("extra_delta", self._poly_extra_delta))
        self._poly_adduct_mass = float(mode.get("adduct_mass", self._poly_adduct_mass))
        self._poly_cluster_adduct_mass = float(mode.get("cluster_adduct_mass", self._poly_cluster_adduct_mass))
        self._poly_adduct_na = bool(mode.get("adduct_na", self._poly_adduct_na))
        self._poly_adduct_k = bool(mode.get("adduct_k", self._poly_adduct_k))
        self._poly_adduct_cl = bool(mode.get("adduct_cl", self._poly_adduct_cl))
        self._poly_adduct_formate = bool(mode.get("adduct_formate", self._poly_adduct_formate))
        self._poly_adduct_acetate = bool(mode.get("adduct_acetate", self._poly_adduct_acetate))
        self._poly_charges_text = str(mode.get("charges", self._poly_charges_text or "1"))
        self._poly_decarb_enabled = bool(mode.get("decarb", self._poly_decarb_enabled))
        self._poly_oxid_enabled = bool(mode.get("oxid", self._poly_oxid_enabled))
        self._poly_cluster_enabled = bool(mode.get("cluster", self._poly_cluster_enabled))
        self._poly_min_rel_int = float(mode.get("min_rel_int", self._poly_min_rel_int))

        if self._poly_enable_cb is not None:
            try:
                self._poly_enable_cb.blockSignals(True)
                self._poly_enable_cb.setChecked(bool(self._poly_enabled))
            except Exception:
                pass
            finally:
                try:
                    self._poly_enable_cb.blockSignals(False)
                except Exception:
                    pass

    def _on_poly_enabled_changed(self, checked: bool) -> None:
        self._poly_enabled = bool(checked)
        self._render_active_plots()

    def _parse_poly_monomers(self, text: str) -> List[Tuple[str, float]]:
        items: List[Tuple[str, float]] = []
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            name = ""
            mass = None
            if "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        mass = float(parts[1])
                    except Exception:
                        mass = None
            else:
                parts = [p.strip() for p in line.split() if p.strip()]
                if len(parts) == 1:
                    name = parts[0]
                    try:
                        mass = float(parts[0])
                    except Exception:
                        mass = None
                elif len(parts) >= 2:
                    name = parts[0]
                    try:
                        mass = float(parts[-1])
                    except Exception:
                        mass = None
            if mass is None:
                continue
            if not name:
                name = f"{mass:g}"
            items.append((name, float(mass)))
        return items

    def _parse_poly_charges(self, text: str) -> List[int]:
        charges: List[int] = []
        for raw in str(text or "").replace(";", ",").split(","):
            part = raw.strip()
            if not part:
                continue
            try:
                val = int(part)
            except Exception:
                continue
            if val == 0:
                continue
            charges.append(val)
        return charges

    def _open_polymer_match(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Polymer / Reaction Match")
        dlg.setModal(True)

        layout = QVBoxLayout(dlg)
        form = QFormLayout()
        layout.addLayout(form)

        enable_cb = QCheckBox("Enable polymer/reaction matching on spectrum")
        enable_cb.setChecked(bool(self._poly_enabled))
        form.addRow(enable_cb)

        form.addRow(QLabel("Monomers (one per line: name,mass or name mass or mass)"))
        monomers_txt = QTextEdit()
        monomers_txt.setPlainText(self._poly_monomers_text or "")
        monomers_txt.setFixedHeight(90)
        form.addRow(monomers_txt)

        bond_presets = {
            "Dehydration (-H2O)": -18.010565,
            "None (0)": 0.0,
        }
        extra_presets = {
            "None (0)": 0.0,
            "Decarboxylation (-CO2)": -43.989829,
            "Dehydration (-H2O)": -18.010565,
        }

        bond_combo = QComboBox()
        bond_combo.addItems(list(bond_presets.keys()) + ["Custom"])
        bond_edit = QLineEdit(str(self._poly_bond_delta))
        form.addRow("Per-bond delta (polymerization)", bond_combo)
        form.addRow("Custom (Da)", bond_edit)

        extra_combo = QComboBox()
        extra_combo.addItems(list(extra_presets.keys()) + ["Custom"])
        extra_edit = QLineEdit(str(self._poly_extra_delta))
        form.addRow("Extra delta (once per chain)", extra_combo)
        form.addRow("Custom (Da)", extra_edit)

        adduct_edit = QLineEdit(str(self._poly_adduct_mass))
        form.addRow("H adduct mass (Da)", adduct_edit)

        adduct_row = QHBoxLayout()
        cb_na = QCheckBox("+Na")
        cb_k = QCheckBox("+K")
        cb_cl = QCheckBox("+Cl")
        cb_form = QCheckBox("+HCOO")
        cb_ac = QCheckBox("+Ac")
        cb_na.setChecked(bool(self._poly_adduct_na))
        cb_k.setChecked(bool(self._poly_adduct_k))
        cb_cl.setChecked(bool(self._poly_adduct_cl))
        cb_form.setChecked(bool(self._poly_adduct_formate))
        cb_ac.setChecked(bool(self._poly_adduct_acetate))
        adduct_row.addWidget(QLabel("Also match:"))
        adduct_row.addWidget(cb_na)
        adduct_row.addWidget(cb_k)
        adduct_row.addWidget(cb_cl)
        adduct_row.addWidget(cb_form)
        adduct_row.addWidget(cb_ac)
        adduct_row.addStretch(1)
        form.addRow(adduct_row)

        charges_edit = QLineEdit(self._poly_charges_text or "1")
        form.addRow("Charges (comma-separated)", charges_edit)

        decarb_cb = QCheckBox("Also match decarboxylation products (−CO2)")
        decarb_cb.setChecked(bool(self._poly_decarb_enabled))
        oxid_cb = QCheckBox("Also match oxidation products (+O)")
        oxid_cb.setChecked(bool(self._poly_oxid_enabled))
        cluster_cb = QCheckBox("Enable noncovalent polymer dimers (2M−H)")
        cluster_cb.setChecked(bool(self._poly_cluster_enabled))
        form.addRow(decarb_cb)
        form.addRow(oxid_cb)
        form.addRow(cluster_cb)

        cluster_adduct_edit = QLineEdit(str(self._poly_cluster_adduct_mass))
        form.addRow("Cluster H adduct mass (Da)", cluster_adduct_edit)

        max_dp_edit = QLineEdit(str(self._poly_max_dp))
        form.addRow("Max total monomers (DP)", max_dp_edit)

        tol_edit = QLineEdit(str(self._poly_tol_value))
        tol_unit_combo = QComboBox()
        tol_unit_combo.addItems(["Da", "ppm"])
        try:
            tol_unit_combo.setCurrentText(str(self._poly_tol_unit or "Da"))
        except Exception:
            pass
        tol_row = QHBoxLayout()
        tol_row.addWidget(tol_edit)
        tol_row.addWidget(tol_unit_combo)
        form.addRow("Tolerance", tol_row)

        minrel_edit = QLineEdit(str(self._poly_min_rel_int))
        form.addRow("Min peak intensity (fraction of max)", minrel_edit)

        def _sync_combo(combo: QComboBox, presets: Dict[str, float], edit: QLineEdit) -> None:
            choice = combo.currentText()
            if choice in presets:
                edit.setText(str(presets[choice]))

        bond_combo.currentIndexChanged.connect(lambda _i: _sync_combo(bond_combo, bond_presets, bond_edit))
        extra_combo.currentIndexChanged.connect(lambda _i: _sync_combo(extra_combo, extra_presets, extra_edit))

        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        apply_btn = QPushButton("Apply")
        match_btn = QPushButton("Match Region")
        reset_btn = QPushButton("Reset")
        close_btn = QPushButton("Close")
        buttons.addWidget(apply_btn)
        buttons.addWidget(match_btn)
        buttons.addWidget(reset_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)

        def _apply() -> None:
            try:
                self._poly_enabled = bool(enable_cb.isChecked())
                self._poly_monomers_text = monomers_txt.toPlainText().strip()
                self._poly_bond_delta = float(bond_edit.text().strip())
                self._poly_extra_delta = float(extra_edit.text().strip())
                self._poly_adduct_mass = float(adduct_edit.text().strip())
                self._poly_cluster_adduct_mass = float(cluster_adduct_edit.text().strip())
                self._poly_charges_text = str(charges_edit.text().strip() or "1")
                self._poly_decarb_enabled = bool(decarb_cb.isChecked())
                self._poly_oxid_enabled = bool(oxid_cb.isChecked())
                self._poly_cluster_enabled = bool(cluster_cb.isChecked())
                self._poly_adduct_na = bool(cb_na.isChecked())
                self._poly_adduct_k = bool(cb_k.isChecked())
                self._poly_adduct_cl = bool(cb_cl.isChecked())
                self._poly_adduct_formate = bool(cb_form.isChecked())
                self._poly_adduct_acetate = bool(cb_ac.isChecked())
                self._poly_max_dp = int(float(max_dp_edit.text().strip()))
                self._poly_tol_value = float(tol_edit.text().strip())
                self._poly_tol_unit = str(tol_unit_combo.currentText() or "Da")
                self._poly_min_rel_int = float(minrel_edit.text().strip())
            except Exception:
                self.dialogs.error("Polymer Match", "One or more numeric fields are invalid.")
                return

            if self._poly_enable_cb is not None:
                try:
                    self._poly_enable_cb.blockSignals(True)
                    self._poly_enable_cb.setChecked(bool(self._poly_enabled))
                except Exception:
                    pass
                finally:
                    try:
                        self._poly_enable_cb.blockSignals(False)
                    except Exception:
                        pass
            self._render_active_plots()

        def _reset() -> None:
            enable_cb.setChecked(False)
            monomers_txt.setPlainText("")
            bond_combo.setCurrentText("Dehydration (-H2O)")
            bond_edit.setText(str(-18.010565))
            extra_combo.setCurrentText("None (0)")
            extra_edit.setText("0.0")
            adduct_edit.setText("1.007276")
            cb_na.setChecked(False)
            cb_k.setChecked(False)
            cb_cl.setChecked(False)
            cb_form.setChecked(False)
            cb_ac.setChecked(False)
            decarb_cb.setChecked(False)
            oxid_cb.setChecked(False)
            cluster_cb.setChecked(False)
            cluster_adduct_edit.setText("-1.007276")
            charges_edit.setText("1")
            max_dp_edit.setText("12")
            tol_edit.setText("0.02")
            tol_unit_combo.setCurrentText("Da")
            minrel_edit.setText("0.01")
            _apply()

        apply_btn.clicked.connect(_apply)
        match_btn.clicked.connect(_apply)
        reset_btn.clicked.connect(_reset)
        close_btn.clicked.connect(dlg.close)

        _sync_combo(bond_combo, bond_presets, bond_edit)
        _sync_combo(extra_combo, extra_presets, extra_edit)

        dlg.resize(620, 520)
        dlg.exec()
