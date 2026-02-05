from __future__ import annotations

import datetime
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lab_gui.external_tools import run_fiji_open, run_fiji_macro
from lab_gui.microscopy_model import MicroscopyDataset, MicroscopyWorkspace
from lab_gui.settings import load_settings, save_settings, validate_imagej_exe_path
from qt_app.services import DialogService, StatusService
from qt_app.services.worker import run_in_worker


SUPPORTED_INPUT_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".czi"}
SUPPORTED_OUTPUT_EXTS = {".csv", ".xlsx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt", ".pdf"}

PRESETS: List[Dict[str, str]] = [
    {
        "id": "particle_size",
        "name": "Particle size (Analyze Particles)",
        "what": "Thresholds the image, converts to mask, runs Analyze Particles, and saves a CSV table.",
        "mode": "particles",
    },
    {
        "id": "droplet_count",
        "name": "Droplet count",
        "what": "Thresholds and counts droplet-like particles via Analyze Particles (Count + sizes).",
        "mode": "particles",
    },
    {
        "id": "area_fraction",
        "name": "Area/Coverage fraction",
        "what": "Thresholds and computes area fraction (foreground fraction) and writes a one-row CSV.",
        "mode": "area_fraction",
    },
    {
        "id": "quick_qc",
        "name": "Quick QC (saturation + focus proxy)",
        "what": "Computes saturation fraction and a simple edge-based focus proxy; writes a one-row CSV.",
        "mode": "qc",
    },
]


class MicroscopyTab(QWidget):
    def __init__(self, status: StatusService, dialogs: DialogService, worker_runner=None) -> None:
        super().__init__()
        self.status = status
        self.dialogs = dialogs
        self.worker_runner = worker_runner or run_in_worker

        self._workspaces: Dict[str, MicroscopyWorkspace] = {}
        self._workspace_order: List[str] = []
        self._active_workspace_id: Optional[str] = None
        self._active_dataset_id: Optional[str] = None
        self._output_cache: Dict[str, List[Path]] = {}
        self._last_workspace_path: Optional[str] = None

        self._preset_combo: Optional[QComboBox] = None
        self._preset_desc: Optional[QLabel] = None
        self._thr_method_cb: Optional[QComboBox] = None
        self._thr_min_edit: Optional[QLineEdit] = None
        self._thr_max_edit: Optional[QLineEdit] = None
        self._min_size_edit: Optional[QLineEdit] = None
        self._max_size_edit: Optional[QLineEdit] = None
        self._circ_min_edit: Optional[QLineEdit] = None
        self._circ_max_edit: Optional[QLineEdit] = None
        self._exclude_edge_cb: Optional[QCheckBox] = None
        self._overlay_cb: Optional[QCheckBox] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._build_ui())

        self._ensure_default_workspace()
        self._refresh_workspace_combo()
        self._refresh_dataset_tree()
        self._refresh_outputs()

    def _build_ui(self) -> QWidget:
        root = QSplitter(Qt.Horizontal)
        root.setChildrenCollapsible(False)
        self._root_splitter = root

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)

        ws = QGroupBox("Workspace")
        ws_layout = QVBoxLayout(ws)
        row = QHBoxLayout()
        row.addWidget(QLabel("Active"))
        self._ws_combo = QComboBox()
        self._ws_combo.currentIndexChanged.connect(self._on_workspace_selected)
        row.addWidget(self._ws_combo, 1)
        ws_layout.addLayout(row)

        ws_btns = QHBoxLayout()
        b_add_ws = QPushButton("Add Workspace…")
        b_add_ws.clicked.connect(self._add_workspace)
        ws_btns.addWidget(b_add_ws)
        b_rename_ws = QPushButton("Rename")
        b_rename_ws.clicked.connect(self._rename_workspace)
        ws_btns.addWidget(b_rename_ws)
        b_del_ws = QPushButton("Delete")
        b_del_ws.clicked.connect(self._delete_workspace)
        ws_btns.addWidget(b_del_ws)
        ws_btns.addStretch(1)
        ws_layout.addLayout(ws_btns)
        left_layout.addWidget(ws)

        ds = QGroupBox("Datasets")
        ds_layout = QVBoxLayout(ds)
        self._ds_tree = QTreeWidget()
        self._ds_tree.setHeaderLabels(["Name", "File", "Outputs"])
        self._ds_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._ds_tree.itemSelectionChanged.connect(self._on_dataset_select)
        ds_layout.addWidget(self._ds_tree, 1)

        ds_btns = QHBoxLayout()
        b_add = QPushButton("Load Files…")
        b_add.clicked.connect(self._load_files)
        ds_btns.addWidget(b_add)
        b_remove = QPushButton("Remove Selected")
        b_remove.clicked.connect(self._remove_selected)
        ds_btns.addWidget(b_remove)
        b_open = QPushButton("Open in Fiji/ImageJ")
        b_open.clicked.connect(self._open_selected_in_imagej)
        ds_btns.addWidget(b_open)
        b_macro = QPushButton("Run Macro (Selected)…")
        b_macro.clicked.connect(self._run_macro_on_selected)
        ds_btns.addWidget(b_macro)
        b_macro_active = QPushButton("Run Macro (Active)…")
        b_macro_active.clicked.connect(self._run_macro_on_active)
        ds_btns.addWidget(b_macro_active)
        b_macro_all = QPushButton("Run Macro (All)…")
        b_macro_all.clicked.connect(self._run_macro_on_all)
        ds_btns.addWidget(b_macro_all)
        b_out = QPushButton("Open Output Folder")
        b_out.clicked.connect(self._open_output_folder)
        ds_btns.addWidget(b_out)
        ds_btns.addStretch(1)
        ds_layout.addLayout(ds_btns)
        left_layout.addWidget(ds, 1)

        settings = QGroupBox("ImageJ/Fiji")
        settings_layout = QHBoxLayout(settings)
        b_set_path = QPushButton("Set Fiji/ImageJ Path…")
        b_set_path.clicked.connect(self._set_imagej_path)
        settings_layout.addWidget(b_set_path)
        self._headless_cb = QCheckBox("Headless")
        self._headless_cb.toggled.connect(self._on_headless_toggled)
        settings_layout.addWidget(self._headless_cb)
        self._imagej_label = QLabel("")
        settings_layout.addWidget(self._imagej_label, 1)
        left_layout.addWidget(settings)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)

        out_box = QGroupBox("Outputs")
        out_layout = QVBoxLayout(out_box)
        self._out_tree = QTreeWidget()
        self._out_tree.setHeaderLabels(["File", "Type", "Size", "Modified"])
        self._out_tree.itemSelectionChanged.connect(self._on_output_select)
        out_layout.addWidget(self._out_tree, 1)

        out_btns = QHBoxLayout()
        b_refresh = QPushButton("Refresh Outputs")
        b_refresh.clicked.connect(self._refresh_outputs)
        out_btns.addWidget(b_refresh)
        b_open_out = QPushButton("Open Selected Output")
        b_open_out.clicked.connect(self._open_selected_output)
        out_btns.addWidget(b_open_out)
        b_import = QPushButton("Import Outputs…")
        b_import.clicked.connect(self._import_outputs)
        out_btns.addWidget(b_import)
        b_export = QPushButton("Export Summary…")
        b_export.clicked.connect(self._export_summary)
        out_btns.addWidget(b_export)
        out_btns.addStretch(1)
        out_layout.addLayout(out_btns)
        right_layout.addWidget(out_box, 1)

        preset_box = self._build_preset_panel()
        right_layout.addWidget(preset_box)

        root.addWidget(left)
        root.addWidget(right)
        root.setStretchFactor(0, 2)
        root.setStretchFactor(1, 3)
        return root

    def _build_preset_panel(self) -> QGroupBox:
        box = QGroupBox("Preset Analysis")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Preset"))
        self._preset_combo = QComboBox()
        for p in PRESETS:
            self._preset_combo.addItem(p["name"], p["id"])
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        row.addWidget(self._preset_combo, 1)
        layout.addLayout(row)

        self._preset_desc = QLabel("")
        self._preset_desc.setWordWrap(True)
        layout.addWidget(self._preset_desc)

        form = QFormLayout()
        self._thr_method_cb = QComboBox()
        self._thr_method_cb.addItems(["Otsu", "Yen", "Triangle", "Manual"])
        self._thr_method_cb.currentIndexChanged.connect(self._update_thr_manual_state)
        form.addRow("Threshold", self._thr_method_cb)

        self._thr_min_edit = QLineEdit("50")
        self._thr_max_edit = QLineEdit("200")
        form.addRow("Manual min", self._thr_min_edit)
        form.addRow("Manual max", self._thr_max_edit)

        self._min_size_edit = QLineEdit("10")
        self._max_size_edit = QLineEdit("Infinity")
        form.addRow("Min size (px^2)", self._min_size_edit)
        form.addRow("Max size (px^2)", self._max_size_edit)

        self._circ_min_edit = QLineEdit("0.00")
        self._circ_max_edit = QLineEdit("1.00")
        form.addRow("Circ min", self._circ_min_edit)
        form.addRow("Circ max", self._circ_max_edit)

        self._exclude_edge_cb = QCheckBox("Exclude edge particles")
        self._exclude_edge_cb.setChecked(True)
        self._overlay_cb = QCheckBox("Create overlay output")
        self._overlay_cb.setChecked(True)
        form.addRow(self._exclude_edge_cb)
        form.addRow(self._overlay_cb)
        layout.addLayout(form)

        btns = QHBoxLayout()
        run_active = QPushButton("Run on Active Image")
        run_active.clicked.connect(self._run_preset_on_active)
        btns.addWidget(run_active)
        run_sel = QPushButton("Run on Selected Images…")
        run_sel.clicked.connect(self._run_preset_on_selected)
        btns.addWidget(run_sel)
        run_all = QPushButton("Run on ALL in Workspace")
        run_all.clicked.connect(self._run_preset_on_all)
        btns.addWidget(run_all)
        btns.addStretch(1)
        layout.addLayout(btns)

        self._on_preset_changed()
        self._update_thr_manual_state()
        return box

    def reset_layout(self) -> None:
        try:
            if getattr(self, "_root_splitter", None) is not None:
                self._root_splitter.setSizes([520, 880])
        except Exception:
            pass

    def _ensure_default_workspace(self) -> None:
        if self._workspaces:
            return
        wid = str(uuid.uuid4())
        ws = MicroscopyWorkspace(id=wid, name="Default")
        self._workspaces[wid] = ws
        self._workspace_order.append(wid)
        self._active_workspace_id = wid

    def _current_workspace(self) -> Optional[MicroscopyWorkspace]:
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
        self._update_imagej_label()

    def _refresh_dataset_tree(self) -> None:
        self._ds_tree.clear()
        ws = self._current_workspace()
        if ws is None:
            return
        for ds in ws.datasets:
            outs = self._output_cache.get(ds.id, [])
            item = QTreeWidgetItem([ds.display_name, str(Path(ds.file_path).name), str(len(outs))])
            item.setData(0, Qt.UserRole, ds.id)
            self._ds_tree.addTopLevelItem(item)

    def _on_workspace_selected(self, _index: int = 0) -> None:
        wid = self._ws_combo.currentData()
        if wid and str(wid) in self._workspaces:
            self._active_workspace_id = str(wid)
            self._active_dataset_id = None
            self._refresh_dataset_tree()
            self._refresh_outputs()

    def _add_workspace(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Workspace", "Workspace name")
        if not ok or not str(name).strip():
            return
        wid = str(uuid.uuid4())
        ws = MicroscopyWorkspace(id=wid, name=str(name).strip())
        self._workspaces[wid] = ws
        self._workspace_order.append(wid)
        self._active_workspace_id = wid
        self._refresh_workspace_combo()
        self._refresh_dataset_tree()

    def _rename_workspace(self) -> None:
        ws = self._current_workspace()
        if ws is None:
            return
        name, ok = QInputDialog.getText(self, "Rename Workspace", "Name", text=ws.name)
        if not ok or not str(name).strip():
            return
        ws.name = str(name).strip()
        self._workspaces[ws.id] = ws
        self._refresh_workspace_combo()

    def _delete_workspace(self) -> None:
        if len(self._workspace_order) <= 1:
            self.dialogs.info("Microscopy", "At least one workspace is required.")
            return
        ws = self._current_workspace()
        if ws is None:
            return
        self._workspaces.pop(ws.id, None)
        if ws.id in self._workspace_order:
            self._workspace_order.remove(ws.id)
        self._active_workspace_id = self._workspace_order[0] if self._workspace_order else None
        self._refresh_workspace_combo()
        self._refresh_dataset_tree()
        self._refresh_outputs()

    def _load_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Microscopy Files",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.czi);;All files (*.*)",
        )
        if not paths:
            return
        ws = self._current_workspace()
        if ws is None:
            return
        for p in paths:
            path = Path(p)
            if path.suffix.lower() not in SUPPORTED_INPUT_EXTS:
                continue
            ds = MicroscopyDataset(
                id=str(uuid.uuid4()),
                display_name=path.stem,
                file_path=str(path),
                workspace_id=ws.id,
                created_at=self._utc_now_iso(),
                notes="",
                output_dir=self._dataset_output_dir(ws, path),
            )
            try:
                Path(ds.output_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            ws.datasets.append(ds)
        self._workspaces[ws.id] = ws
        self._refresh_dataset_tree()
        self._refresh_outputs()
        self.status.set_status(f"Loaded {len(paths)} file(s)")

    def _remove_selected(self) -> None:
        ws = self._current_workspace()
        if ws is None:
            return
        items = self._ds_tree.selectedItems()
        if not items:
            return
        ids = {str(it.data(0, Qt.UserRole) or "") for it in items}
        ws.datasets = [d for d in ws.datasets if d.id not in ids]
        for did in ids:
            self._output_cache.pop(did, None)
        self._workspaces[ws.id] = ws
        self._active_dataset_id = None
        self._refresh_dataset_tree()
        self._refresh_outputs()

    def _on_dataset_select(self) -> None:
        items = self._ds_tree.selectedItems()
        if not items:
            self._active_dataset_id = None
            self._refresh_outputs()
            return
        ds_id = str(items[0].data(0, Qt.UserRole) or "")
        self._active_dataset_id = ds_id or None
        self._refresh_outputs()

    def _on_output_select(self) -> None:
        pass

    def _dataset_output_dir(self, ws: MicroscopyWorkspace, path: Path) -> str:
        ws_part = self._safe_dirname(ws.name) or self._safe_dirname(ws.id)
        ds_part = self._safe_dirname(path.stem)
        base = Path.cwd() / "microscopy" / ws_part / ds_part
        return str(base)

    def _refresh_outputs(self) -> None:
        self._out_tree.clear()
        ds = self._get_active_dataset()
        if ds is None or not ds.output_dir:
            return
        files = self._discover_outputs_limited(ds.output_dir)
        self._output_cache[ds.id] = files
        for f in files:
            try:
                size = f.stat().st_size
                mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                size = 0
                mtime = ""
            item = QTreeWidgetItem([f.name, f.suffix.lower(), self._format_size(size), mtime])
            item.setData(0, Qt.UserRole, str(f))
            self._out_tree.addTopLevelItem(item)
        self._refresh_dataset_tree()

    def _open_selected_output(self) -> None:
        items = self._out_tree.selectedItems()
        if not items:
            return
        path = str(items[0].data(0, Qt.UserRole) or "")
        if not path:
            return
        try:
            os.startfile(path)
        except Exception as exc:
            self.dialogs.error("Microscopy", f"Failed to open file:\n\n{exc}")

    def _open_output_folder(self) -> None:
        ds = self._get_active_dataset()
        if ds is None or not ds.output_dir:
            return
        try:
            os.startfile(str(ds.output_dir))
        except Exception as exc:
            self.dialogs.error("Microscopy", f"Failed to open folder:\n\n{exc}")

    def _import_outputs(self) -> None:
        ds = self._get_active_dataset()
        if ds is None or not ds.output_dir:
            self.dialogs.info("Microscopy", "Select a dataset first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Import Outputs", "")
        if not folder:
            return
        src = Path(folder)
        dst = Path(ds.output_dir)
        copied = 0
        for f in src.rglob("*"):
            try:
                if not f.is_file():
                    continue
                if f.suffix.lower() not in SUPPORTED_OUTPUT_EXTS:
                    continue
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(f), str(dst / f.name))
                copied += 1
            except Exception:
                continue
        self._refresh_outputs()
        self.status.set_status(f"Imported {copied} file(s)")

    def _export_summary(self) -> None:
        ws = self._current_workspace()
        if ws is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Summary",
            "",
            "Excel (*.xlsx);;CSV (*.csv);;All files (*.*)",
        )
        if not path:
            return
        rows = []
        for ds in ws.datasets:
            outs = self._output_cache.get(ds.id)
            if outs is None:
                outs = self._discover_outputs_limited(ds.output_dir) if ds.output_dir else []
            rows.append(
                {
                    "workspace": ws.name,
                    "dataset": ds.display_name,
                    "file_path": ds.file_path,
                    "output_dir": ds.output_dir,
                    "output_count": len(outs or []),
                }
            )
        df = pd.DataFrame(rows)
        try:
            if path.lower().endswith(".csv"):
                df.to_csv(path, index=False)
            else:
                df.to_excel(path, index=False)
            self.status.set_status("Summary exported")
        except Exception as exc:
            self.dialogs.error("Microscopy", f"Failed to export summary:\n\n{exc}")

    def _set_imagej_path(self) -> None:
        settings = load_settings()
        path, _ = QFileDialog.getOpenFileName(self, "Select ImageJ/Fiji executable", "", "Executable (*.exe)")
        if not path:
            return
        valid = validate_imagej_exe_path(path)
        if not valid:
            self.dialogs.error("Microscopy", "Invalid ImageJ/Fiji executable.")
            return
        settings["fiji_exe_path"] = valid
        save_settings(settings)
        self._update_imagej_label()
        self.status.set_status("ImageJ/Fiji path saved")

    def _open_selected_in_imagej(self) -> None:
        settings = load_settings()
        exe = validate_imagej_exe_path(settings.get("fiji_exe_path"))
        if not exe:
            self.dialogs.error("Microscopy", "Set ImageJ/Fiji path first.")
            return
        items = self._ds_tree.selectedItems()
        if not items:
            return
        for it in items:
            ds_id = str(it.data(0, Qt.UserRole) or "")
            ds = self._find_dataset(ds_id)
            if ds is None:
                continue
            try:
                run_fiji_open(exe, ds.file_path)
            except Exception as exc:
                self.dialogs.error("Microscopy", f"Failed to open file:\n\n{exc}")
                return

    def _run_macro_on_selected(self) -> None:
        settings = load_settings()
        exe = validate_imagej_exe_path(settings.get("fiji_exe_path"))
        if not exe:
            self.dialogs.error("Microscopy", "Set ImageJ/Fiji path first.")
            return
        items = self._ds_tree.selectedItems()
        if not items:
            self.dialogs.info("Microscopy", "Select one or more datasets first.")
            return
        macro_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ImageJ Macro",
            "",
            "ImageJ Macro (*.ijm);;All files (*.*)",
        )
        if not macro_path:
            return
        ds_list: List[MicroscopyDataset] = []
        for it in items:
            ds_id = str(it.data(0, Qt.UserRole) or "")
            ds = self._find_dataset(ds_id)
            if ds is not None:
                ds_list.append(ds)
        if not ds_list:
            return
        self._run_macro_batch(exe, macro_path, ds_list)

    def _run_macro_on_active(self) -> None:
        settings = load_settings()
        exe = validate_imagej_exe_path(settings.get("fiji_exe_path"))
        if not exe:
            self.dialogs.error("Microscopy", "Set ImageJ/Fiji path first.")
            return
        ds = self._get_active_dataset()
        if ds is None:
            self.dialogs.info("Microscopy", "Select an active dataset first.")
            return
        macro_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ImageJ Macro",
            "",
            "ImageJ Macro (*.ijm);;All files (*.*)",
        )
        if not macro_path:
            return
        self._run_macro_batch(exe, macro_path, [ds])

    def _run_macro_on_all(self) -> None:
        settings = load_settings()
        exe = validate_imagej_exe_path(settings.get("fiji_exe_path"))
        if not exe:
            self.dialogs.error("Microscopy", "Set ImageJ/Fiji path first.")
            return
        ws = self._current_workspace()
        if ws is None or not ws.datasets:
            self.dialogs.info("Microscopy", "No datasets in workspace.")
            return
        macro_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ImageJ Macro",
            "",
            "ImageJ Macro (*.ijm);;All files (*.*)",
        )
        if not macro_path:
            return
        self._run_macro_batch(exe, macro_path, list(ws.datasets))

    def _run_macro_batch(self, exe: str, macro_path: str, ds_list: List[MicroscopyDataset]) -> None:
        settings = load_settings()
        if not ds_list:
            return

        progress_state = {"value": 0}
        cancel_state = {"cancel": False}
        handle_ref = {"handle": None}

        progress = QProgressDialog("Running ImageJ macro…", "Cancel", 0, len(ds_list), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)

        def _on_cancel() -> None:
            cancel_state["cancel"] = True
            handle = handle_ref.get("handle")
            if handle is not None:
                try:
                    handle.cancel()
                except Exception:
                    pass

        progress.canceled.connect(_on_cancel)

        timer = QTimer(progress)

        def _tick() -> None:
            try:
                progress.setValue(int(progress_state["value"]))
            except Exception:
                pass

        timer.timeout.connect(_tick)
        timer.start(200)

        def _work(_h):
            for i, ds in enumerate(ds_list):
                if cancel_state.get("cancel") or getattr(_h, "cancelled", False):
                    break
                out_dir = ds.output_dir or ""
                try:
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                try:
                    proc = run_fiji_macro(
                        exe_path=exe,
                        macro_path=macro_path,
                        image_path=ds.file_path,
                        out_dir=out_dir,
                        headless=bool(settings.get("microscopy_run_headless", True)),
                    )
                    try:
                        proc.wait(timeout=1800)
                    except Exception:
                        try:
                            proc.wait()
                        except Exception:
                            pass
                except Exception:
                    continue
                progress_state["value"] = i + 1
            return True

        def _done(_res) -> None:
            try:
                timer.stop()
            except Exception:
                pass
            try:
                progress.setValue(len(ds_list))
                progress.close()
            except Exception:
                pass
            self._refresh_outputs()
            self.status.set_status("Macro run finished")

        handle = self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Running ImageJ macro",
            group="microscopy_macro",
            cancel_previous=True,
        )
        handle_ref["handle"] = handle

    def _update_imagej_label(self) -> None:
        settings = load_settings()
        exe = validate_imagej_exe_path(settings.get("fiji_exe_path"))
        self._imagej_label.setText(exe or "ImageJ/Fiji path not set")
        try:
            self._headless_cb.blockSignals(True)
            self._headless_cb.setChecked(bool(settings.get("microscopy_run_headless", True)))
        except Exception:
            pass
        finally:
            try:
                self._headless_cb.blockSignals(False)
            except Exception:
                pass

    def _on_headless_toggled(self, checked: bool) -> None:
        settings = load_settings()
        settings["microscopy_run_headless"] = bool(checked)
        save_settings(settings)

    def _on_preset_changed(self) -> None:
        preset = self._current_preset()
        if self._preset_desc is not None:
            self._preset_desc.setText(str(preset.get("what") or ""))

    def _current_preset(self) -> Dict[str, str]:
        if self._preset_combo is None:
            return PRESETS[0]
        pid = str(self._preset_combo.currentData() or "")
        for p in PRESETS:
            if str(p.get("id")) == pid:
                return p
        return PRESETS[0]

    def _update_thr_manual_state(self) -> None:
        if self._thr_method_cb is None:
            return
        is_manual = str(self._thr_method_cb.currentText() or "").lower() == "manual"
        for w in (self._thr_min_edit, self._thr_max_edit):
            if w is None:
                continue
            w.setEnabled(bool(is_manual))

    def _gather_preset_params(self) -> Dict[str, object]:
        return {
            "threshold_method": ("Otsu" if self._thr_method_cb is None else str(self._thr_method_cb.currentText() or "Otsu")),
            "manual_min": ("50" if self._thr_min_edit is None else str(self._thr_min_edit.text() or "")),
            "manual_max": ("200" if self._thr_max_edit is None else str(self._thr_max_edit.text() or "")),
            "min_size": ("10" if self._min_size_edit is None else str(self._min_size_edit.text() or "")),
            "max_size": ("Infinity" if self._max_size_edit is None else str(self._max_size_edit.text() or "")),
            "circ_min": ("0.00" if self._circ_min_edit is None else str(self._circ_min_edit.text() or "")),
            "circ_max": ("1.00" if self._circ_max_edit is None else str(self._circ_max_edit.text() or "")),
            "exclude_edge": bool(self._exclude_edge_cb.isChecked()) if self._exclude_edge_cb is not None else True,
            "overlay": bool(self._overlay_cb.isChecked()) if self._overlay_cb is not None else True,
        }

    def _selected_dataset_ids(self) -> List[str]:
        ids: List[str] = []
        for it in self._ds_tree.selectedItems() or []:
            did = str(it.data(0, Qt.UserRole) or "")
            if did:
                ids.append(did)
        return ids

    def _run_preset_on_active(self) -> None:
        ds = self._get_active_dataset()
        if ds is None:
            self.dialogs.info("Preset Analysis", "Select an active dataset first.")
            return
        self._run_preset_on_dataset_ids([ds.id])

    def _run_preset_on_selected(self) -> None:
        ids = self._selected_dataset_ids()
        if not ids:
            self.dialogs.info("Preset Analysis", "Select one or more datasets first.")
            return
        self._run_preset_on_dataset_ids(ids)

    def _run_preset_on_all(self) -> None:
        ws = self._current_workspace()
        if ws is None or not ws.datasets:
            self.dialogs.info("Preset Analysis", "No datasets in the active workspace.")
            return
        self._run_preset_on_dataset_ids([str(d.id) for d in ws.datasets])

    def _find_workspace_by_id(self, wid: str) -> Optional[MicroscopyWorkspace]:
        return self._workspaces.get(str(wid))

    def _run_preset_on_dataset_ids(self, dataset_ids: List[str]) -> None:
        exe = validate_imagej_exe_path(load_settings().get("fiji_exe_path"))
        if not exe:
            self.dialogs.error("Preset Analysis", "Set ImageJ/Fiji path first.")
            return

        preset = self._current_preset()
        params = self._gather_preset_params()
        headless = bool(load_settings().get("microscopy_run_headless", True))

        progress_state = {"value": 0}
        cancel_state = {"cancel": False}
        handle_ref = {"handle": None}

        progress = QProgressDialog(f"Running preset: {preset['name']}", "Cancel", 0, len(dataset_ids), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)

        def _on_cancel() -> None:
            cancel_state["cancel"] = True
            handle = handle_ref.get("handle")
            if handle is not None:
                try:
                    handle.cancel()
                except Exception:
                    pass

        progress.canceled.connect(_on_cancel)

        timer = QTimer(progress)

        def _tick() -> None:
            try:
                progress.setValue(int(progress_state["value"]))
            except Exception:
                pass

        timer.timeout.connect(_tick)
        timer.start(200)

        def _work(_h):
            failures: List[str] = []
            for i, did in enumerate(dataset_ids):
                if cancel_state.get("cancel") or getattr(_h, "cancelled", False):
                    break
                ds = self._find_dataset(did)
                if ds is None:
                    continue
                ws = self._find_workspace_by_id(getattr(ds, "workspace_id", "")) or self._current_workspace()
                if ws is None:
                    continue
                in_path = Path(str(ds.file_path))
                if not in_path.exists():
                    failures.append(f"{ds.display_name}: input file not found")
                    continue

                run_dir, ts = self._make_run_output_dir(ws, ds, preset["id"])
                try:
                    run_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                macro_path = run_dir / "preset.ijm"
                log_path = run_dir / "run_log.txt"
                results_path = run_dir / "results.csv"

                try:
                    macro_text = self._render_preset_macro(preset_id=preset["id"], params=params)
                    macro_path.write_text(macro_text, encoding="utf-8")
                except Exception as exc:
                    failures.append(f"{ds.display_name}: failed to write macro ({exc})")
                    continue

                proc = None
                try:
                    proc = run_fiji_macro(
                        exe,
                        str(macro_path),
                        str(in_path),
                        str(run_dir),
                        headless=headless,
                        log_name="run_log.txt",
                    )
                    proc.wait()
                except Exception as exc:
                    failures.append(f"{ds.display_name}: macro failed ({exc})")
                if cancel_state.get("cancel"):
                    try:
                        if proc is not None and proc.poll() is None:
                            proc.terminate()
                    except Exception:
                        pass
                    break

                if not results_path.exists():
                    failures.append(f"{ds.display_name}: results.csv not created (log: {log_path})")

                ds.last_macro_run = str(ts)
                progress_state["value"] = i + 1

            return failures

        def _done(failures: List[str]) -> None:
            try:
                timer.stop()
            except Exception:
                pass
            try:
                progress.setValue(len(dataset_ids))
                progress.close()
            except Exception:
                pass
            self._refresh_outputs()
            self._refresh_dataset_tree()
            if failures and not cancel_state.get("cancel"):
                self.dialogs.warn("Preset Analysis", "\n".join(failures[:6]))
            elif cancel_state.get("cancel"):
                self.dialogs.info("Preset Analysis", "Run cancelled (best-effort).")
            else:
                self.status.set_status("Preset run finished")

        handle = self.worker_runner(
            _work,
            on_result=_done,
            status=self.status,
            description="Running preset analysis",
            group="microscopy_preset",
            cancel_previous=True,
        )
        handle_ref["handle"] = handle

    def _make_run_output_dir(self, ws: MicroscopyWorkspace, ds: MicroscopyDataset, preset_id: str) -> Tuple[Path, str]:
        base = Path(str(ds.output_dir or "")).expanduser() if ds.output_dir else Path.cwd()
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        preset_part = self._safe_dirname(str(preset_id))
        ws_part = self._safe_dirname(str(getattr(ws, "name", "workspace")))
        ds_part = self._safe_dirname(str(getattr(ds, "display_name", "dataset")))
        out_dir = base / "MicroscopyResults" / ws_part / ds_part / preset_part / ts
        return out_dir, ts

    def _render_preset_macro(self, *, preset_id: str, params: Dict[str, object]) -> str:
        thr_method = str(params.get("threshold_method") or "Otsu").strip()
        manual_min = str(params.get("manual_min") or "50").strip()
        manual_max = str(params.get("manual_max") or "200").strip()
        min_size = str(params.get("min_size") or "10").strip()
        max_size = str(params.get("max_size") or "Infinity").strip()
        circ_min = str(params.get("circ_min") or "0.00").strip()
        circ_max = str(params.get("circ_max") or "1.00").strip()
        exclude_edge = "true" if bool(params.get("exclude_edge", True)) else "false"
        overlay = "true" if bool(params.get("overlay", True)) else "false"

        def q(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', "\\\"")

        thr_method_q = q(thr_method)

        common = f"""
// PresetAnalysis (generated)

function getKV(args, key) {{
    k = key + "=";
    i = indexOf(args, k);
    if (i < 0) return "";
    i = i + lengthOf(k);
    if (substring(args, i, i+1) == "\"") {{
        j = indexOf(args, "\"", i+1);
        if (j < 0) return substring(args, i+1);
        return substring(args, i+1, j);
    }}
    j = indexOf(args, " " , i);
    if (j < 0) j = lengthOf(args);
    return substring(args, i, j);
}}

args = getArgument();
input = getKV(args, "input");
out = getKV(args, "output");
if (input=="" || out=="") {{
    print("Missing input/output args");
    exit(1);
}}

function logLine(s) {{
    File.append(s+"\n", out + "/run_log.txt");
    print(s);
}}

File.makeDirectory(out);
logLine("Input: " + input);
logLine("Output: " + out);

setBatchMode(true);
open(input);
title = getTitle();

run("8-bit");

method = "{thr_method_q}";
if (toLowerCase(method) == "manual") {{
    setThreshold(parseFloat("{q(manual_min)}"), parseFloat("{q(manual_max)}"));
    setOption("BlackBackground", true);
    run("Convert to Mask");
}} else {{
    setAutoThreshold(method + " dark");
    setOption("BlackBackground", true);
    run("Convert to Mask");
}}

run("Make Binary");
"""

        if preset_id in ("particle_size", "droplet_count"):
            return (
                common
                + f"""
run("Set Measurements...", "area mean min perimeter shape feret's redirect=None decimal=3");

edgeOpt = {exclude_edge};
overlayOpt = {overlay};
ap = "size={q(min_size)}-{q(max_size)} circularity={q(circ_min)}-{q(circ_max)}";
if (edgeOpt) ap = ap + " exclude";
if (overlayOpt) ap = ap + " show=Overlay"; else ap = ap + " show=Nothing";
ap = ap + " clear";
run("Analyze Particles...", ap);

saveAs("Results", out + "/results.csv");
if (overlayOpt) {{
    selectWindow(title);
    saveAs("PNG", out + "/overlay.png");
}}

close();
run("Close All");
setBatchMode(false);
logLine("Done");
"""
            )

        if preset_id == "area_fraction":
            return (
                common
                + """
run("Set Measurements...", "area area_fraction redirect=None decimal=6");
run("Measure");
saveAs("Results", out + "/results.csv");

close();
run("Close All");
setBatchMode(false);
logLine("Done");
"""
            )

        if preset_id == "quick_qc":
            return """
// PresetAnalysis (generated)

function getKV(args, key) {
    k = key + "=";
    i = indexOf(args, k);
    if (i < 0) return "";
    i = i + lengthOf(k);
    if (substring(args, i, i+1) == "\"") {
        j = indexOf(args, "\"", i+1);
        if (j < 0) return substring(args, i+1);
        return substring(args, i+1, j);
    }
    j = indexOf(args, " " , i);
    if (j < 0) j = lengthOf(args);
    return substring(args, i, j);
}

args = getArgument();
input = getKV(args, "input");
out = getKV(args, "output");
if (input=="" || out=="") {
    print("Missing input/output args");
    exit(1);
}

function logLine(s) {
    File.append(s+"\n", out + "/run_log.txt");
    print(s);
}

File.makeDirectory(out);
setBatchMode(true);
open(input);
run("8-bit");

getHistogram(values, counts, 256);
total = 0;
for (i=0; i<256; i++) total = total + counts[i];
sat = counts[255];
sat_frac = (total>0) ? (sat/total) : 0;

run("Duplicate...", "title=edges");
run("Find Edges");
getStatistics(area, mean, min, max, std);
focus_proxy = std;
close();

File.saveString("saturation_fraction,focus_proxy\n" + sat_frac + "," + focus_proxy + "\n", out + "/results.csv");
logLine("saturation_fraction=" + sat_frac);
logLine("focus_proxy=" + focus_proxy);

close();
run("Close All");
setBatchMode(false);
logLine("Done");
"""

        return common

    def _find_dataset(self, dataset_id: str) -> Optional[MicroscopyDataset]:
        for ws in self._workspaces.values():
            for ds in ws.datasets:
                if str(ds.id) == str(dataset_id):
                    return ds
        return None

    def _get_active_dataset(self) -> Optional[MicroscopyDataset]:
        if not self._active_dataset_id:
            return None
        return self._find_dataset(self._active_dataset_id)

    def _discover_outputs_limited(self, out_dir: str) -> List[Path]:
        p = Path(out_dir)
        if not p.exists() or not p.is_dir():
            return []
        found: List[Path] = []
        start = datetime.datetime.now().timestamp()
        max_seconds = 2.0
        max_files = 5000
        try:
            for f in p.rglob("*"):
                if (datetime.datetime.now().timestamp() - start) > max_seconds:
                    break
                if len(found) >= max_files:
                    break
                try:
                    if not f.is_file():
                        continue
                    if f.suffix.lower() in SUPPORTED_OUTPUT_EXTS:
                        found.append(f)
                except Exception:
                    continue
        except Exception:
            return []
        found.sort(key=lambda x: x.name.lower())
        return found

    def _safe_dirname(self, name: str) -> str:
        s = "".join(ch for ch in str(name) if ch.isalnum() or ch in ("-", "_", " ")).strip()
        s = s.replace(" ", "_")
        return s or "workspace"

    def _format_size(self, size: int) -> str:
        try:
            if size > 1024 * 1024:
                return f"{size / (1024 * 1024):.1f} MB"
            if size > 1024:
                return f"{size / 1024:.1f} KB"
            return f"{size} B"
        except Exception:
            return ""

    def _utc_now_iso(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

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
        ws_rows: List[Dict[str, object]] = []
        for wid in self._workspace_order:
            ws = self._workspaces.get(wid)
            if ws is None:
                continue
            ds_rows: List[Dict[str, object]] = []
            for d in ws.datasets:
                ds_rows.append(
                    {
                        "id": str(d.id),
                        "display_name": str(d.display_name),
                        "file_path": str(d.file_path),
                        "workspace_id": str(d.workspace_id),
                        "created_at": str(d.created_at),
                        "notes": str(d.notes),
                        "output_dir": str(d.output_dir),
                        "last_macro_run": (None if d.last_macro_run is None else str(d.last_macro_run)),
                    }
                )
            ws_rows.append({"id": ws.id, "name": ws.name, "datasets": ds_rows})
        return {
            "schema_version": 1,
            "app": "MFP lab analysis tool",
            "created_utc": self._utc_now_iso(),
            "workspace": {
                "microscopy_workspaces": ws_rows,
                "active_microscopy_workspace_id": self._active_workspace_id,
            },
        }

    def _save_workspace(self) -> Optional[str]:
        if not self._workspaces:
            self.dialogs.info("Microscopy", "No workspaces to save.")
            return None
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Microscopy Workspace",
            "",
            "Microscopy Workspace (*.microscopy.workspace.json);;JSON (*.json);;All files (*.*)",
        )
        if not path:
            return None
        payload = self._encode_workspace()
        try:
            with open(path, "w", encoding="utf-8") as f:
                import json

                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.dialogs.error("Microscopy", f"Failed to save workspace:\n\n{exc}")
            return None
        self._last_workspace_path = str(path)
        self.status.set_status("Microscopy workspace saved")
        return str(path)

    def _load_workspace(self) -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Microscopy Workspace",
            "",
            "Microscopy Workspace (*.microscopy.workspace.json);;JSON (*.json);;All files (*.*)",
        )
        if not path:
            return None
        return self._load_workspace_path(str(path))

    def _load_workspace_path(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                import json

                payload = json.load(f)
        except Exception as exc:
            self.dialogs.error("Microscopy", f"Failed to read workspace JSON:\n\n{exc}")
            return None
        if not isinstance(payload, dict):
            self.dialogs.error("Microscopy", "Workspace JSON must be an object.")
            return None

        ws_obj = payload.get("workspace") if isinstance(payload.get("workspace"), dict) else payload
        if not isinstance(ws_obj, dict):
            self.dialogs.error("Microscopy", "Workspace JSON missing workspace data.")
            return None

        rows = ws_obj.get("microscopy_workspaces") if isinstance(ws_obj.get("microscopy_workspaces"), list) else []
        active_id = ws_obj.get("active_microscopy_workspace_id")

        self._workspaces = {}
        self._workspace_order = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            wid = str(row.get("id") or uuid.uuid4())
            ws = MicroscopyWorkspace(id=wid, name=str(row.get("name") or "Workspace"))
            ds_rows = row.get("datasets") if isinstance(row.get("datasets"), list) else []
            for d in ds_rows:
                if not isinstance(d, dict):
                    continue
                ds = MicroscopyDataset(
                    id=str(d.get("id") or uuid.uuid4()),
                    display_name=str(d.get("display_name") or "Dataset"),
                    file_path=str(d.get("file_path") or ""),
                    workspace_id=str(d.get("workspace_id") or wid),
                    created_at=str(d.get("created_at") or ""),
                    notes=str(d.get("notes") or ""),
                    output_dir=str(d.get("output_dir") or ""),
                    last_macro_run=(None if d.get("last_macro_run") in (None, "") else str(d.get("last_macro_run"))),
                )
                ws.datasets.append(ds)
            self._workspaces[wid] = ws
            self._workspace_order.append(wid)

        if active_id and str(active_id) in self._workspaces:
            self._active_workspace_id = str(active_id)
        elif self._workspace_order:
            self._active_workspace_id = self._workspace_order[0]
        else:
            self._ensure_default_workspace()

        self._refresh_workspace_combo()
        self._refresh_dataset_tree()
        self._refresh_outputs()
        self._last_workspace_path = str(path)
        self.status.set_status("Microscopy workspace loaded")
        return str(path)
