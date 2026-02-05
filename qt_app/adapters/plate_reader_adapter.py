# === PLATE READER TK INVENTORY (from lab_gui/plate_reader_view.py) ===
# UI Controls & Workflows (PlateReaderView)
# Workspace panel:
# - Add Files… -> PlateReaderView._add_files (multi-file Excel/CSV)
# - Load… -> PlateReaderView._load_plate_reader_workspace
# - Save… -> PlateReaderView._save_plate_reader_workspace
# - Remove -> PlateReaderView._remove_selected
# - Rename… -> PlateReaderView._rename_selected
# - Clear -> PlateReaderView._clear_workspace
# - Workspace tree (datasets) -> PlateReaderView._on_tree_select/_set_active_dataset
#
# Preview:
# - Preview button opens _DataPreviewWindow (read-only table preview)
# - No inline table preview in main tab
#
# Run / Wizard:
# - Run button opens PlateReaderRunWizard
# - Step 1: analysis selection (only MIC enabled; others disabled)
# - Step 2: MIC configuration
#   * "First row is headers" toggle re-reads cached dfs
#   * Preview table (first rows)
#   * Select sample rows (replicates)
#   * Select control rows (optional)
#   * Select concentration columns (in sheet order)
#   * Tick labels entry (comma-separated, must match selected columns)
#   * Auto tick labels helper (2-fold dilution ending in 0)
#   * Plot type: bar/line/scatter
#   * Control style: bars/line
#   * Editable title/x/y labels
#   * Apply -> builds PlateReaderMICWizardConfig/Result and renders plot
#
# Analysis behaviors:
# - Mean±std combining via coerce_numeric_matrix
# - High NaN ratio warning if > 35%
# - Re-run keeps plot styling from previous config
# - Each file analyzed independently without reloading (cached dfs)
#
# Plotting:
# - Matplotlib FigureCanvasTkAgg + NavigationToolbar2Tk
# - render_current_plot uses PlateReaderMICWizardResult.render
#
# Plot Editor:
# - PlateReaderPlotEditor edits labels, colors, line/marker/bar widths,
#   fonts, grid/legend, axis limits, invert x
#
# Workspace persistence:
# - Save/Load JSON: schema_version=1, kind=plate_reader_workspace
# - Fields: active_dataset_id, datasets with wizard_mic_config/result
# === END PLATE READER TK INVENTORY ===

from __future__ import annotations

import json
import traceback
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

from lab_gui.plate_reader_io import read_plate_file
from lab_gui.plate_reader_model import (
    PlateReaderDataset,
    PlateReaderMICWizardConfig,
    PlateReaderMICWizardResult,
    _utc_now_iso,
)
from qt_app.services import DialogService, StatusService


class PlateReaderAdapter:
    def __init__(self, *, status: StatusService, dialogs: DialogService) -> None:
        self.status = status
        self.dialogs = dialogs
        self.plate_reader_datasets: List[PlateReaderDataset] = []
        self.active_plate_reader_id: Optional[str] = None

    def list_datasets(self) -> List[PlateReaderDataset]:
        return list(self.plate_reader_datasets or [])

    def get_active_dataset(self) -> Optional[PlateReaderDataset]:
        if not self.plate_reader_datasets:
            return None
        if self.active_plate_reader_id:
            for ds in self.plate_reader_datasets:
                if str(getattr(ds, "id", "")) == str(self.active_plate_reader_id):
                    return ds
        return self.plate_reader_datasets[-1]

    def set_active_dataset_id(self, dataset_id: Optional[str]) -> None:
        self.active_plate_reader_id = (None if dataset_id is None else str(dataset_id))

    def _safe_dataclass_from_dict(self, cls: Any, data: Optional[dict]) -> Optional[Any]:
        if data is None or not isinstance(data, dict):
            return None
        try:
            fields = getattr(cls, "__dataclass_fields__", {}) or {}
            allowed = set(fields.keys())
            clean = {k: v for k, v in data.items() if k in allowed}
            return cls(**clean)
        except Exception:
            return None

    def _encode_dataset(self, ds: PlateReaderDataset) -> dict:
        cfg = getattr(ds, "wizard_mic_config", None)
        res = getattr(ds, "wizard_mic_result", None)
        return {
            "id": str(getattr(ds, "id", "")),
            "display_name": str(getattr(ds, "display_name", "") or ""),
            "path": (str(getattr(ds, "path", "")) if getattr(ds, "path", None) is not None else ""),
            "sheet_name": (None if getattr(ds, "sheet_name", None) is None else str(getattr(ds, "sheet_name"))),
            "header_row": (None if getattr(ds, "header_row", None) is None else int(getattr(ds, "header_row"))),
            "imported_at_utc": str(getattr(ds, "imported_at_utc", "") or ""),
            "wizard_last_analysis": (None if getattr(ds, "wizard_last_analysis", None) is None else str(getattr(ds, "wizard_last_analysis"))),
            "wizard_mic_config": (None if cfg is None else dict(getattr(cfg, "__dict__", {}) or {})),
            "wizard_mic_result": (None if res is None else dict(getattr(res, "__dict__", {}) or {})),
        }

    def save_workspace(self, path: str) -> bool:
        dss = list(self.plate_reader_datasets or [])
        if not dss:
            self.dialogs.info("Plate Reader", "No Plate Reader files to save.")
            return False

        payload = {
            "schema_version": 1,
            "kind": "plate_reader_workspace",
            "created_utc": _utc_now_iso(),
            "active_dataset_id": (None if self.active_plate_reader_id is None else str(self.active_plate_reader_id)),
            "datasets": [self._encode_dataset(d) for d in dss],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            msg = traceback.format_exc()
            self.dialogs.error("Plate Reader", "Failed to save workspace.\n\n" + msg)
            return False

        return True

    def load_workspace(self, path: str) -> Tuple[List[PlateReaderDataset], Optional[str], List[str]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            msg = traceback.format_exc()
            self.dialogs.error("Plate Reader", "Failed to read workspace JSON.\n\n" + msg)
            return [], None, []

        if not isinstance(payload, dict):
            self.dialogs.error("Plate Reader", "Workspace JSON must be an object.")
            return [], None, []

        rows = payload.get("datasets", [])
        if not isinstance(rows, list):
            rows = []

        loaded: List[PlateReaderDataset] = []
        failures: List[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            p_str = str(row.get("path", "") or "")
            if not p_str:
                continue
            p = Path(p_str)
            if not p.exists():
                failures.append(f"Missing: {p}")
                continue

            ds_id = str(row.get("id", "") or "") or str(uuid.uuid4())
            display_name = str(row.get("display_name", "") or "")
            if not display_name:
                display_name = str(p.name)

            ds = PlateReaderDataset(
                id=ds_id,
                name=str(p.stem),
                path=p,
                display_name=display_name,
                sheet_name=(None if row.get("sheet_name", None) is None else str(row.get("sheet_name"))),
                header_row=(None if row.get("header_row", None) is None else int(row.get("header_row"))),
                imported_at_utc=str(row.get("imported_at_utc", "") or ""),
            )

            try:
                ds.df_header0 = read_plate_file(ds.path, sheet_name=ds.sheet_name, header_row=0)
                ds.df_header_none = read_plate_file(ds.path, sheet_name=ds.sheet_name, header_row=None)
            except Exception:
                failures.append(f"Failed to load: {p}")
                continue

            ds.wizard_last_analysis = row.get("wizard_last_analysis", None)
            cfg = self._safe_dataclass_from_dict(PlateReaderMICWizardConfig, row.get("wizard_mic_config", None))
            res = self._safe_dataclass_from_dict(PlateReaderMICWizardResult, row.get("wizard_mic_result", None))
            ds.wizard_mic_config = cfg
            ds.wizard_mic_result = res
            if cfg is not None:
                try:
                    ds.header_row = 0 if bool(getattr(cfg, "use_first_row_as_header", True)) else None
                except Exception:
                    pass

            loaded.append(ds)

        active_id = payload.get("active_dataset_id", None)
        active_id = (None if active_id is None else str(active_id))
        if active_id is not None and not any(str(getattr(d, "id", "")) == active_id for d in loaded):
            active_id = None

        return loaded, active_id, failures

    def unique_display_name(self, base: str) -> str:
        base = str(base or "Dataset").strip() or "Dataset"
        try:
            existing = {str(getattr(d, "display_name", "") or getattr(d, "name", "") or "") for d in (self.plate_reader_datasets or [])}
        except Exception:
            existing = set()
        if base not in existing:
            return base
        i = 2
        while True:
            cand = f"{base} ({i})"
            if cand not in existing:
                return cand
            i += 1

    def load_files(self, paths: List[str]) -> Tuple[List[PlateReaderDataset], Optional[str]]:
        dss = list(self.plate_reader_datasets or [])
        last_id: Optional[str] = None
        for path in list(paths):
            p = Path(path)
            ds = PlateReaderDataset(
                id=str(uuid.uuid4()),
                name=p.stem,
                display_name=self.unique_display_name(str(p.name)),
                path=p,
                sheet_name=None,
                header_row=0,
                imported_at_utc=_utc_now_iso(),
            )

            try:
                ds.df_header0 = read_plate_file(ds.path, sheet_name=ds.sheet_name, header_row=0)
                ds.df_header_none = read_plate_file(ds.path, sheet_name=ds.sheet_name, header_row=None)
            except Exception:
                msg = traceback.format_exc()
                self.dialogs.error("Plate Reader", f"Failed to load file:\n\n{p}\n\n{msg}")
                continue

            dss.append(ds)
            last_id = ds.id

        self.plate_reader_datasets = dss
        if last_id is not None:
            self.active_plate_reader_id = last_id
        return dss, last_id

    def get_dataset_df(self, ds: PlateReaderDataset) -> Optional[Any]:
        try:
            df = ds.current_df()
            if df is not None:
                return df
        except Exception:
            pass

        try:
            ds.df_header0 = read_plate_file(ds.path, sheet_name=ds.sheet_name, header_row=0)
        except Exception:
            ds.df_header0 = None
        try:
            ds.df_header_none = read_plate_file(ds.path, sheet_name=ds.sheet_name, header_row=None)
        except Exception:
            ds.df_header_none = None

        try:
            return ds.current_df()
        except Exception:
            return None
