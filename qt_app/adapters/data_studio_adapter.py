# === DATA STUDIO TK INVENTORY (from lab_gui/data_studio_view.py etc.) ===
# Workspace / dataset loading:
# - Add Files… (CSV/TSV/Excel) via filedialog.askopenfilenames
# - DataStudioDataset stored in DataStudioWorkspace.datasets + order
# - Each dataset has display_name, sheet_name, header_row, columns, schema_hash
# - Async schema inference uses load_table + column_type_map + schema_hash_from_columns
# - Dataframes cached in _df_cache; reuses cached frames for plotting/preview
#
# Plot types supported:
# - Line, Scatter, Line + markers, Bar (grouped), Bar (stacked), Area, Histogram,
#   Box plot, Violin plot, Heatmap, Bubble, Step, Stem, Errorbar
#
# Axis picking behavior:
# - Default axis picks based on time/index keywords, else numeric columns
# - Y columns chosen via popup dialog; apply only on Apply
# - X is "(Index)" or chosen column; map display name to column
#
# Overlay behavior:
# - Overlay selected datasets only after each plotted
# - Requires same plot type across datasets
# - Overlay offset modes: Normal / Offset Y / Offset X
#
# Export behavior:
# - Export button opens DataStudioExportEditor (matplotlib + style controls)
# - Export editor can save plot and export data
#
# Preview table popup:
# - Opens _PreviewWindow with sheet selector; previews first 500 rows
#
# Workspace save/load:
# - encode_workspace/decode_workspace JSON schema_version=1, kind=data_studio_workspace
# === END DATA STUDIO TK INVENTORY ===

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lab_gui.data_studio_io import (
    column_type_map,
    get_sheet_names,
    load_table,
    normalize_series,
    schema_hash_from_columns,
)
from lab_gui.data_studio_model import DataStudioDataset, DataStudioPlotDef, DataStudioWorkspace


class DataStudioAdapter:
    def __init__(self) -> None:
        self.ws = DataStudioWorkspace()
        self.df_cache: Dict[str, pd.DataFrame] = {}

    def reset(self) -> None:
        self.ws = DataStudioWorkspace()
        self.df_cache = {}

    def get_sheet_names(self, path: Path) -> List[str]:
        return get_sheet_names(path)

    def infer_schema(self, ds: DataStudioDataset, *, decimal_comma: bool, auto_cast: bool) -> Tuple[Dict[str, str], str]:
        df = load_table(ds.path, sheet_name=ds.sheet_name, header_row=ds.header_row, decimal_comma=decimal_comma, auto_cast=auto_cast)
        cols = column_type_map(df)
        schema_hash = schema_hash_from_columns(cols)
        return cols, schema_hash

    def load_df(self, ds: DataStudioDataset, *, decimal_comma: bool, auto_cast: bool) -> pd.DataFrame:
        if ds.dataset_id in self.df_cache:
            return self.df_cache[ds.dataset_id]
        df = load_table(ds.path, sheet_name=ds.sheet_name, header_row=ds.header_row, decimal_comma=decimal_comma, auto_cast=auto_cast)
        self.df_cache[ds.dataset_id] = df
        return df

    def ensure_plot_def_for_dataset(self, dataset_id: str, *, plot_type: str) -> None:
        if not dataset_id:
            return
        for pd in self.ws.plot_defs.values():
            if pd.dataset_id == dataset_id:
                return
        pid = str(__import__("uuid").uuid4())
        self.ws.plot_defs[pid] = DataStudioPlotDef(plot_id=pid, dataset_id=dataset_id, plot_type=plot_type)
        if not self.ws.active_plot_id:
            self.ws.active_plot_id = pid

    def remove_plot_defs_for_dataset(self, dataset_id: str) -> None:
        to_drop = [pid for pid, pd in self.ws.plot_defs.items() if pd.dataset_id == dataset_id]
        for pid in to_drop:
            self.ws.plot_defs.pop(pid, None)
            if self.ws.active_plot_id == pid:
                self.ws.active_plot_id = None

    def build_plot_series(
        self,
        *,
        active_plot_def: DataStudioPlotDef,
        overlay_ids: List[str],
        plot_defs: Dict[str, DataStudioPlotDef],
        drop_na_default: bool,
        normalize_default: str,
        decimal_comma: bool,
        auto_cast: bool,
        xcol_override: Optional[str] = None,
        ycols_override: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        sid_list = overlay_ids if overlay_ids else ([active_plot_def.dataset_id] if active_plot_def.dataset_id else [])
        if not sid_list:
            raise ValueError("No active dataset selected.")

        if overlay_ids:
            defs = [pd for pd in plot_defs.values() if pd.dataset_id in sid_list]
            if len(defs) != len(sid_list):
                raise ValueError("Overlay requires plotting each dataset first (store X/Y selections).")
            pt = str(defs[0].plot_type)
            if any(str(d.plot_type) != pt for d in defs if d is not None):
                raise ValueError("Overlay requires the same plot type across datasets.")

        xcol = active_plot_def.x_col
        ycols = list(active_plot_def.y_cols or [])
        plot_type = str(active_plot_def.plot_type or "Line")
        opts: Dict[str, Any] = dict(active_plot_def.options or {})
        group_col = opts.get("group_col")

        if xcol_override is not None:
            xcol = xcol_override
        if ycols_override is not None:
            ycols = list(ycols_override)

        if not ycols:
            raise ValueError("Select at least one Y column.")

        if group_col == "(None)":
            group_col = None

        series: List[Dict[str, Any]] = []
        meta = {
            "title": "",
            "xlabel": (xcol if xcol and xcol != "(Index)" else "Index"),
            "ylabel": ", ".join(ycols),
            "plot_type": plot_type,
            "xcats": [],
        }

        for sid in sid_list:
            ds = self.ws.datasets.get(sid)
            if ds is None:
                continue
            if overlay_ids:
                for opd in plot_defs.values():
                    if opd.dataset_id == sid:
                        xcol = str(opd.x_col or "(Index)")
                        ycols = list(opd.y_cols or [])
                        plot_type = str(opd.plot_type or plot_type)
                        opts = dict(opd.options or {})
                        group_col = opts.get("group_col")
                        break
            df = self.load_df(ds, decimal_comma=decimal_comma, auto_cast=auto_cast)
            drop_na = bool(opts.get("drop_na", drop_na_default))
            if drop_na:
                df = df.dropna()

            if group_col and group_col not in df.columns:
                group_col = None

            if xcol in (None, "(Index)"):
                xvals = np.asarray(df.index, dtype=float)
            else:
                if xcol not in df.columns:
                    raise ValueError(f"X column '{xcol}' not found.")
                xvals = np.asarray(df[xcol], dtype=float)

            groups = [(None, df)] if not group_col else list(df.groupby(group_col))
            for gval, gdf in groups:
                for y in ycols:
                    if y not in gdf.columns:
                        continue
                    yvals = np.asarray(gdf[y], dtype=float)
                    norm = str(opts.get("normalize", normalize_default))
                    if norm != "None":
                        yvals = normalize_series(yvals, norm)
                    label = f"{ds.display_name}:{y}" + (f" | {group_col}={gval}" if gval is not None else "")
                    kind = "line"

                    if plot_type == "Scatter":
                        kind = "scatter"
                    elif plot_type == "Line + markers":
                        kind = "line"
                    elif plot_type == "Bar (grouped)":
                        kind = "bar"
                    elif plot_type == "Bar (stacked)":
                        kind = "bar"
                    elif plot_type == "Area":
                        kind = "area"
                    elif plot_type == "Histogram":
                        kind = "line"
                    elif plot_type == "Box plot":
                        kind = "line"
                    elif plot_type == "Violin plot":
                        kind = "line"
                    elif plot_type == "Heatmap":
                        kind = "line"
                    elif plot_type == "Bubble":
                        kind = "scatter"
                    elif plot_type == "Step":
                        kind = "step"
                    elif plot_type == "Stem":
                        kind = "stem"
                    elif plot_type == "Errorbar":
                        kind = "errorbar"

                    series.append(
                        {
                            "id": f"{sid}:{y}:{gval}",
                            "kind": kind,
                            "x": xvals,
                            "y": yvals,
                            "label": label,
                            "xerr": None,
                            "yerr": None,
                            "size": None,
                        }
                    )

        if plot_type == "Histogram":
            series = []
            for sid in sid_list:
                ds = self.ws.datasets.get(sid)
                if ds is None:
                    continue
                df = self.load_df(ds, decimal_comma=decimal_comma, auto_cast=auto_cast)
                for y in ycols:
                    if y not in df.columns:
                        continue
                    vals = np.asarray(df[y], dtype=float)
                    series.append({"id": f"{sid}:{y}", "kind": "hist", "x": None, "y": vals, "label": f"{ds.display_name}:{y}"})

        if plot_type == "Heatmap":
            ds = self.ws.datasets.get(sid_list[0])
            df = self.load_df(ds, decimal_comma=decimal_comma, auto_cast=auto_cast) if ds else pd.DataFrame()
            r = str(opts.get("heatmap_row") or "(None)")
            c = str(opts.get("heatmap_col") or "(None)")
            v = str(opts.get("heatmap_val") or "(None)")
            if r == "(None)" or c == "(None)" or v == "(None)":
                raise ValueError("Select Row/Col/Value for heatmap.")
            pv = pd.pivot_table(df, index=r, columns=c, values=v, aggfunc=str(opts.get("heatmap_agg") or "mean"))
            meta["heatmap"] = {"rows": list(pv.index), "cols": list(pv.columns), "values": pv.values}
            series = []

        if plot_type in ("Bar (grouped)", "Bar (stacked)"):
            xcats: List[Any] = []
            grouped_series: List[Dict[str, Any]] = []
            for s in series:
                x = s.get("x")
                y = s.get("y")
                if x is None or y is None:
                    continue
                df = pd.DataFrame({"x": x, "y": y})
                g = df.groupby("x").mean(numeric_only=True).reset_index()
                if not xcats:
                    xcats = g["x"].tolist()
                grouped_series.append({"id": s["id"], "label": s["label"], "y": g["y"].to_numpy(dtype=float)})
            meta["xcats"] = xcats
            series = grouped_series

        if plot_type == "Bubble":
            size_col = str(opts.get("size_col") or "(None)")
            for s in series:
                sid = str(s["id"]).split(":", 1)[0]
                ds = self.ws.datasets.get(sid)
                if ds is None:
                    continue
                df = self.load_df(ds, decimal_comma=decimal_comma, auto_cast=auto_cast)
                if size_col in df.columns:
                    s["size"] = np.asarray(df[size_col], dtype=float)

        if plot_type == "Errorbar":
            xerr_col = str(opts.get("x_err_col") or "(None)")
            yerr_col = str(opts.get("y_err_col") or "(None)")
            for s in series:
                sid = str(s["id"]).split(":", 1)[0]
                ds = self.ws.datasets.get(sid)
                if ds is None:
                    continue
                df = self.load_df(ds, decimal_comma=decimal_comma, auto_cast=auto_cast)
                if xerr_col in df.columns:
                    s["xerr"] = np.asarray(df[xerr_col], dtype=float)
                if yerr_col in df.columns:
                    s["yerr"] = np.asarray(df[yerr_col], dtype=float)

        return series, meta

    def apply_overlay_offset(self, series: List[Dict[str, Any]], *, overlay_ids: List[str], mode: str, offset: float) -> None:
        if not overlay_ids or not series:
            return
        if mode == "Normal" or offset == 0.0:
            return
        ordered_ids: List[str] = []
        for s in series:
            sid = str(s.get("id", "")).split(":", 1)[0]
            if sid and sid not in ordered_ids:
                ordered_ids.append(sid)
        idx_map = {sid: i for i, sid in enumerate(ordered_ids)}
        for s in series:
            sid = str(s.get("id", "")).split(":", 1)[0]
            idx = idx_map.get(sid, 0)
            if mode == "Offset Y":
                s["y"] = np.asarray(s.get("y", []), dtype=float) + (idx * offset)
            elif mode == "Offset X":
                s["x"] = np.asarray(s.get("x", []), dtype=float) + (idx * offset)

    def open_export_editor(self, payload: Dict[str, Any]) -> None:
        return None
