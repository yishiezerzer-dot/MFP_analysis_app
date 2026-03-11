from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import ttkbootstrap as tb
import tkinter.ttk as ttk_native

ttk: Any = tb
ttk.LabelFrame = ttk_native.LabelFrame  # type: ignore[attr-defined]

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Keep this import aligned with Matplotlib stubs to avoid Pylance false-positives.
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure

from lab_gui.ui_widgets import MatplotlibNavigator, ToolTip
from lab_gui.ui_theme import style_primary, style_success, style_danger, style_secondary
from lab_gui.plot_card import PlotCard
from lab_gui.data_studio_model import DataStudioDataset, DataStudioWorkspace, DataStudioPlotDef
from lab_gui.data_studio_io import (
    apply_transform_steps,
    column_type_map,
    get_sheet_names,
    load_table,
    normalize_series,
    numeric_columns,
    schema_hash_from_columns,
)
from lab_gui.data_studio_export import DataStudioExportEditor
from lab_gui.data_studio_workspace_io import decode_workspace, encode_workspace


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


class _PreviewWindow(tk.Toplevel):
    def __init__(self, parent: tk.Widget, *, path: Path, dataset: DataStudioDataset, df: pd.DataFrame) -> None:
        super().__init__(parent)
        self.title(f"Preview — {path.name}")
        try:
            self.geometry("1000x700")
        except Exception:
            pass

        self._path = path
        self._dataset = dataset
        self._df = df
        self._view_df = df
        self._render_after: Optional[str] = None
        self._filter_after: Optional[str] = None
        self._sort_col: Optional[str] = None
        self._sort_asc: bool = True
        self._preview_cap = 5000
        self._numeric_cols: List[str] = []
        self._all_filter_label = "(All columns)"
        self._search_series: Optional[pd.Series] = None
        self._suppress_filter_traces: bool = False

        summary = ttk.Frame(self, style="ShellPanel.TFrame", padding=(14, 12))
        summary.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        summary.columnconfigure(0, weight=1)
        ttk.Label(summary, text="Data Preview", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            summary,
            text="Inspect sheets, filter rows, and validate column quality before promoting a dataset into plotting workflows.",
            style="CardHint.TLabel",
            wraplength=920,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        body = ttk.Frame(self, padding=10)
        body.grid(row=1, column=0, sticky="nsew")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)
        body.columnconfigure(0, weight=1)

        info = ttk.Frame(body)
        info.grid(row=0, column=0, sticky="ew")
        info.columnconfigure(1, weight=1)
        self._rows_var = tk.StringVar(value=f"Rows: {len(df)}")
        self._cols_var = tk.StringVar(value=f"Columns: {len(df.columns)}")
        self._notice_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self._rows_var).grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self._cols_var).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(info, textvariable=self._notice_var, style="Muted.TLabel").grid(row=0, column=2, sticky="e", padx=(10, 0))

        sheets = get_sheet_names(path)
        if sheets:
            ttk.Label(info, text="Sheet").grid(row=0, column=3, sticky="e", padx=(10, 0))
            self._sheet_var = tk.StringVar(value=str(dataset.sheet_name or sheets[0]))
            sheet_cb = ttk.Combobox(info, values=sheets, textvariable=self._sheet_var, state="readonly", width=24)
            sheet_cb.grid(row=0, column=4, sticky="e")
            sheet_cb.bind("<<ComboboxSelected>>", lambda _e: self._reload_sheet())
        else:
            self._sheet_var = None

        filter_bar = ttk.LabelFrame(body, text="Filter", padding=8)
        filter_bar.grid(row=1, column=0, sticky="ew")
        filter_bar.columnconfigure(6, weight=1)

        self._filter_col_var = tk.StringVar(value=self._all_filter_label)
        self._filter_text_var = tk.StringVar(value="")
        self._min_var = tk.StringVar(value="")
        self._max_var = tk.StringVar(value="")

        ttk.Label(filter_bar, text="Filter column").grid(row=0, column=0, sticky="w")
        self._filter_col_cb = ttk.Combobox(filter_bar, textvariable=self._filter_col_var, state="readonly", width=24)
        self._filter_col_cb.grid(row=0, column=1, sticky="w", padx=(6, 10))

        ttk.Label(filter_bar, text="Filter text").grid(row=0, column=2, sticky="w")
        self._filter_text_entry = ttk.Entry(filter_bar, textvariable=self._filter_text_var, width=24)
        self._filter_text_entry.grid(row=0, column=3, sticky="w", padx=(6, 10))
        self._filter_text_entry.bind("<Return>", lambda _e: self._apply_filters())

        ttk.Label(filter_bar, text="Min").grid(row=0, column=4, sticky="w")
        self._min_entry = ttk.Entry(filter_bar, textvariable=self._min_var, width=10)
        self._min_entry.grid(row=0, column=5, sticky="w", padx=(6, 6))
        self._min_entry.bind("<Return>", lambda _e: self._apply_filters())

        ttk.Label(filter_bar, text="Max").grid(row=0, column=6, sticky="w")
        self._max_entry = ttk.Entry(filter_bar, textvariable=self._max_var, width=10)
        self._max_entry.grid(row=0, column=7, sticky="w", padx=(6, 10))
        self._max_entry.bind("<Return>", lambda _e: self._apply_filters())

        self._clear_filters_btn = ttk.Button(filter_bar, text="Clear filters", command=self._clear_filters)
        self._clear_filters_btn.grid(row=0, column=8, sticky="e", padx=(6, 0))
        style_danger(self._clear_filters_btn)

        self._copy_btn = ttk.Button(filter_bar, text="Copy selection", command=self._copy_selection)
        self._copy_btn.grid(row=0, column=9, sticky="e", padx=(6, 0))
        style_secondary(self._copy_btn)

        self._copy_filtered_btn = ttk.Button(filter_bar, text="Copy filtered", command=self._copy_filtered)
        self._copy_filtered_btn.grid(row=0, column=10, sticky="e", padx=(6, 0))
        style_secondary(self._copy_filtered_btn)

        self._help_btn = ttk.Button(filter_bar, text="?", width=3, command=self._open_help)
        self._help_btn.grid(row=0, column=11, sticky="e", padx=(6, 0))

        ToolTip.attach(self._filter_col_cb, "Filter column")
        ToolTip.attach(self._filter_text_entry, "Filter text")
        ToolTip.attach(self._min_entry, "Min")
        ToolTip.attach(self._max_entry, "Max")
        ToolTip.attach(self._clear_filters_btn, "Clear filters")
        ToolTip.attach(self._copy_btn, "Copy selection")
        ToolTip.attach(self._copy_filtered_btn, "Copy filtered")
        ToolTip.attach(self._min_entry, "Range filter is available only when filtering a numeric column")
        ToolTip.attach(self._max_entry, "Range filter is available only when filtering a numeric column")

        stats_bar = ttk.LabelFrame(body, text="Quick stats", padding=8)
        stats_bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        stats_bar.columnconfigure(1, weight=1)

        self._stats_col_var = tk.StringVar(value="")
        ttk.Label(stats_bar, text="Numeric column").grid(row=0, column=0, sticky="w")
        self._stats_col_cb = ttk.Combobox(stats_bar, textvariable=self._stats_col_var, state="readonly", width=26)
        self._stats_col_cb.grid(row=0, column=1, sticky="w", padx=(6, 10))

        self._stats_var = tk.StringVar(value="Select a numeric column to see stats.")
        ttk.Label(stats_bar, textvariable=self._stats_var).grid(row=0, column=2, sticky="w")

        ttk.Separator(body).grid(row=3, column=0, sticky="ew", pady=8)

        self._tree = ttk.Treeview(body, show="headings")
        self._tree.grid(row=4, column=0, sticky="nsew")
        sb = ttk.Scrollbar(body, orient="vertical", command=self._tree.yview)
        sb.grid(row=4, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=sb.set)

        try:
            self._filter_text_var.trace_add("write", lambda *_: self._on_filter_var_changed())
            self._min_var.trace_add("write", lambda *_: self._on_filter_var_changed())
            self._max_var.trace_add("write", lambda *_: self._on_filter_var_changed())
        except Exception:
            pass
        self._filter_col_cb.bind("<<ComboboxSelected>>", lambda _e: self._schedule_apply_filters())
        self._stats_col_cb.bind("<<ComboboxSelected>>", lambda _e: self._update_stats())

        self._prepare_columns()
        self._apply_filters()

    def _prepare_columns(self) -> None:
        self._numeric_cols = list(numeric_columns(self._df))
        cols = [str(c) for c in self._df.columns]
        self._filter_col_cb["values"] = [self._all_filter_label] + cols
        if self._filter_col_var.get() not in self._filter_col_cb["values"]:
            self._filter_col_var.set(self._all_filter_label)
        self._stats_col_cb["values"] = self._numeric_cols
        if self._stats_col_var.get() not in self._numeric_cols:
            self._stats_col_var.set(self._numeric_cols[0] if self._numeric_cols else "")

        try:
            self._search_series = self._df.astype(str).agg(" | ".join, axis=1)
        except Exception:
            self._search_series = None

        self._tree.delete(*self._tree.get_children(""))
        self._tree["columns"] = cols
        for c in cols:
            self._tree.heading(str(c), text=str(c), command=lambda col=str(c): self._on_sort(col))
            self._tree.column(str(c), width=120, stretch=True)
        self._refresh_sort_indicators()

    def _schedule_apply_filters(self) -> None:
        try:
            if self._filter_after is not None:
                self.after_cancel(self._filter_after)
        except Exception:
            pass
        self._filter_after = self.after(250, self._apply_filters)

    def _on_filter_var_changed(self) -> None:
        if self._suppress_filter_traces:
            return
        self._schedule_apply_filters()

    def _schedule_render(self, df: pd.DataFrame) -> None:
        try:
            if self._render_after is not None:
                self.after_cancel(self._render_after)
        except Exception:
            pass
        self._tree.delete(*self._tree.get_children(""))

        total = int(len(df))
        show_total = min(total, int(self._preview_cap))
        if total > show_total:
            self._notice_var.set(f"Showing first {show_total:,} rows of {total:,}")
        else:
            self._notice_var.set("")

        rows = df.head(show_total)
        data_rows = rows.itertuples(index=False, name=None)

        def insert_chunk(chunk_size: int = 200) -> None:
            count = 0
            for row in data_rows:
                self._tree.insert("", "end", values=list(row))
                count += 1
                if count >= chunk_size:
                    break
            if count >= chunk_size:
                self._render_after = self.after(1, insert_chunk)
            else:
                self._render_after = None

        self._render_after = self.after(1, insert_chunk)

    def _apply_filters(self) -> None:
        df = self._df
        text = str(self._filter_text_var.get() or "").strip()
        col = str(self._filter_col_var.get() or self._all_filter_label)

        self._update_range_state(col)

        if text:
            text_lower = text.lower()
            if col == self._all_filter_label:
                try:
                    if self._search_series is not None:
                        mask = self._search_series.str.contains(text_lower, case=False, na=False)
                        df = df.loc[mask]
                except Exception:
                    pass
            elif col in df.columns:
                try:
                    mask = df[col].astype(str).str.contains(text_lower, case=False, na=False)
                    df = df.loc[mask]
                except Exception:
                    pass

        min_val = str(self._min_var.get() or "").strip()
        max_val = str(self._max_var.get() or "").strip()
        if (min_val or max_val) and col in df.columns and col in self._numeric_cols:
            try:
                series = pd.to_numeric(df[col], errors="coerce")
                if min_val:
                    df = df.loc[series >= float(min_val)]
                    series = pd.to_numeric(df[col], errors="coerce")
                if max_val:
                    df = df.loc[series <= float(max_val)]
            except Exception:
                pass

        self._view_df = df
        self._apply_sort()
        self._update_stats()
        self._schedule_render(self._view_df)

    def _apply_sort(self) -> None:
        if self._sort_col and self._sort_col in self._view_df.columns:
            try:
                self._view_df = self._view_df.sort_values(by=self._sort_col, ascending=self._sort_asc, kind="mergesort")
            except Exception:
                pass

    def _on_sort(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True
        self._apply_sort()
        self._refresh_sort_indicators()
        self._schedule_render(self._view_df)

    def _refresh_sort_indicators(self) -> None:
        cols = [str(c) for c in self._tree["columns"]]
        for c in cols:
            label = str(c)
            if self._sort_col == c:
                label = f"{c} {'▲' if self._sort_asc else '▼'}"
            try:
                self._tree.heading(str(c), text=label)
            except Exception:
                pass

    def _update_stats(self) -> None:
        col = str(self._stats_col_var.get() or "")
        if not col or col not in self._view_df.columns or col not in self._numeric_cols:
            self._stats_var.set("Select a numeric column to see stats.")
            return
        try:
            series = pd.to_numeric(self._view_df[col], errors="coerce")
            series = series.dropna()
            if series.empty:
                self._stats_var.set("No numeric values in current view.")
                return
            self._stats_var.set(
                f"count={int(series.count())}  mean={float(series.mean()):.4g}  std={float(series.std(ddof=1)):.4g}  min={float(series.min()):.4g}  max={float(series.max()):.4g}"
            )
        except Exception:
            self._stats_var.set("Stats unavailable.")

    def _clear_filters(self) -> None:
        self._filter_text_var.set("")
        self._min_var.set("")
        self._max_var.set("")
        self._filter_col_var.set(self._all_filter_label)
        self._apply_filters()

    def _update_range_state(self, col: str) -> None:
        enable = bool(col and col != self._all_filter_label and col in self._numeric_cols)
        state = "normal" if enable else "disabled"
        try:
            self._min_entry.configure(state=state)
            self._max_entry.configure(state=state)
        except Exception:
            pass
        if not enable:
            try:
                self._suppress_filter_traces = True
                if self._min_var.get():
                    self._min_var.set("")
                if self._max_var.get():
                    self._max_var.set("")
            finally:
                self._suppress_filter_traces = False

    def _copy_selection(self) -> None:
        items = list(self._tree.selection() or [])
        if not items:
            messagebox.showinfo("Copy selection", "Select one or more rows to copy.", parent=self)
            return
        cols = [str(c) for c in self._tree["columns"]]
        rows: List[str] = []
        rows.append("\t".join(cols))
        for iid in items:
            vals = self._tree.item(iid, "values") or []
            rows.append("\t".join(str(v) for v in vals))
        data = "\n".join(rows)
        try:
            self.clipboard_clear()
            self.clipboard_append(data)
            self.update()
        except Exception:
            pass

    def _copy_filtered(self) -> None:
        cols = [str(c) for c in self._tree["columns"]]
        rows: List[str] = []
        rows.append("\t".join(cols))
        df = self._view_df.head(int(self._preview_cap))
        for row in df.itertuples(index=False, name=None):
            rows.append("\t".join(str(v) for v in row))
        data = "\n".join(rows)
        try:
            self.clipboard_clear()
            self.clipboard_append(data)
            self.update()
        except Exception:
            pass
        try:
            messagebox.showinfo("Copy filtered", f"Copied {len(df):,} rows.", parent=self)
        except Exception:
            pass

    def _open_help(self) -> None:
        help_text = (
            "The Preview Table shows the dataset in a read-only view (it does not edit the file).\n"
            "Sorting: click a column header to sort ascending; click again to sort descending.\n"
            "Filtering:\n"
            "  Text filter: choose a column and type text; only rows containing that text are shown.\n"
            "  Numeric range: for numeric columns, enter Min/Max to show only rows within range.\n"
            "  Clear filters resets the view to the original dataset.\n"
            "  Filtering is debounced (results update after a short pause while typing).\n"
            "  Range filters only work when a numeric column is selected.\n"
            "Quick stats:\n"
            "  Select a numeric column to see count/mean/std/min/max for the currently filtered rows.\n"
            "Copy to Excel:\n"
            "  Select rows/cells and click “Copy selection” to copy as tab-separated text.\n"
            "  Click “Copy filtered” to copy the currently filtered view (up to the preview limit).\n"
            "  Paste directly into Excel or Google Sheets."
        )
        messagebox.showinfo("Preview Table – How to use", help_text, parent=self)

    def _reload_sheet(self) -> None:
        if self._sheet_var is None:
            return
        sheet = str(self._sheet_var.get())
        df = load_table(self._path, sheet_name=sheet, header_row=self._dataset.header_row)
        self._dataset.sheet_name = sheet
        self._df = df
        self._view_df = df
        self._sort_col = None
        self._sort_asc = True
        self._prepare_columns()
        self._apply_filters()

        # Preview enhancements: debounced filters, cached search, range gating, copy options, and sort indicators.


class DataStudioView(ttk.Frame):
    def __init__(self, parent: tk.Widget, app: Any, workspace: Any) -> None:
        super().__init__(parent)
        self.app = app
        self.workspace = workspace
        self._ws = DataStudioWorkspace()
        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._transform_cache: Dict[str, Tuple[str, pd.DataFrame]] = {}
        self._transform_warnings: Dict[str, List[str]] = {}
        self._plotted_ids: set = set()
        self._status_var = tk.StringVar(value="Ready")
        self._overlay_mode_var = tk.StringVar(value="Normal")
        self._overlay_offset_var = tk.StringVar(value="0.0")
        self._overlay_refresh_job = None
        self._x_display_to_col: Dict[str, str] = {}
        self._x_col_to_display: Dict[str, str] = {}
        self._x_all_values: List[str] = []
        self._y_display_to_col: Dict[str, str] = {}
        self._y_col_to_display: Dict[str, str] = {}
        self._y_summary_var = tk.StringVar(value="Y: (none)")
        self._dirty_var = tk.StringVar(value="")
        self._banner_var = tk.StringVar(value="")
        self._restoring_ui = False
        self._x_search_var = tk.StringVar(value="")
        self._auto_plot_var = tk.BooleanVar(value=True)
        self._recipes_var = tk.StringVar(value="")
        self._x_title_var = tk.StringVar(value="")
        self._y_title_var = tk.StringVar(value="")
        self._transform_enabled_var = tk.BooleanVar(value=False)

        self._build_ui()

    def status_text(self) -> str:
        try:
            return str(self._status_var.get())
        except Exception:
            return ""

    def _mark_dirty(self) -> None:
        try:
            self._dirty_var.set("● Unsaved changes")
        except Exception:
            pass

    def _clear_dirty(self) -> None:
        try:
            self._dirty_var.set("")
        except Exception:
            pass

    def _active_plot_def(self) -> Optional[DataStudioPlotDef]:
        pid = self._ws.active_plot_id
        if not pid:
            return None
        return self._ws.plot_defs.get(pid)

    def _plot_def_name(self, pd: DataStudioPlotDef) -> str:
        ds = self._ws.datasets.get(pd.dataset_id)
        base = str(ds.display_name) if ds is not None else str(pd.dataset_id)
        return f"{base} · {pd.plot_type or 'Plot'}"

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        panes = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        panes.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(panes)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        ws_canvas = tk.Canvas(left, highlightthickness=0, bd=0)
        ws_canvas.grid(row=0, column=0, sticky="nsew")
        ws_sb = ttk.Scrollbar(left, orient="vertical", command=ws_canvas.yview)
        ws_sb.grid(row=0, column=1, sticky="ns")
        ws_canvas.configure(yscrollcommand=ws_sb.set)

        ws = ttk.LabelFrame(ws_canvas, text="Workspace", padding=8, style="Card.TLabelframe")
        ws.columnconfigure(0, weight=1)
        ws.rowconfigure(2, weight=1)

        ws_window = ws_canvas.create_window((0, 0), window=ws, anchor="nw")

        def _sync_ws_width(event: tk.Event) -> None:
            try:
                ws_canvas.itemconfigure(ws_window, width=event.width)
            except Exception:
                pass

        def _update_ws_scroll(_event: Optional[tk.Event] = None) -> None:
            try:
                ws_canvas.configure(scrollregion=ws_canvas.bbox("all"))
            except Exception:
                pass

        ws.bind("<Configure>", _update_ws_scroll)
        ws_canvas.bind("<Configure>", _sync_ws_width)

        def _on_ws_mousewheel(event: tk.Event) -> None:
            try:
                delta = int(-1 * (event.delta / 120)) if event.delta else 0
                if delta:
                    ws_canvas.yview_scroll(delta, "units")
            except Exception:
                pass

        ws_canvas.bind("<MouseWheel>", _on_ws_mousewheel)
        ws.bind("<MouseWheel>", _on_ws_mousewheel)

        workspace_summary = ttk.Frame(ws, style="ShellPanel.TFrame", padding=(14, 12))
        workspace_summary.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        workspace_summary.columnconfigure(0, weight=1)
        ttk.Label(workspace_summary, text="Data Studio", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            workspace_summary,
            text="Manage imported tables, plot definitions, recipes, and overlay context from a single workspace rail.",
            style="CardHint.TLabel",
            wraplength=300,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))
        ttk.Label(workspace_summary, textvariable=self._status_var, style="Muted.TLabel", wraplength=300, justify="left").grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(workspace_summary, textvariable=self._dirty_var, style="Danger.TLabel").grid(row=3, column=0, sticky="w", pady=(6, 0))

        btns = ttk.Frame(ws)
        btns.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            ws,
            text="Import files, curate the workspace, and promote the active dataset before shaping plots.",
            style="CardHint.TLabel",
            wraplength=300,
            justify="left",
        ).grid(row=1, column=1, sticky="e")
        _b_add = ttk.Button(btns, text="Add Files…", command=self._add_files)
        _b_add.grid(row=0, column=0, sticky="w")
        style_primary(_b_add)
        _b_rm = ttk.Button(btns, text="Remove Selected", command=self._remove_selected)
        _b_rm.grid(row=0, column=1, padx=(6, 0))
        style_danger(_b_rm)
        _b_clr = ttk.Button(btns, text="Clear", command=self._clear_workspace)
        _b_clr.grid(row=0, column=2, padx=(6, 0))
        style_danger(_b_clr)
        _b_sv = ttk.Button(btns, text="Save…", command=self._save_workspace)
        _b_sv.grid(row=0, column=3, padx=(6, 0))
        style_success(_b_sv)
        _b_ld = ttk.Button(btns, text="Load…", command=self._load_workspace)
        _b_ld.grid(row=0, column=4, padx=(6, 0))
        style_primary(_b_ld)

        self._ws_tree = ttk.Treeview(ws, columns=("active", "name"), show="headings", height=10, selectmode="browse")
        self._ws_tree.heading("active", text="Active")
        self._ws_tree.heading("name", text="File")
        self._ws_tree.column("active", width=60, stretch=False, anchor="center")
        self._ws_tree.column("name", width=200, stretch=True)
        self._ws_tree.grid(row=2, column=0, sticky="nsew")
        self._ws_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        _b_sa = ttk.Button(ws, text="Set Active", command=self._set_active_from_selection)
        _b_sa.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        style_secondary(_b_sa)
        _b_pt = ttk.Button(ws, text="Preview Table", command=self._preview_data)
        _b_pt.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        style_secondary(_b_pt)

        defs = ttk.LabelFrame(ws, text="Plot Definitions", padding=8)
        defs.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        defs.columnconfigure(0, weight=1)
        ttk.Label(
            defs,
            text="Store repeatable plot setups and switch the active definition without rebuilding settings manually.",
            style="CardHint.TLabel",
            wraplength=300,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self._plot_tree = ttk.Treeview(defs, columns=("active", "name"), show="headings", height=5, selectmode="browse")
        self._plot_tree.heading("active", text="Active")
        self._plot_tree.heading("name", text="Plot")
        self._plot_tree.column("active", width=60, stretch=False, anchor="center")
        self._plot_tree.column("name", width=200, stretch=True)
        self._plot_tree.grid(row=1, column=0, sticky="ew")
        self._plot_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_plot_select())
        defs_btns = ttk.Frame(defs)
        defs_btns.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        _b_np = ttk.Button(defs_btns, text="New Plot", command=self._new_plot_def)
        _b_np.grid(row=0, column=0, sticky="w")
        style_primary(_b_np)
        _b_rp = ttk.Button(defs_btns, text="Remove Plot", command=self._remove_plot_def)
        _b_rp.grid(row=0, column=1, padx=(6, 0))
        style_danger(_b_rp)
        ttk.Button(defs_btns, text="Set Active", command=self._set_active_plot_from_selection).grid(row=0, column=2, padx=(6, 0))

        recipes = ttk.LabelFrame(ws, text="Recipes", padding=8)
        recipes.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        recipes.columnconfigure(0, weight=1)
        ttk.Label(
            recipes,
            text="Capture reusable plotting workflows and reapply them to another dataset in one step.",
            style="CardHint.TLabel",
            wraplength=300,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self._recipes_cb = ttk.Combobox(recipes, textvariable=self._recipes_var, state="readonly")
        self._recipes_cb.grid(row=1, column=0, sticky="ew")

        recipes_btns = ttk.Frame(recipes)
        recipes_btns.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        _b_sr = ttk.Button(recipes_btns, text="Save as recipe…", command=self._save_recipe)
        _b_sr.grid(row=0, column=0, sticky="w")
        style_success(_b_sr)
        _b_ar = ttk.Button(recipes_btns, text="Apply recipe", command=self._apply_recipe)
        _b_ar.grid(row=0, column=1, padx=(6, 0))
        style_success(_b_ar)
        ttk.Button(recipes_btns, text="Rename", command=self._rename_recipe).grid(row=0, column=2, padx=(6, 0))
        _b_dr = ttk.Button(recipes_btns, text="Delete", command=self._delete_recipe)
        _b_dr.grid(row=0, column=3, padx=(6, 0))
        style_danger(_b_dr)
        ttk.Button(recipes_btns, text="?", width=3, command=self._open_recipes_help).grid(row=0, column=4, padx=(6, 0))

        overlay = ttk.LabelFrame(ws, text="Overlay", padding=8)
        overlay.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(
            overlay,
            text="Compose comparison views and decide which file anchors the active overlay stack.",
            style="CardHint.TLabel",
            wraplength=300,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self._overlay_tree = ttk.Treeview(overlay, columns=("sel", "name"), show="headings", height=6, selectmode="browse")
        self._overlay_tree.heading("sel", text="Overlay")
        self._overlay_tree.heading("name", text="File")
        self._overlay_tree.column("sel", width=70, stretch=False, anchor="center")
        self._overlay_tree.column("name", width=180, stretch=True)
        self._overlay_tree.grid(row=1, column=0, sticky="ew")
        self._overlay_tree.bind("<Button-1>", self._on_overlay_click, add=True)
        _b_ov = ttk.Button(overlay, text="Overlay Selected", command=self._apply_overlay)
        _b_ov.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        style_success(_b_ov)
        ttk.Button(overlay, text="Select all", command=self._overlay_select_all).grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(overlay, text="Select none", command=self._overlay_select_none).grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(overlay, text="Make active = first overlay", command=self._overlay_make_active_first).grid(row=5, column=0, sticky="ew", pady=(6, 0))
        _b_co = ttk.Button(overlay, text="Clear Overlay", command=self._clear_overlay)
        _b_co.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        style_danger(_b_co)
        ttk.Button(overlay, text="?", command=self._open_overlay_help).grid(row=7, column=0, sticky="ew", pady=(6, 0))


        right = ttk.Frame(panes)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

        panes.add(left, weight=1)
        panes.add(right, weight=4)

        top = ttk.LabelFrame(right, text="Workflow Controls", padding=8, style="Card.TLabelframe")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)
        ttk.Label(
            top,
            text="Control plot generation, export transformed data, and keep draft-state feedback visible while iterating.",
            style="CardHint.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Button(top, text="Apply", command=self._apply_plot).grid(row=1, column=0, sticky="ew")
        ttk.Button(top, text="Reset", command=self._reset_plot_builder).grid(row=1, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(top, text="Export…", command=self._export_plot).grid(row=1, column=2, sticky="ew")
        ttk.Button(top, text="Export transformed CSV…", command=self._export_transformed_csv).grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Checkbutton(top, text="Auto-plot", variable=self._auto_plot_var).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(top, textvariable=self._dirty_var, style="Danger.TLabel").grid(row=2, column=2, sticky="e", pady=(8, 0))

        controls = ttk.Frame(right)
        controls.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(controls, text="X").pack(side=tk.LEFT)
        self._x_var = tk.StringVar(value="")
        self._x_cb = ttk.Combobox(controls, textvariable=self._x_var, state="readonly", width=24)
        self._x_cb.pack(side=tk.LEFT, padx=(6, 10))
        self._x_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_x_changed())

        ToolTip.attach(self._x_cb, "X controls the horizontal axis")
        ToolTip.attach(self._x_cb, "(Index) means use row index")

        ttk.Label(controls, text="Plot").pack(side=tk.LEFT)
        self._plot_type_var = tk.StringVar(value=PLOT_TYPES[0])
        self._plot_cb = ttk.Combobox(controls, textvariable=self._plot_type_var, values=PLOT_TYPES, state="readonly", width=18)
        self._plot_cb.pack(side=tk.LEFT, padx=(6, 10))
        self._plot_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_plot_type_changed())

        ttk.Button(controls, text="Y columns…", command=self._open_y_selector).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Data options…", command=self._open_data_options).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Advanced…", command=self._toggle_advanced_panel).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(controls, textvariable=self._y_summary_var, style="Muted.TLabel").pack(side=tk.RIGHT)

        try:
            self._x_title_var.trace_add("write", lambda *_a: self._mark_dirty())
            self._y_title_var.trace_add("write", lambda *_a: self._mark_dirty())
        except Exception:
            pass

        stage_hdr = ttk.Frame(right, style="Surface.TFrame", padding=(14, 12))
        stage_hdr.grid(row=2, column=0, sticky="ew", pady=(8, 6))
        stage_hdr.columnconfigure(0, weight=1)
        ttk.Label(stage_hdr, text="Data Studio Stage", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            stage_hdr,
            text="Shape columns, switch plot strategies, and inspect overlay behavior with the chart kept at the center of the workflow.",
            style="CardHint.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(stage_hdr, textvariable=self._status_var, style="CardStatus.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        banner = ttk.Label(right, textvariable=self._banner_var, style="Info.TLabel", anchor="w")
        banner.grid(row=3, column=0, sticky="ew", pady=(2, 4))

        body = ttk.Frame(right)
        body.grid(row=4, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        builder = ttk.LabelFrame(body, text="Advanced options", padding=8)
        builder.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        builder.columnconfigure(0, weight=1)
        self._builder_panel = builder

        adv_axes = ttk.LabelFrame(builder, text="Axes + Overlay", padding=6)
        adv_axes.grid(row=0, column=0, sticky="ew")
        adv_axes.columnconfigure(1, weight=1)

        ttk.Label(adv_axes, text="Search columns").grid(row=0, column=0, sticky="w")
        self._x_search_entry = ttk.Entry(adv_axes, textvariable=self._x_search_var)
        self._x_search_entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        self._x_search_entry.bind("<KeyRelease>", lambda _e: self._refresh_x_values(self._x_search_var.get()))
        self._x_search_clear_btn = ttk.Button(adv_axes, text="Clear", command=self._clear_x_search)
        self._x_search_clear_btn.grid(row=0, column=2, sticky="e")
        ToolTip.attach(self._x_search_entry, "Search columns by name")
        ToolTip.attach(self._x_search_clear_btn, "Clear search")

        ttk.Button(adv_axes, text="Auto X/Y", command=self._auto_axes_both).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(adv_axes, text="Auto Y", command=self._auto_axes_y_only).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Button(adv_axes, text="?", width=3, command=self._open_axes_help).grid(row=1, column=2, sticky="e", pady=(6, 0))

        ttk.Label(adv_axes, text="X title").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._x_title_entry = ttk.Entry(adv_axes, textvariable=self._x_title_var)
        self._x_title_entry.grid(row=2, column=1, sticky="ew", padx=(6, 6), pady=(6, 0))
        ttk.Label(adv_axes, text="Y title").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self._y_title_entry = ttk.Entry(adv_axes, textvariable=self._y_title_var)
        self._y_title_entry.grid(row=3, column=1, sticky="ew", padx=(6, 6), pady=(6, 0))

        ttk.Label(adv_axes, text="Offset").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ov_mode = ttk.Combobox(adv_axes, textvariable=self._overlay_mode_var, values=["Normal", "Offset Y", "Offset X"], state="readonly", width=12)
        ov_mode.grid(row=4, column=1, sticky="w", padx=(6, 6), pady=(6, 0))
        ov_mode.bind("<<ComboboxSelected>>", lambda _e: self._on_overlay_mode_changed())
        ov_off = ttk.Entry(adv_axes, textvariable=self._overlay_offset_var, width=8)
        ov_off.grid(row=4, column=2, sticky="e", pady=(6, 0))
        ov_off.bind("<KeyRelease>", lambda _e: self._schedule_overlay_refresh())
        ov_off.bind("<Return>", lambda _e: self._on_overlay_mode_changed())
        ov_off.bind("<FocusOut>", lambda _e: self._on_overlay_mode_changed())
        try:
            self._overlay_offset_var.trace_add("write", lambda *_a: self._schedule_overlay_refresh())
            self._overlay_mode_var.trace_add("write", lambda *_a: self._schedule_overlay_refresh())
        except Exception:
            pass

        ttk.Label(builder, text="Group / Series column").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self._group_var = tk.StringVar(value="(None)")
        self._group_cb = ttk.Combobox(builder, textvariable=self._group_var, state="readonly")
        self._group_cb.grid(row=2, column=0, sticky="ew")

        self._extra = ttk.Frame(builder)
        self._extra.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self._extra.columnconfigure(0, weight=1)

        transform = ttk.LabelFrame(builder, text="Transform", padding=6)
        transform.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        transform.columnconfigure(0, weight=1)

        self._transform_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            transform,
            text="Use transformed data for plotting",
            variable=self._transform_enabled_var,
            command=self._on_transform_toggle,
        ).grid(row=0, column=0, sticky="w")

        self._transform_list = tk.Listbox(transform, height=6)
        self._transform_list.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        tbtns = ttk.Frame(transform)
        tbtns.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(tbtns, text="Add step…", command=self._add_transform_step).grid(row=0, column=0, sticky="w")
        ttk.Button(tbtns, text="Edit step…", command=self._edit_transform_step).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(tbtns, text="Remove", command=self._remove_transform_step).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(tbtns, text="Move Up", command=lambda: self._move_transform_step(-1)).grid(row=0, column=3, padx=(6, 0))
        ttk.Button(tbtns, text="Move Down", command=lambda: self._move_transform_step(1)).grid(row=0, column=4, padx=(6, 0))
        ttk.Button(tbtns, text="Clear all", command=self._clear_transform_steps).grid(row=0, column=5, padx=(6, 0))
        ttk.Button(tbtns, text="?", width=3, command=self._open_transform_help).grid(row=0, column=6, padx=(6, 0))

        self._size_var = tk.StringVar(value="(None)")
        self._heat_row_var = tk.StringVar(value="(None)")
        self._heat_col_var = tk.StringVar(value="(None)")
        self._heat_val_var = tk.StringVar(value="(None)")
        self._heat_agg_var = tk.StringVar(value="mean")
        self._bins_var = tk.IntVar(value=20)
        self._roll_var = tk.IntVar(value=5)
        self._xerr_var = tk.StringVar(value="(None)")
        self._yerr_var = tk.StringVar(value="(None)")

        self._toggle_extra_fields()

        self._drop_na_var = tk.BooleanVar(value=True)
        self._decimal_var = tk.BooleanVar(value=False)
        self._autocast_var = tk.BooleanVar(value=True)
        self._norm_var = tk.StringVar(value="None")

        self._builder_panel.grid_remove()

        for v in [
            self._x_var,
            self._group_var,
            self._plot_type_var,
            self._size_var,
            self._heat_row_var,
            self._heat_col_var,
            self._heat_val_var,
            self._heat_agg_var,
            self._xerr_var,
            self._yerr_var,
            self._drop_na_var,
            self._decimal_var,
            self._autocast_var,
            self._norm_var,
            self._bins_var,
            self._roll_var,
        ]:
            try:
                v.trace_add("write", lambda *_a: self._mark_dirty())
            except Exception:
                pass

        plot_card = PlotCard(body, title="Data Studio Analysis", status_text="Live view", show_header=True)
        plot_card.grid(row=0, column=1, sticky="nsew")
        plot = plot_card.body

        self._fig = Figure(figsize=(10.5, 7.5), dpi=110)
        self._ax = self._fig.add_subplot(1, 1, 1)
        try:
            self._fig.subplots_adjust(left=0.06, right=0.8, top=0.98, bottom=0.08)
        except Exception:
            pass
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot)
        self._canvas.draw()
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.grid(row=0, column=0, sticky="nsew")
        plot_card.register_canvas(self._canvas)
        try:
            self._toolbar = NavigationToolbar2Tk(self._canvas, plot, pack_toolbar=False)
            self._toolbar.update()
            self._toolbar.grid(row=1, column=0, sticky="ew")
            try:
                if hasattr(self.app, "_style_mpl_toolbar"):
                    self.app._style_mpl_toolbar(self._toolbar)
            except Exception:
                pass
        except Exception:
            self._toolbar = None

        self._coord_var = tk.StringVar(value="")
        self._coord_label = ttk.Label(plot, textvariable=self._coord_var, anchor="w")
        self._coord_label.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        try:
            self._coord_label.configure(style="Muted.TLabel", padding=(2, 6, 2, 0))
        except Exception:
            pass
        try:
            getattr(self._canvas_widget, "lift", lambda *_a, **_k: None)()
            if self._toolbar is not None:
                getattr(self._toolbar, "lift", lambda *_a, **_k: None)()
            getattr(self._coord_label, "lift", lambda *_a, **_k: None)()
        except Exception:
            pass
        try:
            self._nav = MatplotlibNavigator(canvas=self._canvas, ax=self._ax, status_label=self._coord_var)
            self._nav.attach()
        except Exception:
            self._nav = None

    def _toggle_y_panel(self) -> None:
        self._open_y_selector()

    def _toggle_advanced_panel(self) -> None:
        if getattr(self, "_builder_panel", None) is None:
            return
        try:
            if self._builder_panel.winfo_ismapped():
                self._builder_panel.grid_remove()
            else:
                self._builder_panel.grid()
        except Exception:
            pass

    def _open_data_options(self) -> None:
        win = getattr(self, "_data_options_win", None)
        if win is not None:
            try:
                if win.winfo_exists():
                    win.lift()
                    win.focus_set()
                    return
            except Exception:
                pass

        win = tk.Toplevel(self)
        win.title("Data options")
        win.resizable(False, False)
        win.transient(self.winfo_toplevel())
        self._data_options_win = win

        ttk.Label(
            win,
            text="These settings control how data is parsed before plotting.",
            style="Info.TLabel",
            wraplength=360,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

        opts = ttk.Frame(win)
        opts.grid(row=1, column=0, sticky="ew", padx=12)
        ttk.Checkbutton(opts, text="Drop NaNs", variable=self._drop_na_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Comma → dot numeric", variable=self._decimal_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Auto-cast numeric", variable=self._autocast_var).grid(row=2, column=0, sticky="w")
        ttk.Label(opts, text="Normalize Y").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(opts, textvariable=self._norm_var, values=["None", "Min-Max", "Z-score"], state="readonly").grid(
            row=4, column=0, sticky="ew"
        )

        btns = ttk.Frame(win)
        btns.grid(row=2, column=0, sticky="e", padx=12, pady=(8, 10))
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)

    def _open_y_selector(self) -> None:
        pd = self._active_plot_def()
        if pd is None:
            return

        win = getattr(self, "_y_selector_win", None)
        if win is not None:
            try:
                if win.winfo_exists():
                    win.lift()
                    win.focus_set()
                    return
            except Exception:
                pass

        win = tk.Toplevel(self)
        win.title("Select Y columns")
        win.transient(self.winfo_toplevel())
        win.geometry("420x420")
        self._y_selector_win = win

        ttk.Label(
            win,
            text="Pick one or more Y columns. Changes apply when you press Apply.",
            style="Info.TLabel",
            wraplength=380,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

        filter_frame = ttk.Frame(win)
        filter_frame.grid(row=1, column=0, sticky="ew", padx=12)
        filter_frame.columnconfigure(1, weight=1)

        ttk.Label(filter_frame, text="Search").grid(row=0, column=0, sticky="w")
        filter_var = tk.StringVar(value="")
        filter_entry = ttk.Entry(filter_frame, textvariable=filter_var)
        filter_entry.grid(row=0, column=1, sticky="ew", padx=(6, 8))
        ToolTip.attach(filter_entry, "Search columns by name")

        numeric_only_var = tk.BooleanVar(value=(str(self._plot_type_var.get()) not in ("Heatmap",)))
        numeric_only_cb = ttk.Checkbutton(filter_frame, text="Numeric only", variable=numeric_only_var)
        numeric_only_cb.grid(row=0, column=2, sticky="e")
        ToolTip.attach(numeric_only_cb, "Show only numeric columns")

        listbox = tk.Listbox(win, selectmode="extended", height=14)
        listbox.grid(row=2, column=0, sticky="nsew", padx=12, pady=(6, 0))
        sb = ttk.Scrollbar(win, orient="vertical", command=listbox.yview)
        sb.grid(row=2, column=1, sticky="ns", pady=(6, 0))
        listbox.configure(yscrollcommand=sb.set)

        quick = ttk.Frame(win)
        quick.grid(row=3, column=0, sticky="ew", padx=12, pady=(6, 0))
        quick.columnconfigure(2, weight=1)
        btn_select_all = ttk.Button(quick, text="Select all numeric", command=lambda: _select_all_numeric_visible())
        btn_select_all.grid(row=0, column=0, sticky="w")
        btn_clear = ttk.Button(quick, text="Clear selection", command=lambda: _clear_selection_list())
        btn_clear.grid(row=0, column=1, sticky="w", padx=(6, 0))
        btn_select_visible = ttk.Button(quick, text="Select visible", command=lambda: _select_visible())
        btn_select_visible.grid(row=0, column=2, sticky="w", padx=(6, 0))
        ToolTip.attach(btn_select_all, "Select all numeric columns")
        ToolTip.attach(btn_clear, "Clear current selection")
        ToolTip.attach(btn_select_visible, "Select all visible items")

        counts_var = tk.StringVar(value="Visible: 0   Selected: 0")
        ttk.Label(quick, textvariable=counts_var, style="Muted.TLabel").grid(row=0, column=3, sticky="e")

        win.columnconfigure(0, weight=1)
        win.rowconfigure(2, weight=1)

        def _numeric_set() -> set:
            sid = self._ws.active_id
            ds = self._ws.datasets.get(sid) if sid else None
            cols_map = dict(ds.columns or {}) if ds else {}
            numeric = set()
            for name, dtype in cols_map.items():
                if "int" in str(dtype) or "float" in str(dtype) or "double" in str(dtype):
                    numeric.add(str(name))
            return numeric

        def _available_items(filter_text: str) -> List[str]:
            items = list(self._y_display_to_col.keys())
            filt = str(filter_text or "").strip().lower()
            if filt:
                items = [d for d in items if filt in d.lower()]
            if bool(numeric_only_var.get()):
                numeric = _numeric_set()
                items = [d for d in items if self._y_display_to_col.get(d, d) in numeric or not numeric]
            return items

        def _refresh() -> None:
            items = _available_items(filter_var.get())
            listbox.delete(0, "end")
            for d in items:
                listbox.insert("end", d)
            selected = set(pd.y_cols or [])
            for idx, d in enumerate(items):
                col = self._y_display_to_col.get(d, d)
                if col in selected:
                    listbox.selection_set(idx)
            _update_counts(items)

        def _update_counts(items: Optional[List[str]] = None) -> None:
            visible = len(items) if items is not None else int(listbox.size())
            selected = int(len(listbox.curselection()))
            counts_var.set(f"Visible: {visible}   Selected: {selected}")

        def _select_visible() -> None:
            listbox.select_clear(0, "end")
            for i in range(int(listbox.size())):
                listbox.selection_set(i)
            _update_counts()

        def _clear_selection_list() -> None:
            listbox.select_clear(0, "end")
            _update_counts()

        def _select_all_numeric_visible() -> None:
            numeric = _numeric_set()
            listbox.select_clear(0, "end")
            for idx in range(int(listbox.size())):
                disp = listbox.get(idx)
                col = self._y_display_to_col.get(disp, disp)
                if col in numeric or not numeric:
                    listbox.selection_set(idx)
            _update_counts()

        def _apply() -> None:
            selected: List[str] = []
            for i in listbox.curselection():
                disp = listbox.get(i)
                mapped = self._y_display_to_col.get(disp)
                selected.append(str(mapped if mapped else disp))
            if not selected:
                messagebox.showwarning("Y columns", "Select at least one Y column.", parent=win)
                return
            pd.y_cols = list(selected)
            self._ws.plot_defs[pd.plot_id] = pd
            self._update_y_summary()
            self._mark_dirty()
            self._store_current_config()
            try:
                win.destroy()
            except Exception:
                pass

        filter_var.trace_add("write", lambda *_a: _refresh())
        numeric_only_var.trace_add("write", lambda *_a: _refresh())
        listbox.bind("<<ListboxSelect>>", lambda _e: _update_counts())
        _refresh()

        btns = ttk.Frame(win)
        btns.grid(row=4, column=0, sticky="e", padx=12, pady=(8, 10))
        ttk.Button(btns, text="Apply", command=_apply).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0, 6))

    def _on_canvas_resize(self, event: tk.Event) -> None:
        try:
            w = max(1, int(event.width))
            h = max(1, int(event.height))
            dpi = float(self._fig.get_dpi() or 100.0)
            self._fig.set_size_inches(w / dpi, h / dpi, forward=False)
            self._canvas.draw_idle()
        except Exception:
            return

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Add files",
            filetypes=[("Data", "*.csv *.tsv *.xlsx *.xls"), ("All", "*.*")],
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
            self._ensure_plot_def_for_dataset(sid)
            if self._ws.active_id is None:
                self._ws.active_id = sid
            self._infer_schema_async(sid)
        self._refresh_workspace()
        self._status_var.set(f"Loaded {len(paths)} file(s)")

    def _ensure_plot_def_for_dataset(self, dataset_id: str) -> None:
        if not dataset_id:
            return
        for pd in self._ws.plot_defs.values():
            if pd.dataset_id == dataset_id:
                return
        pid = str(uuid.uuid4())
        x_def, y_def = self._pick_default_axes(dataset_id)
        self._ws.plot_defs[pid] = DataStudioPlotDef(
            plot_id=pid,
            dataset_id=dataset_id,
            plot_type=PLOT_TYPES[0],
            x_col=x_def,
            y_cols=list(y_def),
        )
        if not self._ws.active_plot_id:
            self._ws.active_plot_id = pid

    def _remove_plot_defs_for_dataset(self, dataset_id: str) -> None:
        to_drop = [pid for pid, pd in self._ws.plot_defs.items() if pd.dataset_id == dataset_id]
        for pid in to_drop:
            self._ws.plot_defs.pop(pid, None)
            if self._ws.active_plot_id == pid:
                self._ws.active_plot_id = None

    def _infer_schema_async(self, dataset_id: str) -> None:
        ds = self._ws.datasets.get(dataset_id)
        if ds is None:
            return

        def _worker() -> None:
            try:
                df = load_table(
                    ds.path,
                    sheet_name=ds.sheet_name,
                    header_row=ds.header_row,
                    decimal_comma=bool(self._decimal_var.get()),
                    auto_cast=bool(self._autocast_var.get()),
                )
                cols = column_type_map(df)
                schema_hash = schema_hash_from_columns(cols)
            except Exception:
                cols = {}
                schema_hash = ""

            def _apply() -> None:
                d = self._ws.datasets.get(dataset_id)
                if d is None:
                    return
                d.columns = cols
                d.schema_hash = schema_hash
                self._ws.datasets[dataset_id] = d
                if self._ws.active_id == dataset_id:
                    self._populate_columns()
                    self._restore_config_for_active()

            try:
                self.after(0, _apply)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _set_active_dataset(self, dataset_id: str) -> None:
        if not dataset_id:
            return
        self._ws.active_id = dataset_id
        ds = self._ws.datasets.get(dataset_id)
        if ds is not None:
            try:
                self._transform_enabled_var.set(bool(getattr(ds, "derived_enabled", False)))
            except Exception:
                pass
        self._refresh_transform_list()
        self._ensure_plot_def_for_dataset(dataset_id)
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
        if sid in self._df_cache:
            self._df_cache.pop(sid, None)
        self._remove_plot_defs_for_dataset(sid)
        if sid in self._plotted_ids:
            self._plotted_ids.discard(sid)
        if self._ws.active_id == sid:
            self._ws.active_id = self._ws.order[0] if self._ws.order else None
        self._refresh_workspace()

    def _clear_workspace(self) -> None:
        self._ws = DataStudioWorkspace()
        self._df_cache = {}
        self._plotted_ids = set()
        self._refresh_workspace()

    def _save_workspace(self) -> None:
        if not self._ws.datasets:
            messagebox.showinfo("Data Studio", "No datasets to save.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            title="Save Data Studio Workspace",
            defaultextension=".data_studio.workspace.json",
            filetypes=[("Data Studio Workspace", "*.data_studio.workspace.json"), ("JSON", "*.json"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        payload = encode_workspace(self._ws)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            messagebox.showerror("Data Studio", f"Failed to save workspace:\n\n{exc}", parent=self)
            return
        self._status_var.set("Workspace saved")

    def _load_workspace(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Data Studio Workspace",
            filetypes=[("Data Studio Workspace", "*.data_studio.workspace.json"), ("JSON", "*.json"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            messagebox.showerror("Data Studio", f"Failed to load workspace:\n\n{exc}", parent=self)
            return
        if not isinstance(payload, dict):
            messagebox.showerror("Data Studio", "Workspace JSON must be an object.", parent=self)
            return

        ws, errors = decode_workspace(payload)
        self._ws = ws
        self._df_cache = {}
        self._plotted_ids = set()
        self._refresh_workspace()
        for sid in list(self._ws.order):
            ds = self._ws.datasets.get(sid)
            if ds is not None and not ds.columns:
                self._infer_schema_async(sid)
        if errors:
            messagebox.showwarning("Data Studio", "Workspace loaded with some issues:\n\n" + "\n".join(errors[:10]), parent=self)
        self._status_var.set("Workspace loaded")

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
        x_def, y_def = self._pick_default_axes(ds_id)
        pd = DataStudioPlotDef(
            plot_id=pid,
            dataset_id=ds_id,
            plot_type=str(self._plot_type_var.get()),
            x_col=x_def,
            y_cols=list(y_def),
        )
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
            messagebox.showinfo("Preview", "No active dataset.")
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        df = self._load_df(ds)
        _PreviewWindow(self, path=ds.path, dataset=ds, df=df)

    def _apply_overlay(self) -> None:
        overlay_ids = [sid for sid in self._ws.order if self._overlay_tree.exists(str(sid)) and self._overlay_tree.set(str(sid), "sel") == "✔"]
        if not overlay_ids:
            self._ws.overlay_ids = []
            self._status_var.set("Overlay: 0 dataset(s)")
            return

        for sid in overlay_ids:
            self._ensure_plot_def_for_dataset(sid)

        cfgs = [pd for pd in self._ws.plot_defs.values() if pd.dataset_id in overlay_ids]
        if len(cfgs) != len(overlay_ids):
            self._ws.overlay_ids = []
            self._refresh_workspace()
            return

        current_plot_type = str(self._plot_type_var.get() or "Line")
        if any(str(c.plot_type) != current_plot_type for c in cfgs if c is not None):
            ok = messagebox.askyesno(
                "Overlay",
                "Selected datasets have different plot types. Use the current plot type for all overlays?",
                parent=self,
            )
            if not ok:
                return
            for c in cfgs:
                try:
                    c.plot_type = str(current_plot_type)
                    self._ws.plot_defs[c.plot_id] = c
                except Exception:
                    continue

        self._ws.overlay_ids = overlay_ids
        if self._ws.active_id not in self._ws.overlay_ids:
            self._ws.active_id = self._ws.overlay_ids[0]
        self._status_var.set(f"Overlay: {len(self._ws.overlay_ids)} dataset(s)")
        self._plot()

    def _clear_overlay(self) -> None:
        self._ws.overlay_ids = []
        self._refresh_workspace()
        self._auto_plot_for_selection()

    def _overlay_select_all(self) -> None:
        for sid in list(self._ws.order):
            if self._overlay_tree.exists(str(sid)):
                self._overlay_tree.set(str(sid), "sel", "✔")

    def _overlay_select_none(self) -> None:
        for sid in list(self._ws.order):
            if self._overlay_tree.exists(str(sid)):
                self._overlay_tree.set(str(sid), "sel", "")

    def _overlay_make_active_first(self) -> None:
        if not self._ws.overlay_ids:
            return
        first = self._ws.overlay_ids[0]
        if first in self._ws.datasets:
            self._set_active_dataset(first)

    def _open_overlay_help(self) -> None:
        help_text = (
            "Overlay lets you plot multiple datasets on the same chart.\n"
            "1) Check datasets in the list.\n"
            "2) Click Overlay Selected to apply.\n"
            "3) Use Select all/none to manage checkboxes quickly.\n"
            "4) Make active = first overlay sets the active dataset to the first overlay item.\n"
            "If plot types differ, you can unify them to the current plot type."
        )
        messagebox.showinfo("Overlay – How to use", help_text, parent=self)

    def _on_overlay_mode_changed(self) -> None:
        pd = self._active_plot_def()
        if pd is None or not pd.y_cols:
            return
        try:
            self._plot()
            self._restore_y_selection_only()
        except Exception:
            pass

    def _restore_y_selection_only(self) -> None:
        self._update_y_summary()

    def _schedule_overlay_refresh(self) -> None:
        try:
            if self._overlay_refresh_job is not None:
                self.after_cancel(self._overlay_refresh_job)
        except Exception:
            pass
        try:
            self._overlay_refresh_job = self.after(180, self._on_overlay_mode_changed)
        except Exception:
            self._overlay_refresh_job = None

    def _on_overlay_click(self, evt) -> None:
        row = self._overlay_tree.identify_row(evt.y)
        col = self._overlay_tree.identify_column(evt.x)
        if not row or col != "#1":
            return
        cur = self._overlay_tree.set(row, "sel")
        self._overlay_tree.set(row, "sel", "" if cur == "✔" else "✔")

    def _refresh_workspace(self) -> None:
        self._ws_tree.delete(*self._ws_tree.get_children(""))
        self._overlay_tree.delete(*self._overlay_tree.get_children(""))
        try:
            self._plot_tree.delete(*self._plot_tree.get_children(""))
        except Exception:
            pass
        for sid in self._ws.order:
            ds = self._ws.datasets.get(sid)
            if ds is None:
                continue
            active = "●" if sid == self._ws.active_id else ""
            self._ws_tree.insert("", "end", iid=str(sid), values=(active, ds.display_name))
            ov = "✔" if sid in self._ws.overlay_ids else ""
            self._overlay_tree.insert("", "end", iid=str(sid), values=(ov, ds.display_name))

        self._refresh_recipes_ui()

    def _refresh_recipes_ui(self) -> None:
        try:
            items: List[str] = []
            for rid in (self._ws.recipe_order or list((self._ws.recipes or {}).keys())):
                rec = (self._ws.recipes or {}).get(rid)
                if not isinstance(rec, dict):
                    continue
                name = str(rec.get("name") or "Recipe")
                items.append(f"{name} :: {rid}")
            self._recipes_cb["values"] = items
            if items and self._recipes_var.get() not in items:
                self._recipes_var.set(items[0])
            if not items:
                self._recipes_var.set("")
        except Exception:
            pass

    def _selected_recipe_id(self) -> Optional[str]:
        raw = str(self._recipes_var.get() or "").strip()
        if "::" in raw:
            return raw.split("::", 1)[1].strip()
        return None

    def _save_recipe(self) -> None:
        pd = self._active_plot_def()
        if pd is None:
            return
        name = simpledialog.askstring("Save recipe", "Recipe name:", parent=self)
        if not name:
            return
        rid = uuid.uuid4().hex
        self._ws.recipes[rid] = {
            "name": str(name),
            "plot_type": str(pd.plot_type or "Line"),
            "x_col": (None if pd.x_col in (None, "") else str(pd.x_col)),
            "y_cols": list(pd.y_cols or []),
            "options": dict(pd.options or {}),
        }
        self._ws.recipe_order.append(rid)
        self._refresh_recipes_ui()

    def _rename_recipe(self) -> None:
        rid = self._selected_recipe_id()
        if not rid or rid not in (self._ws.recipes or {}):
            return
        name = simpledialog.askstring("Rename recipe", "New name:", parent=self)
        if not name:
            return
        self._ws.recipes[rid]["name"] = str(name)
        self._refresh_recipes_ui()

    def _delete_recipe(self) -> None:
        rid = self._selected_recipe_id()
        if not rid or rid not in (self._ws.recipes or {}):
            return
        ok = messagebox.askyesno("Delete recipe", "Delete this recipe?", parent=self)
        if not ok:
            return
        self._ws.recipes.pop(rid, None)
        if rid in self._ws.recipe_order:
            self._ws.recipe_order.remove(rid)
        self._refresh_recipes_ui()

    def _apply_recipe(self) -> None:
        rid = self._selected_recipe_id()
        if not rid or rid not in (self._ws.recipes or {}):
            return
        pd = self._active_plot_def()
        if pd is None:
            return
        rec = self._ws.recipes[rid]
        sid = self._ws.active_id
        ds = self._ws.datasets.get(sid) if sid else None
        cols = [str(c) for c in (ds.columns or {}).keys()] if ds else []
        lower_map = {str(c).lower(): str(c) for c in cols}

        def _map_col(c: Optional[str]) -> Optional[str]:
            if c is None:
                return None
            if c in cols:
                return c
            lc = str(c).lower()
            return lower_map.get(lc)

        def _as_opt_str(v: object) -> Optional[str]:
            if v in (None, ""):
                return None
            try:
                return str(v)
            except Exception:
                return None

        x_col = _map_col(_as_opt_str(rec.get("x_col")))
        y_cols_raw = rec.get("y_cols")
        y_cols_in: List[object] = list(y_cols_raw) if isinstance(y_cols_raw, list) else []
        y_cols = [_map_col(_as_opt_str(c)) for c in y_cols_in]
        y_cols = [c for c in y_cols if c]

        if (rec.get("x_col") and not x_col) or (rec.get("y_cols") and not y_cols):
            x_def, y_def = self._pick_default_axes(sid) if sid else (None, [])
            x_col = x_col or x_def
            y_cols = y_cols or list(y_def)
            self._banner_var.set("Some recipe columns were missing; used best defaults for this dataset.")

        pd.plot_type = str(rec.get("plot_type") or pd.plot_type or "Line")
        pd.x_col = x_col
        pd.y_cols = list(y_cols)
        try:
            opts_in = rec.get("options")
            opts_raw: Dict[str, Any] = (dict(opts_in) if isinstance(opts_in, dict) else {})
            opts: Dict[str, Any] = {str(k): v for k, v in (opts_raw or {}).items()}
            for k in ["group_col", "size_col", "x_err_col", "y_err_col", "heatmap_row", "heatmap_col", "heatmap_val"]:
                v = opts.get(k)
                if v not in (None, ""):
                    mapped = _map_col(_as_opt_str(v))
                    opts[k] = mapped if mapped else None
            pd.options = opts
        except Exception:
            pass
        self._ws.plot_defs[pd.plot_id] = pd
        self._populate_columns()
        self._restore_config_for_active()
        self._update_y_summary()
        self._mark_dirty()
        self._store_current_config()
        if self._auto_plot_var.get():
            try:
                self._plot()
            except Exception:
                self._banner_var.set("Auto-plot failed. Check axes or data.")

    def _open_recipes_help(self) -> None:
        help_text = (
            "Recipes save chart settings (type, axes, options) for reuse.\n"
            "1) Set X/Y and chart type.\n"
            "2) Click Save as recipe… to store it.\n"
            "3) Select a recipe and click Apply recipe.\n"
            "If columns are missing, best defaults are used automatically."
        )
        messagebox.showinfo("Recipes – How to use", help_text, parent=self)
        for pid, pd in self._ws.plot_defs.items():
            name = f"{self._plot_def_name(pd)}"
            active = "●" if pid == self._ws.active_plot_id else ""
            self._plot_tree.insert("", "end", iid=str(pid), values=(active, name))
        self._populate_columns()
        self._restore_config_for_active()

    def _selected_id(self, tree: Any) -> Optional[str]:
        try:
            sel = tree.selection()
            return str(sel[0]) if sel else None
        except Exception:
            return None

    def _load_df(self, ds: DataStudioDataset) -> pd.DataFrame:
        if ds.dataset_id in self._df_cache:
            return self._df_cache[ds.dataset_id]
        df = load_table(
            ds.path,
            sheet_name=ds.sheet_name,
            header_row=ds.header_row,
            decimal_comma=bool(self._decimal_var.get()),
            auto_cast=bool(self._autocast_var.get()),
        )
        self._df_cache[ds.dataset_id] = df
        return df

    def _transform_steps_hash(self, steps: List[Dict[str, Any]]) -> str:
        try:
            payload = json.dumps(steps or [], sort_keys=True, ensure_ascii=False)
        except Exception:
            payload = str(steps)
        return str(hash(payload))

    def _get_transformed_df(self, ds: DataStudioDataset) -> pd.DataFrame:
        base = self._load_df(ds)
        steps = list(getattr(ds, "transform_steps", []) or [])
        if not steps:
            return base.copy()
        step_hash = self._transform_steps_hash(steps)
        cached = self._transform_cache.get(ds.dataset_id)
        if cached and cached[0] == step_hash:
            return cached[1]
        out = apply_transform_steps(base, steps)
        warnings = list(out.attrs.get("transform_warnings") or [])
        self._transform_warnings[ds.dataset_id] = warnings
        self._transform_cache[ds.dataset_id] = (step_hash, out)
        return out

    def _get_plot_df(self, ds: DataStudioDataset) -> pd.DataFrame:
        if getattr(ds, "derived_enabled", False):
            return self._get_transformed_df(ds)
        return self._load_df(ds)

    def _refresh_transform_list(self) -> None:
        try:
            self._transform_list.delete(0, "end")
        except Exception:
            return
        sid = self._ws.active_id
        if not sid:
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        steps = list(getattr(ds, "transform_steps", []) or [])
        for i, step in enumerate(steps, start=1):
            self._transform_list.insert("end", f"{i}. {self._format_transform_step(step)}")

    def _format_transform_step(self, step: Dict[str, Any]) -> str:
        stype = str(step.get("type") or "")
        cols = ", ".join([str(c) for c in (step.get("columns") or [])])
        if stype == "select_columns":
            mode = str(step.get("mode") or "keep")
            return f"Select columns ({mode}): {cols or '(all)'}"
        if stype == "rename":
            mapping = step.get("mapping") or {}
            pairs = ", ".join([f"{k}->{v}" for k, v in mapping.items()])
            return f"Rename: {pairs or '(none)'}"
        if stype == "to_numeric":
            errs = str(step.get("errors") or "coerce")
            return f"To numeric ({errs}): {cols or '(all)'}"
        if stype == "fillna":
            val = step.get("value")
            return f"Fill NA ({val}): {cols or '(all)'}"
        if stype == "normalize":
            mode = str(step.get("mode") or "minmax")
            return f"Normalize ({mode}): {cols or '(all)'}"
        if stype == "baseline":
            method = str(step.get("method") or "first")
            rng = step.get("range")
            return f"Baseline ({method}{' ' + str(rng) if rng else ''}): {cols or '(all)'}"
        if stype == "log":
            base = step.get("base")
            offset = step.get("offset")
            return f"Log (base={base}, offset={offset}): {cols or '(all)'}"
        if stype == "rolling_mean":
            window = step.get("window")
            center = step.get("center")
            return f"Rolling mean (w={window}, center={center}): {cols or '(all)'}"
        return f"{stype or 'Step'}: {cols}"

    def _selected_transform_index(self) -> Optional[int]:
        try:
            sel = self._transform_list.curselection()
            return int(sel[0]) if sel else None
        except Exception:
            return None

    def _on_transform_toggle(self) -> None:
        sid = self._ws.active_id
        if not sid:
            try:
                self._transform_enabled_var.set(False)
            except Exception:
                pass
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        ds.derived_enabled = bool(self._transform_enabled_var.get())
        self._ws.datasets[sid] = ds
        self._transform_cache.pop(sid, None)
        self._transform_warnings.pop(sid, None)
        self._populate_columns()
        self._mark_dirty()
        if self._auto_plot_var.get():
            try:
                self._plot()
            except Exception:
                self._banner_var.set("Auto-plot failed. Check axes or data.")

    def _open_transform_step_editor(self, step: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        win = tk.Toplevel(self)
        win.title("Transform step")
        win.transient(self.winfo_toplevel())
        win.grab_set()
        win.columnconfigure(1, weight=1)

        type_var = tk.StringVar(value=str((step or {}).get("type") or "select_columns"))
        cols_var = tk.StringVar(value=", ".join([str(c) for c in (step or {}).get("columns", [])]))
        mode_var = tk.StringVar(value=str((step or {}).get("mode") or "keep"))
        errors_var = tk.StringVar(value=str((step or {}).get("errors") or "coerce"))
        fill_var = tk.StringVar(value=str((step or {}).get("value") or ""))
        norm_var = tk.StringVar(value=str((step or {}).get("mode") or "minmax"))
        base_var = tk.StringVar(value=str((step or {}).get("base") or 10))
        offset_var = tk.StringVar(value=str((step or {}).get("offset") or 0))
        roll_var = tk.StringVar(value=str((step or {}).get("window") or 5))
        center_var = tk.BooleanVar(value=bool((step or {}).get("center", True)))
        baseline_method_var = tk.StringVar(value=str((step or {}).get("method") or "first"))
        range_var = tk.StringVar(value="")
        r_raw = (step or {}).get("range")
        if isinstance(r_raw, list) and len(r_raw) == 2:
            range_var.set(f"{r_raw[0]}:{r_raw[1]}")
        rename_var = tk.StringVar(value="")
        mapping = (step or {}).get("mapping") or {}
        if isinstance(mapping, dict) and mapping:
            rename_var.set(", ".join([f"{k}:{v}" for k, v in mapping.items()]))

        ttk.Label(win, text="Type").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        type_cb = ttk.Combobox(
            win,
            textvariable=type_var,
            state="readonly",
            values=[
                "select_columns",
                "rename",
                "to_numeric",
                "fillna",
                "normalize",
                "baseline",
                "log",
                "rolling_mean",
            ],
        )
        type_cb.grid(row=0, column=1, sticky="ew", padx=10, pady=(10, 4))

        ttk.Label(win, text="Columns (comma-separated)").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(win, textvariable=cols_var).grid(row=1, column=1, sticky="ew", padx=10, pady=4)

        ttk.Label(win, text="Select mode").grid(row=2, column=0, sticky="w", padx=10, pady=4)
        ttk.Combobox(win, textvariable=mode_var, state="readonly", values=["keep", "drop"]).grid(
            row=2, column=1, sticky="ew", padx=10, pady=4
        )

        ttk.Label(win, text="Rename map (old:new)").grid(row=3, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(win, textvariable=rename_var).grid(row=3, column=1, sticky="ew", padx=10, pady=4)

        ttk.Label(win, text="To numeric errors").grid(row=4, column=0, sticky="w", padx=10, pady=4)
        ttk.Combobox(win, textvariable=errors_var, state="readonly", values=["coerce", "ignore", "raise"]).grid(
            row=4, column=1, sticky="ew", padx=10, pady=4
        )

        ttk.Label(win, text="Fill NA value (mean/ffill/const)").grid(row=5, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(win, textvariable=fill_var).grid(row=5, column=1, sticky="ew", padx=10, pady=4)

        ttk.Label(win, text="Normalize mode").grid(row=6, column=0, sticky="w", padx=10, pady=4)
        ttk.Combobox(win, textvariable=norm_var, state="readonly", values=["minmax", "zscore"]).grid(
            row=6, column=1, sticky="ew", padx=10, pady=4
        )

        ttk.Label(win, text="Baseline method").grid(row=7, column=0, sticky="w", padx=10, pady=4)
        ttk.Combobox(win, textvariable=baseline_method_var, state="readonly", values=["first", "mean_range"]).grid(
            row=7, column=1, sticky="ew", padx=10, pady=4
        )

        ttk.Label(win, text="Baseline range (start:end)").grid(row=8, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(win, textvariable=range_var).grid(row=8, column=1, sticky="ew", padx=10, pady=4)

        ttk.Label(win, text="Log base").grid(row=9, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(win, textvariable=base_var).grid(row=9, column=1, sticky="ew", padx=10, pady=4)

        ttk.Label(win, text="Log offset").grid(row=10, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(win, textvariable=offset_var).grid(row=10, column=1, sticky="ew", padx=10, pady=4)

        ttk.Label(win, text="Rolling window").grid(row=11, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(win, textvariable=roll_var).grid(row=11, column=1, sticky="ew", padx=10, pady=4)

        ttk.Checkbutton(win, text="Center rolling window", variable=center_var).grid(
            row=12, column=1, sticky="w", padx=10, pady=(4, 8)
        )

        result: Dict[str, Any] = {}

        def _parse_columns(raw: str) -> List[str]:
            return [c.strip() for c in (raw or "").split(",") if c.strip()]

        def _parse_mapping(raw: str) -> Dict[str, str]:
            out: Dict[str, str] = {}
            for part in (raw or "").split(","):
                if ":" not in part:
                    continue
                k, v = part.split(":", 1)
                k = k.strip()
                v = v.strip()
                if k and v:
                    out[k] = v
            return out

        def _parse_range(raw: str) -> Optional[List[int]]:
            if not raw or ":" not in raw:
                return None
            try:
                a, b = raw.split(":", 1)
                return [int(a.strip()), int(b.strip())]
            except Exception:
                return None

        def _save() -> None:
            stype = str(type_var.get()).strip()
            if not stype:
                messagebox.showwarning("Transform", "Choose a step type.", parent=win)
                return
            result["type"] = stype
            cols = _parse_columns(cols_var.get())
            if cols:
                result["columns"] = cols

            if stype == "select_columns":
                result["mode"] = str(mode_var.get() or "keep")
            elif stype == "rename":
                result["mapping"] = _parse_mapping(rename_var.get())
            elif stype == "to_numeric":
                result["errors"] = str(errors_var.get() or "coerce")
            elif stype == "fillna":
                val = fill_var.get()
                try:
                    if str(val).strip() and str(val).strip().lower() not in ("mean", "ffill"):
                        val = float(val)
                except Exception:
                    pass
                result["value"] = val
            elif stype == "normalize":
                result["mode"] = str(norm_var.get() or "minmax")
            elif stype == "baseline":
                result["method"] = str(baseline_method_var.get() or "first")
                rng = _parse_range(range_var.get())
                if rng:
                    result["range"] = rng
            elif stype == "log":
                try:
                    result["base"] = float(base_var.get())
                except Exception:
                    result["base"] = 10.0
                try:
                    result["offset"] = float(offset_var.get())
                except Exception:
                    result["offset"] = 0.0
            elif stype == "rolling_mean":
                try:
                    result["window"] = int(roll_var.get())
                except Exception:
                    result["window"] = 5
                result["center"] = bool(center_var.get())

            win.destroy()

        def _cancel() -> None:
            result.clear()
            win.destroy()

        btns = ttk.Frame(win)
        btns.grid(row=13, column=0, columnspan=2, sticky="e", padx=10, pady=(4, 10))
        ttk.Button(btns, text="Cancel", command=_cancel).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Save", command=_save).pack(side=tk.RIGHT, padx=(0, 6))

        win.wait_window()
        return result if result else None

    def _add_transform_step(self) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        step = self._open_transform_step_editor()
        if not step:
            return
        ds.transform_steps.append(step)
        self._ws.datasets[sid] = ds
        self._transform_cache.pop(sid, None)
        self._transform_warnings.pop(sid, None)
        self._refresh_transform_list()
        self._populate_columns()
        self._mark_dirty()

    def _edit_transform_step(self) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        idx = self._selected_transform_index()
        if idx is None or idx < 0 or idx >= len(ds.transform_steps):
            messagebox.showinfo("Transform", "Select a step to edit.", parent=self)
            return
        step = self._open_transform_step_editor(ds.transform_steps[idx])
        if not step:
            return
        ds.transform_steps[idx] = step
        self._ws.datasets[sid] = ds
        self._transform_cache.pop(sid, None)
        self._transform_warnings.pop(sid, None)
        self._refresh_transform_list()
        self._populate_columns()
        self._mark_dirty()

    def _remove_transform_step(self) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        idx = self._selected_transform_index()
        if idx is None or idx < 0 or idx >= len(ds.transform_steps):
            return
        ds.transform_steps.pop(idx)
        self._ws.datasets[sid] = ds
        self._transform_cache.pop(sid, None)
        self._transform_warnings.pop(sid, None)
        self._refresh_transform_list()
        self._populate_columns()
        self._mark_dirty()

    def _move_transform_step(self, delta: int) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        idx = self._selected_transform_index()
        if idx is None:
            return
        new_idx = idx + int(delta)
        if new_idx < 0 or new_idx >= len(ds.transform_steps):
            return
        ds.transform_steps[idx], ds.transform_steps[new_idx] = ds.transform_steps[new_idx], ds.transform_steps[idx]
        self._ws.datasets[sid] = ds
        self._transform_cache.pop(sid, None)
        self._transform_warnings.pop(sid, None)
        self._refresh_transform_list()
        try:
            self._transform_list.selection_set(new_idx)
        except Exception:
            pass
        self._populate_columns()
        self._mark_dirty()

    def _clear_transform_steps(self) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        if not ds.transform_steps:
            return
        if not messagebox.askyesno("Transform", "Clear all transform steps?", parent=self):
            return
        ds.transform_steps = []
        self._ws.datasets[sid] = ds
        self._transform_cache.pop(sid, None)
        self._transform_warnings.pop(sid, None)
        self._refresh_transform_list()
        self._populate_columns()
        self._mark_dirty()

    def _open_transform_help(self) -> None:
        help_text = (
            "Transform steps let you create a non-destructive pipeline.\n"
            "Use Add step… to build the pipeline, then enable\n"
            "'Use transformed data for plotting'.\n\n"
            "Common steps:\n"
            "• select_columns (keep/drop)\n"
            "• rename\n"
            "• to_numeric / fillna\n"
            "• normalize / baseline / log / rolling_mean"
        )
        messagebox.showinfo("Transform – Help", help_text, parent=self)

    def _populate_columns(self) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            return
        self._restoring_ui = True
        cols_map = dict(ds.columns or {})
        try:
            df = self._get_plot_df(ds) if getattr(ds, "derived_enabled", False) else self._load_df(ds)
        except Exception:
            df = pd.DataFrame()
        if df is not None and not df.empty:
            cols_map = column_type_map(df)
        elif not cols_map:
            cols_map = {}
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
        self._x_all_values = list(x_values)
        self._refresh_x_values(self._x_search_var.get())

        self._y_display_to_col = {}
        self._y_col_to_display = {}
        for c in cols:
            d = _disp(c, cols_map.get(c, ""))
            self._y_display_to_col[d] = c
            self._y_col_to_display[c] = d
        self._refresh_y_list()

        group_vals = ["(None)"] + cols
        self._group_cb["values"] = group_vals
        if self._group_var.get() not in group_vals:
            self._group_var.set("(None)")

        # extra selectors
        self._size_var.set("(None)")
        self._xerr_var.set("(None)")
        self._yerr_var.set("(None)")
        self._heat_row_var.set("(None)")
        self._heat_col_var.set("(None)")
        self._heat_val_var.set("(None)")
        self._restoring_ui = False

    def _restore_config_for_active(self) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        pd = self._active_plot_def()
        if pd is None:
            return
        self._restoring_ui = True

        # Auto-pick defaults if needed
        if not pd.x_col or not pd.y_cols:
            x_def, y_def = self._pick_default_axes(sid)
            if not pd.x_col:
                pd.x_col = x_def
            if not pd.y_cols:
                pd.y_cols = list(y_def)

        try:
            disp = self._x_col_to_display.get(str(pd.x_col or ""), "(Index)") if pd.x_col else "(Index)"
            self._x_var.set(str(disp))
        except Exception:
            pass
        self._refresh_x_values(self._x_search_var.get())
        try:
            self._plot_type_var.set(str(pd.plot_type or PLOT_TYPES[0]))
        except Exception:
            pass
        self._toggle_extra_fields()

        opts = dict(pd.options or {})
        self._group_var.set(str(opts.get("group_col") or "(None)"))
        self._size_var.set(str(opts.get("size_col") or "(None)"))
        self._xerr_var.set(str(opts.get("x_err_col") or "(None)"))
        self._yerr_var.set(str(opts.get("y_err_col") or "(None)"))
        self._heat_row_var.set(str(opts.get("heatmap_row") or "(None)"))
        self._heat_col_var.set(str(opts.get("heatmap_col") or "(None)"))
        self._heat_val_var.set(str(opts.get("heatmap_val") or "(None)"))
        self._heat_agg_var.set(str(opts.get("heatmap_agg") or "mean"))
        bins_raw = opts.get("hist_bins", 20)
        bins_val = 20
        if isinstance(bins_raw, (int, float, str)):
            try:
                bins_val = int(bins_raw)
            except Exception:
                bins_val = 20
        self._bins_var.set(int(bins_val))

        roll_raw = opts.get("rolling_window", 5)
        roll_val = 5
        if isinstance(roll_raw, (int, float, str)):
            try:
                roll_val = int(roll_raw)
            except Exception:
                roll_val = 5
        self._roll_var.set(int(roll_val))
        self._x_title_var.set(str(opts.get("x_title") or ""))
        self._y_title_var.set(str(opts.get("y_title") or ""))

        self._update_y_summary()

        try:
            self._drop_na_var.set(bool(opts.get("drop_na", True)))
            self._decimal_var.set(bool(opts.get("decimal_comma", False)))
            self._autocast_var.set(bool(opts.get("auto_cast", True)))
            self._norm_var.set(str(opts.get("normalize") or "None"))
        except Exception:
            pass
        self._restoring_ui = False

    def _auto_plot_for_selection(self) -> None:
        sid = self._ws.active_id
        if not sid:
            return
        if not self._auto_plot_var.get():
            return
        pd = self._active_plot_def()
        if pd is None or not pd.y_cols:
            self._banner_var.set("Select at least one Y column to plot.")
            return
        ds = self._ws.datasets.get(sid)
        cols = set((ds.columns or {}).keys()) if ds else set()
        if pd.x_col and pd.x_col not in cols:
            self._banner_var.set("X column is missing for this dataset.")
            return
        if any((y not in cols) for y in (pd.y_cols or [])):
            self._banner_var.set("Some Y columns are missing for this dataset.")
            return
        if self._ws.overlay_ids:
            try:
                self._plot()
            except Exception:
                self._banner_var.set("Auto-plot failed. Check axes or data.")
                return
            return
        if sid in self._plotted_ids:
            try:
                self._plot()
            except Exception:
                self._banner_var.set("Auto-plot failed. Check axes or data.")
                return
        else:
            try:
                self._plot()
            except Exception:
                self._banner_var.set("Auto-plot failed. Check axes or data.")

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

        # Preferred axes memory wins if valid
        pref = self._ws.preferred_axes_by_dataset.get(dataset_id)
        if pref:
            px, py = pref
            if px in cols:
                x_col = px
            if py and all(y in cols for y in py):
                y_cols = list(py)

        return x_col, y_cols

    def _refresh_x_values(self, filter_text: str) -> None:
        filt = str(filter_text or "").strip().lower()
        values = list(self._x_all_values or [])
        if filt:
            values = [v for v in values if v == "(Index)" or filt in v.lower()]
            if "(Index)" not in values:
                values.insert(0, "(Index)")
        self._x_cb["values"] = values
        if self._x_var.get() not in values:
            self._x_var.set("(Index)")

    def _clear_x_search(self) -> None:
        self._x_search_var.set("")
        self._refresh_x_values("")

    def _auto_axes_both(self) -> None:
        sid = self._ws.active_id
        pd = self._active_plot_def()
        if not sid or pd is None:
            return
        x_def, y_def = self._pick_default_axes(sid)
        pd.x_col = x_def
        pd.y_cols = list(y_def)
        self._ws.plot_defs[pd.plot_id] = pd
        try:
            disp = self._x_col_to_display.get(str(pd.x_col or ""), "(Index)") if pd.x_col else "(Index)"
            self._x_var.set(str(disp))
        except Exception:
            pass
        self._update_y_summary()
        self._mark_dirty()
        self._store_current_config()

    def _auto_axes_y_only(self) -> None:
        sid = self._ws.active_id
        pd = self._active_plot_def()
        if not sid or pd is None:
            return
        _x_def, y_def = self._pick_default_axes(sid)
        pd.y_cols = list(y_def)
        self._ws.plot_defs[pd.plot_id] = pd
        self._update_y_summary()
        self._mark_dirty()
        self._store_current_config()

    def _open_axes_help(self) -> None:
        help_text = (
            "X axis: choose one column for the horizontal axis; Index uses row numbers.\n"
            "Y columns: choose one or more columns to plot.\n"
            "Search: use search boxes to quickly find columns by name.\n"
            "Numeric-only: keeps the list clean for typical plots; turn off if needed.\n"
            "Auto buttons: restore recommended defaults if columns change or you’re unsure."
        )
        messagebox.showinfo("Axes Selection – How to use", help_text, parent=self)

    def _toggle_extra_fields(self) -> None:
        for w in self._extra.winfo_children():
            w.destroy()
        kind = str(self._plot_type_var.get())

        if kind == "Bubble":
            ttk.Label(self._extra, text="Size column").grid(row=0, column=0, sticky="w")
            ttk.Combobox(self._extra, textvariable=self._size_var, state="readonly").grid(row=1, column=0, sticky="ew")
        if kind == "Heatmap":
            ttk.Label(self._extra, text="Row").grid(row=0, column=0, sticky="w")
            ttk.Combobox(self._extra, textvariable=self._heat_row_var, state="readonly").grid(row=1, column=0, sticky="ew")
            ttk.Label(self._extra, text="Col").grid(row=2, column=0, sticky="w")
            ttk.Combobox(self._extra, textvariable=self._heat_col_var, state="readonly").grid(row=3, column=0, sticky="ew")
            ttk.Label(self._extra, text="Value").grid(row=4, column=0, sticky="w")
            ttk.Combobox(self._extra, textvariable=self._heat_val_var, state="readonly").grid(row=5, column=0, sticky="ew")
            ttk.Label(self._extra, text="Agg").grid(row=6, column=0, sticky="w")
            ttk.Combobox(self._extra, textvariable=self._heat_agg_var, values=["mean", "sum", "median"], state="readonly").grid(
                row=7, column=0, sticky="ew"
            )
        if kind == "Histogram":
            ttk.Label(self._extra, text="Bins").grid(row=0, column=0, sticky="w")
            ttk.Spinbox(self._extra, from_=5, to=200, textvariable=self._bins_var, width=8).grid(row=1, column=0, sticky="w")
        if kind == "Rolling mean":
            ttk.Label(self._extra, text="Window").grid(row=0, column=0, sticky="w")
            ttk.Spinbox(self._extra, from_=2, to=200, textvariable=self._roll_var, width=8).grid(row=1, column=0, sticky="w")
        if kind == "Errorbar":
            ttk.Label(self._extra, text="Y error").grid(row=0, column=0, sticky="w")
            ttk.Combobox(self._extra, textvariable=self._yerr_var, state="readonly").grid(row=1, column=0, sticky="ew")
            ttk.Label(self._extra, text="X error").grid(row=2, column=0, sticky="w")
            ttk.Combobox(self._extra, textvariable=self._xerr_var, state="readonly").grid(row=3, column=0, sticky="ew")

        # refresh values
        cols = ["(None)"]
        sid = self._ws.active_id
        if sid and sid in self._ws.datasets:
            df = self._load_df(self._ws.datasets[sid])
            cols += [str(c) for c in df.columns]
        for v in [self._size_var, self._heat_row_var, self._heat_col_var, self._heat_val_var, self._xerr_var, self._yerr_var]:
            try:
                if v.get() not in cols:
                    v.set("(None)")
            except Exception:
                pass
        for cb in self._extra.winfo_children():
            if isinstance(cb, ttk.Combobox):
                cb["values"] = cols

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

        numeric_only = str(self._plot_type_var.get()) not in ("Heatmap",)
        if numeric_only:
            sid = self._ws.active_id
            ds = self._ws.datasets.get(sid) if sid else None
            numeric = set()
            try:
                df = self._get_plot_df(ds) if ds is not None else pd.DataFrame()
                numeric = set(numeric_columns(df))
            except Exception:
                cols_map = dict(ds.columns or {}) if ds else {}
                for name, dtype in cols_map.items():
                    if "int" in str(dtype) or "float" in str(dtype) or "double" in str(dtype):
                        numeric.add(str(name))
            items = [d for d in items if self._y_display_to_col.get(d, d) in numeric or not numeric]

        return items

    def _update_y_summary(self) -> None:
        pd = self._active_plot_def()
        if pd is None or not pd.y_cols:
            self._y_summary_var.set("Y: (none)")
            return
        disp = [self._y_col_to_display.get(str(y), str(y)) for y in pd.y_cols]
        if len(disp) <= 3:
            self._y_summary_var.set("Y: " + ", ".join(disp))
        else:
            self._y_summary_var.set(f"Y: {len(disp)} selected")

    def _select_all_numeric_y(self) -> None:
        sid = self._ws.active_id
        ds = self._ws.datasets.get(sid) if sid else None
        if ds is None:
            return
        cols_map = dict(ds.columns or {})
        numeric = set()
        for name, dtype in cols_map.items():
            if "int" in str(dtype) or "float" in str(dtype) or "double" in str(dtype):
                numeric.add(str(name))
        pd = self._active_plot_def()
        if pd is None:
            return
        pd.y_cols = [c for c in numeric]
        self._ws.plot_defs[pd.plot_id] = pd
        self._update_y_summary()
        self._mark_dirty()
        self._store_current_config()

    def _clear_y_selection(self) -> None:
        pd = self._active_plot_def()
        if pd is None:
            return
        pd.y_cols = []
        self._ws.plot_defs[pd.plot_id] = pd
        self._update_y_summary()
        self._mark_dirty()
        self._store_current_config()

    def _on_plot_type_changed(self) -> None:
        if self._restoring_ui:
            return
        self._toggle_extra_fields()
        self._refresh_y_list()
        self._mark_dirty()
        self._store_current_config()
        if self._ws.overlay_ids:
            new_type = str(self._plot_type_var.get())
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

    def _on_y_changed(self) -> None:
        if self._restoring_ui:
            return
        self._mark_dirty()
        self._store_current_config()

    def _reset_plot_builder(self) -> None:
        self._plot_type_var.set(PLOT_TYPES[0])
        self._drop_na_var.set(True)
        self._decimal_var.set(False)
        self._autocast_var.set(True)
        self._norm_var.set("None")
        self._populate_columns()
        self._restore_config_for_active()
        self._clear_dirty()

    def _collect_selected_y(self) -> List[str]:
        pd = self._active_plot_def()
        if pd is None:
            return []
        return list(pd.y_cols or [])

    def _plot(self) -> None:
        try:
            base_series, meta = self._build_plot_series()
        except Exception as exc:
            messagebox.showerror("Plot", str(exc), parent=self)
            return

        series = [dict(s) for s in (base_series or [])]
        self._apply_overlay_offset(series)

        self._ax.clear()
        plot_type = str(meta.get("plot_type", "Line"))

        if plot_type in ("Box plot", "Violin plot"):
            data = [np.asarray(s.get("y", []), dtype=float) for s in series]
            labels = [str(s.get("label", "")) for s in series]
            if plot_type == "Box plot":
                try:
                    self._ax.boxplot(data, tick_labels=labels, showfliers=True)
                except TypeError:
                    # Older Matplotlib accepts `labels=...`; call dynamically to avoid stub mismatch.
                    boxplot_any = getattr(self._ax, "boxplot")
                    boxplot_any(data, labels=labels, showfliers=True)
            else:
                self._ax.violinplot(data, showmeans=True, showmedians=True)
                self._ax.set_xticks(range(1, len(labels) + 1))
                self._ax.set_xticklabels(labels, rotation=45, ha="right")
        elif plot_type == "Histogram":
            for s in series:
                y = np.asarray(s.get("y", []), dtype=float)
                self._ax.hist(
                    y,
                    bins=int(self._bins_var.get()),
                    alpha=0.5,
                    label=str(s.get("label", "")),
                    color=s.get("color"),
                )
        elif plot_type in ("Bar (grouped)", "Bar (stacked)"):
            # group by x positions
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
                    self._ax.bar(
                        x + i * width - (len(series) - 1) * width / 2,
                        y,
                        width=width,
                        label=labels[i],
                        color=s.get("color"),
                    )
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

        hm = meta.get("heatmap")
        if hm is not None:
            hm_d = hm if isinstance(hm, dict) else {}
            self._ax.clear()
            im = self._ax.imshow(hm_d.get("values", []), aspect="auto")
            cols = list(hm_d.get("cols", []) or [])
            rows = list(hm_d.get("rows", []) or [])
            self._ax.set_xticks(range(len(cols)))
            self._ax.set_xticklabels([str(c) for c in cols], rotation=45, ha="right")
            self._ax.set_yticks(range(len(rows)))
            self._ax.set_yticklabels([str(r) for r in rows])
            self._fig.colorbar(im, ax=self._ax, fraction=0.046, pad=0.04)

        self._ax.set_title(meta.get("title", ""))
        self._ax.set_xlabel(meta.get("xlabel", ""))
        self._ax.set_ylabel(meta.get("ylabel", ""))
        if len(series) > 1:
            self._ax.legend(loc="best")
        self._ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()
        self._last_payload = {"series": base_series, "overlay_mode": str(self._overlay_mode_var.get()), "overlay_offset": self._safe_float(self._overlay_offset_var.get()), **meta}
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
                    sid_s = str(ds.dataset_id)
                    x_def, y_def = self._pick_default_axes(sid_s)
                    if not pd.x_col:
                        pd.x_col = x_def
                    if not pd.y_cols:
                        pd.y_cols = list(y_def)
                    msg = f"Columns changed; auto-selected X={pd.x_col or 'Index'}, Y={', '.join(pd.y_cols or [])}"
                if msg:
                    try:
                        self._banner_var.set(msg)
                    except Exception:
                        pass
                    self._restore_config_for_active()
            pd.last_validated_schema_hash = str(ds.schema_hash or "")
            self._ws.plot_defs[pd.plot_id] = pd
            try:
                self._plot_tree.item(pd.plot_id, values=("●", self._plot_def_name(pd)))
            except Exception:
                pass

        self._plot()
        self._clear_dirty()

    def _build_plot_series(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        sid_list = self._ws.overlay_ids if self._ws.overlay_ids else ([self._ws.active_id] if self._ws.active_id else [])
        if not sid_list:
            raise ValueError("No active dataset selected.")

        # If overlay is active, require stored plot defs and same plot type
        if self._ws.overlay_ids:
            plot_defs = [d for d in self._ws.plot_defs.values() if d.dataset_id in sid_list]
            if len(plot_defs) != len(sid_list):
                raise ValueError("Overlay requires plotting each dataset first (store X/Y selections).")
            pt = str(plot_defs[0].plot_type)
            if any(str(d.plot_type) != pt for d in plot_defs if d is not None):
                raise ValueError("Overlay requires the same plot type across datasets.")

        plot_def = self._active_plot_def()
        xcol = None
        ycols: List[str] = []
        group_col = None
        plot_type = str(self._plot_type_var.get())
        opts: Dict[str, Any] = {}

        if plot_def is not None:
            xcol = plot_def.x_col
            ycols = list(plot_def.y_cols or [])
            plot_type = str(plot_def.plot_type or plot_type)
            opts = dict(plot_def.options or {})
            group_col = opts.get("group_col")

        if xcol is None:
            x_disp = self._x_var.get()
            xcol = None if x_disp == "(Index)" else self._x_display_to_col.get(x_disp, x_disp)

        if not ycols:
            ycols = self._collect_selected_y()

        if not ycols:
            raise ValueError("Select at least one Y column.")

        if group_col == "(None)":
            group_col = None
        series: List[Dict[str, Any]] = []
        x_title = str(opts.get("x_title") or "").strip()
        y_title = str(opts.get("y_title") or "").strip()
        meta = {
            "title": "",
            "xlabel": (x_title if x_title else (xcol if xcol and xcol != "(Index)" else "Index")),
            "ylabel": (y_title if y_title else ", ".join(ycols)),
            "plot_type": plot_type,
            "xcats": [],
        }

        for sid in sid_list:
            ds = self._ws.datasets.get(sid)
            if ds is None:
                continue
            if self._ws.overlay_ids:
                for opd in self._ws.plot_defs.values():
                    if opd.dataset_id == sid:
                        xcol = str(opd.x_col or "(Index)")
                        ycols = list(opd.y_cols or [])
                        plot_type = str(opd.plot_type or plot_type)
                        opts = dict(opd.options or {})
                        group_col = opts.get("group_col")
                        break
            df = self._load_df(ds)
            drop_na = bool(opts.get("drop_na", self._drop_na_var.get()))
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
                    norm = str(opts.get("normalize", self._norm_var.get()))
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

                    series.append({
                        "id": f"{sid}:{y}:{gval}",
                        "kind": kind,
                        "x": xvals,
                        "y": yvals,
                        "label": label,
                        "xerr": None,
                        "yerr": None,
                        "size": None,
                    })

        # Special plot types
        if plot_type == "Histogram":
            series = []
            for sid in sid_list:
                ds = self._ws.datasets.get(sid)
                if ds is None:
                    continue
                df = self._load_df(ds)
                for y in ycols:
                    if y not in df.columns:
                        continue
                    vals = np.asarray(df[y], dtype=float)
                    series.append({"id": f"{sid}:{y}", "kind": "hist", "x": None, "y": vals, "label": f"{ds.display_name}:{y}"})

        if plot_type == "Heatmap":
            ds = self._ws.datasets.get(sid_list[0])
            df = self._load_df(ds) if ds else pd.DataFrame()
            r = str(opts.get("heatmap_row") or self._heat_row_var.get())
            c = str(opts.get("heatmap_col") or self._heat_col_var.get())
            v = str(opts.get("heatmap_val") or self._heat_val_var.get())
            if r == "(None)" or c == "(None)" or v == "(None)":
                raise ValueError("Select Row/Col/Value for heatmap.")
            agg = str(opts.get("heatmap_agg") or self._heat_agg_var.get())
            pv = pd.pivot_table(df, index=r, columns=c, values=v, aggfunc=cast(Any, agg))
            meta["heatmap"] = {"rows": list(pv.index), "cols": list(pv.columns), "values": pv.values}
            series = []

        if plot_type in ("Bar (grouped)", "Bar (stacked)"):
            # Aggregate Y by X categories
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
            size_col = str(opts.get("size_col") or self._size_var.get())
            for s in series:
                sid = str(s["id"]).split(":", 1)[0]
                ds = self._ws.datasets.get(sid)
                if ds is None:
                    continue
                df = self._load_df(ds)
                if size_col in df.columns:
                    s["size"] = np.asarray(df[size_col], dtype=float)

        if plot_type == "Errorbar":
            xerr_col = str(opts.get("x_err_col") or self._xerr_var.get())
            yerr_col = str(opts.get("y_err_col") or self._yerr_var.get())
            for s in series:
                sid = str(s["id"]).split(":", 1)[0]
                ds = self._ws.datasets.get(sid)
                if ds is None:
                    continue
                df = self._load_df(ds)
                if xerr_col in df.columns:
                    s["xerr"] = np.asarray(df[xerr_col], dtype=float)
                if yerr_col in df.columns:
                    s["yerr"] = np.asarray(df[yerr_col], dtype=float)

        return series, meta

    def _apply_overlay_offset(self, series: List[Dict[str, Any]]) -> None:
        if not self._ws.overlay_ids or not series:
            return
        try:
            mode = str(self._overlay_mode_var.get() or "Normal")
            offset = self._safe_float(self._overlay_offset_var.get())
        except Exception:
            mode = "Normal"
            offset = 0.0
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

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(str(value).strip())
        except Exception:
            return 0.0

    def _store_current_config(self) -> None:
        pd = self._active_plot_def()
        if pd is None:
            return
        x_disp = self._x_var.get()
        x_col = None if x_disp == "(Index)" else self._x_display_to_col.get(x_disp, x_disp)
        y_cols = self._collect_selected_y()
        pd.x_col = x_col
        pd.y_cols = list(y_cols)
        pd.plot_type = str(self._plot_type_var.get())
        pd.options = {
            "group_col": (None if self._group_var.get() == "(None)" else self._group_var.get()),
            "y_err_col": (None if self._yerr_var.get() == "(None)" else self._yerr_var.get()),
            "x_err_col": (None if self._xerr_var.get() == "(None)" else self._xerr_var.get()),
            "size_col": (None if self._size_var.get() == "(None)" else self._size_var.get()),
            "heatmap_row": (None if self._heat_row_var.get() == "(None)" else self._heat_row_var.get()),
            "heatmap_col": (None if self._heat_col_var.get() == "(None)" else self._heat_col_var.get()),
            "heatmap_val": (None if self._heat_val_var.get() == "(None)" else self._heat_val_var.get()),
            "heatmap_agg": str(self._heat_agg_var.get()),
            "hist_bins": int(self._bins_var.get()),
            "rolling_window": int(self._roll_var.get()),
            "drop_na": bool(self._drop_na_var.get()),
            "decimal_comma": bool(self._decimal_var.get()),
            "auto_cast": bool(self._autocast_var.get()),
            "normalize": str(self._norm_var.get()),
            "x_title": str(self._x_title_var.get() or ""),
            "y_title": str(self._y_title_var.get() or ""),
        }
        self._ws.plot_defs[pd.plot_id] = pd
        if pd.dataset_id:
            self._ws.preferred_axes_by_dataset[pd.dataset_id] = (pd.x_col, list(pd.y_cols or []))

    def _export_plot(self) -> None:
        payload = getattr(self, "_last_payload", None)
        if not payload:
            messagebox.showinfo("Export", "Plot something first.")
            return
        DataStudioExportEditor(self, payload=payload)

    def _export_transformed_csv(self) -> None:
        sid = self._ws.active_id
        if not sid:
            messagebox.showinfo("Export", "Select a dataset first.", parent=self)
            return
        ds = self._ws.datasets.get(sid)
        if ds is None:
            messagebox.showinfo("Export", "Select a dataset first.", parent=self)
            return
        steps = list(getattr(ds, "transform_steps", []) or [])
        if not steps:
            messagebox.showinfo("Export", "No transform steps to export.", parent=self)
            return

        base_name = Path(str(getattr(ds, "display_name", "data"))).stem or "data"
        suffix = str(getattr(ds, "derived_name_suffix", " (transformed)"))
        default_name = f"{base_name}{suffix}.csv"

        path = filedialog.asksaveasfilename(
            title="Export transformed CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return

        try:
            df = self._get_transformed_df(ds)
            df.to_csv(path, index=False)
        except Exception as exc:
            messagebox.showerror("Export", f"Failed to export CSV:\n\n{exc}", parent=self)
            return

        warnings = self._transform_warnings.get(ds.dataset_id) or []
        if warnings:
            self._banner_var.set("Exported with transform warnings. Check Transform help for details.")
        else:
            self._banner_var.set("Exported transformed CSV.")
