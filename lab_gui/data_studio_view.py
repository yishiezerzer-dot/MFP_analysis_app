from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

import ttkbootstrap as tb
import tkinter.ttk as ttk_native

ttk = tb
ttk.LabelFrame = ttk_native.LabelFrame

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from lab_gui.ui_widgets import MatplotlibNavigator, ToolTip
from lab_gui.data_studio_model import DataStudioDataset, DataStudioWorkspace, DataStudioPlotDef
from lab_gui.data_studio_io import (
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

        body = ttk.Frame(self, padding=10)
        body.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)
        body.columnconfigure(0, weight=1)

        info = ttk.Frame(body)
        info.grid(row=0, column=0, sticky="ew")
        info.columnconfigure(1, weight=1)
        ttk.Label(info, text=f"Rows: {len(df)}").grid(row=0, column=0, sticky="w")
        ttk.Label(info, text=f"Columns: {len(df.columns)}").grid(row=0, column=1, sticky="w", padx=(12, 0))

        sheets = get_sheet_names(path)
        if sheets:
            ttk.Label(info, text="Sheet").grid(row=0, column=2, sticky="e", padx=(10, 0))
            self._sheet_var = tk.StringVar(value=str(dataset.sheet_name or sheets[0]))
            sheet_cb = ttk.Combobox(info, values=sheets, textvariable=self._sheet_var, state="readonly", width=24)
            sheet_cb.grid(row=0, column=3, sticky="e")
            sheet_cb.bind("<<ComboboxSelected>>", lambda _e: self._reload_sheet())
        else:
            self._sheet_var = None

        ttk.Separator(body).grid(row=1, column=0, sticky="ew", pady=8)

        self._tree = ttk.Treeview(body, show="headings")
        self._tree.grid(row=2, column=0, sticky="nsew")
        sb = ttk.Scrollbar(body, orient="vertical", command=self._tree.yview)
        sb.grid(row=2, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=sb.set)

        self._render_table(df)

    def _render_table(self, df: pd.DataFrame) -> None:
        self._tree.delete(*self._tree.get_children(""))
        self._tree["columns"] = [str(c) for c in df.columns]
        for c in df.columns:
            self._tree.heading(str(c), text=str(c))
            self._tree.column(str(c), width=120, stretch=True)
        for _, row in df.head(500).iterrows():
            self._tree.insert("", "end", values=[row.get(c, "") for c in df.columns])

    def _reload_sheet(self) -> None:
        if self._sheet_var is None:
            return
        sheet = str(self._sheet_var.get())
        df = load_table(self._path, sheet_name=sheet, header_row=self._dataset.header_row)
        self._dataset.sheet_name = sheet
        self._df = df
        self._render_table(df)


class DataStudioView(ttk.Frame):
    def __init__(self, parent: tk.Widget, app: Any, workspace: Any) -> None:
        super().__init__(parent)
        self.app = app
        self.workspace = workspace
        self._ws = DataStudioWorkspace()
        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._plotted_ids: set = set()
        self._status_var = tk.StringVar(value="Ready")
        self._overlay_mode_var = tk.StringVar(value="Normal")
        self._overlay_offset_var = tk.StringVar(value="0.0")
        self._overlay_refresh_job = None
        self._x_display_to_col: Dict[str, str] = {}
        self._x_col_to_display: Dict[str, str] = {}
        self._y_display_to_col: Dict[str, str] = {}
        self._y_col_to_display: Dict[str, str] = {}
        self._y_summary_var = tk.StringVar(value="Y: (none)")
        self._dirty_var = tk.StringVar(value="")
        self._banner_var = tk.StringVar(value="")
        self._restoring_ui = False

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
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        ws = ttk.LabelFrame(self, text="Workspace", padding=8)
        ws.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ws.columnconfigure(0, weight=1)
        ws.rowconfigure(1, weight=1)

        btns = ttk.Frame(ws)
        btns.grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Add Files…", command=self._add_files).grid(row=0, column=0, sticky="w")
        ttk.Button(btns, text="Remove Selected", command=self._remove_selected).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(btns, text="Clear", command=self._clear_workspace).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(btns, text="Save…", command=self._save_workspace).grid(row=0, column=3, padx=(6, 0))
        ttk.Button(btns, text="Load…", command=self._load_workspace).grid(row=0, column=4, padx=(6, 0))

        self._ws_tree = ttk.Treeview(ws, columns=("active", "name"), show="headings", height=10, selectmode="browse")
        self._ws_tree.heading("active", text="Active")
        self._ws_tree.heading("name", text="File")
        self._ws_tree.column("active", width=60, stretch=False, anchor="center")
        self._ws_tree.column("name", width=200, stretch=True)
        self._ws_tree.grid(row=1, column=0, sticky="nsew")
        self._ws_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        ttk.Button(ws, text="Set Active", command=self._set_active_from_selection).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(ws, text="Preview Table", command=self._preview_data).grid(row=3, column=0, sticky="ew", pady=(6, 0))

        defs = ttk.LabelFrame(ws, text="Plot Definitions", padding=8)
        defs.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        defs.columnconfigure(0, weight=1)
        self._plot_tree = ttk.Treeview(defs, columns=("active", "name"), show="headings", height=5, selectmode="browse")
        self._plot_tree.heading("active", text="Active")
        self._plot_tree.heading("name", text="Plot")
        self._plot_tree.column("active", width=60, stretch=False, anchor="center")
        self._plot_tree.column("name", width=200, stretch=True)
        self._plot_tree.grid(row=0, column=0, sticky="ew")
        self._plot_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_plot_select())
        defs_btns = ttk.Frame(defs)
        defs_btns.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(defs_btns, text="New Plot", command=self._new_plot_def).grid(row=0, column=0, sticky="w")
        ttk.Button(defs_btns, text="Remove Plot", command=self._remove_plot_def).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(defs_btns, text="Set Active", command=self._set_active_plot_from_selection).grid(row=0, column=2, padx=(6, 0))

        overlay = ttk.LabelFrame(ws, text="Overlay", padding=8)
        overlay.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        self._overlay_tree = ttk.Treeview(overlay, columns=("sel", "name"), show="headings", height=6, selectmode="browse")
        self._overlay_tree.heading("sel", text="Overlay")
        self._overlay_tree.heading("name", text="File")
        self._overlay_tree.column("sel", width=70, stretch=False, anchor="center")
        self._overlay_tree.column("name", width=180, stretch=True)
        self._overlay_tree.grid(row=0, column=0, sticky="ew")
        self._overlay_tree.bind("<Button-1>", self._on_overlay_click, add=True)
        ttk.Button(overlay, text="Overlay Selected", command=self._apply_overlay).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(overlay, text="Clear Overlay", command=self._clear_overlay).grid(row=2, column=0, sticky="ew", pady=(6, 0))


        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        top = ttk.Frame(right)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, textvariable=self._status_var).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self._dirty_var, foreground="#b00020").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(top, text="Apply", command=self._apply_plot).pack(side=tk.RIGHT)
        ttk.Button(top, text="Reset", command=self._reset_plot_builder).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(top, text="Export…", command=self._export_plot).pack(side=tk.RIGHT, padx=(8, 0))

        controls = ttk.Frame(right)
        controls.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(controls, text="X").pack(side=tk.LEFT)
        self._x_var = tk.StringVar(value="")
        self._x_cb = ttk.Combobox(controls, textvariable=self._x_var, state="readonly", width=24)
        self._x_cb.pack(side=tk.LEFT, padx=(6, 10))
        self._x_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_x_changed())

        ttk.Label(controls, text="Plot").pack(side=tk.LEFT)
        self._plot_type_var = tk.StringVar(value=PLOT_TYPES[0])
        self._plot_cb = ttk.Combobox(controls, textvariable=self._plot_type_var, values=PLOT_TYPES, state="readonly", width=18)
        self._plot_cb.pack(side=tk.LEFT, padx=(6, 10))
        self._plot_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_plot_type_changed())

        ttk.Button(controls, text="Y columns…", command=self._open_y_selector).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Data options…", command=self._open_data_options).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Advanced…", command=self._toggle_advanced_panel).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(controls, textvariable=self._y_summary_var, foreground="#555").pack(side=tk.RIGHT)

        ttk.Label(controls, text="Offset").pack(side=tk.LEFT)
        ov_mode = ttk.Combobox(controls, textvariable=self._overlay_mode_var, values=["Normal", "Offset Y", "Offset X"], state="readonly", width=10)
        ov_mode.pack(side=tk.LEFT, padx=(6, 6))
        ov_mode.bind("<<ComboboxSelected>>", lambda _e: self._on_overlay_mode_changed())
        ov_off = ttk.Entry(controls, textvariable=self._overlay_offset_var, width=8)
        ov_off.pack(side=tk.LEFT)
        ov_off.bind("<KeyRelease>", lambda _e: self._schedule_overlay_refresh())
        ov_off.bind("<Return>", lambda _e: self._on_overlay_mode_changed())
        ov_off.bind("<FocusOut>", lambda _e: self._on_overlay_mode_changed())
        try:
            self._overlay_offset_var.trace_add("write", lambda *_a: self._schedule_overlay_refresh())
            self._overlay_mode_var.trace_add("write", lambda *_a: self._schedule_overlay_refresh())
        except Exception:
            pass

        banner = ttk.Label(right, textvariable=self._banner_var, foreground="#0b5394", anchor="w")
        banner.grid(row=2, column=0, sticky="ew", pady=(2, 4))

        body = ttk.Frame(right)
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        builder = ttk.LabelFrame(body, text="Advanced options", padding=8)
        builder.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        builder.columnconfigure(0, weight=1)
        self._builder_panel = builder

        ttk.Label(builder, text="Group / Series column").grid(row=0, column=0, sticky="w", pady=(6, 0))
        self._group_var = tk.StringVar(value="(None)")
        self._group_cb = ttk.Combobox(builder, textvariable=self._group_var, state="readonly")
        self._group_cb.grid(row=1, column=0, sticky="ew")

        self._extra = ttk.Frame(builder)
        self._extra.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._extra.columnconfigure(0, weight=1)

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

        plot = ttk.Frame(body)
        plot.grid(row=0, column=1, sticky="nsew")
        plot.columnconfigure(0, weight=1)
        plot.rowconfigure(0, weight=1)
        plot.rowconfigure(1, weight=0)
        plot.rowconfigure(2, weight=0)

        self._fig = Figure(figsize=(10.5, 7.5), dpi=110)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot)
        self._canvas.draw()
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.grid(row=0, column=0, sticky="nsew")
        self._canvas_widget.bind("<Configure>", self._on_canvas_resize)
        try:
            self._toolbar = NavigationToolbar2Tk(self._canvas, plot, pack_toolbar=False)
            self._toolbar.update()
            self._toolbar.grid(row=1, column=0, sticky="ew")
        except Exception:
            self._toolbar = None

        self._coord_var = tk.StringVar(value="")
        ttk.Label(plot, textvariable=self._coord_var, anchor="w").grid(row=2, column=0, sticky="ew", pady=(2, 0))
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
            foreground="#0b5394",
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
            foreground="#0b5394",
            wraplength=380,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

        filter_var = tk.StringVar(value="")
        ttk.Entry(win, textvariable=filter_var).grid(row=1, column=0, sticky="ew", padx=12)

        listbox = tk.Listbox(win, selectmode="extended", height=14)
        listbox.grid(row=2, column=0, sticky="nsew", padx=12, pady=(6, 0))
        sb = ttk.Scrollbar(win, orient="vertical", command=listbox.yview)
        sb.grid(row=2, column=1, sticky="ns", pady=(6, 0))
        listbox.configure(yscrollcommand=sb.set)

        win.columnconfigure(0, weight=1)
        win.rowconfigure(2, weight=1)

        def _refresh() -> None:
            items = self._available_y_items(filter_var.get())
            listbox.delete(0, "end")
            for d in items:
                listbox.insert("end", d)
            selected = set(pd.y_cols or [])
            for idx, d in enumerate(items):
                col = self._y_display_to_col.get(d, d)
                if col in selected:
                    listbox.selection_set(idx)

        def _apply() -> None:
            selected: List[str] = []
            for i in listbox.curselection():
                disp = listbox.get(i)
                selected.append(self._y_display_to_col.get(disp, disp))
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
        _refresh()

        btns = ttk.Frame(win)
        btns.grid(row=3, column=0, sticky="e", padx=12, pady=(8, 10))
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
        self._ws.plot_defs[pid] = DataStudioPlotDef(plot_id=pid, dataset_id=dataset_id, plot_type=PLOT_TYPES[0])
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
        pd = DataStudioPlotDef(plot_id=pid, dataset_id=ds_id, plot_type=str(self._plot_type_var.get()))
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

        missing = [sid for sid in overlay_ids if sid not in self._plotted_ids]
        if missing:
            messagebox.showinfo("Overlay", "Plot each dataset first before overlay.")
            self._ws.overlay_ids = []
            self._refresh_workspace()
            return

        cfgs = [pd for pd in self._ws.plot_defs.values() if pd.dataset_id in overlay_ids]
        if len(cfgs) != len(overlay_ids):
            messagebox.showinfo("Overlay", "Overlay requires plotting each dataset first.")
            self._ws.overlay_ids = []
            self._refresh_workspace()
            return

        plot_type = str(cfgs[0].plot_type)
        if any(str(c.plot_type) != plot_type for c in cfgs if c is not None):
            messagebox.showinfo("Overlay", "Overlay requires the same plot type across datasets.")
            self._ws.overlay_ids = []
            self._refresh_workspace()
            return

        self._ws.overlay_ids = overlay_ids
        if self._ws.active_id not in self._ws.overlay_ids:
            self._ws.active_id = self._ws.overlay_ids[0]
        self._status_var.set(f"Overlay: {len(self._ws.overlay_ids)} dataset(s)")
        self._plot()

    def _clear_overlay(self) -> None:
        self._ws.overlay_ids = []
        self._refresh_workspace()
        self._auto_plot_for_selection()

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
        for pid, pd in self._ws.plot_defs.items():
            name = f"{self._plot_def_name(pd)}"
            active = "●" if pid == self._ws.active_plot_id else ""
            self._plot_tree.insert("", "end", iid=str(pid), values=(active, name))
        self._populate_columns()
        self._restore_config_for_active()

    def _selected_id(self, tree: ttk.Treeview) -> Optional[str]:
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
            try:
                df = self._load_df(ds)
            except Exception:
                df = pd.DataFrame()
            cols_map = column_type_map(df)
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
        self._x_cb["values"] = x_values
        if self._x_var.get() not in x_values:
            self._x_var.set("(Index)")

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
        self._bins_var.set(int(opts.get("hist_bins") or 20))
        self._roll_var.set(int(opts.get("rolling_window") or 5))

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

        # Preferred axes memory wins if valid
        pref = self._ws.preferred_axes_by_dataset.get(dataset_id)
        if pref:
            px, py = pref
            if px in cols:
                x_col = px
            if py and all(y in cols for y in py):
                y_cols = list(py)

        return x_col, y_cols

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
                self._ax.boxplot(data, labels=labels, showfliers=True)
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

        if meta.get("heatmap") is not None:
            hm = meta.get("heatmap")
            self._ax.clear()
            im = self._ax.imshow(hm["values"], aspect="auto")
            self._ax.set_xticks(range(len(hm["cols"])) )
            self._ax.set_xticklabels([str(c) for c in hm["cols"]], rotation=45, ha="right")
            self._ax.set_yticks(range(len(hm["rows"])) )
            self._ax.set_yticklabels([str(r) for r in hm["rows"]])
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
                    x_def, y_def = self._pick_default_axes(sid)
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
            defs = [pd for pd in self._ws.plot_defs.values() if pd.dataset_id in sid_list]
            if len(defs) != len(sid_list):
                raise ValueError("Overlay requires plotting each dataset first (store X/Y selections).")
            pt = str(defs[0].plot_type)
            if any(str(d.plot_type) != pt for d in defs if d is not None):
                raise ValueError("Overlay requires the same plot type across datasets.")

        pd = self._active_plot_def()
        xcol = None
        ycols: List[str] = []
        group_col = None
        plot_type = str(self._plot_type_var.get())
        opts: Dict[str, Any] = {}

        if pd is not None:
            xcol = pd.x_col
            ycols = list(pd.y_cols or [])
            plot_type = str(pd.plot_type or plot_type)
            opts = dict(pd.options or {})
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
        meta = {
            "title": "",
            "xlabel": (xcol if xcol and xcol != "(Index)" else "Index"),
            "ylabel": ", ".join(ycols),
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
            pv = pd.pivot_table(df, index=r, columns=c, values=v, aggfunc=str(opts.get("heatmap_agg") or self._heat_agg_var.get()))
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
