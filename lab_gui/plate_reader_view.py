from __future__ import annotations

import json
import traceback
import uuid
from pathlib import Path
from typing import Any, List, Optional

import tkinter as tk
from tkinter import simpledialog
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from lab_gui.plate_reader_io import coerce_numeric_matrix, preview_dataframe, read_plate_file
from lab_gui.plate_reader_model import (
    PlateReaderDataset,
    PlateReaderMICWizardConfig,
    PlateReaderMICWizardResult,
    _utc_now_iso,
)
from lab_gui.ui_widgets import ToolTip


class _DataPreviewWindow(tk.Toplevel):
    def __init__(self, parent: tk.Widget, *, max_rows_default: int = 120) -> None:
        super().__init__(parent)
        self._max_rows = int(max_rows_default)
        self._df: Optional[pd.DataFrame] = None

        self.title("Plate Reader Preview")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(2, weight=1)

        ttk.Label(top, text="Rows:").grid(row=0, column=0, sticky="w")
        self._rows_var = tk.IntVar(value=self._max_rows)
        ttk.Spinbox(top, from_=10, to=2000, width=6, textvariable=self._rows_var, command=self._refresh).grid(
            row=0, column=1, sticky="w", padx=(6, 12)
        )
        self._info_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self._info_var).grid(row=0, column=2, sticky="e")
        ttk.Button(top, text="Close", command=self._on_close).grid(row=0, column=3, sticky="e", padx=(12, 0))

        table = ttk.Frame(self, padding=(8, 0, 8, 8))
        table.grid(row=1, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)

        self._tree = ttk.Treeview(table, show="headings")
        self._tree.grid(row=0, column=0, sticky="nsew")
        ysb = ttk.Scrollbar(table, orient=tk.VERTICAL, command=self._tree.yview)
        ysb.grid(row=0, column=1, sticky="ns")
        xsb = ttk.Scrollbar(table, orient=tk.HORIZONTAL, command=self._tree.xview)
        xsb.grid(row=1, column=0, sticky="ew")
        self._tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

        try:
            sw = int(self.winfo_screenwidth())
            sh = int(self.winfo_screenheight())
            self.geometry(f"{max(900, int(sw * 0.62))}x{max(520, int(sh * 0.58))}")
        except Exception:
            pass

        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

    def _on_close(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    def set_df(self, df: Optional[pd.DataFrame], *, title_suffix: str = "") -> None:
        self._df = df
        if title_suffix:
            self.title(f"Plate Reader Preview — {title_suffix}")
        else:
            self.title("Plate Reader Preview")
        self._refresh()

    def _refresh(self) -> None:
        try:
            self._max_rows = int(self._rows_var.get() or 120)
        except Exception:
            self._max_rows = 120
        self._render(self._df)

    def _render(self, df: Optional[pd.DataFrame]) -> None:
        tree = self._tree
        try:
            tree.delete(*tree.get_children())
        except Exception:
            pass

        if df is None or df.empty:
            self._info_var.set("(no data)")
            tree["columns"] = ["(empty)"]
            tree.heading("(empty)", text="(empty)")
            tree.column("(empty)", width=300, stretch=True)
            return

        self._info_var.set(f"{int(df.shape[0])} rows × {int(df.shape[1])} cols")
        cols, rows = preview_dataframe(df, max_rows=self._max_rows)
        tree["columns"] = cols
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(80, min(240, 10 * len(str(c)))), stretch=True)
        for r in rows:
            try:
                tree.insert("", tk.END, values=r)
            except Exception:
                pass


class PlateReaderRunWizard(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        dataset: PlateReaderDataset,
        df: pd.DataFrame,
        on_apply: Any,
    ) -> None:
        super().__init__(parent)
        self._dataset = dataset
        self._df = df
        self._on_apply = on_apply

        self.title("Run Analysis")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._analysis_var = tk.StringVar(value="mic")

        self._stack = ttk.Frame(self, padding=10)
        self._stack.grid(row=0, column=0, sticky="nsew")
        self._stack.columnconfigure(0, weight=1)
        self._stack.rowconfigure(0, weight=1)

        self._step_select = ttk.Frame(self._stack)
        self._step_mic = ttk.Frame(self._stack)

        for f in (self._step_select, self._step_mic):
            f.grid(row=0, column=0, sticky="nsew")

        self._build_step_select()
        self._build_step_mic()

        self._show_step("select")

        try:
            sw = int(self.winfo_screenwidth())
            sh = int(self.winfo_screenheight())
            self.geometry(f"{max(980, int(sw * 0.72))}x{max(640, int(sh * 0.72))}")
        except Exception:
            pass

        try:
            self.transient(parent.winfo_toplevel())
        except Exception:
            pass

    def _show_step(self, which: str) -> None:
        try:
            self._step_select.tkraise() if which == "select" else self._step_mic.tkraise()
        except Exception:
            pass

    def _build_step_select(self) -> None:
        f = self._step_select
        f.columnconfigure(0, weight=1)

        ttk.Label(f, text="Choose analysis", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(f, text="Only MIC is implemented right now.").grid(row=1, column=0, sticky="w", pady=(4, 10))

        box = ttk.LabelFrame(f, text="Analysis", padding=10)
        box.grid(row=2, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)

        ttk.Radiobutton(box, text="MIC", variable=self._analysis_var, value="mic").grid(row=0, column=0, sticky="w")

        for i, name in enumerate(["Fluorescence", "UV/Absorbance", "Kinetics", "Custom"]):
            rb = ttk.Radiobutton(box, text=name + " (coming soon)", variable=self._analysis_var, value=name.lower())
            rb.state(["disabled"])
            rb.grid(row=i + 1, column=0, sticky="w")

        btns = ttk.Frame(f)
        btns.grid(row=3, column=0, sticky="e", pady=(14, 0))
        ttk.Button(btns, text="Cancel", command=self._cancel).grid(row=0, column=0)
        ttk.Button(btns, text="Continue", command=lambda: self._show_step("mic")).grid(row=0, column=1, padx=(8, 0))

    def _build_step_mic(self) -> None:
        f = self._step_mic
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=1)

        ttk.Label(f, text="MIC configuration", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            f,
            text="Select sample/control replicate rows, select concentration columns, and (optionally) remap tick labels.",
            wraplength=880,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))

        top = ttk.Frame(f)
        top.grid(row=2, column=0, sticky="nsew")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)
        top.rowconfigure(1, weight=1)

        self._use_header_var = tk.BooleanVar(value=True)
        if self._dataset.wizard_mic_config is not None:
            try:
                self._use_header_var.set(bool(self._dataset.wizard_mic_config.use_first_row_as_header))
            except Exception:
                pass
        ttk.Checkbutton(top, text="First row is headers", variable=self._use_header_var, command=self._reload_df).grid(
            row=0, column=0, sticky="w"
        )
        ToolTip.attach(top.winfo_children()[-1], "If unchecked, the first row is treated as data (no headers).")

        prev_box = ttk.LabelFrame(top, text="Preview (first rows)", padding=6)
        prev_box.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        prev_box.columnconfigure(0, weight=1)
        prev_box.rowconfigure(0, weight=1)
        self._prev_tree = ttk.Treeview(prev_box, show="headings", height=8)
        self._prev_tree.grid(row=0, column=0, sticky="nsew")
        ysb = ttk.Scrollbar(prev_box, orient=tk.VERTICAL, command=self._prev_tree.yview)
        ysb.grid(row=0, column=1, sticky="ns")
        self._prev_tree.configure(yscrollcommand=ysb.set)

        rows_box = ttk.LabelFrame(top, text="Rows", padding=6)
        rows_box.grid(row=1, column=1, sticky="nsew", padx=(0, 8))
        rows_box.columnconfigure(0, weight=1)
        rows_box.columnconfigure(1, weight=1)
        rows_box.rowconfigure(1, weight=1)

        ttk.Label(rows_box, text="Sample rows (replicates)").grid(row=0, column=0, sticky="w")
        ttk.Label(rows_box, text="Control rows (optional)").grid(row=0, column=1, sticky="w")

        self._sample_rows_lb = tk.Listbox(rows_box, selectmode=tk.EXTENDED, exportselection=False, height=10)
        self._control_rows_lb = tk.Listbox(rows_box, selectmode=tk.EXTENDED, exportselection=False, height=10)
        self._sample_rows_lb.grid(row=1, column=0, sticky="nsew")
        self._control_rows_lb.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        ToolTip.attach(self._sample_rows_lb, "Select the replicate rows for your sample.")
        ToolTip.attach(self._control_rows_lb, "Optional control replicate rows.")

        for lb, col in ((self._sample_rows_lb, 0), (self._control_rows_lb, 1)):
            sb = ttk.Scrollbar(rows_box, orient=tk.VERTICAL, command=lb.yview)
            sb.grid(row=1, column=col + 2, sticky="ns", padx=(4, 0))
            lb.configure(yscrollcommand=sb.set)

        cols_box = ttk.LabelFrame(top, text="Columns = concentrations", padding=6)
        cols_box.grid(row=1, column=2, sticky="nsew")
        cols_box.columnconfigure(0, weight=1)
        cols_box.rowconfigure(1, weight=1)

        ttk.Label(cols_box, text="Select concentration columns (in sheet order)").grid(row=0, column=0, sticky="w")
        self._cols_lb = tk.Listbox(cols_box, selectmode=tk.EXTENDED, exportselection=False, height=10)
        self._cols_lb.grid(row=1, column=0, sticky="nsew")
        sbc = ttk.Scrollbar(cols_box, orient=tk.VERTICAL, command=self._cols_lb.yview)
        sbc.grid(row=1, column=1, sticky="ns")
        self._cols_lb.configure(yscrollcommand=sbc.set)
        ToolTip.attach(self._cols_lb, "Choose the columns that contain the concentration series.")

        bot = ttk.Frame(f)
        bot.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        bot.columnconfigure(1, weight=1)
        bot.columnconfigure(3, weight=1)

        ttk.Label(bot, text="Tick labels:").grid(row=0, column=0, sticky="w")
        self._ticks_var = tk.StringVar(value="")
        self._ticks_entry = ttk.Entry(bot, textvariable=self._ticks_var, width=50)
        self._ticks_entry.grid(row=0, column=1, sticky="ew", padx=(6, 10))
        ToolTip.attach(self._ticks_entry, "Comma-separated labels for the selected columns (must match count).")

        # Default helper: powers-of-two labels (1,2,4,8,...)
        self._auto_ticks_var = tk.BooleanVar(value=True)
        if self._dataset.wizard_mic_config is not None:
            try:
                self._auto_ticks_var.set(bool(getattr(self._dataset.wizard_mic_config, "auto_tick_labels_power2", True)))
            except Exception:
                pass
        self._auto_ticks_cb = ttk.Checkbutton(
            bot,
            text="Default 1024,512,…,0",
            variable=self._auto_ticks_var,
            command=self._on_auto_ticks_toggle,
        )
        self._auto_ticks_cb.grid(row=0, column=2, sticky="w", padx=(0, 10))
        ToolTip.attach(self._auto_ticks_cb, "Auto-fill tick labels as a 2-fold dilution series ending with 0, based on selected columns.")

        ttk.Label(bot, text="Plot type:").grid(row=0, column=3, sticky="w")
        self._plot_type_var = tk.StringVar(value="bar")
        ttk.Combobox(bot, textvariable=self._plot_type_var, state="readonly", width=10, values=["bar", "line", "scatter"]).grid(
            row=0, column=4, sticky="w", padx=(6, 10)
        )

        ttk.Label(bot, text="Control style:").grid(row=0, column=5, sticky="w")
        self._control_style_var = tk.StringVar(value="bars")
        ttk.Combobox(bot, textvariable=self._control_style_var, state="readonly", width=10, values=["bars", "line"]).grid(
            row=0, column=6, sticky="w", padx=(6, 0)
        )

        titles = ttk.Frame(f)
        titles.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        titles.columnconfigure(1, weight=1)
        titles.columnconfigure(3, weight=1)
        titles.columnconfigure(5, weight=1)

        self._title_var = tk.StringVar(value="MIC")
        self._xlabel_var = tk.StringVar(value="Concentration (µM)")
        self._ylabel_var = tk.StringVar(value="OD 600nm")

        ttk.Label(titles, text="Title:").grid(row=0, column=0, sticky="w")
        ttk.Entry(titles, textvariable=self._title_var).grid(row=0, column=1, sticky="ew", padx=(6, 10))
        ttk.Label(titles, text="X label:").grid(row=0, column=2, sticky="w")
        ttk.Entry(titles, textvariable=self._xlabel_var).grid(row=0, column=3, sticky="ew", padx=(6, 10))
        ttk.Label(titles, text="Y label:").grid(row=0, column=4, sticky="w")
        ttk.Entry(titles, textvariable=self._ylabel_var).grid(row=0, column=5, sticky="ew", padx=(6, 0))

        btns = ttk.Frame(f)
        btns.grid(row=5, column=0, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Back", command=lambda: self._show_step("select")).grid(row=0, column=0)
        ttk.Button(btns, text="Cancel", command=self._cancel).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(btns, text="Apply", command=self._apply).grid(row=0, column=2, padx=(8, 0))

        self._load_from_dataset()
        self._render_preview_table()
        self._wire_tick_autofill()
        self._on_auto_ticks_toggle()

    def _wire_tick_autofill(self) -> None:
        # Update tick labels when column selection changes.
        try:
            self._cols_lb.bind("<<ListboxSelect>>", lambda _e: self._update_auto_ticks())
        except Exception:
            pass

    def _on_auto_ticks_toggle(self) -> None:
        # Enable/disable manual entry and update auto ticks immediately.
        try:
            if bool(self._auto_ticks_var.get()):
                self._ticks_entry.state(["disabled"])
            else:
                self._ticks_entry.state(["!disabled"])
        except Exception:
            pass
        self._update_auto_ticks()

    def _update_auto_ticks(self) -> None:
        if not bool(getattr(self, "_auto_ticks_var", tk.BooleanVar(value=False)).get()):
            return
        try:
            n = len(self._selected_columns())
        except Exception:
            n = 0
        if n <= 0:
            try:
                self._ticks_var.set("")
            except Exception:
                pass
            return
        # Label concentrations in descending order (highest -> lowest), with control (0) as the LAST column.
        # Example (n=12): 1024,512,...,2,1,0
        if n == 1:
            labels = ["0"]
        else:
            labels = [str(int(2 ** (n - 2 - i))) for i in range(n - 1)] + ["0"]
        try:
            self._ticks_var.set(",".join(labels))
        except Exception:
            pass

    def _cancel(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    def _reload_df(self) -> None:
        try:
            use_header = bool(self._use_header_var.get())
        except Exception:
            use_header = True

        # Prefer cached in-memory dataframes (avoid disk IO).
        try:
            if use_header:
                self._df = getattr(self._dataset, "df_header0", None)
            else:
                self._df = getattr(self._dataset, "df_header_none", None)
        except Exception:
            self._df = None

        # Backward-compatible fallback if caches are missing.
        if self._df is None:
            try:
                header = 0 if use_header else None
                self._df = read_plate_file(self._dataset.path, sheet_name=self._dataset.sheet_name, header_row=header)
            except Exception:
                msg = traceback.format_exc()
                messagebox.showerror("Run Analysis", "Failed to reload data.\n\n" + msg, parent=self)
                return
        self._render_preview_table()
        self._populate_row_and_col_lists()

    def _load_from_dataset(self) -> None:
        cfg = self._dataset.wizard_mic_config
        if cfg is None:
            self._populate_row_and_col_lists()
            return

        try:
            self._use_header_var.set(bool(cfg.use_first_row_as_header))
        except Exception:
            pass
        self._ticks_var.set(",".join(cfg.tick_labels or []))
        try:
            self._auto_ticks_var.set(bool(getattr(cfg, "auto_tick_labels_power2", True)))
        except Exception:
            pass
        self._plot_type_var.set(cfg.plot_type or "bar")
        self._control_style_var.set(cfg.control_style or "bars")
        self._title_var.set(cfg.title or "MIC")
        self._xlabel_var.set(cfg.x_label or "Concentration")
        self._ylabel_var.set(cfg.y_label or "OD 600nm")

        self._populate_row_and_col_lists()

        try:
            for i in cfg.sample_rows:
                self._sample_rows_lb.selection_set(int(i))
        except Exception:
            pass
        try:
            for i in cfg.control_rows:
                self._control_rows_lb.selection_set(int(i))
        except Exception:
            pass
        try:
            cols = [str(c) for c in self._df.columns]
            want = set([str(c) for c in (cfg.concentration_columns or [])])
            for idx, c in enumerate(cols):
                if c in want:
                    self._cols_lb.selection_set(idx)
        except Exception:
            pass

    def _populate_row_and_col_lists(self) -> None:
        for lb in (self._sample_rows_lb, self._control_rows_lb):
            try:
                lb.delete(0, tk.END)
            except Exception:
                pass
        try:
            n = int(self._df.shape[0])
        except Exception:
            n = 0
        for i in range(n):
            self._sample_rows_lb.insert(tk.END, f"Row {i+1}")
            self._control_rows_lb.insert(tk.END, f"Row {i+1}")

        try:
            self._cols_lb.delete(0, tk.END)
        except Exception:
            pass
        for c in [str(c) for c in self._df.columns]:
            self._cols_lb.insert(tk.END, c)

    def _render_preview_table(self) -> None:
        tree = self._prev_tree
        try:
            tree.delete(*tree.get_children())
        except Exception:
            pass

        if self._df is None or self._df.empty:
            tree["columns"] = ["(empty)"]
            tree.heading("(empty)", text="(empty)")
            tree.column("(empty)", width=300, stretch=True)
            return

        cols, rows = preview_dataframe(self._df, max_rows=12)
        tree["columns"] = cols
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(80, min(220, 9 * len(str(c)))), stretch=True)
        for r in rows:
            try:
                tree.insert("", tk.END, values=r)
            except Exception:
                pass

    def _selected_rows(self, lb: tk.Listbox) -> List[int]:
        try:
            return [int(i) for i in lb.curselection()]
        except Exception:
            return []

    def _selected_columns(self) -> List[str]:
        try:
            idxs = [int(i) for i in self._cols_lb.curselection()]
        except Exception:
            idxs = []
        cols = [str(c) for c in self._df.columns]
        return [cols[i] for i in idxs if 0 <= i < len(cols)]

    def _apply(self) -> None:
        sample_rows = self._selected_rows(self._sample_rows_lb)
        control_rows = self._selected_rows(self._control_rows_lb)
        conc_cols = self._selected_columns()

        if not sample_rows:
            messagebox.showerror("MIC", "Select at least one sample row.", parent=self)
            return
        if not conc_cols:
            messagebox.showerror("MIC", "Select at least one concentration column.", parent=self)
            return

        tick_text = (self._ticks_var.get() or "").strip()
        tick_labels: List[str] = []
        if tick_text:
            tick_labels = [t.strip() for t in tick_text.split(",") if t.strip()]
            if len(tick_labels) != len(conc_cols):
                messagebox.showerror(
                    "MIC",
                    f"Tick labels count ({len(tick_labels)}) must match selected columns ({len(conc_cols)}).",
                    parent=self,
                )
                return

        # Persist the auto-tick setting (for next wizard run) and auto-fill if needed.
        auto_power2 = bool(getattr(self, "_auto_ticks_var", tk.BooleanVar(value=False)).get())
        if auto_power2 and (not tick_labels) and conc_cols:
            if len(conc_cols) == 1:
                tick_labels = ["0"]
            else:
                n = int(len(conc_cols))
                tick_labels = [str(int(2 ** (n - 2 - i))) for i in range(n - 1)] + ["0"]

        sample_mat, sample_nan = coerce_numeric_matrix(self._df, row_indices=sample_rows, columns=conc_cols)
        if sample_mat.size == 0:
            messagebox.showerror("MIC", "Selected sample cells are empty.", parent=self)
            return
        sample_mean = np.nanmean(sample_mat, axis=0)
        sample_std = np.nanstd(sample_mat, axis=0, ddof=1) if sample_mat.shape[0] > 1 else np.zeros(sample_mat.shape[1])

        control_mean = None
        control_std = None
        if control_rows:
            ctrl_mat, _ctrl_nan = coerce_numeric_matrix(self._df, row_indices=control_rows, columns=conc_cols)
            if ctrl_mat.size:
                control_mean = np.nanmean(ctrl_mat, axis=0)
                control_std = np.nanstd(ctrl_mat, axis=0, ddof=1) if ctrl_mat.shape[0] > 1 else np.zeros(ctrl_mat.shape[1])

        if sample_nan > 0.35:
            messagebox.showwarning("MIC", f"Many selected sample cells are non-numeric (NaN ratio: {sample_nan:.0%}).", parent=self)

        prev_cfg = self._dataset.wizard_mic_config

        cfg = PlateReaderMICWizardConfig(
            use_first_row_as_header=bool(self._use_header_var.get()),
            sample_rows=list(sample_rows),
            control_rows=list(control_rows),
            concentration_columns=list(conc_cols),
            tick_labels=list(tick_labels),
            auto_tick_labels_power2=bool(auto_power2),
            title=str(self._title_var.get() or "MIC"),
            x_label=str(self._xlabel_var.get() or "Concentration"),
            y_label=str(self._ylabel_var.get() or "OD 600nm"),
            plot_type=str(self._plot_type_var.get() or "bar"),
            control_style=str(self._control_style_var.get() or "bars"),
        )

        # Important UX: re-running MIC should update the data points but keep the plot styling
        # the user edited in the Plot Editor.
        if prev_cfg is not None:
            for k in (
                "invert_x",
                "sample_color",
                "control_color",
                "line_width",
                "marker_size",
                "bar_width",
                "capsize",
                "errorbar_linewidth",
                "title_fontsize",
                "label_fontsize",
                "tick_fontsize",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "grid_on",
                "legend_on",
            ):
                try:
                    if hasattr(prev_cfg, k):
                        setattr(cfg, k, getattr(prev_cfg, k))
                except Exception:
                    pass

        result = PlateReaderMICWizardResult(
            concentrations=[float(i) for i in range(len(conc_cols))],
            x_tick_labels=(tick_labels if tick_labels else [str(c) for c in conc_cols]),
            sample_mean=[float(x) if np.isfinite(x) else float("nan") for x in sample_mean.tolist()],
            sample_std=[float(x) if np.isfinite(x) else float("nan") for x in sample_std.tolist()],
            control_mean=([float(x) if np.isfinite(x) else float("nan") for x in control_mean.tolist()] if control_mean is not None else None),
            control_std=([float(x) if np.isfinite(x) else float("nan") for x in control_std.tolist()] if control_std is not None else None),
        )

        self._on_apply(cfg, result)
        try:
            self.destroy()
        except Exception:
            pass


class PlateReaderView(ttk.Frame):
    """Plate Reader tab (wizard-driven).

    Main tab stays minimal: Load → Preview → Run wizard → plot.
    """

    def __init__(self, parent: tk.Widget, app: Any, workspace: Any) -> None:
        super().__init__(parent)
        self.app = app
        self.workspace = workspace

        self._dataset: Optional[PlateReaderDataset] = None
        self._df: Optional[pd.DataFrame] = None
        self._preview_win: Optional[_DataPreviewWindow] = None

        self._status_var = tk.StringVar(value="Ready")
        self._active_file_var = tk.StringVar(value="Active file: (none)")
        self._active_mic_var = tk.StringVar(value="MIC: —")

        self._build_ui()
        self._restore_from_workspace()

    def status_text(self) -> str:
        try:
            return str(self._status_var.get())
        except Exception:
            return ""

    def _build_ui(self) -> None:
        # Layout: Workspace panel (left) + plot area (right)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        ws = ttk.LabelFrame(self, text="Workspace", padding=8)
        ws.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ws.columnconfigure(0, weight=1)
        ws.rowconfigure(1, weight=1)

        ws_btns = ttk.Frame(ws)
        ws_btns.grid(row=0, column=0, sticky="ew")
        ws_btns.columnconfigure(5, weight=1)

        self._add_files_btn = ttk.Button(ws_btns, text="Add Files…", command=self._add_files)
        self._add_files_btn.grid(row=0, column=0, sticky="w")
        ToolTip.attach(self._add_files_btn, "Add one or more Excel/CSV plate-reader files to the workspace.")

        self._load_ws_btn = ttk.Button(ws_btns, text="Load…", command=self._load_plate_reader_workspace)
        self._load_ws_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ToolTip.attach(self._load_ws_btn, "Load a saved Plate Reader workspace JSON.")

        self._save_ws_btn = ttk.Button(ws_btns, text="Save…", command=self._save_plate_reader_workspace)
        self._save_ws_btn.grid(row=0, column=2, sticky="w", padx=(8, 0))
        ToolTip.attach(self._save_ws_btn, "Save the current Plate Reader workspace to JSON.")

        self._remove_btn = ttk.Button(ws_btns, text="Remove", command=self._remove_selected)
        self._remove_btn.grid(row=0, column=3, sticky="w", padx=(12, 0))

        self._rename_btn = ttk.Button(ws_btns, text="Rename…", command=self._rename_selected)
        self._rename_btn.grid(row=0, column=4, sticky="w", padx=(8, 0))

        self._clear_btn = ttk.Button(ws_btns, text="Clear", command=self._clear_workspace)
        self._clear_btn.grid(row=0, column=5, sticky="e")

        self._ws_tree = ttk.Treeview(ws, columns=("type", "shape", "mic"), show="tree headings", selectmode="browse", height=14)
        self._ws_tree.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self._ws_tree.heading("#0", text="Name")
        self._ws_tree.column("#0", width=220, stretch=True, anchor="w")
        self._ws_tree.heading("type", text="Type")
        self._ws_tree.heading("shape", text="Shape")
        self._ws_tree.heading("mic", text="MIC")
        self._ws_tree.column("type", width=52, stretch=False, anchor="w")
        self._ws_tree.column("shape", width=90, stretch=False, anchor="w")
        self._ws_tree.column("mic", width=40, stretch=False, anchor="center")
        sb = ttk.Scrollbar(ws, orient=tk.VERTICAL, command=self._ws_tree.yview)
        sb.grid(row=1, column=1, sticky="ns", pady=(8, 0), padx=(6, 0))
        self._ws_tree.configure(yscrollcommand=sb.set)
        try:
            self._ws_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        except Exception:
            pass

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        top = ttk.Frame(right, padding=(0, 0, 0, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(3, weight=1)

        self._preview_btn = ttk.Button(top, text="Preview", command=self._open_preview)
        self._preview_btn.grid(row=0, column=0, sticky="w")
        ToolTip.attach(self._preview_btn, "Open a read-only preview of the active dataset.")

        self._run_btn = ttk.Button(top, text="Run", command=self._open_wizard)
        self._run_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ToolTip.attach(self._run_btn, "Open the Analysis Wizard (MIC first) for the active dataset.")

        self._edit_plot_btn = ttk.Button(top, text="Edit Plot…", command=self._open_plot_editor)
        self._edit_plot_btn.grid(row=0, column=2, sticky="w", padx=(8, 0))
        ToolTip.attach(self._edit_plot_btn, "Edit plot styling for the active dataset.")

        info = ttk.Frame(top)
        info.grid(row=0, column=3, sticky="ew", padx=(12, 0))
        info.columnconfigure(0, weight=1)
        ttk.Label(info, textvariable=self._active_file_var).grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self._active_mic_var).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(top, textvariable=self._status_var).grid(row=0, column=4, sticky="e")

        plot = ttk.Frame(right)
        plot.grid(row=1, column=0, sticky="nsew")
        plot.columnconfigure(0, weight=1)
        plot.rowconfigure(0, weight=1)
        plot.rowconfigure(1, weight=0)

        self._fig = Figure(figsize=(9.0, 6.0), dpi=110)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        try:
            self._toolbar = NavigationToolbar2Tk(self._canvas, plot, pack_toolbar=False)
            self._toolbar.update()
            self._toolbar.grid(row=1, column=0, sticky="ew")
        except Exception:
            self._toolbar = None

        self._render_empty_plot()
        self._update_buttons()
        self._refresh_workspace_list()

    def _update_buttons(self) -> None:
        has = self._df is not None and self._dataset is not None
        try:
            self._preview_btn.configure(state=("normal" if has else "disabled"))
            self._run_btn.configure(state=("normal" if has else "disabled"))
            can_edit = bool(
                self._dataset is not None
                and getattr(self._dataset, "wizard_mic_result", None) is not None
                and getattr(self._dataset, "wizard_mic_config", None) is not None
            )
            self._edit_plot_btn.configure(state=("normal" if can_edit else "disabled"))

            has_any = bool((getattr(self.workspace, "plate_reader_datasets", None) or []))
            self._remove_btn.configure(state=("normal" if has_any else "disabled"))
            self._rename_btn.configure(state=("normal" if (self._dataset is not None) else "disabled"))
            self._clear_btn.configure(state=("normal" if has_any else "disabled"))
            self._save_ws_btn.configure(state=("normal" if has_any else "disabled"))
        except Exception:
            pass

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

    def _safe_dataclass_from_dict(self, cls: Any, data: Optional[dict]) -> Optional[Any]:
        if data is None:
            return None
        if not isinstance(data, dict):
            return None
        try:
            fields = getattr(cls, "__dataclass_fields__", {}) or {}
            allowed = set(fields.keys())
            clean = {k: v for k, v in data.items() if k in allowed}
            return cls(**clean)
        except Exception:
            return None

    def _save_plate_reader_workspace(self) -> None:
        try:
            dss = list(getattr(self.workspace, "plate_reader_datasets", []) or [])
        except Exception:
            dss = []
        if not dss:
            messagebox.showinfo("Plate Reader", "No Plate Reader files to save.", parent=self)
            return

        path = filedialog.asksaveasfilename(
            title="Save Plate Reader Workspace",
            defaultextension=".plate_reader.workspace.json",
            filetypes=[
                ("Plate Reader Workspace", "*.plate_reader.workspace.json"),
                ("JSON", "*.json"),
                ("All files", "*.*"),
            ],
            parent=self,
        )
        if not path:
            return

        payload = {
            "schema_version": 1,
            "kind": "plate_reader_workspace",
            "created_utc": _utc_now_iso(),
            "active_dataset_id": (None if getattr(self.workspace, "active_plate_reader_id", None) is None else str(getattr(self.workspace, "active_plate_reader_id"))),
            "datasets": [self._encode_dataset(d) for d in dss],
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            msg = traceback.format_exc()
            messagebox.showerror("Plate Reader", "Failed to save workspace.\n\n" + msg, parent=self)
            return

        self._status_var.set("Workspace saved")

    def _load_plate_reader_workspace(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Plate Reader Workspace",
            filetypes=[
                ("Plate Reader Workspace", "*.plate_reader.workspace.json"),
                ("JSON", "*.json"),
                ("All files", "*.*"),
            ],
            parent=self,
        )
        if not path:
            return

        # Optional confirmation if there is already data.
        try:
            has_any = bool((getattr(self.workspace, "plate_reader_datasets", None) or []))
        except Exception:
            has_any = False
        if has_any:
            if not messagebox.askyesno("Plate Reader", "Load workspace and replace current Plate Reader files?", parent=self):
                return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            msg = traceback.format_exc()
            messagebox.showerror("Plate Reader", "Failed to read workspace JSON.\n\n" + msg, parent=self)
            return

        if not isinstance(payload, dict):
            messagebox.showerror("Plate Reader", "Workspace JSON must be an object.", parent=self)
            return

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

            # Load df caches
            try:
                ds.df_header0 = read_plate_file(ds.path, sheet_name=ds.sheet_name, header_row=0)
                ds.df_header_none = read_plate_file(ds.path, sheet_name=ds.sheet_name, header_row=None)
            except Exception:
                failures.append(f"Failed to load: {p}")
                continue

            # Restore analysis (MIC)
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

        if not loaded:
            msg = "No datasets were loaded."
            if failures:
                msg += "\n\n" + "\n".join(failures[:12])
            messagebox.showerror("Plate Reader", msg, parent=self)
            return

        active_id = payload.get("active_dataset_id", None)
        active_id = (None if active_id is None else str(active_id))
        if active_id is not None and not any(str(getattr(d, "id", "")) == active_id for d in loaded):
            active_id = None

        try:
            self.workspace.plate_reader_datasets = loaded
            self.workspace.active_plate_reader_id = (active_id if active_id is not None else str(getattr(loaded[-1], "id", "")))
        except Exception:
            pass

        self._restore_from_workspace()
        if failures:
            messagebox.showwarning("Plate Reader", "Workspace loaded with some issues:\n\n" + "\n".join(failures[:12]), parent=self)
        self._status_var.set("Workspace loaded")

    def _open_plot_editor(self) -> None:
        if self._dataset is None or self._dataset.wizard_mic_config is None or self._dataset.wizard_mic_result is None:
            return

        PlateReaderPlotEditor(self, dataset=self._dataset, on_apply=self._on_plot_style_applied)

    def _on_plot_style_applied(self) -> None:
        self._render_from_dataset()
        self._status_var.set("Style updated")
        self._update_buttons()

    def _restore_from_workspace(self) -> None:
        try:
            dss = getattr(self.workspace, "plate_reader_datasets", []) or []
        except Exception:
            dss = []

        if not dss:
            self._dataset = None
            self._df = None
            self._set_active_labels()
            return

        # Choose active dataset
        active_id = getattr(self.workspace, "active_plate_reader_id", None)
        ds: Optional[PlateReaderDataset] = None
        if active_id:
            for d in dss:
                if str(getattr(d, "id", "")) == str(active_id):
                    ds = d
                    break
        if ds is None:
            ds = dss[-1]
            try:
                self.workspace.active_plate_reader_id = ds.id
            except Exception:
                pass

        self._dataset = ds

        # Use cached df if available; otherwise fall back to disk load.
        self._df = self._get_dataset_df(ds)
        self._set_active_labels()
        self._render_from_dataset()
        self._refresh_workspace_list()
        self._update_buttons()

    def _get_dataset_df(self, ds: PlateReaderDataset) -> Optional[pd.DataFrame]:
        try:
            df = ds.current_df()
            if df is not None:
                return df
        except Exception:
            pass

        # Backward-compatible fallback (older datasets without cached dfs)
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

    def _set_active_labels(self) -> None:
        ds = self._dataset
        if ds is None:
            self._active_file_var.set("Active file: (none)")
            self._active_mic_var.set("MIC: —")
            return
        nm = str(getattr(ds, "display_name", "") or getattr(ds, "name", "") or "")
        if not nm:
            try:
                nm = str(ds.path.name)
            except Exception:
                nm = "(dataset)"
        self._active_file_var.set(f"Active file: {nm}")
        has_mic = bool(getattr(ds, "wizard_mic_result", None) is not None)
        self._active_mic_var.set("MIC: configured" if has_mic else "MIC: not configured")

    def _refresh_workspace_list(self) -> None:
        tree = getattr(self, "_ws_tree", None)
        if tree is None:
            return
        try:
            tree.delete(*tree.get_children())
        except Exception:
            pass

        try:
            dss = getattr(self.workspace, "plate_reader_datasets", []) or []
        except Exception:
            dss = []

        active_id = getattr(self.workspace, "active_plate_reader_id", None)
        for ds in dss:
            try:
                pid = str(getattr(ds, "id", ""))
            except Exception:
                pid = ""

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
            ftype = (suf if suf else "file")

            shape = ""
            try:
                df0 = getattr(ds, "df_header0", None)
                if df0 is not None:
                    shape = f"{int(df0.shape[0])}×{int(df0.shape[1])}"
            except Exception:
                shape = ""

            mic = "✅" if getattr(ds, "wizard_mic_result", None) is not None else "❌"
            try:
                tree.insert("", tk.END, iid=pid, values=(ftype, shape, mic), text=name)
            except Exception:
                pass

        if active_id and tree.exists(str(active_id)):
            try:
                tree.selection_set(str(active_id))
            except Exception:
                pass

    def _on_tree_select(self, _evt: Any = None) -> None:
        tree = getattr(self, "_ws_tree", None)
        if tree is None:
            return
        try:
            sel = tree.selection()
        except Exception:
            sel = ()
        if not sel:
            return
        ds_id = str(sel[0])
        self._set_active_dataset(ds_id)

    def _set_active_dataset(self, dataset_id: str) -> None:
        try:
            dss = getattr(self.workspace, "plate_reader_datasets", []) or []
        except Exception:
            dss = []
        ds = next((d for d in dss if str(getattr(d, "id", "")) == str(dataset_id)), None)
        if ds is None:
            return

        try:
            self.workspace.active_plate_reader_id = ds.id
        except Exception:
            pass
        self._dataset = ds
        self._df = self._get_dataset_df(ds)
        self._set_active_labels()
        self._render_from_dataset()
        self._status_var.set("Active changed")
        self._update_buttons()

    def _unique_display_name(self, base: str) -> str:
        base = str(base or "Dataset").strip() or "Dataset"
        try:
            existing = {str(getattr(d, "display_name", "") or getattr(d, "name", "") or "") for d in (getattr(self.workspace, "plate_reader_datasets", []) or [])}
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

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Add Plate Reader files",
            filetypes=[
                ("Excel", "*.xlsx;*.xlsm;*.xls"),
                ("CSV", "*.csv"),
                ("All files", "*.*"),
            ],
            parent=self,
        )
        if not paths:
            return

        try:
            dss = list(getattr(self.workspace, "plate_reader_datasets", []) or [])
        except Exception:
            dss = []

        last_id: Optional[str] = None
        for path in list(paths):
            p = Path(path)
            ds = PlateReaderDataset(
                id=str(uuid.uuid4()),
                name=p.stem,
                display_name=self._unique_display_name(str(p.name)),
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
                messagebox.showerror("Plate Reader", f"Failed to load file:\n\n{p}\n\n{msg}", parent=self)
                continue

            dss.append(ds)
            last_id = ds.id

        try:
            self.workspace.plate_reader_datasets = dss
        except Exception:
            pass

        if last_id is not None:
            try:
                self.workspace.active_plate_reader_id = last_id
            except Exception:
                pass

        self._restore_from_workspace()
        self._status_var.set("Files added")

    def _remove_selected(self) -> None:
        tree = getattr(self, "_ws_tree", None)
        if tree is None:
            return
        try:
            sel = tree.selection()
        except Exception:
            sel = ()
        if not sel:
            return
        rm_id = str(sel[0])

        try:
            dss = list(getattr(self.workspace, "plate_reader_datasets", []) or [])
        except Exception:
            dss = []
        if not dss:
            return

        # Remove
        new_dss = [d for d in dss if str(getattr(d, "id", "")) != rm_id]
        try:
            self.workspace.plate_reader_datasets = new_dss
        except Exception:
            pass

        active_id = str(getattr(self.workspace, "active_plate_reader_id", "") or "")
        if active_id == rm_id:
            new_active = None
            if new_dss:
                new_active = str(getattr(new_dss[-1], "id", ""))
            try:
                self.workspace.active_plate_reader_id = new_active
            except Exception:
                pass

        self._restore_from_workspace()
        self._status_var.set("Removed")

    def _clear_workspace(self) -> None:
        try:
            dss = getattr(self.workspace, "plate_reader_datasets", []) or []
        except Exception:
            dss = []
        if not dss:
            return
        if not messagebox.askyesno("Plate Reader", "Clear Plate Reader workspace?", parent=self):
            return

        try:
            self.workspace.plate_reader_datasets = []
            self.workspace.active_plate_reader_id = None
        except Exception:
            pass
        self._dataset = None
        self._df = None
        self._set_active_labels()
        self._render_empty_plot()
        self._refresh_workspace_list()
        self._update_buttons()
        self._status_var.set("Cleared")

    def _rename_selected(self) -> None:
        ds = self._dataset
        if ds is None:
            return
        current = str(getattr(ds, "display_name", "") or getattr(ds, "name", "") or "")
        new = simpledialog.askstring("Rename", "Dataset name:", initialvalue=current, parent=self)
        if new is None:
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
        self._status_var.set("Renamed")

    # Legacy single-load entry removed; use workspace Add Files…

    def _open_preview(self) -> None:
        if self._df is None or self._dataset is None:
            messagebox.showinfo("Plate Reader", "Load or select a file first.", parent=self)
            return
        try:
            if self._preview_win is not None and self._preview_win.winfo_exists():
                self._preview_win.lift()
                self._preview_win.focus_force()
            else:
                self._preview_win = _DataPreviewWindow(self)
        except Exception:
            self._preview_win = None
            raise
        try:
            suffix = str(getattr(self._dataset, "display_name", "") or self._dataset.path.name)
            self._preview_win.set_df(self._df, title_suffix=suffix)
        except Exception:
            pass

    def _open_wizard(self) -> None:
        if self._dataset is None or self._df is None:
            messagebox.showinfo("Plate Reader", "Load or select a file first.", parent=self)
            return

        def on_apply(cfg: PlateReaderMICWizardConfig, result: PlateReaderMICWizardResult) -> None:
            self._dataset.wizard_last_analysis = "mic"
            self._dataset.wizard_mic_config = cfg
            self._dataset.wizard_mic_result = result
            self._dataset.header_row = 0 if cfg.use_first_row_as_header else None

            # Keep in-memory data only; no reload on apply.
            self._df = self._dataset.current_df()

            self._render_from_dataset()
            self._status_var.set("Applied")
            self._set_active_labels()
            self._refresh_workspace_list()
            self._update_buttons()

        PlateReaderRunWizard(self, dataset=self._dataset, df=self._df, on_apply=on_apply)

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


class PlateReaderPlotEditor(tk.Toplevel):
    def __init__(self, parent: tk.Widget, *, dataset: PlateReaderDataset, on_apply: Any) -> None:
        super().__init__(parent)
        self._dataset = dataset
        self._on_apply = on_apply

        self.title("Plate Reader — Edit Plot")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        cfg = self._dataset.wizard_mic_config
        if cfg is None:
            raise RuntimeError("No wizard config to edit")

        body = ttk.Frame(self, padding=10)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)

        # Labels
        ttk.Label(body, text="Labels", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")

        self._title_txt = tk.StringVar(value=str(getattr(cfg, "title", "MIC")))
        self._xlabel_txt = tk.StringVar(value=str(getattr(cfg, "x_label", "Concentration")))
        self._ylabel_txt = tk.StringVar(value=str(getattr(cfg, "y_label", "OD 600nm")))

        ttk.Label(body, text="Title").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(body, textvariable=self._title_txt).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(body, text="X label").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(body, textvariable=self._xlabel_txt).grid(row=2, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(body, text="Y label").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(body, textvariable=self._ylabel_txt).grid(row=3, column=1, sticky="ew", pady=(6, 0))

        self._invert_x = tk.BooleanVar(value=bool(getattr(cfg, "invert_x", False)))
        ttk.Checkbutton(body, text="Invert X axis", variable=self._invert_x).grid(row=4, column=0, sticky="w", pady=(8, 0))

        # Colors
        ttk.Separator(body).grid(row=5, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(body, text="Colors", font=("TkDefaultFont", 10, "bold")).grid(row=6, column=0, sticky="w")

        self._sample_color = tk.StringVar(value=str(getattr(cfg, "sample_color", "#1f77b4")))
        self._control_color = tk.StringVar(value=str(getattr(cfg, "control_color", "#ff7f0e")))

        colors_row = ttk.Frame(body)
        colors_row.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(colors_row, text="Sample color…", command=self._pick_sample_color).grid(row=0, column=0, sticky="w")
        ttk.Label(colors_row, textvariable=self._sample_color).grid(row=0, column=1, sticky="w", padx=(8, 16))
        ttk.Button(colors_row, text="Control color…", command=self._pick_control_color).grid(row=0, column=2, sticky="w")
        ttk.Label(colors_row, textvariable=self._control_color).grid(row=0, column=3, sticky="w", padx=(8, 0))

        # Thickness controls
        ttk.Separator(body).grid(row=8, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(body, text="Thickness", font=("TkDefaultFont", 10, "bold")).grid(row=9, column=0, sticky="w")

        self._line_w = tk.DoubleVar(value=float(getattr(cfg, "line_width", 1.6)))
        self._marker_s = tk.DoubleVar(value=float(getattr(cfg, "marker_size", 6.0)))
        self._bar_w = tk.DoubleVar(value=float(getattr(cfg, "bar_width", 0.65)))
        self._cap = tk.DoubleVar(value=float(getattr(cfg, "capsize", 3.0)))
        self._e_lw = tk.DoubleVar(value=float(getattr(cfg, "errorbar_linewidth", 1.0)))

        ttk.Label(body, text="Line width").grid(row=10, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(body, from_=0.1, to=20.0, increment=0.1, textvariable=self._line_w, width=8).grid(
            row=10, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Label(body, text="Marker size").grid(row=11, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(body, from_=1.0, to=40.0, increment=0.5, textvariable=self._marker_s, width=8).grid(
            row=11, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Label(body, text="Bar width").grid(row=12, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(body, from_=0.05, to=0.9, increment=0.05, textvariable=self._bar_w, width=8).grid(
            row=12, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Label(body, text="Errorbar cap size").grid(row=13, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(body, from_=0.0, to=20.0, increment=0.5, textvariable=self._cap, width=8).grid(
            row=13, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Label(body, text="Errorbar line width").grid(row=14, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(body, from_=0.1, to=10.0, increment=0.1, textvariable=self._e_lw, width=8).grid(
            row=14, column=1, sticky="w", pady=(6, 0)
        )

        # Fonts + toggles
        ttk.Separator(body).grid(row=15, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(body, text="Text & Layout", font=("TkDefaultFont", 10, "bold")).grid(row=16, column=0, sticky="w")

        self._title_fs = tk.IntVar(value=int(getattr(cfg, "title_fontsize", 12)))
        self._label_fs = tk.IntVar(value=int(getattr(cfg, "label_fontsize", 10)))
        self._tick_fs = tk.IntVar(value=int(getattr(cfg, "tick_fontsize", 9)))
        self._grid_on = tk.BooleanVar(value=bool(getattr(cfg, "grid_on", False)))
        self._legend_on = tk.BooleanVar(value=bool(getattr(cfg, "legend_on", True)))

        ttk.Label(body, text="Title font size").grid(row=17, column=0, sticky="w")
        ttk.Spinbox(body, from_=6, to=36, increment=1, textvariable=self._title_fs, width=8).grid(row=17, column=1, sticky="w")
        ttk.Label(body, text="Label font size").grid(row=18, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(body, from_=6, to=30, increment=1, textvariable=self._label_fs, width=8).grid(
            row=18, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Label(body, text="Tick font size").grid(row=19, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(body, from_=6, to=30, increment=1, textvariable=self._tick_fs, width=8).grid(
            row=19, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Checkbutton(body, text="Grid", variable=self._grid_on).grid(row=20, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(body, text="Legend", variable=self._legend_on).grid(row=20, column=1, sticky="w", pady=(8, 0))

        # Limits
        ttk.Separator(body).grid(row=21, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(body, text="Axis limits (optional)", font=("TkDefaultFont", 10, "bold")).grid(row=22, column=0, sticky="w")
        self._x_min = tk.StringVar(value=("" if getattr(cfg, "x_min", None) is None else str(getattr(cfg, "x_min"))))
        self._x_max = tk.StringVar(value=("" if getattr(cfg, "x_max", None) is None else str(getattr(cfg, "x_max"))))
        self._y_min = tk.StringVar(value=("" if getattr(cfg, "y_min", None) is None else str(getattr(cfg, "y_min"))))
        self._y_max = tk.StringVar(value=("" if getattr(cfg, "y_max", None) is None else str(getattr(cfg, "y_max"))))

        limits = ttk.Frame(body)
        limits.grid(row=23, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        for i in range(4):
            limits.columnconfigure(i, weight=1)
        ttk.Label(limits, text="X min").grid(row=0, column=0, sticky="w")
        ttk.Entry(limits, textvariable=self._x_min, width=10).grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Label(limits, text="X max").grid(row=0, column=1, sticky="w")
        ttk.Entry(limits, textvariable=self._x_max, width=10).grid(row=1, column=1, sticky="ew", padx=(0, 6))
        ttk.Label(limits, text="Y min").grid(row=0, column=2, sticky="w")
        ttk.Entry(limits, textvariable=self._y_min, width=10).grid(row=1, column=2, sticky="ew", padx=(0, 6))
        ttk.Label(limits, text="Y max").grid(row=0, column=3, sticky="w")
        ttk.Entry(limits, textvariable=self._y_max, width=10).grid(row=1, column=3, sticky="ew")

        btns = ttk.Frame(body)
        btns.grid(row=24, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(btns, text="Close", command=self._close).grid(row=0, column=0)
        ttk.Button(btns, text="Apply", command=self._apply).grid(row=0, column=1, padx=(8, 0))

        try:
            self.transient(parent.winfo_toplevel())
        except Exception:
            pass

    def _close(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

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
            cfg.title = str(self._title_txt.get() or "")
            cfg.x_label = str(self._xlabel_txt.get() or "")
            cfg.y_label = str(self._ylabel_txt.get() or "")
            cfg.invert_x = bool(self._invert_x.get())
            cfg.sample_color = str(self._sample_color.get() or "#1f77b4")
            cfg.control_color = str(self._control_color.get() or "#ff7f0e")

            cfg.line_width = float(self._line_w.get())
            cfg.marker_size = float(self._marker_s.get())
            cfg.bar_width = float(self._bar_w.get())
            cfg.capsize = float(self._cap.get())
            cfg.errorbar_linewidth = float(self._e_lw.get())

            cfg.title_fontsize = int(self._title_fs.get())
            cfg.label_fontsize = int(self._label_fs.get())
            cfg.tick_fontsize = int(self._tick_fs.get())
            cfg.grid_on = bool(self._grid_on.get())
            cfg.legend_on = bool(self._legend_on.get())

            cfg.x_min = self._parse_opt_float(self._x_min.get())
            cfg.x_max = self._parse_opt_float(self._x_max.get())
            cfg.y_min = self._parse_opt_float(self._y_min.get())
            cfg.y_max = self._parse_opt_float(self._y_max.get())
        except Exception as exc:
            messagebox.showerror("Edit Plot", f"Invalid value: {exc}", parent=self)
            return

        try:
            self._on_apply()
        except Exception:
            pass

    def _pick_sample_color(self) -> None:
        try:
            c = colorchooser.askcolor(title="Sample color", initialcolor=str(self._sample_color.get()), parent=self)
        except Exception:
            c = None
        if c and c[1]:
            self._sample_color.set(str(c[1]))

    def _pick_control_color(self) -> None:
        try:
            c = colorchooser.askcolor(title="Control color", initialcolor=str(self._control_color.get()), parent=self)
        except Exception:
            c = None
        if c and c[1]:
            self._control_color.set(str(c[1]))
