from __future__ import annotations

import colorsys
from typing import Any, Dict, List, Optional, cast

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import numpy as np
import pandas as pd

from matplotlib import cm, colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Keep this import aligned with Matplotlib stubs to avoid Pylance false-positives.
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure

from lab_gui.plot_card import PlotCard
from lab_gui.ui_theme import style_primary, style_secondary, style_success, style_danger
from lab_gui.ui_widgets import MatplotlibNavigator


class DataStudioExportEditor(tk.Toplevel):
    def __init__(self, parent: tk.Widget, *, payload: Dict[str, Any]) -> None:
        """Implement the `__init__` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        super().__init__(parent)
        self._payload = dict(payload or {})
        self._payload_base_series = list(self._payload.get("series", []))
        self.title("Export Editor — Data Studio")
        try:
            self.geometry("1300x850")
        except Exception:
            pass

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        top = ttk.LabelFrame(self, text="Export Actions", padding=8, style="Card.TLabelframe")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)
        top.columnconfigure(3, weight=1)
        ttk.Label(
            top,
            text="Finalize the exported Data Studio figure here while keeping the large preview front and center.",
            style="CardHint.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        controls_btn = ttk.Button(top, text="Controls…", command=self._open_controls)
        controls_btn.grid(row=1, column=0, sticky="ew")
        style_primary(controls_btn)
        save_btn = ttk.Button(top, text="Save As…", command=self._save_as)
        save_btn.grid(row=1, column=1, sticky="ew", padx=(8, 8))
        style_success(save_btn)
        export_btn = ttk.Button(top, text="Export Data…", command=self._export_data)
        export_btn.grid(row=1, column=2, sticky="ew")
        style_secondary(export_btn)
        close_btn = ttk.Button(top, text="Close", command=self.destroy)
        close_btn.grid(row=1, column=3, sticky="ew", padx=(8, 0))
        style_danger(close_btn)

        stage_hdr = ttk.Frame(self, style="Surface.TFrame", padding=(14, 12))
        stage_hdr.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        stage_hdr.columnconfigure(0, weight=1)
        ttk.Label(stage_hdr, text="Data Studio Export Stage", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            stage_hdr,
            text="Keep the finished chart visible, then open the control window only for detailed typography, color, and bounds changes.",
            style="CardHint.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(stage_hdr, text="EXPORT", style="CardStatus.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        plot_card = PlotCard(cast(tk.Widget, self), title="Data Studio Export", status_text="Preview", show_header=True)
        plot_card.grid(row=2, column=0, sticky="nsew")
        plot = plot_card.body
        plot.columnconfigure(0, weight=1)
        plot.rowconfigure(0, weight=1)
        plot.rowconfigure(1, weight=0)
        plot.rowconfigure(2, weight=0)

        self._fig = Figure(figsize=(12.5, 7.2), dpi=110)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        plot_card.register_canvas(self._canvas)
        try:
            self._toolbar = NavigationToolbar2Tk(self._canvas, plot, pack_toolbar=False)
            self._toolbar.update()
            self._toolbar.grid(row=1, column=0, sticky="ew")
        except Exception:
            self._toolbar = None

        self._coord_var = tk.StringVar(value="")
        coord_lbl = ttk.Label(plot, textvariable=self._coord_var, anchor="w")
        coord_lbl.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        try:
            coord_lbl.configure(style="Muted.TLabel", padding=(2, 6, 2, 0))
        except Exception:
            pass

        try:
            self._nav = MatplotlibNavigator(canvas=self._canvas, ax=self._ax, status_label=self._coord_var)
            self._nav.attach()
        except Exception:
            self._nav = None

        # Controls
        self.title_var = tk.StringVar(value=str(self._payload.get("title", "")))
        self.xlabel_var = tk.StringVar(value=str(self._payload.get("xlabel", "")))
        self.ylabel_var = tk.StringVar(value=str(self._payload.get("ylabel", "")))
        self.legend_on_var = tk.BooleanVar(value=True)
        self.grid_on_var = tk.BooleanVar(value=True)
        self.title_fs_var = tk.IntVar(value=12)
        self.label_fs_var = tk.IntVar(value=10)
        self.tick_fs_var = tk.IntVar(value=9)
        self.legend_fs_var = tk.IntVar(value=9)

        self.xmin_var = tk.StringVar(value="")
        self.xmax_var = tk.StringVar(value="")
        self.ymin_var = tk.StringVar(value="")
        self.ymax_var = tk.StringVar(value="")
        self.reverse_x_var = tk.BooleanVar(value=False)

        self.fig_w_var = tk.DoubleVar(value=float(self._fig.get_size_inches()[0]))
        self.fig_h_var = tk.DoubleVar(value=float(self._fig.get_size_inches()[1]))

        self.axes_facecolor_var = tk.StringVar(value="#ffffff")
        self.label_color_var = tk.StringVar(value="#111111")

        self.legend_text_color_var = tk.StringVar(value="#111111")
        self.legend_box_color_var = tk.StringVar(value="#ffffff")
        self.legend_frame_on_var = tk.BooleanVar(value=True)

        self._overlay_scheme_var = tk.StringVar(value="Manual (workspace)")
        self._overlay_single_hue_color = "#1f77b4"
        self._overlay_mode_var = tk.StringVar(value="Normal")
        self._overlay_offset_var = tk.DoubleVar(value=0.0)
        try:
            self._overlay_mode_var.set(str(self._payload.get("overlay_mode") or "Normal"))
            self._overlay_offset_var.set(float(self._payload.get("overlay_offset") or 0.0))
        except Exception:
            pass

        self._series_styles: Dict[str, Dict[str, Any]] = {}
        for s in list(self._payload.get("series", [])):
            sid = str(s.get("id"))
            self._series_styles[sid] = {
                "label": str(s.get("label", sid)),
                "color": str(s.get("color", "")),
            }

        self._controls_win: Optional[tk.Toplevel] = None
        self._replot()

    def _open_controls(self) -> None:
        """Open a file, view, or resource.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        if self._controls_win is not None and bool(self._controls_win.winfo_exists()):
            self._controls_win.deiconify()
            self._controls_win.lift()
            return

        win = tk.Toplevel(self)
        self._controls_win = win
        win.title("Export Controls — Data Studio")
        try:
            win.geometry("520x760")
        except Exception:
            pass

        outer = ttk.Frame(win, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        hdr = ttk.Frame(outer, style="ShellPanel.TFrame", padding=(14, 12))
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        hdr.columnconfigure(0, weight=1)
        ttk.Label(hdr, text="Export Controls", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            hdr,
            text="Detailed styling lives here so the export preview stays spacious and focused.",
            style="CardHint.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(hdr, text="DATA STUDIO", style="CardStatus.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        canvas = tk.Canvas(outer, highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        ysb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        ysb.grid(row=1, column=1, sticky="ns")
        canvas.configure(yscrollcommand=ysb.set)

        inner = ttk.Frame(canvas, padding=6)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_config(_evt=None):
            """Implement the `_on_inner_config` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        def _on_canvas_config(evt=None):
            """Implement the `_on_canvas_config` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            try:
                w = int(evt.width) if evt is not None else int(canvas.winfo_width())
                canvas.itemconfigure(inner_id, width=w)
            except Exception:
                pass

        try:
            inner.bind("<Configure>", _on_inner_config, add=True)
            canvas.bind("<Configure>", _on_canvas_config, add=True)
        except Exception:
            pass

        def _pick_one(var: tk.StringVar, title: str) -> None:
            """Implement the `_pick_one` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            try:
                c = colorchooser.askcolor(color=(var.get() or None), title=title, parent=win)[1]
                if c:
                    var.set(str(c))
            except Exception:
                return

        row = 0

        txt_group = ttk.Labelframe(inner, text="Text", padding=(8, 6))
        txt_group.grid(row=row, column=0, sticky="ew")
        txt_group.columnconfigure(1, weight=1)
        row += 1
        ttk.Label(txt_group, text="Title").grid(row=0, column=0, sticky="w")
        ttk.Entry(txt_group, textvariable=self.title_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(txt_group, text="X label").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(txt_group, textvariable=self.xlabel_var).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Label(txt_group, text="Y label").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(txt_group, textvariable=self.ylabel_var).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        fonts_group = ttk.Labelframe(inner, text="Fonts", padding=(8, 6))
        fonts_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        fonts_group.columnconfigure(1, weight=1)
        row += 1
        ttk.Label(fonts_group, text="Title font size").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(fonts_group, from_=6, to=48, textvariable=self.title_fs_var, width=6, command=self._replot).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Label(fonts_group, text="Axis label font size").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(fonts_group, from_=6, to=48, textvariable=self.label_fs_var, width=6, command=self._replot).grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(fonts_group, text="Tick font size").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(fonts_group, from_=6, to=48, textvariable=self.tick_fs_var, width=6, command=self._replot).grid(
            row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
        )

        leg_group = ttk.Labelframe(inner, text="Legend", padding=(8, 6))
        leg_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        leg_group.columnconfigure(1, weight=1)
        row += 1
        ttk.Checkbutton(leg_group, text="Show legend", variable=self.legend_on_var, command=self._replot).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(leg_group, text="Legend font size").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(leg_group, from_=6, to=48, textvariable=self.legend_fs_var, width=6, command=self._replot).grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(leg_group, text="Legend text color").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(leg_group, textvariable=self.legend_text_color_var).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Button(leg_group, text="Pick…", command=lambda: _pick_one(self.legend_text_color_var, "Legend text"))
        ttk.Label(leg_group, text="Legend box color").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(leg_group, textvariable=self.legend_box_color_var).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Button(leg_group, text="Pick…", command=lambda: _pick_one(self.legend_box_color_var, "Legend box"))
        ttk.Checkbutton(leg_group, text="Legend frame", variable=self.legend_frame_on_var, command=self._replot).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        lim_group = ttk.Labelframe(inner, text="Axis Limits (blank = keep)", padding=(8, 6))
        lim_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        lim_group.columnconfigure(1, weight=1)
        row += 1
        ttk.Label(lim_group, text="X min").grid(row=0, column=0, sticky="w")
        ttk.Entry(lim_group, textvariable=self.xmin_var, width=14).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(lim_group, text="X max").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(lim_group, textvariable=self.xmax_var, width=14).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Label(lim_group, text="Y min").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(lim_group, textvariable=self.ymin_var, width=14).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Label(lim_group, text="Y max").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(lim_group, textvariable=self.ymax_var, width=14).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Checkbutton(lim_group, text="Reverse X axis", variable=self.reverse_x_var, command=self._replot).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

        fig_group = ttk.Labelframe(inner, text="Figure Size", padding=(8, 6))
        fig_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        fig_group.columnconfigure(1, weight=1)
        row += 1
        ttk.Label(fig_group, text="Width (in)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(fig_group, from_=4, to=30, increment=0.5, textvariable=self.fig_w_var, width=8, command=self._replot).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Label(fig_group, text="Height (in)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(fig_group, from_=3, to=20, increment=0.5, textvariable=self.fig_h_var, width=8, command=self._replot).grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
        )

        col_group = ttk.Labelframe(inner, text="Global Colors", padding=(8, 6))
        col_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        col_group.columnconfigure(1, weight=1)
        row += 1
        ttk.Label(col_group, text="Axes background").grid(row=0, column=0, sticky="w")
        ttk.Entry(col_group, textvariable=self.axes_facecolor_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(col_group, text="Pick…", command=lambda: _pick_one(self.axes_facecolor_var, "Axes background"))
        ttk.Label(col_group, text="Label/tick color").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(col_group, textvariable=self.label_color_var).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Button(col_group, text="Pick…", command=lambda: _pick_one(self.label_color_var, "Label color"))
        ttk.Checkbutton(col_group, text="Grid", variable=self.grid_on_var, command=self._replot).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        ov_group = ttk.Labelframe(inner, text="Overlay Styling", padding=(8, 6))
        ov_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        ov_group.columnconfigure(1, weight=1)
        row += 1
        ttk.Label(ov_group, text="Color scheme").grid(row=0, column=0, sticky="w")
        ov_colors = ttk.Combobox(ov_group, textvariable=self._overlay_scheme_var, values=self._overlay_scheme_options(), state="readonly")
        ov_colors.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ov_colors.bind("<<ComboboxSelected>>", lambda _e: self._replot())
        ttk.Button(ov_group, text="Pick hue…", command=self._pick_overlay_single_hue_color).grid(row=1, column=1, sticky="e", pady=(6, 0))
        ttk.Label(ov_group, text="Overlay mode").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ov_mode = ttk.Combobox(ov_group, textvariable=self._overlay_mode_var, values=["Normal", "Offset Y", "Offset X"], state="readonly")
        ov_mode.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ov_mode.bind("<<ComboboxSelected>>", lambda _e: self._replot())
        ttk.Label(ov_group, text="Offset value").grid(row=3, column=0, sticky="w")
        ov_off = ttk.Entry(ov_group, textvariable=self._overlay_offset_var)
        ov_off.grid(row=3, column=1, sticky="ew", padx=(8, 0))
        ov_off.bind("<KeyRelease>", lambda _e: self._replot())

        ttk.Label(inner, text="Series (double-click to edit)").grid(row=row, column=0, sticky="w", pady=(10, 0))
        row += 1
        tree = ttk.Treeview(inner, columns=("label", "color"), show="headings", height=8)
        tree.heading("label", text="Label")
        tree.heading("color", text="Color")
        tree.column("label", width=260)
        tree.column("color", width=80)
        tree.grid(row=row, column=0, sticky="nsew", pady=(6, 0))
        inner.rowconfigure(row, weight=1)
        row += 1

        def _refresh_tree() -> None:
            """Refresh derived state or UI content.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            for it in list(tree.get_children("")):
                tree.delete(it)
            for sid, st in self._series_styles.items():
                tree.insert("", "end", iid=str(sid), values=(st.get("label", ""), st.get("color", "")))

        def _edit_cell(evt=None) -> None:
            """Implement the `_edit_cell` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            sel = tree.selection()
            sid = str(sel[0]) if sel else ""
            if not sid:
                return
            st = self._series_styles.get(sid)
            if st is None:
                return
            col = tree.identify_column(evt.x) if evt is not None else "#1"
            if col == "#1":
                new_label = simpledialog.askstring("Label", "Series label:", initialvalue=str(st.get("label", "")), parent=win)
                if new_label is None:
                    return
                st["label"] = str(new_label)
            elif col == "#2":
                c = colorchooser.askcolor(color=(st.get("color", "") or None), parent=win)[1]
                if c:
                    st["color"] = str(c)
            self._series_styles[sid] = st
            _refresh_tree()
            self._replot()

        tree.bind("<Double-1>", _edit_cell, add=True)
        _refresh_tree()

        btns = ttk.Frame(inner)
        btns.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        btns.columnconfigure(0, weight=1)
        ttk.Button(btns, text="Apply", command=self._replot).grid(row=0, column=0, sticky="w")
        ttk.Button(btns, text="Close", command=win.destroy).grid(row=0, column=1, sticky="e")

    def _parse_optional_float(self, raw: str) -> Optional[float]:
        """Parse raw input into structured values.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        raw = (raw or "").strip()
        if not raw:
            return None
        return float(raw)

    def _overlay_scheme_options(self) -> List[str]:
        """Implement the `_overlay_scheme_options` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        return [
            "Manual (workspace)",
            "Single hue…",
            "Viridis",
            "Plasma",
            "Magma",
            "Cividis",
            "Turbo",
            "Spectral",
            "Set1",
            "Set2",
            "Dark2",
            "Paired",
        ]

    def _pick_overlay_single_hue_color(self) -> None:
        """Implement the `_pick_overlay_single_hue_color` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        try:
            picked = colorchooser.askcolor(color=(self._overlay_single_hue_color or None), title="Pick overlay hue", parent=self)[1]
        except Exception:
            picked = None
        if not picked:
            return
        self._overlay_single_hue_color = str(picked)
        try:
            self._overlay_scheme_var.set("Single hue…")
        except Exception:
            pass
        self._replot()

    def _overlay_colors_for_scheme(self, scheme: str, n: int) -> List[str]:
        """Implement the `_overlay_colors_for_scheme` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        if n <= 0:
            return []
        scheme = str(scheme or "").strip()
        if scheme in ("", "Manual (workspace)"):
            return []
        if scheme == "Single hue…":
            try:
                base_rgb = mcolors.to_rgb(str(self._overlay_single_hue_color or "#1f77b4"))
            except Exception:
                base_rgb = (0.12, 0.47, 0.71)
            try:
                    h, _, s = colorsys.rgb_to_hls(*base_rgb)
            except Exception:
                    h, _, s = (0.58, 0.45, 0.65)
            lo = 0.22
            hi = 0.88
            vals = np.linspace(lo, hi, n)
            return [mcolors.to_hex(colorsys.hls_to_rgb(float(h), float(v), float(s))) for v in vals]
        try:
            cmap = cm.get_cmap(str(scheme).lower())
        except Exception:
            try:
                cmap = cm.get_cmap(str(scheme))
            except Exception:
                return []
        xs = np.linspace(0.06, 0.94, n)
        return [mcolors.to_hex(cmap(float(x))) for x in xs]

    def _replot(self) -> None:
        """Implement the `_replot` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        try:
            self._fig.set_size_inches(float(self.fig_w_var.get()), float(self.fig_h_var.get()), forward=True)
        except Exception:
            pass

        self._ax.clear()
        self._ax.set_title(self.title_var.get())
        self._ax.set_xlabel(self.xlabel_var.get())
        self._ax.set_ylabel(self.ylabel_var.get())

        try:
            self._ax.title.set_fontsize(int(self.title_fs_var.get()))
            self._ax.xaxis.label.set_fontsize(int(self.label_fs_var.get()))
            self._ax.yaxis.label.set_fontsize(int(self.label_fs_var.get()))
            self._ax.tick_params(axis="both", labelsize=int(self.tick_fs_var.get()))
        except Exception:
            pass

        try:
            lbl_color = str(self.label_color_var.get() or "")
            if lbl_color:
                self._ax.title.set_color(lbl_color)
                self._ax.xaxis.label.set_color(lbl_color)
                self._ax.yaxis.label.set_color(lbl_color)
                self._ax.tick_params(axis="both", colors=lbl_color)
        except Exception:
            pass

        try:
            self._ax.set_facecolor(str(self.axes_facecolor_var.get() or "#ffffff"))
        except Exception:
            pass

        series = [dict(s) for s in list(self._payload_base_series or [])]
        plot_type = str(self._payload.get("plot_type", "Line"))

        # Overlay color scheme (by dataset id)
        scheme = str(self._overlay_scheme_var.get() or "").strip()
        color_map: Dict[str, str] = {}
        if scheme not in ("", "Manual (workspace)"):
            ordered_ids: List[str] = []
            for s in series:
                sid = str(s.get("id", "")).split(":", 1)[0]
                if sid and sid not in ordered_ids:
                    ordered_ids.append(sid)
            colors = self._overlay_colors_for_scheme(scheme, len(ordered_ids))
            color_map = {sid: c for sid, c in zip(ordered_ids, colors)}

        # Overlay offset
        try:
            mode = str(self._overlay_mode_var.get() or "Normal")
            offset = float(self._overlay_offset_var.get())
        except Exception:
            mode = "Normal"
            offset = 0.0
        if mode != "Normal" and offset != 0.0:
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

        if plot_type in ("Box plot", "Violin plot"):
            data = [np.asarray(s.get("y", []), dtype=float) for s in series]
            labels = [str(self._series_styles.get(str(s.get("id")), {}).get("label", s.get("label", ""))) for s in series]
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
                sid = str(s.get("id"))
                st = self._series_styles.get(sid, {})
                label = str(st.get("label", s.get("label", sid)))
                color = str(st.get("color", "")).strip() or color_map.get(sid.split(":", 1)[0], None)
                y = np.asarray(s.get("y", []), dtype=float)
                self._ax.hist(y, bins=20, alpha=0.5, label=label, color=color)
        elif plot_type == "Heatmap":
            hm = self._payload.get("heatmap")
            if isinstance(hm, dict):
                im = self._ax.imshow(hm.get("values", []), aspect="auto")
                cols = list(hm.get("cols", []) or [])
                rows = list(hm.get("rows", []) or [])
                self._ax.set_xticks(range(len(cols)))
                self._ax.set_xticklabels([str(c) for c in cols], rotation=45, ha="right")
                self._ax.set_yticks(range(len(rows)))
                self._ax.set_yticklabels([str(r) for r in rows])
                self._fig.colorbar(im, ax=self._ax, fraction=0.046, pad=0.04)
        else:
            for s in series:
                sid = str(s.get("id"))
                st = self._series_styles.get(sid, {})
                label = str(st.get("label", s.get("label", sid)))
                color = str(st.get("color", s.get("color", ""))).strip() or color_map.get(sid.split(":", 1)[0], None)
                color = color or None
                x = np.asarray(s.get("x", []), dtype=float)
                y = np.asarray(s.get("y", []), dtype=float)
                kind = str(s.get("kind", "line"))
                if kind == "scatter":
                    self._ax.scatter(x, y, s=18, label=label, color=color)
                elif kind == "bar":
                    self._ax.bar(x, y, label=label, color=color)
                else:
                    self._ax.plot(x, y, label=label, color=color)

        if bool(self.grid_on_var.get()):
            self._ax.grid(True, alpha=0.3)
        if bool(self.legend_on_var.get()) and len(series) > 1:
            try:
                leg = self._ax.legend(loc="best", frameon=bool(self.legend_frame_on_var.get()))
                leg.set_title("")
                try:
                    leg.get_frame().set_facecolor(str(self.legend_box_color_var.get() or "#ffffff"))
                except Exception:
                    pass
                try:
                    for t in leg.get_texts():
                        t.set_color(str(self.legend_text_color_var.get() or "#111111"))
                        t.set_fontsize(int(self.legend_fs_var.get()))
                except Exception:
                    pass
            except Exception:
                pass

        try:
            xmin = self._parse_optional_float(self.xmin_var.get())
            xmax = self._parse_optional_float(self.xmax_var.get())
            ymin = self._parse_optional_float(self.ymin_var.get())
            ymax = self._parse_optional_float(self.ymax_var.get())
            if xmin is not None or xmax is not None:
                self._ax.set_xlim(left=xmin, right=xmax)
            if ymin is not None or ymax is not None:
                self._ax.set_ylim(bottom=ymin, top=ymax)
        except Exception:
            pass

        if bool(self.reverse_x_var.get()):
            try:
                self._ax.invert_xaxis()
            except Exception:
                pass

        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _save_as(self) -> None:
        """Save output/state to persistent storage.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        path = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        try:
            self._fig.savefig(path)
        except Exception as exc:
            messagebox.showerror("Save plot", f"Failed to save plot:\n{exc}", parent=self)

    def _export_data(self) -> None:
        """Export data in an external-friendly format.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        series = list(self._payload.get("series", []))
        if not series:
            return
        path = filedialog.asksaveasfilename(
            title="Export data",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return

        try:
            if str(path).lower().endswith(".csv"):
                df = pd.DataFrame()
                for s in series:
                    label = str(s.get("label", s.get("id")))
                    df[f"{label}_x"] = np.asarray(s.get("x", []), dtype=float)
                    df[f"{label}_y"] = np.asarray(s.get("y", []), dtype=float)
                df.to_csv(path, index=False)
            else:
                with pd.ExcelWriter(path) as w:
                    for s in series:
                        label = str(s.get("label", s.get("id")))
                        df = pd.DataFrame({"x": np.asarray(s.get("x", []), dtype=float), "y": np.asarray(s.get("y", []), dtype=float)})
                        df.to_excel(w, sheet_name=label[:30], index=False)
        except Exception as exc:
            messagebox.showerror("Export", f"Failed to export data:\n{exc}", parent=self)
