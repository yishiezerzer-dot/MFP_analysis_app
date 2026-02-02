from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from lab_gui.ui_widgets import ToolTip, MatplotlibNavigator


class ExportEditor(tk.Toplevel):
    def __init__(
        self,
        app: Any,
        *,
        kind: str,
        default_stem: str,
        tooltip_text: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(app)
        self.app = app
        self.kind = str(kind)
        self.default_stem = str(default_stem)
        self._tooltip_text = dict(tooltip_text or {})

        try:
            self._init_ui()
        except Exception:
            msg = traceback.format_exc()
            try:
                messagebox.showerror(
                    "Export Editor",
                    "Export Editor failed to open.\n\n"
                    "This usually means an unexpected exception occurred during window construction.\n\n"
                    + msg,
                    parent=app,
                )
            except Exception:
                pass
            try:
                self.destroy()
            except Exception:
                pass
            return

    def _tt(self, key: str) -> str:
        return str(self._tooltip_text.get(str(key), "") or "")

    def _init_ui(self) -> None:

        self._controls_win: Optional[tk.Toplevel] = None
        self._controls_scroll_canvas: Optional[tk.Canvas] = None

        self.title(f"Export Editor — {self.kind.upper()}")
        try:
            sw = int(self.winfo_screenwidth())
            sh = int(self.winfo_screenheight())
            self.geometry(f"{max(1100, int(sw * 0.92))}x{max(700, int(sh * 0.82))}")
        except Exception:
            self.geometry("1500x900")

        # Layout: keep the export window focused on the plot.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=6)
        top.grid(row=0, column=0, sticky="ew")
        controls_btn = ttk.Button(top, text="Controls…", command=self._open_controls_window)
        controls_btn.pack(side=tk.LEFT)
        saveas_btn = ttk.Button(top, text="Save As…", command=self._save_as)
        saveas_btn.pack(side=tk.LEFT, padx=(8, 0))
        close_btn = ttk.Button(top, text="Close", command=self._on_close_export)
        close_btn.pack(side=tk.RIGHT)

        ToolTip.attach(controls_btn, self._tt("exp_controls"))
        ToolTip.attach(saveas_btn, self._tt("exp_saveas"))
        ToolTip.attach(close_btn, self._tt("exp_close"))

        plot = ttk.Frame(self)
        plot.grid(row=1, column=0, sticky="nsew")
        plot.columnconfigure(0, weight=1)
        plot.rowconfigure(0, weight=1)
        plot.rowconfigure(1, weight=0)
        plot.rowconfigure(2, weight=0)

        self._fig = Figure(figsize=(14.0, 7.5), dpi=110)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._table_artist = None

        self._canvas = FigureCanvasTkAgg(self._fig, master=plot)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # IMPORTANT: Avoid mixing geometry managers in the same container.
        # The toolbar packs itself by default; we use grid here.
        try:
            self._toolbar = NavigationToolbar2Tk(self._canvas, plot, pack_toolbar=False)
            try:
                self._toolbar.update()
            except Exception:
                pass
            try:
                self._toolbar.grid(row=1, column=0, sticky="ew")
            except Exception:
                pass
        except Exception:
            self._toolbar = None

        self._coord_var = tk.StringVar(value="")
        self._coord_label = ttk.Label(plot, textvariable=self._coord_var, anchor="w")
        self._coord_label.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        try:
            self._mpl_nav = MatplotlibNavigator(
                canvas=self._canvas,
                ax=self._ax,
                status_label=self._coord_var,
            )
            self._mpl_nav.attach()
        except Exception:
            self._mpl_nav = None

        # Editable controls
        self.title_var = tk.StringVar(value="")
        self.xlabel_var = tk.StringVar(value="")
        self.ylabel_var = tk.StringVar(value="")
        self.title_fs_var = tk.IntVar(value=int(self.app.title_fontsize_var.get()))
        self.label_fs_var = tk.IntVar(value=int(self.app.label_fontsize_var.get()))
        self.tick_fs_var = tk.IntVar(value=int(self.app.tick_fontsize_var.get()))
        self.ann_fs_var = tk.IntVar(value=max(6, int(self.app.tick_fontsize_var.get()) - 1))

        self.xmin_var = tk.StringVar(value="")
        self.xmax_var = tk.StringVar(value="")
        self.ymin_var = tk.StringVar(value="")
        self.ymax_var = tk.StringVar(value="")

        self.fig_w_var = tk.DoubleVar(value=float(self._fig.get_size_inches()[0]))
        self.fig_h_var = tk.DoubleVar(value=float(self._fig.get_size_inches()[1]))

        self.number_labels_var = tk.BooleanVar(value=False)

        self._annotations: List[Any] = []
        self._ann_original_text: Dict[int, str] = {}
        self._active_ann: Optional[Any] = None

        # Table + overrides (for editing the table text)
        self._num_to_ann: Dict[int, Any] = {}
        self._table_rt_override: Dict[int, str] = {}

        # Table placement (axes coordinates)
        self.tbl_x_var = tk.DoubleVar(value=0.56)
        self.tbl_y_var = tk.DoubleVar(value=0.56)
        self.tbl_w_var = tk.DoubleVar(value=0.43)
        self.tbl_h_var = tk.DoubleVar(value=0.43)

        # Colors
        self.plot_color_var = tk.StringVar(value="#1f77b4")
        self.label_color_var = tk.StringVar(value="#111111")
        self.axes_facecolor_var = tk.StringVar(value="#ffffff")
        self.table_facecolor_var = tk.StringVar(value="#ffffff")
        self.table_text_color_var = tk.StringVar(value="#111111")

        # Legend (overlay)
        self.legend_on_var = tk.BooleanVar(value=True)
        self.legend_fs_var = tk.IntVar(value=max(6, int(self.app.tick_fontsize_var.get()) - 1))
        self.legend_text_color_var = tk.StringVar(value="#111111")
        self.legend_box_color_var = tk.StringVar(value="#ffffff")
        self.legend_frame_on_var = tk.BooleanVar(value=True)
        self._legend_artist: Any = None
        self._legend_handles: List[Any] = []
        self._legend_labels: List[str] = []
        self._legend_handle_by_sid: Dict[str, Any] = {}
        self._legend_entries: List[Tuple[str, str]] = []
        self._legend_label_override: Dict[str, str] = {}

        self._preserve_plot_colors = False
        try:
            if self.kind in ("tic", "uv", "spectrum") and hasattr(self.app, "_is_overlay_active") and bool(self.app._is_overlay_active()):
                self._preserve_plot_colors = True
                self.plot_color_var.set("")
        except Exception:
            self._preserve_plot_colors = False

        self._install_color_traces()
        self._live_style_job: Optional[str] = None
        self._install_live_style_traces()

        try:
            self._build_initial_plot()
        except Exception as exc:
            # Avoid a blank export editor window if plot construction fails.
            try:
                self._ax.clear()
                self._ax.text(
                    0.5,
                    0.5,
                    f"Failed to build export plot:\n\n{exc}",
                    ha="center",
                    va="center",
                    transform=self._ax.transAxes,
                )
            except Exception:
                pass
            try:
                messagebox.showerror("Export Editor", f"Failed to build export plot:\n\n{exc}", parent=self)
            except Exception:
                pass

        # Force an initial render (helps prevent a blank-looking window on some TkAgg setups).
        try:
            self._canvas.draw()
        except Exception:
            pass
        try:
            self.after(0, self._canvas.draw_idle)
        except Exception:
            pass

        self._cid_press = self._canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_motion = self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_release = self._canvas.mpl_connect("button_release_event", self._on_release)

        try:
            # Non-modal: avoids Windows focus issues when minimized.
            self.transient(self.app)
        except Exception:
            pass

        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close_export)
        except Exception:
            pass

        # Open the controls window by default (so all options remain accessible).
        try:
            self.after(0, self._open_controls_window)
        except Exception:
            pass

    def _on_close_export(self) -> None:
        try:
            if self._controls_win is not None and bool(self._controls_win.winfo_exists()):
                self._controls_win.destroy()
        except Exception:
            pass
        try:
            tk.Toplevel.destroy(self)
        except Exception:
            try:
                self.destroy()
            except Exception:
                pass

    def _on_controls_closed(self) -> None:
        self._controls_win = None
        self._controls_scroll_canvas = None

    def _open_controls_window(self) -> None:
        # Reuse if already open.
        if self._controls_win is not None:
            try:
                if bool(self._controls_win.winfo_exists()):
                    self._controls_win.deiconify()
                    self._controls_win.lift()
                    try:
                        self._controls_win.focus_force()
                    except Exception:
                        pass
                    return
            except Exception:
                pass

        win = tk.Toplevel(self)
        self._controls_win = win
        win.title(f"Export Controls — {self.kind.upper()}")
        try:
            win.geometry("520x900")
        except Exception:
            pass

        # Place controls next to the export plot window (best-effort).
        try:
            self.update_idletasks()
            win.update_idletasks()
            sx = int(self.winfo_rootx())
            sy = int(self.winfo_rooty())
            sw = int(self.winfo_width())
            x = sx + sw + 8
            y = sy
            screen_w = int(win.winfo_screenwidth())
            screen_h = int(win.winfo_screenheight())
            w = 520
            h = 900
            if x + w > screen_w - 10:
                x = max(10, screen_w - w - 10)
            if y + h > screen_h - 60:
                y = max(10, screen_h - h - 60)
            win.geometry(f"{w}x{h}+{int(x)}+{int(y)}")
        except Exception:
            pass
        try:
            win.transient(self)
        except Exception:
            pass
        try:
            win.protocol("WM_DELETE_WINDOW", lambda: (self._on_controls_closed(), win.destroy()))
        except Exception:
            pass

        outer = ttk.Frame(win, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        # Scrollable area
        canvas = tk.Canvas(outer, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        self._controls_scroll_canvas = canvas
        ysb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        ysb.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=ysb.set)

        inner = ttk.Frame(canvas, padding=6)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_config(_evt=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        def _on_canvas_config(evt=None):
            try:
                w = int(evt.width) if evt is not None else int(canvas.winfo_width())
                canvas.itemconfigure(inner_id, width=w)
            except Exception:
                pass

        try:
            inner.bind("<Configure>", _on_inner_config, add=True)
        except Exception:
            pass
        try:
            canvas.bind("<Configure>", _on_canvas_config, add=True)
        except Exception:
            pass

        # --- Controls ---
        row = 0

        # Titles/labels
        ttk.Label(inner, text="Title").grid(row=row, column=0, sticky="w")
        ent_title = ttk.Entry(inner, textvariable=self.title_var)
        ent_title.grid(row=row, column=1, sticky="ew", padx=(8, 0))
        row += 1

        ttk.Label(inner, text="X label").grid(row=row, column=0, sticky="w")
        ent_xlab = ttk.Entry(inner, textvariable=self.xlabel_var)
        ent_xlab.grid(row=row, column=1, sticky="ew", padx=(8, 0))
        row += 1

        ttk.Label(inner, text="Y label").grid(row=row, column=0, sticky="w")
        ent_ylab = ttk.Entry(inner, textvariable=self.ylabel_var)
        ent_ylab.grid(row=row, column=1, sticky="ew", padx=(8, 0))
        row += 1

        ttk.Separator(inner).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Font sizes
        def _add_slider(
            parent: tk.Widget,
            *,
            variable: Union[tk.IntVar, tk.DoubleVar],
            from_: float,
            to: float,
            step: Optional[float] = None,
            fmt: Optional[str] = None,
        ) -> ttk.Scale:
            holder = ttk.Frame(parent)
            holder.columnconfigure(0, weight=1)

            scale_var = tk.DoubleVar(value=float(variable.get()))
            lbl = ttk.Label(holder, text="")

            def _format_value(v: float) -> str:
                if fmt:
                    try:
                        return fmt.format(v)
                    except Exception:
                        pass
                if isinstance(variable, tk.IntVar):
                    return str(int(round(v)))
                return f"{float(v):.2f}"

            def _apply(v: Any = None) -> None:
                try:
                    fv = float(v if v is not None else scale_var.get())
                except Exception:
                    return
                if step is not None and step > 0:
                    try:
                        fv = round(fv / float(step)) * float(step)
                    except Exception:
                        pass
                if isinstance(variable, tk.IntVar):
                    iv = int(round(fv))
                    try:
                        variable.set(iv)
                    except Exception:
                        return
                    try:
                        scale_var.set(float(iv))
                    except Exception:
                        pass
                    try:
                        lbl.configure(text=str(iv))
                    except Exception:
                        pass
                    return

                try:
                    variable.set(float(fv))
                except Exception:
                    return
                try:
                    lbl.configure(text=_format_value(float(variable.get())))
                except Exception:
                    try:
                        lbl.configure(text=_format_value(fv))
                    except Exception:
                        pass

            s = ttk.Scale(holder, from_=float(from_), to=float(to), variable=scale_var, command=_apply)
            s.grid(row=0, column=0, sticky="ew")
            lbl.grid(row=0, column=1, sticky="e", padx=(8, 0))
            _apply(scale_var.get())

            holder.grid(row=row, column=1, sticky="ew", padx=(8, 0))
            return s

        ttk.Label(inner, text="Title font size").grid(row=row, column=0, sticky="w")
        s_title_fs = _add_slider(inner, variable=self.title_fs_var, from_=6, to=48, step=1)
        row += 1

        ttk.Label(inner, text="Axis label font size").grid(row=row, column=0, sticky="w")
        s_label_fs = _add_slider(inner, variable=self.label_fs_var, from_=6, to=48, step=1)
        row += 1

        ttk.Label(inner, text="Tick font size").grid(row=row, column=0, sticky="w")
        s_tick_fs = _add_slider(inner, variable=self.tick_fs_var, from_=6, to=48, step=1)
        row += 1

        ttk.Label(inner, text="Annotation font size").grid(row=row, column=0, sticky="w")
        s_ann_fs = _add_slider(inner, variable=self.ann_fs_var, from_=6, to=48, step=1)
        row += 1

        ttk.Separator(inner).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Axis limits
        ttk.Label(inner, text="X min").grid(row=row, column=0, sticky="w")
        xmn = ttk.Entry(inner, textvariable=self.xmin_var, width=14)
        xmn.grid(row=row, column=1, sticky="w", padx=(8, 0))
        row += 1

        ttk.Label(inner, text="X max").grid(row=row, column=0, sticky="w")
        xmx = ttk.Entry(inner, textvariable=self.xmax_var, width=14)
        xmx.grid(row=row, column=1, sticky="w", padx=(8, 0))
        row += 1

        ttk.Label(inner, text="Y min").grid(row=row, column=0, sticky="w")
        ymn = ttk.Entry(inner, textvariable=self.ymin_var, width=14)
        ymn.grid(row=row, column=1, sticky="w", padx=(8, 0))
        row += 1

        ttk.Label(inner, text="Y max").grid(row=row, column=0, sticky="w")
        ymx = ttk.Entry(inner, textvariable=self.ymax_var, width=14)
        ymx.grid(row=row, column=1, sticky="w", padx=(8, 0))
        row += 1

        ttk.Separator(inner).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Figure size
        ttk.Label(inner, text="Figure width (in)").grid(row=row, column=0, sticky="w")
        s_fig_w = _add_slider(inner, variable=self.fig_w_var, from_=4.0, to=30.0, step=0.1, fmt="{:.1f}")
        row += 1

        ttk.Label(inner, text="Figure height (in)").grid(row=row, column=0, sticky="w")
        s_fig_h = _add_slider(inner, variable=self.fig_h_var, from_=3.0, to=20.0, step=0.1, fmt="{:.1f}")
        row += 1

        ttk.Separator(inner).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Legend (overlay only)
        leg_group = ttk.Labelframe(inner, text="Legend (overlay)", padding=(8, 6))
        leg_group.grid(row=row, column=0, columnspan=2, sticky="ew")
        leg_group.columnconfigure(1, weight=1)
        row += 1

        def _pick_legend_color(var: tk.StringVar, title: str) -> None:
            try:
                c = colorchooser.askcolor(color=(var.get() or None), title=title, parent=win)[1]
                if c:
                    var.set(str(c))
            except Exception:
                return

        ttk.Checkbutton(leg_group, text="Show legend", variable=self.legend_on_var, command=self._apply_style_and_limits).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(leg_group, text="Legend font size").grid(row=1, column=0, sticky="w", pady=(6, 0))
        s_leg_fs = _add_slider(leg_group, variable=self.legend_fs_var, from_=6, to=36, step=1)

        ttk.Checkbutton(leg_group, text="Show legend box", variable=self.legend_frame_on_var, command=self._apply_style_and_limits).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

        ttk.Label(leg_group, text="Legend text color").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ent_leg_txt = ttk.Entry(leg_group, textvariable=self.legend_text_color_var)
        ent_leg_txt.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        btn_leg_txt = ttk.Button(leg_group, text="Pick…", command=lambda: _pick_legend_color(self.legend_text_color_var, "Legend text"))
        btn_leg_txt.grid(row=3, column=2, sticky="e", padx=(8, 0), pady=(6, 0))

        ttk.Label(leg_group, text="Legend box color").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ent_leg_bg = ttk.Entry(leg_group, textvariable=self.legend_box_color_var)
        ent_leg_bg.grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        btn_leg_bg = ttk.Button(leg_group, text="Pick…", command=lambda: _pick_legend_color(self.legend_box_color_var, "Legend box"))
        btn_leg_bg.grid(row=4, column=2, sticky="e", padx=(8, 0), pady=(6, 0))

        ttk.Separator(inner).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Legend labels (overlay only)
        leg_labels_group = ttk.Labelframe(inner, text="Legend labels (overlay)", padding=(8, 6))
        leg_labels_group.grid(row=row, column=0, columnspan=2, sticky="nsew")
        leg_labels_group.columnconfigure(0, weight=1)
        row += 1

        leg_tree = ttk.Treeview(leg_labels_group, columns=("label",), show="headings", height=5, selectmode="browse")
        leg_tree.heading("label", text="Label")
        leg_tree.column("label", width=320, stretch=True)
        leg_tree.grid(row=0, column=0, sticky="nsew")
        leg_sb = ttk.Scrollbar(leg_labels_group, orient="vertical", command=leg_tree.yview)
        leg_sb.grid(row=0, column=1, sticky="ns")
        leg_tree.configure(yscrollcommand=leg_sb.set)
        self._legend_tree = leg_tree  # type: ignore[attr-defined]

        def _refresh_legend_tree() -> None:
            tv = getattr(self, "_legend_tree", None)
            if tv is None:
                return
            try:
                for it in list(tv.get_children("")):
                    tv.delete(it)
            except Exception:
                pass
            for sid, label in list(self._legend_entries):
                try:
                    tv.insert("", "end", iid=str(sid), values=(str(label),))
                except Exception:
                    continue

        def _edit_legend_label(evt=None) -> None:
            tv = getattr(self, "_legend_tree", None)
            if tv is None:
                return
            try:
                sel = tv.selection()
                sid = str(sel[0]) if sel else ""
            except Exception:
                sid = ""
            if not sid:
                return
            cur = ""
            for s, lbl in self._legend_entries:
                if str(s) == str(sid):
                    cur = str(lbl)
                    break
            new_label = simpledialog.askstring("Legend label", "Label:", initialvalue=cur, parent=win)
            if new_label is None:
                return
            new_label = str(new_label).strip()
            self._legend_label_override[str(sid)] = new_label
            try:
                h = self._legend_handle_by_sid.get(str(sid))
                if h is not None and hasattr(h, "set_label"):
                    h.set_label(str(new_label))
            except Exception:
                pass
            # Update cached labels/entries
            self._legend_entries = [(s, (new_label if str(s) == str(sid) else lbl)) for s, lbl in self._legend_entries]
            self._legend_labels = [str(getattr(h, "get_label")()) for h in list(self._legend_handles) if h is not None]
            self._apply_legend()
            try:
                self._canvas.draw_idle()
            except Exception:
                pass
            _refresh_legend_tree()

        try:
            leg_tree.bind("<Double-1>", _edit_legend_label, add=True)
        except Exception:
            pass

        _refresh_legend_tree()

        ttk.Separator(inner).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Numbering/table
        num_cb = ttk.Checkbutton(
            inner,
            text="Number labels + show table",
            variable=self.number_labels_var,
            command=lambda: self._apply_numbering(redraw_only=False),
        )
        num_cb.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        # Table placement + editing
        tbl_group = ttk.Labelframe(inner, text="Table", padding=(8, 6))
        tbl_group.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tbl_group.columnconfigure(1, weight=1)
        row += 1

        def _nudge_table(_evt=None):
            self._apply_numbering(redraw_only=True)
            try:
                self._canvas.draw_idle()
            except Exception:
                pass

        ttk.Label(tbl_group, text="X").grid(row=0, column=0, sticky="w")
        s_tbl_x = ttk.Scale(tbl_group, from_=0.0, to=1.0, variable=self.tbl_x_var, command=lambda _v=None: _nudge_table())
        s_tbl_x.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(tbl_group, text="Y").grid(row=1, column=0, sticky="w")
        s_tbl_y = ttk.Scale(tbl_group, from_=0.0, to=1.0, variable=self.tbl_y_var, command=lambda _v=None: _nudge_table())
        s_tbl_y.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(tbl_group, text="W").grid(row=2, column=0, sticky="w")
        s_tbl_w = ttk.Scale(tbl_group, from_=0.10, to=1.0, variable=self.tbl_w_var, command=lambda _v=None: _nudge_table())
        s_tbl_w.grid(row=2, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(tbl_group, text="H").grid(row=3, column=0, sticky="w")
        s_tbl_h = ttk.Scale(tbl_group, from_=0.10, to=1.0, variable=self.tbl_h_var, command=lambda _v=None: _nudge_table())
        s_tbl_h.grid(row=3, column=1, sticky="ew", padx=(8, 0))

        # Editable table rows (Label / RT)
        rows_group = ttk.Labelframe(inner, text="Table rows (double-click to edit)", padding=(8, 6))
        rows_group.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        rows_group.columnconfigure(0, weight=1)
        rows_group.rowconfigure(0, weight=1)
        row += 1

        tv = ttk.Treeview(rows_group, columns=("num", "label", "rt"), show="headings", height=8)
        tv.heading("num", text="#")
        tv.heading("label", text="Label")
        tv.heading("rt", text="RT")
        tv.column("num", width=44, anchor="w", stretch=False)
        tv.column("label", width=280, anchor="w", stretch=True)
        tv.column("rt", width=110, anchor="w", stretch=False)
        tv.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(rows_group, orient="vertical", command=tv.yview)
        sb.grid(row=0, column=1, sticky="ns")
        tv.configure(yscrollcommand=sb.set)
        self._tbl_tree = tv

        def _edit_table_cell(evt=None):
            try:
                row_id = tv.identify_row(evt.y)
                col_id = tv.identify_column(evt.x)
                if not row_id or col_id not in ("#2", "#3"):
                    return
                vals = tv.item(row_id, "values")
                if not vals:
                    return
                n = int(vals[0])
            except Exception:
                return

            if col_id == "#3":
                current = str(self._table_rt_override.get(n) or vals[2] or "")
                new_rt = simpledialog.askstring("Edit RT", f"RT for #{n}:", initialvalue=current, parent=win)
                if new_rt is None:
                    return
                self._table_rt_override[int(n)] = str(new_rt).strip()
                self._apply_numbering(redraw_only=True)
                try:
                    self._canvas.draw_idle()
                except Exception:
                    pass
                self._refresh_table_tree()
                return

            # Label column
            ann = self._num_to_ann.get(int(n))
            if ann is None:
                return
            current = str(self._ann_original_text.get(id(ann), vals[1] or ""))
            new_label = simpledialog.askstring("Edit label", f"Label for #{n}:", initialvalue=current, parent=win)
            if new_label is None:
                return
            self._ann_original_text[id(ann)] = str(new_label)
            self._apply_numbering(redraw_only=True)
            try:
                self._canvas.draw_idle()
            except Exception:
                pass
            self._refresh_table_tree()

        try:
            tv.bind("<Double-1>", _edit_table_cell, add=True)
        except Exception:
            pass

        try:
            self._refresh_table_tree()
        except Exception:
            pass

        ttk.Separator(inner).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Colors (live)
        colors_group = ttk.Labelframe(inner, text="Colors (live)", padding=(8, 6))
        colors_group.grid(row=row, column=0, columnspan=2, sticky="ew")
        colors_group.columnconfigure(1, weight=1)
        row += 1

        def _pick_one(var: tk.StringVar, title: str) -> None:
            try:
                c = colorchooser.askcolor(color=(var.get() or None), title=title, parent=win)[1]
                if c:
                    var.set(str(c))
            except Exception:
                return

        ttk.Label(colors_group, text="Plot").grid(row=0, column=0, sticky="w")
        ent_plot_c = ttk.Entry(colors_group, textvariable=self.plot_color_var)
        ent_plot_c.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        btn_plot_c = ttk.Button(colors_group, text="Pick…", command=lambda: _pick_one(self.plot_color_var, "Plot color"))
        btn_plot_c.grid(row=0, column=2, sticky="e", padx=(8, 0))

        ttk.Label(colors_group, text="Labels").grid(row=1, column=0, sticky="w")
        ent_label_c = ttk.Entry(colors_group, textvariable=self.label_color_var)
        ent_label_c.grid(row=1, column=1, sticky="ew", padx=(8, 0))
        btn_label_c = ttk.Button(colors_group, text="Pick…", command=lambda: _pick_one(self.label_color_var, "Label color"))
        btn_label_c.grid(row=1, column=2, sticky="e", padx=(8, 0))

        ttk.Label(colors_group, text="Axes bg").grid(row=2, column=0, sticky="w")
        ent_axes_bg = ttk.Entry(colors_group, textvariable=self.axes_facecolor_var)
        ent_axes_bg.grid(row=2, column=1, sticky="ew", padx=(8, 0))
        btn_axes_bg = ttk.Button(colors_group, text="Pick…", command=lambda: _pick_one(self.axes_facecolor_var, "Axes background"))
        btn_axes_bg.grid(row=2, column=2, sticky="e", padx=(8, 0))

        ttk.Label(colors_group, text="Table bg").grid(row=3, column=0, sticky="w")
        ent_tbl_bg = ttk.Entry(colors_group, textvariable=self.table_facecolor_var)
        ent_tbl_bg.grid(row=3, column=1, sticky="ew", padx=(8, 0))
        btn_tbl_bg = ttk.Button(colors_group, text="Pick…", command=lambda: _pick_one(self.table_facecolor_var, "Table background"))
        btn_tbl_bg.grid(row=3, column=2, sticky="e", padx=(8, 0))

        ttk.Label(colors_group, text="Table text").grid(row=4, column=0, sticky="w")
        ent_tbl_txt = ttk.Entry(colors_group, textvariable=self.table_text_color_var)
        ent_tbl_txt.grid(row=4, column=1, sticky="ew", padx=(8, 0))
        btn_tbl_txt = ttk.Button(colors_group, text="Pick…", command=lambda: _pick_one(self.table_text_color_var, "Table text"))
        btn_tbl_txt.grid(row=4, column=2, sticky="e", padx=(8, 0))

        ttk.Separator(inner).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Buttons
        btns = ttk.Frame(inner)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        btns.columnconfigure(0, weight=1)

        apply_btn = ttk.Button(btns, text="Apply", command=self._apply_style_and_limits)
        apply_btn.pack(side=tk.LEFT)
        close_btn = ttk.Button(btns, text="Close", command=lambda: (self._on_controls_closed(), win.destroy()))
        close_btn.pack(side=tk.RIGHT)

        inner.columnconfigure(1, weight=1)

        # Tooltips (best-effort)
        try:
            ToolTip.attach(ent_title, self._tt("exp_title"))
            ToolTip.attach(ent_xlab, self._tt("exp_xlabel"))
            ToolTip.attach(ent_ylab, self._tt("exp_ylabel"))
            ToolTip.attach(apply_btn, self._tt("exp_apply"))
            ToolTip.attach(close_btn, self._tt("exp_close"))
            ToolTip.attach(s_title_fs, "Title font size (export-only).")
            ToolTip.attach(s_label_fs, "Axis label font size (export-only).")
            ToolTip.attach(s_tick_fs, "Tick label font size (export-only).")
            ToolTip.attach(s_ann_fs, "Annotation font size (export-only).")
            ToolTip.attach(s_fig_w, "Figure width in inches (export-only).")
            ToolTip.attach(s_fig_h, "Figure height in inches (export-only).")
            ToolTip.attach(s_leg_fs, "Legend font size (overlay export-only).")
            ToolTip.attach(ent_leg_txt, "Legend text color (overlay export-only).")
            ToolTip.attach(ent_leg_bg, "Legend box color (overlay export-only).")
            ToolTip.attach(num_cb, "Replace labels with numbers and show a table (export-only).")
            ToolTip.attach(s_tbl_x, "Table X position (axes coords, export-only).")
            ToolTip.attach(s_tbl_y, "Table Y position (axes coords, export-only).")
            ToolTip.attach(s_tbl_w, "Table width (axes coords, export-only).")
            ToolTip.attach(s_tbl_h, "Table height (axes coords, export-only).")
            ToolTip.attach(ent_plot_c, "Line/trace color (live).")
            ToolTip.attach(ent_label_c, "Annotation/label color (live).")
            ToolTip.attach(ent_axes_bg, "Axes facecolor (live).")
            ToolTip.attach(ent_tbl_bg, "Table background (live).")
            ToolTip.attach(ent_tbl_txt, "Table text color (live).")
        except Exception:
            pass

    def _install_color_traces(self) -> None:
        if bool(getattr(self, "_color_traces_installed", False)):
            return
        self._color_traces_installed = True

        def _cb(*_args) -> None:
            try:
                self._apply_colors()
            except Exception:
                pass

        for var in (
            self.plot_color_var,
            self.label_color_var,
            self.axes_facecolor_var,
            self.table_facecolor_var,
            self.table_text_color_var,
            self.legend_text_color_var,
            self.legend_box_color_var,
        ):
            try:
                var.trace_add("write", _cb)
            except Exception:
                try:
                    var.trace("w", _cb)
                except Exception:
                    pass

    def _pick_colors(self) -> None:
        # Minimal color picker: reuse existing vars, apply immediately.
        try:
            c = colorchooser.askcolor(title="Pick plot color", parent=self)[1]
            if c:
                self.plot_color_var.set(str(c))
        except Exception:
            pass
        try:
            c = colorchooser.askcolor(title="Pick label color", parent=self)[1]
            if c:
                self.label_color_var.set(str(c))
        except Exception:
            pass
        try:
            c = colorchooser.askcolor(title="Pick axes background color", parent=self)[1]
            if c:
                self.axes_facecolor_var.set(str(c))
        except Exception:
            pass
        try:
            c = colorchooser.askcolor(title="Pick table background color", parent=self)[1]
            if c:
                self.table_facecolor_var.set(str(c))
        except Exception:
            pass
        try:
            c = colorchooser.askcolor(title="Pick table text color", parent=self)[1]
            if c:
                self.table_text_color_var.set(str(c))
        except Exception:
            pass
        self._apply_colors()

    def _refresh_table_tree(self) -> None:
        tv = getattr(self, "_tbl_tree", None)
        if tv is None:
            return
        try:
            for it in tv.get_children():
                tv.delete(it)
        except Exception:
            return

        if not bool(self.number_labels_var.get()):
            return

        # Build from current numbering order
        nums = sorted(self._num_to_ann.keys())
        for n in nums:
            ann = self._num_to_ann.get(int(n))
            label = ""
            if ann is not None:
                label = str(self._ann_original_text.get(id(ann), ""))
            rt = self._table_rt_override.get(int(n))
            if not rt:
                rt = self._label_rt_for_number(int(n))
            tv.insert("", "end", values=(str(n), label, rt))

    def _label_rt_for_number(self, n: int) -> str:
        ann = self._num_to_ann.get(int(n))
        if ann is None:
            return ""
        if self.kind in ("tic", "uv"):
            try:
                return f"{float(ann.xy[0]):.4f}"
            except Exception:
                return ""
        meta = self.app._current_spectrum_meta
        if meta is None:
            return ""
        return f"{float(meta.rt_min):.4f}"

    def _apply_colors(self) -> None:
        plot_c = (self.plot_color_var.get() or "").strip() or None
        label_c = (self.label_color_var.get() or "").strip() or None
        bg_c = (self.axes_facecolor_var.get() or "").strip() or None
        leg_txt = (self.legend_text_color_var.get() or "").strip() or None
        leg_bg = (self.legend_box_color_var.get() or "").strip() or None

        try:
            if bg_c:
                self._ax.set_facecolor(bg_c)
        except Exception:
            pass

        # Plot artists
        if plot_c:
            try:
                for ln in list(getattr(self._ax, "lines", [])):
                    ln.set_color(plot_c)
            except Exception:
                pass
            try:
                for coll in list(getattr(self._ax, "collections", [])):
                    if hasattr(coll, "set_color"):
                        coll.set_color(plot_c)
            except Exception:
                pass

        # Labels
        if label_c:
            for ann in list(self._annotations):
                try:
                    ann.set_color(label_c)
                except Exception:
                    pass
                try:
                    arr = ann.arrow_patch
                    if arr is not None and hasattr(arr, "set_color"):
                        arr.set_color(label_c)
                except Exception:
                    pass

        # Table (if present)
        self._apply_numbering(redraw_only=True)

        # Legend colors (if present)
        try:
            if self._legend_artist is not None:
                if leg_txt:
                    for txt in list(self._legend_artist.get_texts()):
                        try:
                            txt.set_color(leg_txt)
                        except Exception:
                            pass
                if leg_bg:
                    try:
                        frame = self._legend_artist.get_frame()
                        if frame is not None:
                            frame.set_facecolor(leg_bg)
                    except Exception:
                        pass
        except Exception:
            pass
        self._canvas.draw_idle()

    def _parse_optional_float(self, raw: str) -> Optional[float]:
        raw = (raw or "").strip()
        if not raw:
            return None
        return float(raw)

    def _clear_annotations(self) -> None:
        for ann in list(self._annotations):
            try:
                ann.remove()
            except Exception:
                pass
        self._annotations = []
        self._ann_original_text = {}

    def _add_annotation(self, text: str, *, xy: Tuple[float, float], xytext: Tuple[float, float]) -> Any:
        ann = self._ax.annotate(
            str(text),
            xy=(float(xy[0]), float(xy[1])),
            xytext=(float(xytext[0]), float(xytext[1])),
            textcoords="data",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=int(self.ann_fs_var.get()),
            arrowprops={"arrowstyle": "-", "lw": 0.9},
            clip_on=True,
        )
        try:
            ann.set_picker(True)
        except Exception:
            pass
        self._annotations.append(ann)
        self._ann_original_text[id(ann)] = str(text)
        return ann

    def _build_initial_plot(self) -> None:
        # NOTE: This relies on App-provided attributes/methods; kept identical to original behavior.
        self._ax.clear()
        if self._table_artist is not None:
            try:
                self._table_artist.remove()
            except Exception:
                pass
        self._table_artist = None
        self._clear_annotations()
        if self._legend_artist is not None:
            try:
                self._legend_artist.remove()
            except Exception:
                pass
        self._legend_artist = None
        self._legend_handles = []
        self._legend_labels = []
        self._legend_handle_by_sid = {}
        self._legend_entries = []

        if self.kind == "tic":
            if hasattr(self.app, "_is_overlay_active") and bool(self.app._is_overlay_active()):
                base_title = (self.app.tic_title_var.get() or "TIC (MS1)").strip()
                pol = str(getattr(self.app, "polarity_var").get())
                mode = str(getattr(self.app, "_overlay_mode_var").get() or "Stacked")
                self.title_var.set(f"{base_title} — overlay ({mode}) | polarity: {pol}")
                self.xlabel_var.set(self.app.tic_xlabel_var.get())
                self.ylabel_var.set(self.app.tic_ylabel_var.get())

                ids = list(self.app._overlay_dataset_ids())
                name_map = self.app._overlay_display_names(ids)
                max_global = 0.0
                per_max: Dict[str, float] = {}
                for sid in ids:
                    _meta, rts, tics = self.app._overlay_meta_for_session(str(sid), pol)
                    if tics is None or tics.size == 0:
                        per_max[str(sid)] = 0.0
                        continue
                    m = float(np.max(tics)) if tics.size else 0.0
                    per_max[str(sid)] = m
                    max_global = max(max_global, m)

                if not any((per_max.get(str(sid), 0.0) or 0.0) > 0 for sid in ids):
                    self._ax.text(0.5, 0.5, "No TIC data loaded", ha="center", va="center", transform=self._ax.transAxes)
                else:
                    offset_step = 0.12 * max_global if max_global > 0 else 1.0
                    for i, sid in enumerate(ids):
                        _meta, rts, tics = self.app._overlay_meta_for_session(str(sid), pol)
                        if rts is None or tics is None or rts.size == 0 or tics.size == 0:
                            continue
                        y = np.asarray(tics, dtype=float)
                        if mode in ("Normalized", "Percent of max"):
                            denom = float(per_max.get(str(sid), 0.0) or 0.0)
                            if denom > 0:
                                y = y / denom
                            if mode == "Percent of max":
                                y = y * 100.0
                        elif mode == "Offset":
                            y = y + float(i) * float(offset_step)
                        col = self.app._ensure_overlay_color(str(sid))
                        default_label = str(name_map.get(str(sid), str(sid)))
                        label = str(self._legend_label_override.get(str(sid), default_label))
                        try:
                            (ln,) = self._ax.plot(rts, y, linewidth=1, color=col, alpha=0.85, label=label)
                            self._legend_handles.append(ln)
                            self._legend_labels.append(label)
                            if str(sid) not in self._legend_handle_by_sid:
                                self._legend_handle_by_sid[str(sid)] = ln
                                self._legend_entries.append((str(sid), label))
                        except Exception:
                            self._ax.plot(rts, y, linewidth=1, color=col, alpha=0.85)
            else:
                self.title_var.set((self.app.tic_title_var.get() or "TIC (MS1)").strip())
                self.xlabel_var.set(self.app.tic_xlabel_var.get())
                self.ylabel_var.set(self.app.tic_ylabel_var.get())
                rts = self.app._filtered_rts
                tics = self.app._filtered_tics
                if rts is None or tics is None or rts.size == 0:
                    self._ax.text(0.5, 0.5, "No TIC data loaded", ha="center", va="center", transform=self._ax.transAxes)
                else:
                    self._ax.plot(rts, tics, linewidth=1)

        elif self.kind == "uv":
            if hasattr(self.app, "_is_overlay_active") and bool(self.app._is_overlay_active()) and bool(self.app._overlay_show_uv_var.get()):
                base_title = (self.app.uv_title_var.get() or "UV chromatogram").strip()
                self.title_var.set(f"{base_title} — overlay")
                self.xlabel_var.set(self.app.uv_xlabel_var.get())
                self.ylabel_var.set(self.app.uv_ylabel_var.get())

                ids = list(self.app._overlay_dataset_ids())
                name_map = self.app._overlay_display_names(ids)
                any_uv = False
                for sid in ids:
                    sess = self.app._sessions.get(str(sid))
                    if sess is None:
                        continue
                    uv_id = getattr(sess, "linked_uv_id", None)
                    if not uv_id or str(uv_id) not in self.app._uv_sessions:
                        continue
                    uv_sess = self.app._uv_sessions[str(uv_id)]
                    x = np.asarray(uv_sess.rt_min, dtype=float)
                    y = np.asarray(uv_sess.signal, dtype=float)
                    if x.size == 0 or y.size == 0:
                        continue
                    any_uv = True
                    col = self.app._ensure_overlay_color(str(sid))
                    default_label = str(name_map.get(str(sid), str(sid)))
                    label = str(self._legend_label_override.get(str(sid), default_label))
                    try:
                        (ln,) = self._ax.plot(x, y, linewidth=1, color=col, alpha=0.85, label=label)
                        self._legend_handles.append(ln)
                        self._legend_labels.append(label)
                        if str(sid) not in self._legend_handle_by_sid:
                            self._legend_handle_by_sid[str(sid)] = ln
                            self._legend_entries.append((str(sid), label))
                    except Exception:
                        self._ax.plot(x, y, linewidth=1, color=col, alpha=0.85)

                if not any_uv:
                    self._ax.text(0.5, 0.5, "No UV linked", ha="center", va="center", transform=self._ax.transAxes)
            else:
                base_title = (self.app.uv_title_var.get() or "UV chromatogram").strip()
                uv_sess = self.app._active_uv_session()
                suffix = (" — " + uv_sess.path.name) if uv_sess is not None else ""
                self.title_var.set(f"{base_title}{suffix}")
                self.xlabel_var.set(self.app.uv_xlabel_var.get())
                self.ylabel_var.set(self.app.uv_ylabel_var.get())

                x, y = self.app._active_uv_xy()
                if x is None or y is None or x.size == 0:
                    self._ax.text(0.5, 0.5, "No UV linked", ha="center", va="center", transform=self._ax.transAxes)
                else:
                    self._ax.plot(x, y, linewidth=1)

                    labels_by_uvrt = self.app._active_uv_labels_by_uvrt(create=False)
                    if bool(self.app.uv_label_from_ms_var.get()) and labels_by_uvrt:
                        fs = int(self.ann_fs_var.get())
                        try:
                            min_conf = float(self.app.uv_label_min_conf_var.get())
                        except Exception:
                            min_conf = 0.0
                        min_conf = max(0.0, min(100.0, float(min_conf)))
                        for uv_rt, states in sorted(labels_by_uvrt.items(), key=lambda kv: float(kv[0])):
                            uv_i = int(np.argmin(np.abs(x - float(uv_rt))))
                            x0 = float(x[uv_i])
                            y0 = float(y[uv_i])

                            drawn = 0
                            for st in list(states):
                                if drawn >= 3:
                                    break
                                try:
                                    conf = float(getattr(st, "confidence", 0.0) or 0.0)
                                except Exception:
                                    conf = 0.0
                                if float(conf) < float(min_conf):
                                    continue

                                disp = self.app._format_uv_label_display_text(st)
                                ann = self._ax.annotate(
                                    str(disp),
                                    xy=(x0, y0),
                                    xytext=(float(st.xytext[0]), float(st.xytext[1])),
                                    textcoords="data",
                                    ha="center",
                                    va="bottom",
                                    rotation=90,
                                    fontsize=fs,
                                    arrowprops={"arrowstyle": "-", "lw": 0.9},
                                    clip_on=True,
                                )
                                try:
                                    ann.set_picker(True)
                                except Exception:
                                    pass
                                self._annotations.append(ann)
                                self._ann_original_text[id(ann)] = str(disp)
                                drawn += 1

        else:
            if hasattr(self.app, "_is_overlay_active") and bool(self.app._is_overlay_active()):
                base_title = (self.app.spec_title_var.get() or "Spectrum (MS1)").strip()
                target_rt = getattr(self.app, "_overlay_selected_ms_rt", None)
                if target_rt is None and self.app._current_spectrum_meta is not None:
                    try:
                        target_rt = float(self.app._current_spectrum_meta.rt_min)
                    except Exception:
                        target_rt = None
                if target_rt is None:
                    self.title_var.set("Spectrum")
                    self.xlabel_var.set(self.app.spec_xlabel_var.get())
                    self.ylabel_var.set(self.app.spec_ylabel_var.get())
                    self._ax.text(0.5, 0.5, "No spectrum loaded", ha="center", va="center", transform=self._ax.transAxes)
                else:
                    self.title_var.set(f"{base_title} — overlay at RT={float(target_rt):.4f} min")
                    self.xlabel_var.set(self.app.spec_xlabel_var.get())
                    self.ylabel_var.set(self.app.spec_ylabel_var.get())

                    ids = list(self.app._overlay_dataset_ids())
                    name_map = self.app._overlay_display_names(ids)
                    mode = str(getattr(self.app, "_overlay_mode_var").get() or "Stacked")
                    stack = bool(getattr(self.app, "_overlay_stack_spectra_var").get())

                    max_global = 0.0
                    spectra: List[Tuple[str, Any, np.ndarray, np.ndarray]] = []
                    for sid in ids:
                        got = self.app._get_spectrum_for_rt(str(sid), float(target_rt))
                        if got is None:
                            continue
                        meta, mz_vals, int_vals, _dt = got
                        spectra.append((str(sid), meta, np.asarray(mz_vals, dtype=float), np.asarray(int_vals, dtype=float)))
                        if int_vals is not None and np.asarray(int_vals).size:
                            max_global = max(max_global, float(np.max(int_vals)))

                    if not spectra:
                        self._ax.text(0.5, 0.5, "No spectra near selected RT", ha="center", va="center", transform=self._ax.transAxes)
                    else:
                        offset_step = 0.12 * max_global if max_global > 0 else 1.0
                        active_sid = str(getattr(self.app, "_active_session_id", "") or "")
                        if active_sid not in ids and ids:
                            active_sid = str(ids[0])

                        for i, (sid, meta, mz_vals, int_vals) in enumerate(spectra):
                            col = self.app._ensure_overlay_color(str(sid))
                            y = np.asarray(int_vals, dtype=float)
                            if mode in ("Normalized", "Percent of max"):
                                denom = float(np.max(y)) if y.size else 0.0
                                if denom > 0:
                                    y = y / denom
                                if mode == "Percent of max":
                                    y = y * 100.0
                            if stack:
                                y = y + float(i) * float(offset_step)
                            base = 0.0 + (float(i) * float(offset_step) if stack else 0.0)
                            default_label = str(name_map.get(str(sid), str(sid)))
                            label = str(self._legend_label_override.get(str(sid), default_label))
                            try:
                                coll = self._ax.vlines(mz_vals, base, y, linewidth=0.7, color=col, alpha=0.8, label=label)
                                self._legend_handles.append(coll)
                                self._legend_labels.append(label)
                                if str(sid) not in self._legend_handle_by_sid:
                                    self._legend_handle_by_sid[str(sid)] = coll
                                    self._legend_entries.append((str(sid), label))
                            except Exception:
                                self._ax.vlines(mz_vals, base, y, linewidth=0.7, color=col, alpha=0.8)

                            if str(sid) == active_sid:
                                try:
                                    sess = self.app._sessions.get(str(sid))
                                except Exception:
                                    sess = None
                                custom = getattr(sess, "custom_labels_by_spectrum", None) if sess is not None else None
                                overrides = getattr(sess, "spec_label_overrides", None) if sess is not None else None
                                labels_by_key = self.app._collect_labels_for_spectrum(
                                    str(meta.spectrum_id),
                                    meta,
                                    np.asarray(mz_vals, dtype=float),
                                    np.asarray(int_vals, dtype=float),
                                    custom_labels_by_spectrum=custom,
                                    spec_label_overrides=overrides,
                                )
                                if labels_by_key:
                                    mz_a = np.asarray(mz_vals, dtype=float)
                                    in_a = np.asarray(int_vals, dtype=float)
                                    if mz_a.size and in_a.size:
                                        order = np.argsort(mz_a)
                                        mz_s = mz_a[order]
                                        in_s = in_a[order]
                                        y_off = 0.10 * float(np.max(in_s)) if in_s.size else 1.0

                                        def nearest_peak(target: float) -> Tuple[float, float]:
                                            i2 = int(np.searchsorted(mz_s, float(target)))
                                            cand = []
                                            if 0 <= i2 < mz_s.size:
                                                cand.append(i2)
                                            if i2 - 1 >= 0:
                                                cand.append(i2 - 1)
                                            if i2 + 1 < mz_s.size:
                                                cand.append(i2 + 1)
                                            if not cand:
                                                return float(target), 0.0
                                            j2 = min(cand, key=lambda k: abs(float(mz_s[k]) - float(target)))
                                            return float(mz_s[j2]), float(in_s[j2])

                                        for mz_key in sorted(labels_by_key.keys()):
                                            mz_use, in_use = nearest_peak(float(mz_key))
                                            items = labels_by_key.get(float(mz_key), [])
                                            for j, (_kind, text) in enumerate(items):
                                                self._add_annotation(
                                                    str(text),
                                                    xy=(mz_use, in_use),
                                                    xytext=(mz_use, float(in_use) + y_off * (1.0 + float(j))),
                                                )
            else:
                meta = self.app._current_spectrum_meta
                mz = self.app._current_spectrum_mz
                inten = self.app._current_spectrum_int
                if meta is None or mz is None or inten is None:
                    self.title_var.set("Spectrum")
                    self.xlabel_var.set(self.app.spec_xlabel_var.get())
                    self.ylabel_var.set(self.app.spec_ylabel_var.get())
                    self._ax.text(0.5, 0.5, "No spectrum loaded", ha="center", va="center", transform=self._ax.transAxes)
                else:
                    base_title = (self.app.spec_title_var.get() or "Spectrum (MS1)").strip()
                    self.title_var.set(f"{base_title} at RT={meta.rt_min:.4f} min")
                    self.xlabel_var.set(self.app.spec_xlabel_var.get())
                    self.ylabel_var.set(self.app.spec_ylabel_var.get())

                    self._ax.vlines(mz, 0.0, inten, linewidth=0.8)

                    labels_by_key = self.app._collect_labels_for_export(np.asarray(mz, dtype=float), np.asarray(inten, dtype=float))
                    if labels_by_key:
                        mz_a = np.asarray(mz, dtype=float)
                        in_a = np.asarray(inten, dtype=float)
                        if mz_a.size and in_a.size:
                            order = np.argsort(mz_a)
                            mz_s = mz_a[order]
                            in_s = in_a[order]
                            y_off = 0.10 * float(np.max(in_s)) if in_s.size else 1.0

                            def nearest_peak(target: float) -> Tuple[float, float]:
                                i = int(np.searchsorted(mz_s, float(target)))
                                cand = []
                                if 0 <= i < mz_s.size:
                                    cand.append(i)
                                if i - 1 >= 0:
                                    cand.append(i - 1)
                                if i + 1 < mz_s.size:
                                    cand.append(i + 1)
                                if not cand:
                                    return float(target), 0.0
                                j = min(cand, key=lambda k: abs(float(mz_s[k]) - float(target)))
                                return float(mz_s[j]), float(in_s[j])

                            for mz_key in sorted(labels_by_key.keys()):
                                mz_use, in_use = nearest_peak(float(mz_key))
                                items = labels_by_key.get(float(mz_key), [])
                                for j, (_kind, text) in enumerate(items):
                                    self._add_annotation(
                                        str(text),
                                        xy=(mz_use, in_use),
                                        xytext=(mz_use, float(in_use) + y_off * (1.0 + float(j))),
                                    )

        self._apply_style_and_limits_impl(initial=True)
        self._apply_colors()

    def _apply_style_and_limits_impl(self, *, initial: bool = False) -> None:
        self._apply_style_only_impl()

        try:
            xmin = self._parse_optional_float(self.xmin_var.get())
            xmax = self._parse_optional_float(self.xmax_var.get())
            ymin = self._parse_optional_float(self.ymin_var.get())
            ymax = self._parse_optional_float(self.ymax_var.get())
        except Exception:
            messagebox.showerror("Invalid", "Axis limits must be numbers (or blank).", parent=self)
            return

        if xmin is not None or xmax is not None:
            self._ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            self._ax.set_ylim(bottom=ymin, top=ymax)

        if initial:
            try:
                x0, x1 = self._ax.get_xlim()
                y0, y1 = self._ax.get_ylim()
                self.xmin_var.set("")
                self.xmax_var.set("")
                self.ymin_var.set("")
                self.ymax_var.set("")
                _ = (x0, x1, y0, y1)
            except Exception:
                pass

        self._apply_numbering(redraw_only=True)
        self._canvas.draw_idle()

    def _apply_style_and_limits(self) -> None:
        self._apply_style_and_limits_impl(initial=False)

    def _apply_style_only_impl(self) -> None:
        self._ax.set_title(self.title_var.get())
        self._ax.set_xlabel(self.xlabel_var.get())
        self._ax.set_ylabel(self.ylabel_var.get())

        tfs = int(self.title_fs_var.get())
        lfs = int(self.label_fs_var.get())
        kfs = int(self.tick_fs_var.get())
        afs = int(self.ann_fs_var.get())
        self._ax.title.set_fontsize(tfs)
        self._ax.xaxis.label.set_fontsize(lfs)
        self._ax.yaxis.label.set_fontsize(lfs)
        self._ax.tick_params(axis="both", which="major", labelsize=kfs)
        try:
            self._ax.grid(True, which="major", alpha=0.20)
        except Exception:
            pass

        sci = ScalarFormatter(useMathText=True)
        sci.set_scientific(True)
        sci.set_powerlimits((0, 0))
        sci.set_useOffset(False)
        self._ax.yaxis.set_major_formatter(sci)

        for ann in self._annotations:
            try:
                ann.set_fontsize(afs)
            except Exception:
                pass

        try:
            w = float(self.fig_w_var.get())
            h = float(self.fig_h_var.get())
            if w > 0 and h > 0:
                self._fig.set_size_inches(w, h, forward=True)
        except Exception:
            pass

        self._apply_legend()

    def _apply_legend(self) -> None:
        try:
            if self._legend_artist is not None:
                try:
                    self._legend_artist.remove()
                except Exception:
                    pass
            self._legend_artist = None
        except Exception:
            self._legend_artist = None

        want = bool(self.legend_on_var.get()) and len(self._legend_handles) > 1
        if not want:
            return

        try:
            fs = int(self.legend_fs_var.get())
        except Exception:
            fs = 8

        try:
            self._legend_artist = self._ax.legend(
                handles=list(self._legend_handles),
                labels=list(self._legend_labels),
                loc="best",
                fontsize=fs,
                frameon=bool(self.legend_frame_on_var.get()),
            )
        except Exception:
            self._legend_artist = None
            return

        leg_txt = (self.legend_text_color_var.get() or "").strip() or None
        leg_bg = (self.legend_box_color_var.get() or "").strip() or None
        try:
            if self._legend_artist is not None:
                if leg_txt:
                    for txt in list(self._legend_artist.get_texts()):
                        try:
                            txt.set_color(leg_txt)
                        except Exception:
                            pass
                if leg_bg:
                    frame = self._legend_artist.get_frame()
                    if frame is not None:
                        try:
                            frame.set_facecolor(leg_bg)
                        except Exception:
                            pass
        except Exception:
            pass

    def _install_live_style_traces(self) -> None:
        if bool(getattr(self, "_live_style_traces_installed", False)):
            return
        self._live_style_traces_installed = True

        def _schedule(*_args) -> None:
            try:
                self._schedule_live_style_apply()
            except Exception:
                pass

        for var in (
            self.title_fs_var,
            self.label_fs_var,
            self.tick_fs_var,
            self.ann_fs_var,
            self.fig_w_var,
            self.fig_h_var,
        ):
            try:
                var.trace_add("write", _schedule)
            except Exception:
                try:
                    var.trace("w", _schedule)
                except Exception:
                    pass

    def _schedule_live_style_apply(self) -> None:
        try:
            if self._live_style_job is not None:
                try:
                    self.after_cancel(self._live_style_job)
                except Exception:
                    pass
                self._live_style_job = None
        except Exception:
            pass

        try:
            self._live_style_job = self.after(80, self._apply_live_style_now)
        except Exception:
            self._live_style_job = None

    def _apply_live_style_now(self) -> None:
        self._live_style_job = None
        # Live updates should never pop error dialogs (e.g., while typing axis limits).
        try:
            self._apply_style_only_impl()
            self._apply_numbering(redraw_only=True)
            self._canvas.draw_idle()
        except Exception:
            pass

    def _label_rt_for_annotation(self, ann) -> str:
        if self.kind in ("tic", "uv"):
            try:
                x = float(ann.xy[0])
                return f"{x:.4f}"
            except Exception:
                return ""
        meta = self.app._current_spectrum_meta
        if meta is None:
            return ""
        return f"{float(meta.rt_min):.4f}"

    def _apply_numbering(self, redraw_only: bool = False) -> None:
        enabled = bool(self.number_labels_var.get())
        if not enabled:
            for ann in self._annotations:
                orig = self._ann_original_text.get(id(ann))
                if orig is not None:
                    try:
                        ann.set_text(str(orig))
                    except Exception:
                        pass
            if self._table_artist is not None:
                try:
                    self._table_artist.remove()
                except Exception:
                    pass
            self._table_artist = None
            self._num_to_ann = {}
            self._refresh_table_tree()
            if not redraw_only:
                self._canvas.draw_idle()
            return

        ann_sorted = []
        for ann in self._annotations:
            try:
                ann_sorted.append((float(ann.xy[0]), ann))
            except Exception:
                ann_sorted.append((0.0, ann))
        ann_sorted.sort(key=lambda t: float(t[0]))

        self._num_to_ann = {}
        rows: List[List[str]] = []
        for i, (_x, ann) in enumerate(ann_sorted, start=1):
            orig = self._ann_original_text.get(id(ann), "")
            try:
                ann.set_text(str(i))
            except Exception:
                pass
            self._num_to_ann[int(i)] = ann
            rt_text = self._table_rt_override.get(int(i))
            if not rt_text:
                rt_text = self._label_rt_for_annotation(ann)
            rows.append([str(i), str(orig), str(rt_text)])

        if self._table_artist is not None:
            try:
                self._table_artist.remove()
            except Exception:
                pass
        self._table_artist = None

        if rows:
            tbl = self._ax.table(
                cellText=rows,
                colLabels=["#", "Label", "RT (min)"],
                colLoc="left",
                cellLoc="left",
                bbox=[
                    float(self.tbl_x_var.get()),
                    float(self.tbl_y_var.get()),
                    float(self.tbl_w_var.get()),
                    float(self.tbl_h_var.get()),
                ],
            )
            self._table_artist = tbl
            try:
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(max(7, int(self.tick_fs_var.get()) - 1))
            except Exception:
                pass

            face = (self.table_facecolor_var.get() or "").strip()
            txtc = (self.table_text_color_var.get() or "").strip()
            try:
                for (_r, _c), cell in tbl.get_celld().items():
                    if face:
                        cell.set_facecolor(face)
                    if txtc:
                        cell.get_text().set_color(txtc)
            except Exception:
                pass

        self._refresh_table_tree()
        if not redraw_only:
            self._canvas.draw_idle()

    def _save_as(self) -> None:
        initial = f"{self.default_stem}.png"
        path = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            initialfile=initial,
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            try:
                self.configure(cursor="watch")
                self.update_idletasks()
            except Exception:
                pass
            self._canvas.draw()
            self._fig.savefig(path, bbox_inches="tight", facecolor=self._fig.get_facecolor(), transparent=False)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save plot:\n{exc}", parent=self)
            return
        finally:
            try:
                self.configure(cursor="")
            except Exception:
                pass
        messagebox.showinfo("Saved", f"Saved:\n{path}", parent=self)

    def _open_editor_for_label(self, ann) -> None:
        if ann is None:
            return
        current = ""
        try:
            current = str(self._ann_original_text.get(id(ann), ann.get_text()))
        except Exception:
            current = str(self._ann_original_text.get(id(ann), ""))

        dlg = tk.Toplevel(self)
        dlg.title("Edit label")
        dlg.resizable(False, False)
        dlg.transient(self)

        frm = ttk.Frame(dlg, padding=10)
        frm.grid(row=0, column=0)
        ttk.Label(frm, text="Label text").grid(row=0, column=0, sticky="w")
        var = tk.StringVar(value=current)
        ent = ttk.Entry(frm, textvariable=var, width=46)
        ent.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        try:
            ent.focus_set()
            ent.selection_range(0, tk.END)
        except Exception:
            pass

        def apply() -> None:
            new = (var.get() or "").strip()
            if not new:
                messagebox.showerror("Invalid", "Label cannot be empty (use Delete).", parent=dlg)
                return
            self._ann_original_text[id(ann)] = new
            if not bool(self.number_labels_var.get()):
                try:
                    ann.set_text(new)
                except Exception:
                    pass
            self._apply_numbering(redraw_only=True)
            self._canvas.draw_idle()
            dlg.destroy()

        def delete() -> None:
            try:
                ann.remove()
            except Exception:
                pass
            self._annotations = [a for a in self._annotations if a is not ann]
            self._ann_original_text.pop(id(ann), None)
            self._apply_numbering(redraw_only=True)
            self._canvas.draw_idle()
            dlg.destroy()

        ttk.Button(frm, text="Apply", command=apply).grid(row=2, column=0, pady=(10, 0), padx=(0, 8), sticky="e")
        ttk.Button(frm, text="Delete", command=delete).grid(row=2, column=1, pady=(10, 0), padx=(0, 8), sticky="e")
        ttk.Button(frm, text="Cancel", command=dlg.destroy).grid(row=2, column=2, pady=(10, 0), sticky="e")

    def _on_press(self, event) -> None:
        if event.inaxes != self._ax:
            return
        if getattr(event, "button", None) != 1:
            return
        ann = None
        try:
            for a in list(self._annotations):
                ok, _ = a.contains(event)
                if ok:
                    ann = a
                    break
        except Exception:
            ann = None

        if ann is None:
            self._active_ann = None
            return

        if getattr(event, "dblclick", False) or getattr(event, "button", None) == 3:
            self._open_editor_for_label(ann)
            return

        self._active_ann = ann

    def _on_motion(self, event) -> None:
        if self._active_ann is None:
            return
        if event.inaxes != self._ax:
            return
        try:
            x = float(event.xdata)
            y = float(event.ydata)
        except Exception:
            return
        try:
            self._active_ann.set_position((x, y))
        except Exception:
            return
        self._canvas.draw_idle()

    def _on_release(self, event) -> None:
        self._active_ann = None
