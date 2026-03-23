from __future__ import annotations

import math
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import numpy as np

from matplotlib import colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Keep this import aligned with Matplotlib stubs to avoid Pylance false-positives.
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox

from lab_gui.plot_card import PlotCard
from lab_gui.ui_theme import style_primary, style_secondary, style_success
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
        """Implement the `__init__` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Implement the `_tt` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        return str(self._tooltip_text.get(str(key), "") or "")

    def _source_axis(self) -> Optional[Any]:
        """Implement the `_source_axis` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        attr = {"tic": "_ax_tic", "uv": "_ax_uv", "spectrum": "_ax_spec"}.get(str(self.kind), "")
        if not attr:
            return None
        return getattr(self.app, attr, None)

    def _to_hex_color(self, value: Any, fallback: str) -> str:
        """Implement the `_to_hex_color` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        try:
            return str(mcolors.to_hex(value, keep_alpha=False))
        except Exception:
            return str(fallback)

    def _grid_style_from_lines(self, lines: List[Any], *, fallback_color: str, fallback_alpha: float, fallback_lw: float) -> Dict[str, Any]:
        """Implement the `_grid_style_from_lines` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        for ln in list(lines or []):
            try:
                return {
                    "color": ln.get_color(),
                    "alpha": float(ln.get_alpha() if ln.get_alpha() is not None else fallback_alpha),
                    "linewidth": float(ln.get_linewidth()),
                    "linestyle": str(ln.get_linestyle() or "-"),
                }
            except Exception:
                continue
        return {
            "color": fallback_color,
            "alpha": float(fallback_alpha),
            "linewidth": float(fallback_lw),
            "linestyle": "-",
        }

    def _capture_live_style_snapshot(self) -> Dict[str, Any]:
        """Implement the `_capture_live_style_snapshot` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        defaults: Dict[str, Any] = {
            "figure_facecolor": "#F4F7F7",
            "axes_facecolor": "#FCFEFE",
            "title_color": "#0F4C46",
            "label_color": "#111827",
            "tick_color": "#6B7280",
            "spine_color": "#C7D6D2",
            "spine_width": 0.9,
            "title_family": "Segoe UI",
            "label_family": "Segoe UI",
            "tick_family": "Segoe UI",
            "title_weight": "semibold",
            "label_weight": "semibold",
            "grid_x": {"color": "#EEF4F3", "alpha": 1.0, "linewidth": 0.6, "linestyle": "-"},
            "grid_y": {"color": "#D6E2DF", "alpha": 0.95, "linewidth": 0.8, "linestyle": "-"},
            "plot_color": {"tic": "#0F766E", "uv": "#0F8AA6", "spectrum": "#0B5D6B"}.get(str(self.kind), "#0F766E"),
            "fill_color": {"tic": "#CDEEE6", "uv": "#D9F1F7"}.get(str(self.kind), "#CDEEE6"),
            "line_width": {"tic": 1.8, "uv": 1.65, "spectrum": 1.0}.get(str(self.kind), 1.5),
            "line_alpha": {"tic": 0.99, "uv": 0.98, "spectrum": 0.96}.get(str(self.kind), 0.94),
            "fill_alpha": {"tic": 0.24, "uv": 0.28}.get(str(self.kind), 0.24),
            "line_capstyle": "round",
            "collection_linewidth": 1.0,
            "legend_text_color": "#111827",
            "legend_box_color": "#FCFEFE",
            "table_facecolor": "#FFFFFF",
            "table_text_color": "#111827",
        }

        src_fig = getattr(self.app, "_fig", None)
        src_ax = self._source_axis()

        try:
            if src_fig is not None:
                defaults["figure_facecolor"] = src_fig.get_facecolor()
        except Exception:
            pass

        if src_ax is None:
            return defaults

        try:
            defaults["axes_facecolor"] = src_ax.get_facecolor()
        except Exception:
            pass
        try:
            defaults["title_color"] = src_ax.title.get_color()
        except Exception:
            pass
        try:
            defaults["label_color"] = src_ax.xaxis.label.get_color()
        except Exception:
            pass
        try:
            defaults["title_family"] = str((src_ax.title.get_fontfamily() or [defaults["title_family"]])[0])
        except Exception:
            pass
        try:
            defaults["label_family"] = str((src_ax.xaxis.label.get_fontfamily() or [defaults["label_family"]])[0])
        except Exception:
            pass
        try:
            tick_labels = list(src_ax.get_xticklabels()) + list(src_ax.get_yticklabels())
            for lbl in tick_labels:
                txt = str(lbl.get_text() or "").strip()
                if txt:
                    defaults["tick_color"] = lbl.get_color()
                    fam = lbl.get_fontfamily() or [defaults["tick_family"]]
                    defaults["tick_family"] = str(fam[0])
                    break
        except Exception:
            pass
        try:
            defaults["title_weight"] = str(src_ax.title.get_fontweight() or defaults["title_weight"])
        except Exception:
            pass
        try:
            defaults["label_weight"] = str(src_ax.xaxis.label.get_fontweight() or defaults["label_weight"])
        except Exception:
            pass
        try:
            spine = src_ax.spines.get("bottom")
            if spine is not None:
                defaults["spine_color"] = spine.get_edgecolor()
                defaults["spine_width"] = float(spine.get_linewidth())
        except Exception:
            pass

        try:
            defaults["grid_x"] = self._grid_style_from_lines(
                list(src_ax.get_xgridlines()),
                fallback_color="#EEF4F3",
                fallback_alpha=1.0,
                fallback_lw=0.6,
            )
        except Exception:
            pass
        try:
            defaults["grid_y"] = self._grid_style_from_lines(
                list(src_ax.get_ygridlines()),
                fallback_color="#D6E2DF",
                fallback_alpha=0.95,
                fallback_lw=0.8,
            )
        except Exception:
            pass

        try:
            lines = list(getattr(src_ax, "lines", []))
            if lines:
                ln = lines[0]
                defaults["plot_color"] = ln.get_color()
                defaults["line_width"] = float(ln.get_linewidth())
                defaults["line_alpha"] = float(ln.get_alpha() if ln.get_alpha() is not None else defaults["line_alpha"])
                try:
                    defaults["line_capstyle"] = str(ln.get_solid_capstyle() or defaults["line_capstyle"])
                except Exception:
                    pass
        except Exception:
            pass

        try:
            for coll in list(getattr(src_ax, "collections", [])):
                facecolors = coll.get_facecolors() if hasattr(coll, "get_facecolors") else []
                edgecolors = coll.get_edgecolors() if hasattr(coll, "get_edgecolors") else []
                linewidths = coll.get_linewidths() if hasattr(coll, "get_linewidths") else []
                if linewidths is not None and len(linewidths):
                    defaults["collection_linewidth"] = float(linewidths[0])
                if self.kind in ("tic", "uv") and facecolors is not None and len(facecolors):
                    alpha = coll.get_alpha() if hasattr(coll, "get_alpha") else None
                    if alpha is None:
                        try:
                            alpha = float(facecolors[0][-1])
                        except Exception:
                            alpha = defaults["fill_alpha"]
                    if float(alpha or 0.0) < 0.8:
                        defaults["fill_color"] = facecolors[0]
                        defaults["fill_alpha"] = float(alpha or defaults["fill_alpha"])
                        break
                if self.kind == "spectrum":
                    if edgecolors is not None and len(edgecolors):
                        defaults["plot_color"] = edgecolors[0]
                        break
                    if facecolors is not None and len(facecolors):
                        defaults["plot_color"] = facecolors[0]
                        break
        except Exception:
            pass

        return defaults

    def _apply_live_plot_theme(self) -> None:
        """Prepare plotting data and visual elements.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        snap = dict(getattr(self, "_live_style_snapshot", {}) or {})
        fig_bg = snap.get("figure_facecolor", "#F4F7F7")
        ax_bg = (self.axes_facecolor_var.get() or "").strip() or self._to_hex_color(snap.get("axes_facecolor"), "#FCFEFE")
        title_color = snap.get("title_color", "#0F4C46")
        label_color = snap.get("label_color", "#111827")
        tick_color = snap.get("tick_color", "#6B7280")
        spine_color = snap.get("spine_color", "#C7D6D2")
        spine_width = float(snap.get("spine_width", 0.9) or 0.9)
        grid_x = dict(snap.get("grid_x", {}) or {})
        grid_y = dict(snap.get("grid_y", {}) or {})

        try:
            self._fig.patch.set_facecolor(fig_bg)
        except Exception:
            pass

        try:
            self._fig.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.94)
        except Exception:
            pass

        try:
            self._ax.set_facecolor(ax_bg)
            self._ax.set_axisbelow(True)
            self._ax.margins(x=0.01)
        except Exception:
            pass

        try:
            self._ax.title.set_color(title_color)
            self._ax.title.set_fontfamily(str(snap.get("title_family", "Segoe UI")))
            self._ax.title.set_fontweight(str(snap.get("title_weight", "semibold")))
            self._ax.title.set_position((0.5, 1.02))
        except Exception:
            pass
        try:
            self._ax.xaxis.label.set_color(label_color)
            self._ax.yaxis.label.set_color(label_color)
            self._ax.xaxis.label.set_fontfamily(str(snap.get("label_family", "Segoe UI")))
            self._ax.yaxis.label.set_fontfamily(str(snap.get("label_family", "Segoe UI")))
            self._ax.xaxis.label.set_fontweight(str(snap.get("label_weight", "semibold")))
            self._ax.yaxis.label.set_fontweight(str(snap.get("label_weight", "semibold")))
            self._ax.xaxis.labelpad = 12
            self._ax.yaxis.labelpad = 10
        except Exception:
            pass
        try:
            self._ax.tick_params(axis="both", colors=tick_color, which="major", length=4, width=0.8, pad=7)
            for lbl in list(self._ax.get_xticklabels()) + list(self._ax.get_yticklabels()):
                try:
                    lbl.set_fontfamily(str(snap.get("tick_family", "Segoe UI")))
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self._ax.spines["top"].set_visible(False)
            self._ax.spines["right"].set_visible(False)
            self._ax.spines["left"].set_color(spine_color)
            self._ax.spines["bottom"].set_color(spine_color)
            self._ax.spines["left"].set_linewidth(spine_width)
            self._ax.spines["bottom"].set_linewidth(spine_width)
        except Exception:
            pass

        try:
            self._ax.grid(True, axis="y", which="major", color=grid_y.get("color", "#D6E2DF"), alpha=float(grid_y.get("alpha", 0.95) or 0.95), linewidth=float(grid_y.get("linewidth", 0.8) or 0.8), linestyle=str(grid_y.get("linestyle", "-") or "-"))
            self._ax.grid(True, axis="x", which="major", color=grid_x.get("color", "#EEF4F3"), alpha=float(grid_x.get("alpha", 1.0) or 1.0), linewidth=float(grid_x.get("linewidth", 0.6) or 0.6), linestyle=str(grid_x.get("linestyle", "-") or "-"))
        except Exception:
            pass

    def _init_ui(self) -> None:

        """Implement the `_init_ui` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        self.rowconfigure(2, weight=1)

        top = ttk.LabelFrame(self, text="Export Actions", padding=8, style="Card.TLabelframe")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)
        top.columnconfigure(3, weight=1)
        top.columnconfigure(4, weight=1)
        ttk.Label(
            top,
            text="Tune the export-specific presentation here while keeping the final figure large and readable.",
            style="CardHint.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))
        controls_btn = ttk.Button(top, text="Controls…", command=self._open_controls_window)
        controls_btn.grid(row=1, column=0, sticky="ew")
        style_primary(controls_btn)
        arrange_btn = ttk.Button(top, text="Auto Arrange Labels", command=self._auto_arrange_labels)
        arrange_btn.grid(row=1, column=1, sticky="ew", padx=(8, 8))
        style_secondary(arrange_btn)
        distribute_btn = ttk.Button(top, text="Distribute Labels…", command=self._open_distribute_labels_dialog)
        distribute_btn.grid(row=1, column=2, sticky="ew", padx=(0, 8))
        style_secondary(distribute_btn)
        saveas_btn = ttk.Button(top, text="Save As…", command=self._save_as)
        saveas_btn.grid(row=1, column=3, sticky="ew", padx=(0, 8))
        style_success(saveas_btn)
        close_btn = ttk.Button(top, text="Close", command=self._on_close_export)
        close_btn.grid(row=1, column=4, sticky="ew")
        style_secondary(close_btn)

        ToolTip.attach(controls_btn, self._tt("exp_controls"))
        ToolTip.attach(arrange_btn, self._tt("annotate_peaks"))
        ToolTip.attach(distribute_btn, "Distribute export labels like PowerPoint: equal center spacing on X/Y for all labels or a selected index range.")
        ToolTip.attach(saveas_btn, self._tt("exp_saveas"))
        ToolTip.attach(close_btn, self._tt("exp_close"))

        stage_hdr = ttk.Frame(self, style="Surface.TFrame", padding=(14, 12))
        stage_hdr.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        stage_hdr.columnconfigure(0, weight=1)
        ttk.Label(stage_hdr, text="Export Stage", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            stage_hdr,
            text="Review the final composition at full size, then open controls only when you need detailed styling changes.",
            style="CardHint.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(stage_hdr, text=str(self.kind).upper(), style="CardStatus.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        plot_card = PlotCard(cast(tk.Widget, self), title=f"{str(self.kind).upper()} Export", status_text="Preview", show_header=True)
        plot_card.grid(row=2, column=0, sticky="nsew")
        plot = plot_card.body
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
        plot_card.register_canvas(self._canvas)

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

        self._live_style_snapshot = self._capture_live_style_snapshot()

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
        self.ann_fs_var = tk.IntVar(value=10)
        self.ann_orientation_var = tk.StringVar(value="vertical")

        self.xmin_var = tk.StringVar(value="")
        self.xmax_var = tk.StringVar(value="")
        self.ymin_var = tk.StringVar(value="")
        self.ymax_var = tk.StringVar(value="")

        self.fig_w_var = tk.DoubleVar(value=float(self._fig.get_size_inches()[0]))
        self.fig_h_var = tk.DoubleVar(value=float(self._fig.get_size_inches()[1]))
        self.overlay_gap_var = tk.DoubleVar(value=float(getattr(self.app, "_overlay_offset_scale", 0.12) or 0.12))

        self.number_labels_var = tk.BooleanVar(value=False)

        self._annotations: List[Any] = []
        self._ann_original_text: Dict[int, str] = {}
        self._active_ann: Optional[Any] = None
        self._label_distribution_win: Optional[tk.Toplevel] = None

        # Table + overrides (for editing the table text)
        self._num_to_ann: Dict[int, Any] = {}
        self._table_rt_override: Dict[int, str] = {}

        # Table placement (axes coordinates)
        self.tbl_x_var = tk.DoubleVar(value=0.56)
        self.tbl_y_var = tk.DoubleVar(value=0.56)
        self.tbl_w_var = tk.DoubleVar(value=0.43)
        self.tbl_h_var = tk.DoubleVar(value=0.43)

        # Colors
        self.plot_color_var = tk.StringVar(value="")
        self.label_color_var = tk.StringVar(value=self._to_hex_color(self._live_style_snapshot.get("label_color"), "#111111"))
        self.axes_facecolor_var = tk.StringVar(value=self._to_hex_color(self._live_style_snapshot.get("axes_facecolor"), "#FCFEFE"))
        self.table_facecolor_var = tk.StringVar(value=self._to_hex_color(self._live_style_snapshot.get("table_facecolor"), "#FFFFFF"))
        self.table_text_color_var = tk.StringVar(value=self._to_hex_color(self._live_style_snapshot.get("table_text_color"), "#111111"))

        # Legend (overlay)
        self.legend_on_var = tk.BooleanVar(value=True)
        self.legend_fs_var = tk.IntVar(value=max(6, int(self.app.tick_fontsize_var.get()) - 1))
        self.legend_text_color_var = tk.StringVar(value=self._to_hex_color(self._live_style_snapshot.get("legend_text_color"), "#111111"))
        self.legend_box_color_var = tk.StringVar(value=self._to_hex_color(self._live_style_snapshot.get("legend_box_color"), "#FCFEFE"))
        self.legend_frame_on_var = tk.BooleanVar(value=True)
        self._legend_artist: Any = None
        self._legend_handles: List[Any] = []
        self._legend_labels: List[str] = []
        self._legend_handle_by_sid: Dict[str, Any] = {}
        self._legend_entries: List[Tuple[str, str]] = []
        self._legend_label_override: Dict[str, str] = {}

        self._preserve_plot_colors = self.kind in ("tic", "uv", "spectrum")

        try:
            self._fig.patch.set_facecolor(self._live_style_snapshot.get("figure_facecolor", "#F4F7F7"))
        except Exception:
            pass

        self._install_color_traces()
        self._plot_rebuild_job: Optional[str] = None
        self._install_overlay_gap_trace()
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
        """Close resources and finalize state.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Implement the `_on_controls_closed` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._controls_win = None
        self._controls_scroll_canvas = None

    def _open_controls_window(self) -> None:
        # Reuse if already open.
        """Open a file, view, or resource.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        hdr = ttk.Frame(outer, style="ShellPanel.TFrame", padding=(14, 12))
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        hdr.columnconfigure(0, weight=1)
        ttk.Label(hdr, text="Export Controls", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            hdr,
            text="This window holds detailed styling controls so the main export preview can stay uncluttered.",
            style="CardHint.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(hdr, text=str(self.kind).upper(), style="CardStatus.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        # Scrollable area
        canvas = tk.Canvas(outer, highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        self._controls_scroll_canvas = canvas
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
            """Implement the `_add_slider` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            holder = ttk.Frame(parent)
            holder.columnconfigure(0, weight=1)

            scale_var = tk.DoubleVar(value=float(variable.get()))
            lbl = ttk.Label(holder, text="")

            def _format_value(v: float) -> str:
                """Implement the `_format_value` behavior for this module.

                Text-only documentation note: modify internal logic here to change behavior.
                """
                if fmt:
                    try:
                        return fmt.format(v)
                    except Exception:
                        pass
                if isinstance(variable, tk.IntVar):
                    return str(int(round(v)))
                return f"{float(v):.2f}"

            def _apply(v: Any = None) -> None:
                """Implement the `_apply` behavior for this module.

                Text-only documentation note: modify internal logic here to change behavior.
                """
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

        ttk.Label(inner, text="Label orientation").grid(row=row, column=0, sticky="w")
        ann_orientation_box = ttk.Combobox(inner, textvariable=self.ann_orientation_var, values=("vertical", "horizontal"), state="readonly")
        ann_orientation_box.grid(row=row, column=1, sticky="ew", padx=(8, 0))
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

        ttk.Label(inner, text="Overlay gap × max").grid(row=row, column=0, sticky="w")
        s_overlay_gap = _add_slider(inner, variable=self.overlay_gap_var, from_=0.0, to=1.5, step=0.01, fmt="{:.2f}")
        row += 1

        ttk.Separator(inner).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Legend (overlay only)
        leg_group = ttk.Labelframe(inner, text="Legend (overlay)", padding=(8, 6))
        leg_group.grid(row=row, column=0, columnspan=2, sticky="ew")
        leg_group.columnconfigure(1, weight=1)
        row += 1

        def _pick_legend_color(var: tk.StringVar, title: str) -> None:
            """Implement the `_pick_legend_color` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
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
            """Refresh derived state or UI content.

            Text-only documentation note: modify internal logic here to change behavior.
            """
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
            """Implement the `_edit_legend_label` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
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
            """Implement the `_nudge_table` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
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

        def _edit_table_cell(evt: Any = None):
            """Implement the `_edit_table_cell` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            if evt is None:
                return
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
            """Implement the `_pick_one` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
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
            ToolTip.attach(ann_orientation_box, "Export-only label orientation. Switch between vertical and horizontal annotation text.")
            ToolTip.attach(s_fig_w, "Figure width in inches (export-only).")
            ToolTip.attach(s_fig_h, "Figure height in inches (export-only).")
            ToolTip.attach(s_overlay_gap, "Overlay gap as a fraction of the panel maximum. Used by Offset mode and stacked spectra.")
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
        """Implement the `_install_color_traces` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        if bool(getattr(self, "_color_traces_installed", False)):
            return
        self._color_traces_installed = True

        def _cb(*_args) -> None:
            """Implement the `_cb` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
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
        """Implement the `_pick_colors` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Refresh derived state or UI content.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Implement the `_label_rt_for_number` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Implement the `_apply_colors` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Parse raw input into structured values.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        raw = (raw or "").strip()
        if not raw:
            return None
        return float(raw)

    def _annotation_rotation(self) -> float:
        """Implement the `_annotation_rotation` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        return 0.0 if str(self.ann_orientation_var.get() or "vertical").strip().lower() == "horizontal" else 90.0

    def _apply_annotation_orientation(self) -> None:
        """Implement the `_apply_annotation_orientation` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        rotation = float(self._annotation_rotation())
        for ann in list(self._annotations):
            try:
                ann.set_rotation(rotation)
            except Exception:
                pass
            try:
                ann.set_rotation_mode("anchor")
            except Exception:
                pass

    def _clear_annotations(self) -> None:
        """Implement the `_clear_annotations` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        for ann in list(self._annotations):
            try:
                ann.remove()
            except Exception:
                pass
        self._annotations = []
        self._ann_original_text = {}

    def _add_annotation(self, text: str, *, xy: Tuple[float, float], xytext: Tuple[float, float]) -> Any:
        """Implement the `_add_annotation` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        ann_color = (self.label_color_var.get() or "").strip() or self._to_hex_color(self._live_style_snapshot.get("label_color"), "#111111")
        ann = self._ax.annotate(
            str(text),
            xy=(float(xy[0]), float(xy[1])),
            xytext=(float(xytext[0]), float(xytext[1])),
            textcoords="data",
            ha="center",
            va="bottom",
            rotation=float(self._annotation_rotation()),
            fontsize=int(self.ann_fs_var.get()),
            color=ann_color,
            arrowprops={"arrowstyle": "-", "lw": 0.95, "color": ann_color, "alpha": 0.9},
            clip_on=True,
        )
        try:
            ann.set_picker(True)
        except Exception:
            pass
        self._annotations.append(ann)
        self._ann_original_text[id(ann)] = str(text)
        return ann

    def _auto_arrange_labels(self) -> None:
        """Implement the `_auto_arrange_labels` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        if not self._annotations:
            try:
                messagebox.showinfo("Auto Arrange Labels", "There are no export labels to arrange.", parent=self)
            except Exception:
                pass
            return
        helper = getattr(self.app, "_auto_arrange_annotation_artists", None)
        if not callable(helper):
            return
        items: List[Dict[str, Any]] = []
        for idx, ann in enumerate(list(self._annotations)):
            try:
                anchor_xy = ann.xy
            except Exception:
                continue
            priority = 50.0
            try:
                priority += max(0.0, float(anchor_xy[1]))
            except Exception:
                pass
            priority -= float(idx) * 0.05
            items.append(
                {
                    "artist": ann,
                    "anchor_xy": (float(anchor_xy[0]), float(anchor_xy[1])),
                    "locked": False,
                    "priority": float(priority),
                }
            )
        try:
            result = helper(canvas=self._canvas, ax=self._ax, items=items, full_reflow=True)
            changed = bool(result.get("changed", False)) if isinstance(result, dict) else bool(result)
        except Exception:
            changed = False
        if changed:
            try:
                self._canvas.draw_idle()
            except Exception:
                pass

    def _sorted_export_annotations(self) -> List[Any]:
        """Export data in an external-friendly format.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        anns: List[Tuple[float, Any]] = []
        for ann in list(self._annotations):
            try:
                anns.append((float(ann.xy[0]), ann))
            except Exception:
                anns.append((0.0, ann))
        anns.sort(key=lambda item: float(item[0]))
        return [ann for _, ann in anns]

    def _parse_annotation_selection(self, raw: str, total: int) -> List[int]:
        """Parse raw input into structured values.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        raw = str(raw or "").strip()
        if not raw:
            return []
        selected: List[int] = []
        for chunk in raw.split(","):
            part = str(chunk).strip()
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                start = int(str(left).strip())
                end = int(str(right).strip())
                lo = min(start, end)
                hi = max(start, end)
                for value in range(lo, hi + 1):
                    if 1 <= value <= total and value not in selected:
                        selected.append(int(value))
                continue
            value = int(part)
            if 1 <= value <= total and value not in selected:
                selected.append(int(value))
        return selected

    def _resolve_selected_annotations(self, *, scope: str, selection_text: str) -> List[Any]:
        """Implement the `_resolve_selected_annotations` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        ordered = self._sorted_export_annotations()
        if not ordered:
            return []
        if str(scope) != "selected":
            return list(ordered)
        selected_indices = self._parse_annotation_selection(selection_text, len(ordered))
        if not selected_indices:
            raise ValueError("Enter one or more label indices, for example 1-4,7")
        selected_annotations = [ordered[idx - 1] for idx in selected_indices if 1 <= idx <= len(ordered)]
        if not selected_annotations:
            raise ValueError("The selected label indices are out of range")
        return selected_annotations

    def _annotation_is_locked(self, ann: Any) -> bool:
        """Implement the `_annotation_is_locked` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        for name in ("_mfp_locked", "mfp_locked", "locked"):
            try:
                if bool(getattr(ann, name, False)):
                    return True
            except Exception:
                continue
        try:
            locked_ids = getattr(self, "_locked_annotation_ids", None)
            if isinstance(locked_ids, set) and id(ann) in locked_ids:
                return True
        except Exception:
            pass
        return False

    def _distribute_annotations(
        self,
        *,
        scope: str,
        selection_text: str,
        x_spacing: float = 1.0,
        y_spacing: float = 1.0,
        apply_x: bool = True,
        apply_y: bool = True,
    ) -> Dict[str, int]:
        """Implement the `_distribute_annotations` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        selected_annotations = self._resolve_selected_annotations(scope=scope, selection_text=selection_text)
        if not selected_annotations:
            return {"moved": 0, "locked": 0, "selected": 0, "movable": 0, "axis_x": 0, "axis_y": 0}

        x_spacing = float(x_spacing)
        y_spacing = float(y_spacing)
        if bool(apply_x) and x_spacing < 0.0:
            raise ValueError("X spacing must be zero or greater.")
        if bool(apply_y) and y_spacing < 0.0:
            raise ValueError("Y spacing must be zero or greater.")

        try:
            self._canvas.draw()
        except Exception:
            pass
        try:
            renderer = self._canvas.get_renderer()
        except Exception as exc:
            raise RuntimeError(f"Renderer unavailable: {exc}")

        axis_bbox = self._ax.get_window_extent(renderer)
        to_disp = self._ax.transData.transform
        from_disp = self._ax.transData.inverted().transform

        entries: List[Dict[str, Any]] = []
        locked = 0
        for ann in selected_annotations:
            bbox: Optional[Bbox] = None
            try:
                current_data_x, current_data_y = ann.get_position()
                anchor_disp = tuple(to_disp((float(current_data_x), float(current_data_y))))
                bbox = ann.get_window_extent(renderer)
            except Exception:
                continue
            if bbox is None:
                continue
            is_locked = self._annotation_is_locked(ann)
            if is_locked:
                locked += 1
            entries.append(
                {
                    "ann": ann,
                    "locked": bool(is_locked),
                    "current_data": (float(current_data_x), float(current_data_y)),
                    "anchor_disp": (float(anchor_disp[0]), float(anchor_disp[1])),
                    "center_disp": (float((bbox.x0 + bbox.x1) * 0.5), float((bbox.y0 + bbox.y1) * 0.5)),
                    "size_disp": (float(max(1e-9, bbox.width)), float(max(1e-9, bbox.height))),
                }
            )

        movable = [entry for entry in entries if not bool(entry.get("locked", False))]
        moved = 0
        axis_x_applied = 0
        axis_y_applied = 0

        if bool(apply_x) and len(movable) >= 3:
            ordered_x = sorted(movable, key=lambda item: float(item["center_disp"][0]))
            left = float(ordered_x[0]["center_disp"][0])
            right = float(ordered_x[-1]["center_disp"][0])
            step_x = float((right - left) / max(1, len(ordered_x) - 1))
            midpoint_x = float((left + right) * 0.5)
            for idx, entry in enumerate(ordered_x):
                width = float(entry["size_disp"][0])
                ppt_center = float(left + (idx * step_x))
                raw_center = float(midpoint_x + ((ppt_center - midpoint_x) * x_spacing))
                clamped_center = min(max(raw_center, float(axis_bbox.x0 + (width * 0.5))), float(axis_bbox.x1 - (width * 0.5)))
                entry["target_center_x"] = float(clamped_center)
            axis_x_applied = 1

        if bool(apply_y) and len(movable) >= 3:
            ordered_y = sorted(movable, key=lambda item: float(item["center_disp"][1]))
            bottom = float(ordered_y[0]["center_disp"][1])
            top = float(ordered_y[-1]["center_disp"][1])
            step_y = float((top - bottom) / max(1, len(ordered_y) - 1))
            midpoint_y = float((bottom + top) * 0.5)
            for idx, entry in enumerate(ordered_y):
                height = float(entry["size_disp"][1])
                ppt_center = float(bottom + (idx * step_y))
                raw_center = float(midpoint_y + ((ppt_center - midpoint_y) * y_spacing))
                clamped_center = min(max(raw_center, float(axis_bbox.y0 + (height * 0.5))), float(axis_bbox.y1 - (height * 0.5)))
                entry["target_center_y"] = float(clamped_center)
            axis_y_applied = 1

        for entry in movable:
            ann = entry.get("ann")
            if ann is None:
                continue
            current_data = entry.get("current_data", (0.0, 0.0))
            anchor_disp = entry.get("anchor_disp", (0.0, 0.0))
            size_disp = entry.get("size_disp", (1.0, 1.0))
            target_center_x = float(entry.get("target_center_x", entry.get("center_disp", (anchor_disp[0], anchor_disp[1]))[0]))
            target_center_y = float(entry.get("target_center_y", entry.get("center_disp", (anchor_disp[0], anchor_disp[1]))[1]))
            target_anchor_x = float(target_center_x) if bool(axis_x_applied) else float(anchor_disp[0])
            target_anchor_y = float(target_center_y - (size_disp[1] * 0.5)) if bool(axis_y_applied) else float(anchor_disp[1])

            try:
                next_data = from_disp((float(target_anchor_x), float(target_anchor_y)))
                final_x = float(next_data[0])
                final_y = float(next_data[1])
                ann.set_position((final_x, final_y))
                if abs(final_x - float(current_data[0])) > 1e-9 or abs(final_y - float(current_data[1])) > 1e-9:
                    moved += 1
            except Exception:
                continue

        if moved:
            self._apply_numbering(redraw_only=True)
            try:
                self._canvas.draw_idle()
            except Exception:
                pass

        return {
            "moved": int(moved),
            "locked": int(locked),
            "selected": int(len(entries)),
            "movable": int(len(movable)),
            "axis_x": int(axis_x_applied),
            "axis_y": int(axis_y_applied),
        }

    def _shift_annotations(self, *, scope: str, selection_text: str, shift_x: float, shift_y: float, apply_x: bool = True, apply_y: bool = True) -> int:
        """Implement the `_shift_annotations` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        selected_annotations = self._resolve_selected_annotations(scope=scope, selection_text=selection_text)
        if not selected_annotations:
            return 0

        moved = 0
        try:
            x0, x1 = self._ax.get_xlim()
            x_min = min(float(x0), float(x1))
            x_max = max(float(x0), float(x1))
            x_padding = max(abs(float(shift_x)) * 0.25, (x_max - x_min) * 0.01)
        except Exception:
            x_min = float("-inf")
            x_max = float("inf")
            x_padding = 0.0
        try:
            y0, y1 = self._ax.get_ylim()
            y_min = min(float(y0), float(y1))
            y_max = max(float(y0), float(y1))
            y_padding = max(abs(float(shift_y)) * 0.25, (y_max - y_min) * 0.01)
        except Exception:
            y_min = float("-inf")
            y_max = float("inf")
            y_padding = 0.0

        for ann in selected_annotations:
            try:
                current_x, current_y = ann.get_position()
            except Exception:
                continue
            next_x = float(current_x) + (float(shift_x) if bool(apply_x) else 0.0)
            next_y = float(current_y) + (float(shift_y) if bool(apply_y) else 0.0)
            if math.isfinite(x_min) and math.isfinite(x_max):
                next_x = min(max(next_x, float(x_min + x_padding)), float(x_max - x_padding))
            if math.isfinite(y_min) and math.isfinite(y_max):
                next_y = min(max(next_y, float(y_min + y_padding)), float(y_max - y_padding))
            try:
                ann.set_position((float(next_x), float(next_y)))
                moved += 1
            except Exception:
                continue

        if moved:
            self._apply_numbering(redraw_only=True)
            try:
                self._canvas.draw_idle()
            except Exception:
                pass
        return int(moved)

    def _align_annotations(self, *, scope: str, selection_text: str, horizontal: bool = False, vertical: bool = False) -> Dict[str, int]:
        """Implement the `_align_annotations` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        selected_annotations = self._resolve_selected_annotations(scope=scope, selection_text=selection_text)
        if not selected_annotations:
            return {"moved": 0, "locked": 0, "selected": 0, "movable": 0, "aligned_h": 0, "aligned_v": 0}

        try:
            self._canvas.draw()
        except Exception:
            pass
        to_disp = self._ax.transData.transform

        entries: List[Dict[str, Any]] = []
        locked = 0
        for ann in selected_annotations:
            try:
                current_data_x, current_data_y = ann.get_position()
                anchor_disp = tuple(to_disp((float(current_data_x), float(current_data_y))))
            except Exception:
                continue
            is_locked = self._annotation_is_locked(ann)
            if is_locked:
                locked += 1
            entries.append(
                {
                    "ann": ann,
                    "locked": bool(is_locked),
                    "current_data": (float(current_data_x), float(current_data_y)),
                    "anchor_disp": (float(anchor_disp[0]), float(anchor_disp[1])),
                }
            )

        movable = [entry for entry in entries if not bool(entry.get("locked", False))]
        if not movable:
            return {
                "moved": 0,
                "locked": int(locked),
                "selected": int(len(entries)),
                "movable": 0,
                "aligned_h": 0,
                "aligned_v": 0,
            }

        reference = movable[0]
        reference_data = reference.get("current_data", (0.0, 0.0))
        reference_x = float(reference_data[0])
        reference_y = float(reference_data[1])
        moved = 0
        aligned_h = 0
        aligned_v = 0

        for entry in movable:
            ann = entry.get("ann")
            if ann is None:
                continue
            current_data = entry.get("current_data", (0.0, 0.0))
            target_data_x = float(current_data[0])
            target_data_y = float(current_data[1])
            if bool(vertical):
                target_data_x = float(reference_x)
                aligned_v = 1
            if bool(horizontal):
                target_data_y = float(reference_y)
                aligned_h = 1

            try:
                final_x = float(target_data_x)
                final_y = float(target_data_y)
                ann.set_position((final_x, final_y))
                if abs(final_x - float(current_data[0])) > 1e-9 or abs(final_y - float(current_data[1])) > 1e-9:
                    moved += 1
            except Exception:
                continue

        if moved:
            self._apply_numbering(redraw_only=True)
            try:
                self._canvas.draw_idle()
            except Exception:
                pass

        return {
            "moved": int(moved),
            "locked": int(locked),
            "selected": int(len(entries)),
            "movable": int(len(movable)),
            "aligned_h": int(aligned_h),
            "aligned_v": int(aligned_v),
        }

    def _open_distribute_labels_dialog(self) -> None:
        """Open a file, view, or resource.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        if self._label_distribution_win is not None:
            try:
                if bool(self._label_distribution_win.winfo_exists()):
                    self._label_distribution_win.deiconify()
                    self._label_distribution_win.lift()
                    self._label_distribution_win.focus_force()
                    return
            except Exception:
                pass

        ordered = self._sorted_export_annotations()
        if not ordered:
            try:
                messagebox.showinfo("Distribute Labels", "There are no export labels to distribute.", parent=self)
            except Exception:
                pass
            return

        dlg = tk.Toplevel(self)
        self._label_distribution_win = dlg
        dlg.title("Distribute Labels")
        dlg.transient(self)
        dlg.resizable(False, False)

        scope_var = tk.StringVar(value="all")
        selection_var = tk.StringVar(value="")
        x_step_var = tk.DoubleVar(value=1.0)
        y_step_var = tk.DoubleVar(value=1.0)
        shift_x_var = tk.DoubleVar(value=0.0)
        shift_y_var = tk.DoubleVar(value=0.0)

        frm = ttk.Frame(dlg, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Scope").grid(row=0, column=0, sticky="w")
        scope_box = ttk.Combobox(frm, textvariable=scope_var, values=("all", "selected"), state="readonly", width=18)
        scope_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(frm, text="Selected indices").grid(row=1, column=0, sticky="w", pady=(8, 0))
        selection_entry = ttk.Entry(frm, textvariable=selection_var)
        selection_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Distribution mode").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(frm, text="PowerPoint-style equal center spacing", style="CardHint.TLabel").grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="X spacing (1.0 = default)").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=x_step_var).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Y spacing (1.0 = default)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=y_step_var).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Move X together").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=shift_x_var).grid(row=5, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Move Y together").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=shift_y_var).grid(row=6, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        preview_lines: List[str] = []
        for idx, ann in enumerate(ordered, start=1):
            try:
                label_text = str(self._ann_original_text.get(id(ann), ann.get_text()))
            except Exception:
                label_text = ""
            preview_lines.append(f"{idx}. {label_text}")
        preview_text = tk.Text(frm, width=42, height=min(10, max(4, len(preview_lines))), wrap="word")
        preview_text.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        try:
            preview_text.insert("1.0", "Use left-to-right label indices, for example 1-4,7\nPowerPoint distribution keeps locked labels fixed and spaces movable label centers equally.\n\n" + "\n".join(preview_lines))
            preview_text.configure(state="disabled")
        except Exception:
            pass

        def _apply(*, apply_x: bool, apply_y: bool) -> None:
            """Implement the `_apply` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            try:
                result = self._distribute_annotations(
                    scope=str(scope_var.get() or "all"),
                    selection_text=str(selection_var.get() or ""),
                    x_spacing=float(x_step_var.get()),
                    y_spacing=float(y_step_var.get()),
                    apply_x=bool(apply_x),
                    apply_y=bool(apply_y),
                )
            except ValueError as exc:
                messagebox.showerror("Distribute Labels", str(exc), parent=dlg)
                return
            except Exception as exc:
                messagebox.showerror("Distribute Labels", f"Failed to distribute labels:\n{exc}", parent=dlg)
                return
            try:
                moved = int(result.get("moved", 0)) if isinstance(result, dict) else int(result)
                locked = int(result.get("locked", 0)) if isinstance(result, dict) else 0
                movable = int(result.get("movable", 0)) if isinstance(result, dict) else 0
                axis_x = int(result.get("axis_x", 0)) if isinstance(result, dict) else int(bool(apply_x))
                axis_y = int(result.get("axis_y", 0)) if isinstance(result, dict) else int(bool(apply_y))
                notes: List[str] = [f"Updated {moved} label(s).", f"Locked labels kept fixed: {locked}."]
                if bool(apply_x) and not bool(axis_x):
                    notes.append("X distribution needs at least 3 movable labels.")
                if bool(apply_y) and not bool(axis_y):
                    notes.append("Y distribution needs at least 3 movable labels.")
                notes.append(f"Movable labels in scope: {movable}.")
                messagebox.showinfo("Distribute Labels", "\n".join(notes), parent=dlg)
            except Exception:
                pass

        def _shift(*, apply_x: bool, apply_y: bool) -> None:
            """Implement the `_shift` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            try:
                moved = self._shift_annotations(
                    scope=str(scope_var.get() or "all"),
                    selection_text=str(selection_var.get() or ""),
                    shift_x=float(shift_x_var.get()),
                    shift_y=float(shift_y_var.get()),
                    apply_x=bool(apply_x),
                    apply_y=bool(apply_y),
                )
            except ValueError as exc:
                messagebox.showerror("Distribute Labels", str(exc), parent=dlg)
                return
            except Exception as exc:
                messagebox.showerror("Distribute Labels", f"Failed to move labels:\n{exc}", parent=dlg)
                return
            try:
                messagebox.showinfo("Distribute Labels", f"Moved {moved} label(s).", parent=dlg)
            except Exception:
                pass

        def _align(*, horizontal: bool, vertical: bool) -> None:
            """Implement the `_align` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            try:
                result = self._align_annotations(
                    scope=str(scope_var.get() or "all"),
                    selection_text=str(selection_var.get() or ""),
                    horizontal=bool(horizontal),
                    vertical=bool(vertical),
                )
            except ValueError as exc:
                messagebox.showerror("Distribute Labels", str(exc), parent=dlg)
                return
            except Exception as exc:
                messagebox.showerror("Distribute Labels", f"Failed to align labels:\n{exc}", parent=dlg)
                return
            try:
                moved = int(result.get("moved", 0)) if isinstance(result, dict) else int(result)
                locked = int(result.get("locked", 0)) if isinstance(result, dict) else 0
                movable = int(result.get("movable", 0)) if isinstance(result, dict) else 0
                notes: List[str] = [f"Aligned {moved} label(s).", f"Locked labels kept fixed: {locked}.", f"Movable labels in scope: {movable}."]
                messagebox.showinfo("Distribute Labels", "\n".join(notes), parent=dlg)
            except Exception:
                pass

        def _close() -> None:
            """Close resources and finalize state.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            self._label_distribution_win = None
            try:
                dlg.destroy()
            except Exception:
                pass

        buttons = ttk.Frame(frm)
        buttons.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)
        buttons.columnconfigure(3, weight=1)
        buttons.columnconfigure(4, weight=1)
        buttons.columnconfigure(5, weight=1)
        buttons.columnconfigure(6, weight=1)

        ttk.Button(buttons, text="Apply X", command=lambda: _apply(apply_x=True, apply_y=False)).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(buttons, text="Apply Y", command=lambda: _apply(apply_x=False, apply_y=True)).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(buttons, text="Apply X + Y", command=lambda: _apply(apply_x=True, apply_y=True)).grid(row=0, column=2, sticky="ew", padx=(0, 6))
        ttk.Button(buttons, text="Move X", command=lambda: _shift(apply_x=True, apply_y=False)).grid(row=0, column=3, sticky="ew", padx=(0, 6))
        ttk.Button(buttons, text="Move Y", command=lambda: _shift(apply_x=False, apply_y=True)).grid(row=0, column=4, sticky="ew", padx=(0, 6))
        ttk.Button(buttons, text="Move X + Y", command=lambda: _shift(apply_x=True, apply_y=True)).grid(row=0, column=5, sticky="ew", padx=(0, 6))
        ttk.Button(buttons, text="Close", command=_close).grid(row=0, column=6, sticky="ew")

        align_buttons = ttk.Frame(frm)
        align_buttons.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        align_buttons.columnconfigure(0, weight=1)
        align_buttons.columnconfigure(1, weight=1)
        align_buttons.columnconfigure(2, weight=1)
        ttk.Button(align_buttons, text="Align Horizontal", command=lambda: _align(horizontal=True, vertical=False)).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(align_buttons, text="Align Vertical", command=lambda: _align(horizontal=False, vertical=True)).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(align_buttons, text="Align H + V", command=lambda: _align(horizontal=True, vertical=True)).grid(row=0, column=2, sticky="ew")

        try:
            dlg.protocol("WM_DELETE_WINDOW", _close)
        except Exception:
            pass

    def _build_initial_plot(self) -> None:
        # NOTE: This relies on App-provided attributes/methods; kept identical to original behavior.
        """Build and return composed application state.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
                    offset_step = self._overlay_gap_step(max_global)
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
                            (ln,) = self._ax.plot(
                                rts,
                                y,
                                linewidth=float(self._live_style_snapshot.get("line_width", 1.5) or 1.5),
                                color=col,
                                alpha=float(self._live_style_snapshot.get("line_alpha", 0.94) or 0.94),
                                solid_capstyle=str(self._live_style_snapshot.get("line_capstyle", "round") or "round"),
                                label=label,
                            )
                            self._legend_handles.append(ln)
                            self._legend_labels.append(label)
                            if str(sid) not in self._legend_handle_by_sid:
                                self._legend_handle_by_sid[str(sid)] = ln
                                self._legend_entries.append((str(sid), label))
                        except Exception:
                            self._ax.plot(
                                rts,
                                y,
                                linewidth=float(self._live_style_snapshot.get("line_width", 1.5) or 1.5),
                                color=col,
                                alpha=float(self._live_style_snapshot.get("line_alpha", 0.94) or 0.94),
                                solid_capstyle=str(self._live_style_snapshot.get("line_capstyle", "round") or "round"),
                            )
            else:
                self.title_var.set((self.app.tic_title_var.get() or "TIC (MS1)").strip())
                self.xlabel_var.set(self.app.tic_xlabel_var.get())
                self.ylabel_var.set(self.app.tic_ylabel_var.get())
                rts = self.app._filtered_rts
                tics = self.app._filtered_tics
                if rts is None or tics is None or rts.size == 0:
                    self._ax.text(0.5, 0.5, "No TIC data loaded", ha="center", va="center", transform=self._ax.transAxes)
                else:
                    self._ax.plot(
                        rts,
                        tics,
                        linewidth=float(self._live_style_snapshot.get("line_width", 1.8) or 1.8),
                        color=self._to_hex_color(self._live_style_snapshot.get("plot_color"), "#0F766E"),
                        alpha=float(self._live_style_snapshot.get("line_alpha", 0.99) or 0.99),
                        solid_capstyle=str(self._live_style_snapshot.get("line_capstyle", "round") or "round"),
                    )
                    try:
                        self._ax.fill_between(
                            rts,
                            tics,
                            0.0,
                            color=self._to_hex_color(self._live_style_snapshot.get("fill_color"), "#CDEEE6"),
                            alpha=float(self._live_style_snapshot.get("fill_alpha", 0.24) or 0.24),
                            linewidth=0,
                        )
                    except Exception:
                        pass

        elif self.kind == "uv":
            if hasattr(self.app, "_is_overlay_active") and bool(self.app._is_overlay_active()) and bool(self.app._overlay_show_uv_var.get()):
                base_title = (self.app.uv_title_var.get() or "UV chromatogram").strip()
                self.title_var.set(f"{base_title} — overlay")
                self.xlabel_var.set(self.app.uv_xlabel_var.get())
                self.ylabel_var.set(self.app.uv_ylabel_var.get())

                ids = list(self.app._overlay_dataset_ids())
                name_map = self.app._overlay_display_names(ids)
                mode = str(getattr(self.app, "_overlay_mode_var").get() or "Stacked")
                max_global = 0.0
                for sid in ids:
                    sess = self.app._sessions.get(str(sid))
                    if sess is None:
                        continue
                    uv_id = getattr(sess, "linked_uv_id", None)
                    if not uv_id or str(uv_id) not in self.app._uv_sessions:
                        continue
                    uv_sess = self.app._uv_sessions[str(uv_id)]
                    y = np.asarray(uv_sess.signal, dtype=float)
                    if y.size == 0:
                        continue
                    try:
                        max_global = max(max_global, float(np.max(y)))
                    except Exception:
                        continue
                offset_step = self._overlay_gap_step(max_global)
                any_uv = False
                for i, sid in enumerate(ids):
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
                    if mode == "Offset":
                        y = y + (float(i) * float(offset_step))
                    any_uv = True
                    col = self.app._ensure_overlay_color(str(sid))
                    default_label = str(name_map.get(str(sid), str(sid)))
                    label = str(self._legend_label_override.get(str(sid), default_label))
                    try:
                        (ln,) = self._ax.plot(
                            x,
                            y,
                            linewidth=float(self._live_style_snapshot.get("line_width", 1.4) or 1.4),
                            color=col,
                            alpha=float(self._live_style_snapshot.get("line_alpha", 0.92) or 0.92),
                            solid_capstyle=str(self._live_style_snapshot.get("line_capstyle", "round") or "round"),
                            label=label,
                        )
                        self._legend_handles.append(ln)
                        self._legend_labels.append(label)
                        if str(sid) not in self._legend_handle_by_sid:
                            self._legend_handle_by_sid[str(sid)] = ln
                            self._legend_entries.append((str(sid), label))
                    except Exception:
                        self._ax.plot(
                            x,
                            y,
                            linewidth=float(self._live_style_snapshot.get("line_width", 1.4) or 1.4),
                            color=col,
                            alpha=float(self._live_style_snapshot.get("line_alpha", 0.92) or 0.92),
                            solid_capstyle=str(self._live_style_snapshot.get("line_capstyle", "round") or "round"),
                        )

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
                    self._ax.plot(
                        x,
                        y,
                        linewidth=float(self._live_style_snapshot.get("line_width", 1.65) or 1.65),
                        color=self._to_hex_color(self._live_style_snapshot.get("plot_color"), "#0F8AA6"),
                        alpha=float(self._live_style_snapshot.get("line_alpha", 0.98) or 0.98),
                        solid_capstyle=str(self._live_style_snapshot.get("line_capstyle", "round") or "round"),
                    )
                    try:
                        self._ax.fill_between(
                            x,
                            y,
                            0.0,
                            color=self._to_hex_color(self._live_style_snapshot.get("fill_color"), "#D9F1F7"),
                            alpha=float(self._live_style_snapshot.get("fill_alpha", 0.28) or 0.28),
                            linewidth=0,
                        )
                    except Exception:
                        pass

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
                                    rotation=float(self._annotation_rotation()),
                                    fontsize=fs,
                                    color=(self.label_color_var.get() or "").strip() or self._to_hex_color(self._live_style_snapshot.get("label_color"), "#111111"),
                                    arrowprops={
                                        "arrowstyle": "-",
                                        "lw": 0.95,
                                        "color": (self.label_color_var.get() or "").strip() or self._to_hex_color(self._live_style_snapshot.get("label_color"), "#111111"),
                                        "alpha": 0.9,
                                    },
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
                        offset_step = self._overlay_gap_step(max_global)
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
                                coll = self._ax.vlines(
                                    mz_vals,
                                    base,
                                    y,
                                    linewidth=float(self._live_style_snapshot.get("collection_linewidth", 1.0) or 1.0),
                                    color=col,
                                    alpha=float(self._live_style_snapshot.get("line_alpha", 0.9) or 0.9),
                                    label=label,
                                )
                                self._legend_handles.append(coll)
                                self._legend_labels.append(label)
                                if str(sid) not in self._legend_handle_by_sid:
                                    self._legend_handle_by_sid[str(sid)] = coll
                                    self._legend_entries.append((str(sid), label))
                            except Exception:
                                self._ax.vlines(
                                    mz_vals,
                                    base,
                                    y,
                                    linewidth=float(self._live_style_snapshot.get("collection_linewidth", 1.0) or 1.0),
                                    color=col,
                                    alpha=float(self._live_style_snapshot.get("line_alpha", 0.9) or 0.9),
                                )

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

                                        # Helper function for `nearest_peak` workflow behavior.
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

                    self._ax.vlines(
                        mz,
                        0.0,
                        inten,
                        linewidth=float(self._live_style_snapshot.get("collection_linewidth", 1.0) or 1.0),
                        color=self._to_hex_color(self._live_style_snapshot.get("plot_color"), "#0B5D6B"),
                        alpha=float(self._live_style_snapshot.get("line_alpha", 0.96) or 0.96),
                    )

                    labels_by_key = self.app._collect_labels_for_export(np.asarray(mz, dtype=float), np.asarray(inten, dtype=float))
                    if labels_by_key:
                        mz_a = np.asarray(mz, dtype=float)
                        in_a = np.asarray(inten, dtype=float)
                        if mz_a.size and in_a.size:
                            order = np.argsort(mz_a)
                            mz_s = mz_a[order]
                            in_s = in_a[order]
                            y_off = 0.10 * float(np.max(in_s)) if in_s.size else 1.0

                            # Helper function for `nearest_peak` workflow behavior.
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
        """Implement the `_apply_style_and_limits_impl` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Implement the `_apply_style_and_limits` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._apply_style_and_limits_impl(initial=False)

    def _apply_style_only_impl(self) -> None:
        """Implement the `_apply_style_only_impl` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._apply_live_plot_theme()

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
        self._apply_annotation_orientation()

        try:
            w = float(self.fig_w_var.get())
            h = float(self.fig_h_var.get())
            if w > 0 and h > 0:
                self._fig.set_size_inches(w, h, forward=True)
        except Exception:
            pass

        self._apply_legend()

    def _apply_legend(self) -> None:
        """Implement the `_apply_legend` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Implement the `_install_live_style_traces` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        if bool(getattr(self, "_live_style_traces_installed", False)):
            return
        self._live_style_traces_installed = True

        def _schedule(*_args) -> None:
            """Implement the `_schedule` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            try:
                self._schedule_live_style_apply()
            except Exception:
                pass

        for var in (
            self.title_fs_var,
            self.label_fs_var,
            self.tick_fs_var,
            self.ann_fs_var,
            self.ann_orientation_var,
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

    def _overlay_gap_step(self, max_global: float) -> float:
        """Implement the `_overlay_gap_step` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        try:
            scale = float(self.overlay_gap_var.get() or 0.12)
        except Exception:
            scale = 0.12
        scale = max(0.0, min(5.0, float(scale)))
        if float(max_global) > 0:
            return float(scale) * float(max_global)
        return 1.0 if scale > 0 else 0.0

    def _install_overlay_gap_trace(self) -> None:
        """Implement the `_install_overlay_gap_trace` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        if bool(getattr(self, "_overlay_gap_trace_installed", False)):
            return
        self._overlay_gap_trace_installed = True

        def _schedule(*_args) -> None:
            """Implement the `_schedule` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            try:
                self._schedule_plot_rebuild()
            except Exception:
                pass

        try:
            self.overlay_gap_var.trace_add("write", _schedule)
        except Exception:
            try:
                self.overlay_gap_var.trace("w", _schedule)
            except Exception:
                pass

    def _schedule_plot_rebuild(self) -> None:
        """Prepare plotting data and visual elements.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        try:
            if self._plot_rebuild_job is not None:
                try:
                    self.after_cancel(self._plot_rebuild_job)
                except Exception:
                    pass
                self._plot_rebuild_job = None
        except Exception:
            pass

        try:
            self._plot_rebuild_job = self.after(90, self._rebuild_plot_now)
        except Exception:
            self._plot_rebuild_job = None

    def _rebuild_plot_now(self) -> None:
        """Prepare plotting data and visual elements.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._plot_rebuild_job = None
        lims = {
            "xmin": self.xmin_var.get(),
            "xmax": self.xmax_var.get(),
            "ymin": self.ymin_var.get(),
            "ymax": self.ymax_var.get(),
        }
        try:
            self._build_initial_plot()
            self.xmin_var.set(lims["xmin"])
            self.xmax_var.set(lims["xmax"])
            self.ymin_var.set(lims["ymin"])
            self.ymax_var.set(lims["ymax"])
            self._apply_style_and_limits_impl(initial=False)
        except Exception:
            pass

    def _schedule_live_style_apply(self) -> None:
        """Implement the `_schedule_live_style_apply` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Implement the `_apply_live_style_now` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._live_style_job = None
        # Live updates should never pop error dialogs (e.g., while typing axis limits).
        try:
            self._apply_style_only_impl()
            self._apply_numbering(redraw_only=True)
            self._canvas.draw_idle()
        except Exception:
            pass

    def _label_rt_for_annotation(self, ann) -> str:
        """Implement the `_label_rt_for_annotation` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Implement the `_apply_numbering` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
            bbox = Bbox.from_bounds(
                float(self.tbl_x_var.get()),
                float(self.tbl_y_var.get()),
                float(self.tbl_w_var.get()),
                float(self.tbl_h_var.get()),
            )
            tbl = self._ax.table(
                cellText=rows,
                colLabels=["#", "Label", "RT (min)"],
                colLoc="left",
                cellLoc="left",
                bbox=bbox,
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
        """Save output/state to persistent storage.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Open a file, view, or resource.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
            """Implement the `apply` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
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
            """Implement the `delete` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
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
        """Implement the `_on_press` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Implement the `_on_motion` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
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
        """Implement the `_on_release` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._active_ann = None
