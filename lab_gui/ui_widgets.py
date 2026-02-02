from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import tkinter as tk

from matplotlib.patches import Rectangle


class ToolTip:
    """Global singleton tooltip bubble (does not steal focus)."""

    _tipwin: Optional[tk.Toplevel] = None
    _label: Optional[tk.Label] = None
    _after_id: Optional[str] = None
    _active_widget: Optional[tk.Widget] = None
    _text: str = ""
    _delay_ms: int = 450
    _wrap_px: int = 360
    _last_xy: Optional[Tuple[int, int]] = None

    # Style defaults; can be overridden by the app to match its theme.
    _bg: str = "#ffffe0"
    _fg: str = "#000000"

    @classmethod
    def set_style(cls, *, background: Optional[str] = None, foreground: Optional[str] = None) -> None:
        if background is not None:
            cls._bg = str(background)
        if foreground is not None:
            cls._fg = str(foreground)

    @classmethod
    def attach(cls, widget: tk.Widget, text: str, *, delay_ms: int = 450, wrap_px: int = 360) -> None:
        if widget is None:
            return
        s = (text or "").strip()
        if not s:
            return

        def on_enter(_evt=None) -> None:
            cls._active_widget = widget
            cls._text = s
            cls._delay_ms = int(delay_ms)
            cls._wrap_px = int(wrap_px)
            try:
                cls._last_xy = (int(widget.winfo_pointerx()), int(widget.winfo_pointery()))
            except Exception:
                cls._last_xy = None
            cls._schedule_show()

        def on_leave(_evt=None) -> None:
            cls._cancel_scheduled()
            cls.hide()

        def on_motion(_evt=None) -> None:
            try:
                cls._last_xy = (int(widget.winfo_pointerx()), int(widget.winfo_pointery()))
            except Exception:
                cls._last_xy = None
            # If already visible, reposition.
            if cls._tipwin is not None:
                cls._place()

        def on_press(_evt=None) -> None:
            cls._cancel_scheduled()
            cls.hide()

        try:
            widget.bind("<Enter>", on_enter, add=True)
            widget.bind("<Leave>", on_leave, add=True)
            widget.bind("<Motion>", on_motion, add=True)
            widget.bind("<ButtonPress>", on_press, add=True)
        except Exception:
            pass

    @classmethod
    def _cancel_scheduled(cls) -> None:
        try:
            if cls._after_id is not None and cls._active_widget is not None:
                cls._active_widget.after_cancel(cls._after_id)
        except Exception:
            pass
        cls._after_id = None

    @classmethod
    def _schedule_show(cls) -> None:
        cls._cancel_scheduled()
        w = cls._active_widget
        if w is None:
            return
        try:
            cls._after_id = w.after(int(cls._delay_ms), cls.show)
        except Exception:
            cls._after_id = None

    @classmethod
    def show(cls) -> None:
        cls._after_id = None
        w = cls._active_widget
        if w is None:
            return
        try:
            if not bool(w.winfo_exists()):
                return
        except Exception:
            return

        if cls._tipwin is None:
            try:
                tw = tk.Toplevel(w)
            except Exception:
                return
            cls._tipwin = tw
            try:
                tw.wm_overrideredirect(True)
            except Exception:
                pass
            try:
                tw.attributes("-topmost", True)
            except Exception:
                pass

            # Best-effort: keep from stealing focus on Windows.
            try:
                tw.attributes("-toolwindow", True)
            except Exception:
                pass

            try:
                lbl = tk.Label(
                    tw,
                    text="",
                    justify="left",
                    relief="solid",
                    borderwidth=1,
                    background=cls._bg,
                    foreground=cls._fg,
                    padx=8,
                    pady=6,
                    wraplength=int(cls._wrap_px),
                )
                lbl.pack(fill="both", expand=True)
                cls._label = lbl
            except Exception:
                cls._label = None

        if cls._label is not None:
            try:
                cls._label.configure(text=str(cls._text), wraplength=int(cls._wrap_px))
            except Exception:
                pass

        cls._place()

    @classmethod
    def _place(cls) -> None:
        if cls._tipwin is None or cls._active_widget is None:
            return
        w = cls._active_widget

        try:
            if cls._last_xy is not None:
                px, py = cls._last_xy
            else:
                px, py = (int(w.winfo_pointerx()), int(w.winfo_pointery()))
        except Exception:
            return

        # Offset so we don't cover the pointer.
        x = int(px) + 14
        y = int(py) + 18

        try:
            cls._tipwin.update_idletasks()
            tw_w = int(cls._tipwin.winfo_reqwidth())
            tw_h = int(cls._tipwin.winfo_reqheight())
        except Exception:
            tw_w, tw_h = 280, 80

        try:
            sw = int(w.winfo_screenwidth())
            sh = int(w.winfo_screenheight())
        except Exception:
            sw, sh = 1920, 1080

        pad = 6
        x = max(pad, min(int(x), int(sw - tw_w - pad)))
        y = max(pad, min(int(y), int(sh - tw_h - pad)))

        try:
            cls._tipwin.wm_geometry(f"+{int(x)}+{int(y)}")
        except Exception:
            pass

    @classmethod
    def hide(cls) -> None:
        if cls._tipwin is None:
            return
        try:
            cls._tipwin.destroy()
        except Exception:
            pass
        cls._tipwin = None
        cls._label = None


class MatplotlibNavigator:
    """Lightweight pan/zoom/reset + coordinate readout for TkAgg plots."""

    def __init__(
        self,
        *,
        canvas: Any,
        ax: Optional[Any] = None,
        axes_provider: Optional[Callable[[], Iterable[Any]]] = None,
        status_label: Optional[Union[tk.Label, tk.StringVar, Callable[[str], None]]] = None,
        home_limits_provider: Optional[Callable[[Any], Optional[Tuple[float, float, float, float]]]] = None,
        enable_zoom: bool = True,
        enable_pan: bool = True,
        enable_reset: bool = True,
        enable_coords: bool = True,
        enable_box_zoom: bool = True,
        box_zoom_enabled_cb: Optional[Callable[[], bool]] = None,
        box_click_callback: Optional[Callable[[Any], None]] = None,
        auto_update_home: bool = True,
    ) -> None:
        self.canvas = canvas
        self._fig = getattr(canvas, "figure", None)
        self._ax = ax
        self._axes_provider = axes_provider
        self._status_label = status_label
        self._home_limits_provider = home_limits_provider
        self._enable_zoom = bool(enable_zoom)
        self._enable_pan = bool(enable_pan)
        self._enable_reset = bool(enable_reset)
        self._enable_coords = bool(enable_coords)
        self._enable_box_zoom = bool(enable_box_zoom)
        self._box_zoom_enabled_cb = box_zoom_enabled_cb
        self._box_click_callback = box_click_callback
        self._auto_update_home = bool(auto_update_home)

        self._home_limits: Dict[Any, Tuple[float, float, float, float]] = {}
        self._pan_active = False
        self._pan_ax: Optional[Any] = None
        self._pan_start_xy: Optional[Tuple[float, float]] = None
        self._pan_start_lim: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self._pan_start_pix: Optional[Tuple[float, float]] = None
        self._pan_anchor_data: Optional[Tuple[float, float]] = None
        self._box_active = False
        self._box_ax: Optional[Any] = None
        self._box_start_data: Optional[Tuple[float, float]] = None
        self._box_start_pix: Optional[Tuple[float, float]] = None
        self._box_rect: Optional[Any] = None
        self._box_pending = False
        self._last_axes: Optional[Any] = None
        self._last_home_update_t: float = 0.0
        self._cid: List[int] = []

    def attach(self) -> None:
        try:
            self._cid.append(self.canvas.mpl_connect("scroll_event", self._on_scroll))
            self._cid.append(self.canvas.mpl_connect("button_press_event", self._on_press))
            self._cid.append(self.canvas.mpl_connect("button_release_event", self._on_release))
            self._cid.append(self.canvas.mpl_connect("motion_notify_event", self._on_motion))
            self._cid.append(self.canvas.mpl_connect("key_press_event", self._on_key))
            if self._auto_update_home:
                self._cid.append(self.canvas.mpl_connect("draw_event", self._on_draw))
        except Exception:
            pass

    def is_box_pending(self) -> bool:
        return bool(self._box_pending)

        # Tk-level wheel fallback (Windows/Linux)
        try:
            w = self.canvas.get_tk_widget()
            w.bind("<MouseWheel>", self._on_tk_mousewheel, add=True)
            w.bind("<Button-4>", self._on_tk_mousewheel, add=True)
            w.bind("<Button-5>", self._on_tk_mousewheel, add=True)
        except Exception:
            pass

    def update_home_from_artists(self) -> None:
        for ax in self._iter_axes():
            lim = self._home_limits_provider(ax) if self._home_limits_provider is not None else None
            if lim is None:
                lim = self._compute_limits_from_artists(ax)
            if lim is None:
                continue
            self._home_limits[ax] = lim

    def _iter_axes(self) -> Iterable[Any]:
        if self._axes_provider is not None:
            try:
                return list(self._axes_provider())
            except Exception:
                return []
        if self._ax is not None:
            return [self._ax]
        if self._fig is not None:
            try:
                return list(self._fig.axes)
            except Exception:
                return []
        return []

    def _on_draw(self, _evt=None) -> None:
        # Throttle to avoid heavy recompute during drag
        now = time.time()
        if now - float(self._last_home_update_t) < 0.20:
            return
        self._last_home_update_t = float(now)
        self.update_home_from_artists()

    def _set_status(self, text: str) -> None:
        lbl = self._status_label
        if lbl is None:
            return
        try:
            if isinstance(lbl, tk.StringVar):
                lbl.set(str(text))
                return
        except Exception:
            pass
        try:
            if isinstance(lbl, tk.Label):
                lbl.configure(text=str(text))
                return
        except Exception:
            pass
        try:
            if callable(lbl):
                lbl(str(text))
        except Exception:
            pass

    def _format_coord(self, x: float, y: float) -> str:
        def _fmt(v: float) -> str:
            if v == 0.0:
                return "0"
            av = abs(float(v))
            if av >= 1e4 or av < 1e-3:
                return f"{float(v):.3g}"
            return f"{float(v):.4g}"

        return f"x={_fmt(x)}, y={_fmt(y)}"

    def _on_motion(self, event) -> None:
        ax = event.inaxes if event is not None else None
        if ax is not None:
            self._last_axes = ax

        # Cursor style
        try:
            if not self._pan_active:
                if ax is not None:
                    self.canvas.get_tk_widget().configure(cursor="crosshair")
                else:
                    self.canvas.get_tk_widget().configure(cursor="")
        except Exception:
            pass

        if self._box_pending and not self._box_active:
            self._maybe_begin_box(event)
        if self._box_active:
            self._update_box(event)

        if self._enable_coords:
            if ax is None or event.xdata is None or event.ydata is None:
                self._set_status("")
            else:
                self._set_status(self._format_coord(float(event.xdata), float(event.ydata)))

        if not self._pan_active:
            return
        if self._pan_ax is None or ax != self._pan_ax:
            return
        if self._pan_start_pix is None or self._pan_start_lim is None or self._pan_anchor_data is None:
            return
        try:
            curx = float(event.x)
            cury = float(event.y)
        except Exception:
            return

        try:
            inv = ax.transData.inverted()
            x0, y0 = self._pan_start_pix
            sx, sy = inv.transform((x0, y0))
            cx, cy = inv.transform((curx, cury))
        except Exception:
            return

        dx = float(cx) - float(sx)
        dy = float(cy) - float(sy)

        (xlim0, xlim1), (ylim0, ylim1) = self._pan_start_lim
        try:
            ax.set_xlim(float(xlim0) - dx, float(xlim1) - dx)
            ax.set_ylim(float(ylim0) - dy, float(ylim1) - dy)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_press(self, event) -> None:
        if event is None:
            return
        ax = event.inaxes
        if ax is not None:
            self._last_axes = ax

        try:
            w = self.canvas.get_tk_widget()
            w.focus_set()
        except Exception:
            pass

        if self._enable_reset and bool(getattr(event, "dblclick", False)):
            self.reset_view(ax)
            return

        if event.button == 1:
            self._start_box_pending(event)

        if not self._enable_pan:
            return
        if ax is None:
            return
        if event.button not in (2, 3):
            return
        if event.button == 3 and self._is_over_text(ax, event):
            return

        try:
            xpix = float(event.x)
            ypix = float(event.y)
        except Exception:
            return

        self._pan_active = True
        self._pan_ax = ax
        self._pan_start_xy = None
        self._pan_start_pix = (float(xpix), float(ypix))
        try:
            self._pan_start_lim = (ax.get_xlim(), ax.get_ylim())
        except Exception:
            self._pan_start_lim = None
        try:
            self._pan_anchor_data = self._event_xy(event, ax)
        except Exception:
            self._pan_anchor_data = None

        try:
            self.canvas.get_tk_widget().configure(cursor="hand2")
        except Exception:
            pass

    def _can_box_zoom(self) -> bool:
        if not bool(self._enable_box_zoom):
            return False
        if self._box_zoom_enabled_cb is not None:
            try:
                return bool(self._box_zoom_enabled_cb())
            except Exception:
                return False
        return True

    def _start_box_pending(self, event) -> None:
        if not self._can_box_zoom():
            return
        ax = event.inaxes
        if ax is None:
            return
        if self._is_over_text(ax, event):
            return
        xdata, ydata = self._event_xy(event, ax)
        if xdata is None or ydata is None:
            return
        try:
            xpix = float(event.x)
            ypix = float(event.y)
        except Exception:
            return

        self._box_pending = True
        self._box_active = False
        self._box_ax = ax
        self._box_start_data = (float(xdata), float(ydata))
        self._box_start_pix = (float(xpix), float(ypix))

    def _maybe_begin_box(self, event) -> None:
        if not self._box_pending or self._box_active:
            return
        if self._box_ax is None or self._box_start_pix is None or self._box_start_data is None:
            return
        if event is None or event.inaxes != self._box_ax:
            return
        try:
            dx = abs(float(event.x) - float(self._box_start_pix[0]))
            dy = abs(float(event.y) - float(self._box_start_pix[1]))
        except Exception:
            return
        if max(dx, dy) < 5.0:
            return

        self._box_active = True
        self._box_pending = False
        ax = self._box_ax
        x0, y0 = self._box_start_data
        try:
            if self._box_rect is not None:
                self._box_rect.remove()
        except Exception:
            pass
        self._box_rect = None

        try:
            self._box_rect = Rectangle(
                (float(x0), float(y0)),
                0.0,
                0.0,
                fill=True,
                facecolor="#99c2ff",
                edgecolor="#3366cc",
                linewidth=1.0,
                alpha=0.25,
            )
            ax.add_patch(self._box_rect)
            self.canvas.draw_idle()
        except Exception:
            self._box_rect = None

    def _update_box(self, event) -> None:
        if not self._box_active or self._box_ax is None or self._box_start_data is None:
            return
        ax = self._box_ax
        if event.inaxes != ax:
            return
        xdata, ydata = self._event_xy(event, ax)
        if xdata is None or ydata is None:
            return
        x0, y0 = self._box_start_data
        x1, y1 = float(xdata), float(ydata)
        try:
            if self._box_rect is not None:
                self._box_rect.set_x(min(x0, x1))
                self._box_rect.set_y(min(y0, y1))
                self._box_rect.set_width(abs(x1 - x0))
                self._box_rect.set_height(abs(y1 - y0))
                self.canvas.draw_idle()
        except Exception:
            pass

    def _finish_box(self, event) -> None:
        ax = self._box_ax
        self._box_active = False
        self._box_ax = None
        start_pix = self._box_start_pix
        start_data = self._box_start_data
        self._box_start_pix = None
        self._box_start_data = None

        try:
            if self._box_rect is not None:
                self._box_rect.remove()
        except Exception:
            pass
        self._box_rect = None

        if ax is None or event is None or start_pix is None or start_data is None:
            return
        try:
            dx = abs(float(event.x) - float(start_pix[0]))
            dy = abs(float(event.y) - float(start_pix[1]))
        except Exception:
            return
        if max(dx, dy) < 5.0:
            return

        x1, y1 = self._event_xy(event, ax)
        if x1 is None or y1 is None:
            return
        x0, y0 = start_data

        try:
            if bool(ax.xaxis_inverted()):
                ax.set_xlim(max(x0, x1), min(x0, x1))
            else:
                ax.set_xlim(min(x0, x1), max(x0, x1))
            ax.set_ylim(min(y0, y1), max(y0, y1))
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_release(self, _event) -> None:
        if self._box_active:
            self._finish_box(_event)
        elif self._box_pending:
            self._box_pending = False
            self._box_ax = None
            self._box_start_data = None
            self._box_start_pix = None
            try:
                if self._box_click_callback is not None:
                    self._box_click_callback(_event)
            except Exception:
                pass
        if not self._pan_active:
            return
        self._pan_active = False
        self._pan_ax = None
        self._pan_start_xy = None
        self._pan_start_lim = None
        self._pan_start_pix = None
        self._pan_anchor_data = None
        try:
            self.canvas.get_tk_widget().configure(cursor="")
        except Exception:
            pass

    def _on_key(self, event) -> None:
        if not self._enable_reset:
            return
        if event is None:
            return
        try:
            key = str(event.key or "").lower()
        except Exception:
            key = ""
        if key == "r":
            self.reset_view(self._last_axes or event.inaxes)

    def _on_scroll(self, event) -> None:
        if not self._enable_zoom:
            return
        if event is None or event.inaxes is None:
            return
        ax = event.inaxes
        self._last_axes = ax
        try:
            step = int(getattr(event, "step", 0) or 0)
        except Exception:
            step = 0
        if step == 0:
            return

        try:
            key = str(getattr(event, "key", "") or "").lower()
        except Exception:
            key = ""
        x_only = ("shift" in key) and ("control" not in key)
        y_only = ("control" in key) and ("shift" not in key)

        xdata, ydata = self._event_xy(event, ax)
        if xdata is None or ydata is None:
            return
        self._zoom_at(ax, float(xdata), float(ydata), step=step, x_only=x_only, y_only=y_only)

    def _on_tk_mousewheel(self, event) -> None:
        if not self._enable_zoom:
            return
        ax = self._pick_axes_from_tk(event)
        if ax is None:
            return

        # Determine scroll direction
        step = 0
        try:
            if getattr(event, "num", None) == 4:
                step = 1
            elif getattr(event, "num", None) == 5:
                step = -1
        except Exception:
            pass
        if step == 0:
            try:
                delta = int(getattr(event, "delta", 0) or 0)
                step = 1 if delta > 0 else (-1 if delta < 0 else 0)
            except Exception:
                step = 0
        if step == 0:
            return

        xdata, ydata = self._tk_event_data(event, ax)
        if xdata is None or ydata is None:
            return

        state = int(getattr(event, "state", 0) or 0)
        shift = bool(state & 0x0001)
        ctrl = bool(state & 0x0004)
        x_only = bool(shift and not ctrl)
        y_only = bool(ctrl and not shift)
        self._zoom_at(ax, float(xdata), float(ydata), step=step, x_only=x_only, y_only=y_only)

    def _zoom_at(self, ax: Any, x: float, y: float, *, step: int, x_only: bool, y_only: bool) -> None:
        scale = 1.0 / 1.15 if step > 0 else 1.15
        try:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        except Exception:
            return

        def _zoom_lim(lim: Tuple[float, float], center: float) -> Tuple[float, float]:
            lo, hi = float(lim[0]), float(lim[1])
            inv = lo > hi
            if inv:
                lo, hi = hi, lo
            span = max(1e-12, float(hi - lo))
            new_span = span * float(scale)
            if new_span <= 0:
                return lim
            c = float(center)
            new_lo = c - (c - lo) * float(scale)
            new_hi = c + (hi - c) * float(scale)
            if inv:
                return (new_hi, new_lo)
            return (new_lo, new_hi)

        try:
            if not y_only:
                ax.set_xlim(_zoom_lim(xlim, x))
            if not x_only:
                ax.set_ylim(_zoom_lim(ylim, y))
            self.canvas.draw_idle()
        except Exception:
            pass

    def reset_view(self, ax: Optional[Any] = None) -> None:
        if ax is None:
            ax = self._last_axes
        if ax is None:
            return
        lim = self._home_limits.get(ax)
        if lim is None:
            lim = self._compute_limits_from_artists(ax)
        if lim is None:
            return
        x0, x1, y0, y1 = lim
        try:
            if bool(ax.xaxis_inverted()):
                ax.set_xlim(max(x0, x1), min(x0, x1))
            else:
                ax.set_xlim(min(x0, x1), max(x0, x1))
            ax.set_ylim(min(y0, y1), max(y0, y1))
            self.canvas.draw_idle()
        except Exception:
            pass

    def _event_xy(self, event, ax: Any) -> Tuple[Optional[float], Optional[float]]:
        if event is None:
            return (None, None)
        if event.xdata is not None and event.ydata is not None:
            return (float(event.xdata), float(event.ydata))
        try:
            inv = ax.transData.inverted()
            xdata, ydata = inv.transform((float(event.x), float(event.y)))
            return (float(xdata), float(ydata))
        except Exception:
            return (None, None)

    def _pick_axes_from_tk(self, _event) -> Optional[Any]:
        if self._last_axes is not None:
            return self._last_axes
        axes = list(self._iter_axes())
        return axes[0] if axes else None

    def _tk_event_data(self, event, ax: Any) -> Tuple[Optional[float], Optional[float]]:
        try:
            x = float(event.x)
            y = float(event.y)
        except Exception:
            return (None, None)
        try:
            inv = ax.transData.inverted()
            xdata, ydata = inv.transform((x, y))
            return (float(xdata), float(ydata))
        except Exception:
            return (None, None)

    def _compute_limits_from_artists(self, ax: Any) -> Optional[Tuple[float, float, float, float]]:
        xs: List[float] = []
        ys: List[float] = []

        # Lines
        for ln in list(getattr(ax, "lines", []) or []):
            try:
                if not bool(getattr(ln, "get_visible")()):
                    continue
                x = getattr(ln, "get_xdata")()
                y = getattr(ln, "get_ydata")()
                if x is None or y is None:
                    continue
                xs.extend([float(v) for v in x if v is not None])
                ys.extend([float(v) for v in y if v is not None])
            except Exception:
                continue

        # Collections (e.g., vlines, scatter)
        for coll in list(getattr(ax, "collections", []) or []):
            try:
                if not bool(getattr(coll, "get_visible")()):
                    continue
            except Exception:
                pass
            try:
                if hasattr(coll, "get_segments"):
                    segs = coll.get_segments()
                    for seg in list(segs or []):
                        for pt in list(seg or []):
                            try:
                                xs.append(float(pt[0]))
                                ys.append(float(pt[1]))
                            except Exception:
                                continue
            except Exception:
                pass
            try:
                if hasattr(coll, "get_offsets"):
                    offs = coll.get_offsets()
                    for pt in list(offs or []):
                        try:
                            xs.append(float(pt[0]))
                            ys.append(float(pt[1]))
                        except Exception:
                            continue
            except Exception:
                pass

        if not xs or not ys:
            return None

        try:
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
        except Exception:
            return None

        # Pad
        x_span = float(x1 - x0)
        y_span = float(y1 - y0)
        if x_span == 0:
            x_span = max(1e-6, abs(float(x0)) * 0.02)
        if y_span == 0:
            y_span = max(1e-6, abs(float(y0)) * 0.02)
        pad_x = x_span * 0.04
        pad_y = y_span * 0.04
        return (float(x0 - pad_x), float(x1 + pad_x), float(y0 - pad_y), float(y1 + pad_y))

    def _is_over_text(self, ax: Any, event) -> bool:
        try:
            texts = list(getattr(ax, "texts", []) or [])
        except Exception:
            texts = []
        for txt in texts:
            try:
                contains, _ = txt.contains(event)
            except Exception:
                contains = False
            if contains:
                return True
        return False
