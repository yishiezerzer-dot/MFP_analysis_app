"""Reusable PlotCard container for matplotlib FigureCanvasTkAgg widgets.

A PlotCard is a styled ``ttk.Frame`` with three rows:
    0. **header** – title label (left) + optional status label (right)
    1. **body**   – the main plot area (canvas goes here, weight=1)
    2. **footer** – toolbar / coord label row

It does NOT create or own the matplotlib Figure/Canvas; callers create
those and grid them directly inside ``PlotCard.body``.  PlotCard only
provides the container structure, consistent padding, a subtle border,
and a ``<Configure>`` resize handler that calls ``draw_idle()`` on the
canvas when the size changes.

Usage in any tab::

    card = PlotCard(parent, title="FTIR")
    card.grid(row=..., column=..., sticky="nsew")
    # create fig, canvas, toolbar, coord_label as before but parent them
    # to card.body instead of a bare ttk.Frame
    canvas_widget.grid(row=0, column=0, sticky="nsew")
    toolbar.grid(row=1, column=0, sticky="ew")
    coord_label.grid(row=2, column=0, sticky="ew")
    # register canvas for resize handling:
    card.register_canvas(canvas)
"""
from __future__ import annotations

import tkinter as tk
from typing import Any, Optional

try:
    import ttkbootstrap as tb
    import tkinter.ttk as ttk_native

    ttk: Any = tb
except ImportError:
    import tkinter.ttk as ttk_native  # type: ignore[no-redef]

    ttk: Any = ttk_native  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Brand colours (keep in sync with app.py / app_chrome.py)
# ---------------------------------------------------------------------------
_CARD_BG = "#F8F4EE"
_HEADER_FG = "#12322F"
_STATUS_FG = "#8B5128"
_BORDER_COLOR = "#D3DBD7"
_SHADOW_COLOR = "#DFE7E3"
_CORNER_RADIUS = 18
_SHADOW_OFFSET = 5


# ---------------------------------------------------------------------------
# Canvas helper – rounded rectangle
# ---------------------------------------------------------------------------
def _draw_rounded_rect(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    **kw: Any,
) -> int:
    """Draw a rounded rectangle on *canvas* via a smooth polygon.

    Duplicate control-points keep the straight edges crisp while
    single corner-points produce gentle curves.
    """
    r = max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [
        x1 + r, y1,     x1 + r, y1,
        x2 - r, y1,     x2 - r, y1,
        x2, y1,
        x2, y1 + r,     x2, y1 + r,
        x2, y2 - r,     x2, y2 - r,
        x2, y2,
        x2 - r, y2,     x2 - r, y2,
        x1 + r, y2,     x1 + r, y2,
        x1, y2,
        x1, y2 - r,     x1, y2 - r,
        x1, y1 + r,     x1, y1 + r,
        x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


class PlotCard(ttk.Frame):
    """Styled card container for a matplotlib plot with rounded corners and drop shadow."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        title: str = "",
        status_text: str = "",
        show_header: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(parent, **kw)

        # ---- outer styling: invisible frame; canvas draws rounded rect ----
        try:
            self.configure(relief="flat", borderwidth=0, padding=(0, 0, 0, 0))
        except Exception:
            try:
                self.configure(relief="flat", borderwidth=0)
            except Exception:
                pass

        self.columnconfigure(0, weight=1)
        # row 0 = header, row 1 = body, row 2 = footer (optional)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        # -- Background canvas for rounded rectangle + drop shadow --
        self._card_canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self._card_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        try:
            self._card_canvas.tk.call("lower", str(self._card_canvas))
        except Exception:
            pass
        # Match parent / theme background outside the rounded rect
        try:
            _s = ttk.Style()
            _pbg = (
                getattr(getattr(_s, "colors", None), "bg", None)
                or _s.lookup("TFrame", "background")
                or "#E9EFEC"
            )
            self._card_canvas.configure(bg=str(_pbg))
        except Exception:
            pass
        self._card_canvas.bind("<Configure>", self._on_card_bg_configure)
        self._card_bg_redraw_id: Optional[str] = None

        # ---- header ----
        self._header_frame = ttk.Frame(self)
        if show_header:
            self._header_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        self._header_frame.columnconfigure(0, weight=1)

        self._eyebrow_label = ttk.Label(
            self._header_frame,
            text="ANALYSIS VIEW",
            style="AppEyebrow.TLabel",
        )
        self._eyebrow_label.grid(row=0, column=0, sticky="w")

        self._title_var = tk.StringVar(value=str(title))
        self._title_label = ttk.Label(
            self._header_frame,
            textvariable=self._title_var,
            font=("Segoe UI Semibold", 12),
        )
        self._title_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
        try:
            self._title_label.configure(foreground=_HEADER_FG)
        except Exception:
            pass

        self._status_var = tk.StringVar(value=str(status_text))
        self._status_label = ttk.Label(
            self._header_frame,
            textvariable=self._status_var,
            font=("Segoe UI Semibold", 8),
            style="CardStatus.TLabel",
        )
        self._status_label.grid(row=0, column=1, rowspan=2, sticky="e", padx=(8, 0))
        try:
            self._status_label.configure(foreground=_STATUS_FG)
        except Exception:
            pass

        # ---- body (callers grid their canvas here) ----
        self._body = ttk.Frame(self, style="Surface.TFrame")
        self._body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(4, 14))
        self._body.columnconfigure(0, weight=1)
        self._body.rowconfigure(0, weight=1)
        self._body.rowconfigure(1, weight=0)
        self._body.rowconfigure(2, weight=0)

        # ---- internal refs ----
        self._canvas: Any = None   # FigureCanvasTkAgg
        self._resize_after_id: Optional[str] = None

    # -- public properties --------------------------------------------------
    @property
    def body(self) -> Any:
        """The inner frame where the canvas/toolbar/coord label are placed."""
        return self._body

    @property
    def title_var(self) -> tk.StringVar:
        return self._title_var

    @property
    def status_var(self) -> tk.StringVar:
        return self._status_var

    # -- public methods -----------------------------------------------------
    def set_title(self, text: str) -> None:
        self._title_var.set(str(text))

    def set_status(self, text: str) -> None:
        self._status_var.set(str(text))

    # -- rounded card background -------------------------------------------
    def _on_card_bg_configure(self, event: Any = None) -> None:
        """Throttled redraw of the rounded-card background."""
        if self._card_bg_redraw_id is not None:
            try:
                self.after_cancel(self._card_bg_redraw_id)
            except Exception:
                pass
        self._card_bg_redraw_id = self.after(30, self._redraw_card_bg)

    def _redraw_card_bg(self) -> None:
        """Draw the rounded rectangle + shadow on the background canvas."""
        self._card_bg_redraw_id = None
        c = getattr(self, "_card_canvas", None)
        if c is None:
            return
        c.delete("card_bg")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 20 or h < 20:
            return
        r = _CORNER_RADIUS
        off = _SHADOW_OFFSET
        # Drop shadow (offset, soft grey)
        _draw_rounded_rect(
            c, off, off, w - 1, h - 1, r,
            fill=_SHADOW_COLOR, outline=_SHADOW_COLOR, tags="card_bg",
        )
        # Main card face
        _draw_rounded_rect(
            c, 0, 0, w - off - 1, h - off - 1, r,
            fill=_CARD_BG, outline=_BORDER_COLOR, tags="card_bg",
        )
        try:
            c.create_line(18, 58, max(18, w - off - 18), 58, fill=_BORDER_COLOR, width=1, tags="card_bg")
        except Exception:
            pass

    def register_canvas(self, canvas: Any) -> None:
        """Register the FigureCanvasTkAgg for automatic resize handling.

        Binds ``<Configure>`` on the canvas widget and calls
        ``fig.set_size_inches(..., forward=False)`` + ``canvas.draw_idle()``
        on resize.  Throttled to at most once per 50 ms to avoid flickering.
        """
        self._canvas = canvas
        try:
            canvas.get_tk_widget().bind("<Configure>", self._on_canvas_configure, add=True)
        except Exception:
            pass

    # -- internal -----------------------------------------------------------
    def _on_canvas_configure(self, event: Any = None) -> None:
        """Throttled resize handler – matches the Data Studio pattern."""
        if self._resize_after_id is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.after(50, self._do_resize)

    def _do_resize(self) -> None:
        self._resize_after_id = None
        canvas = self._canvas
        if canvas is None:
            return
        try:
            w = canvas.get_tk_widget()
            width = int(w.winfo_width())
            height = int(w.winfo_height())
            if width < 20 or height < 20:
                return
            fig = getattr(canvas, "figure", None)
            if fig is None:
                return
            dpi = float(fig.get_dpi())
            fig.set_size_inches(width / dpi, height / dpi, forward=False)
            canvas.draw_idle()
        except Exception:
            pass
