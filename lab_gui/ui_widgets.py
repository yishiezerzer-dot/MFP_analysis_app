from __future__ import annotations

from typing import Optional, Tuple

import tkinter as tk


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
