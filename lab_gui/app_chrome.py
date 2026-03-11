"""Reusable top-bar ("app chrome") for the MFP lab analysis Tkinter app.

This module provides ``AppChrome`` – a thin wrapper that draws a modern top
bar above an existing content widget (typically a ``ttk.Notebook``).  It
does **not** touch any tab layout or analysis logic.
"""
from __future__ import annotations

import tkinter as tk
from typing import Any, Callable, Optional

try:
    import ttkbootstrap as tb
    import tkinter.ttk as ttk_native
    ttk: Any = tb
except ImportError:
    import tkinter.ttk as ttk_native  # type: ignore[no-redef]
    ttk: Any = ttk_native  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------
_BG_BAR = "#173632"
_BG_BAR_LIGHT = "#214742"
_TEXT_BAR = "#F8F7F3"
_BADGE_BG = "#0F5B52"
_BADGE_TEXT = "#EAF3F0"

_STATUS_READY_BG = "#DCEBE3"
_STATUS_READY_FG = "#1A5A42"
_STATUS_BUSY_BG = "#F0E1D1"
_STATUS_BUSY_FG = "#8B5128"
_STATUS_ERROR_BG = "#F4D9D5"
_STATUS_ERROR_FG = "#9C2F26"
_META_TEXT = "#D8E6E2"
_META_TEXT_MUTED = "#A8BFBA"
_EDGE = "#0F2B28"
_ACTION_BG = "#244A45"


class AppChrome(tk.Frame):
    """Top bar that sits above the main content area.

    Usage::

        chrome = AppChrome(root)
        chrome.grid(row=0, column=0, sticky="ew")  # top bar row
        content.grid(row=1, column=0, sticky="nsew")  # existing content
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_view: Optional[Callable[[], None]] = None,
        on_help: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent, bg=_BG_BAR, height=38)
        self.grid_propagate(True)
        self.columnconfigure(1, weight=1)  # spacer stretches

        self._on_view = on_view
        self._on_help = on_help

        # ---- Left: compact badge only ----
        left = tk.Frame(self, bg=_BG_BAR)
        left.grid(row=0, column=0, sticky="w", padx=(16, 0), pady=6)

        badge = tk.Label(
            left,
            text="MFP",
            bg=_BADGE_BG,
            fg=_BADGE_TEXT,
            font=("Segoe UI Semibold", 9, "bold"),
            padx=10,
            pady=4,
            relief="flat",
        )
        badge.pack(side=tk.LEFT)

        # ---- Center: spacer only; module context already exists in the content shell ----
        center = tk.Frame(self, bg=_BG_BAR)
        center.grid(row=0, column=1, sticky="ew", padx=0, pady=0)
        center.columnconfigure(0, weight=1)

        self._context_primary_var = tk.StringVar(value="LCMS")
        self._context_secondary_var = tk.StringVar(value="Ready")

        # ---- Right: status pill + optional quick actions ----
        right = tk.Frame(self, bg=_BG_BAR)
        right.grid(row=0, column=2, sticky="e", padx=(0, 16), pady=6)

        self._status_pill = tk.Label(
            right,
            text="  SYSTEM READY  ",
            bg=_STATUS_READY_BG,
            fg=_STATUS_READY_FG,
            font=("Segoe UI Semibold", 8, "bold"),
            padx=10,
            pady=5,
            relief="flat",
        )
        self._status_pill.pack(side=tk.LEFT)

        self._btn_view = None
        self._btn_help = None
        if self._on_view is not None or self._on_help is not None:
            sep = tk.Frame(right, bg=_BG_BAR_LIGHT, width=1)
            sep.pack(side=tk.LEFT, fill="y", padx=12, pady=4)

            if self._on_view is not None:
                self._btn_view = self._chrome_button(right, "View", self._fire_view)
                self._btn_view.pack(side=tk.LEFT)

            if self._on_help is not None:
                self._btn_help = self._chrome_button(right, "Help", self._fire_help)
                self._btn_help.pack(side=tk.LEFT, padx=(6, 0))

        # Subtle bottom edge for depth
        _edge = tk.Frame(self, bg=_EDGE, height=1)
        _edge.grid(row=1, column=0, columnspan=3, sticky="ew")

    # ------------------------------------------------------------------
    # Status pill API
    # ------------------------------------------------------------------
    def set_status_ready(self, text: str = "SYSTEM READY") -> None:
        self._set_pill(text, _STATUS_READY_BG, _STATUS_READY_FG)

    def set_status_busy(self, text: str = "BUSY") -> None:
        self._set_pill(text, _STATUS_BUSY_BG, _STATUS_BUSY_FG)

    def set_status_error(self, text: str = "ERROR") -> None:
        self._set_pill(text, _STATUS_ERROR_BG, _STATUS_ERROR_FG)

    def set_context(self, primary: str, secondary: str = "") -> None:
        try:
            self._context_primary_var.set(str(primary or ""))
            self._context_secondary_var.set(str(secondary or ""))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _set_pill(self, text: str, bg: str, fg: str) -> None:
        try:
            self._status_pill.configure(text=f"  {text}  ", bg=bg, fg=fg)
        except Exception:
            pass

    def _chrome_button(self, parent: tk.Misc, text: str, command: Callable[[], None]) -> tk.Label:
        """Flat label styled as a clickable button on the top bar."""
        btn = tk.Label(
            parent,
            text=text,
            bg=_ACTION_BG,
            fg=_TEXT_BAR,
            font=("Segoe UI Semibold", 8),
            cursor="hand2",
            padx=10,
            pady=5,
        )
        btn.configure(takefocus=True)
        btn.bind("<Button-1>", lambda _e: command())
        btn.bind("<Return>", lambda _e: command())
        btn.bind("<space>", lambda _e: command())
        btn.bind("<Enter>", lambda _e: btn.configure(bg=_BG_BAR_LIGHT))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=_ACTION_BG))
        return btn

    def _fire_view(self) -> None:
        if self._on_view is not None:
            try:
                self._on_view()
            except Exception:
                pass

    def _fire_help(self) -> None:
        if self._on_help is not None:
            try:
                self._on_help()
            except Exception:
                pass
