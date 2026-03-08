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
# Brand colours (mirrors the constants in app.py)
# ---------------------------------------------------------------------------
_BG_BAR = "#04504A"          # PRIMARY_TEAL
_BG_BAR_LIGHT = "#05756C"    # slightly lighter for hover
_TEXT_BAR = "#FFFFFF"
_BADGE_BG = "#05312E"        # SECONDARY_TEAL

_STATUS_READY_BG = "#16A34A"  # green-600
_STATUS_READY_FG = "#FFFFFF"
_STATUS_BUSY_BG = "#EAB308"   # yellow-500
_STATUS_BUSY_FG = "#1C1917"
_STATUS_ERROR_BG = "#DC2626"  # red-600
_STATUS_ERROR_FG = "#FFFFFF"
_META_TEXT = "#D1FAE5"
_META_TEXT_MUTED = "#A7F3D0"


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
        super().__init__(parent, bg=_BG_BAR, height=48)
        self.grid_propagate(True)
        self.columnconfigure(1, weight=1)  # spacer stretches

        self._on_view = on_view
        self._on_help = on_help

        # ---- Left: badge + label ----
        left = tk.Frame(self, bg=_BG_BAR)
        left.grid(row=0, column=0, sticky="w", padx=(10, 0), pady=4)

        badge = tk.Label(
            left,
            text=" MFP ",
            bg=_BADGE_BG,
            fg=_TEXT_BAR,
            font=("Segoe UI", 9, "bold"),
            padx=6,
            pady=1,
            relief="flat",
        )
        badge.pack(side=tk.LEFT)

        title = tk.Label(
            left,
            text="MFP Lab Tool",
            bg=_BG_BAR,
            fg=_TEXT_BAR,
            font=("Segoe UI", 10),
        )
        title.pack(side=tk.LEFT, padx=(8, 0))

        # ---- Center: active module + short context ----
        center = tk.Frame(self, bg=_BG_BAR)
        center.grid(row=0, column=1, sticky="ew", padx=18, pady=4)
        center.columnconfigure(0, weight=1)

        self._context_primary_var = tk.StringVar(value="LCMS")
        self._context_secondary_var = tk.StringVar(value="Ready")

        primary = tk.Label(
            center,
            textvariable=self._context_primary_var,
            bg=_BG_BAR,
            fg=_TEXT_BAR,
            font=("Segoe UI Semibold", 10),
            anchor="w",
        )
        primary.grid(row=0, column=0, sticky="w")

        secondary = tk.Label(
            center,
            textvariable=self._context_secondary_var,
            bg=_BG_BAR,
            fg=_META_TEXT_MUTED,
            font=("Segoe UI", 8),
            anchor="w",
        )
        secondary.grid(row=1, column=0, sticky="w")

        # ---- Right: status pill + optional quick actions ----
        right = tk.Frame(self, bg=_BG_BAR)
        right.grid(row=0, column=2, sticky="e", padx=(0, 10), pady=4)

        self._status_pill = tk.Label(
            right,
            text="  SYSTEM READY  ",
            bg=_STATUS_READY_BG,
            fg=_STATUS_READY_FG,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2,
            relief="flat",
        )
        self._status_pill.pack(side=tk.LEFT)

        self._btn_view = None
        self._btn_help = None
        if self._on_view is not None or self._on_help is not None:
            sep = tk.Frame(right, bg=_BG_BAR_LIGHT, width=1)
            sep.pack(side=tk.LEFT, fill="y", padx=10, pady=2)

            if self._on_view is not None:
                self._btn_view = self._chrome_button(right, "View", self._fire_view)
                self._btn_view.pack(side=tk.LEFT)

            if self._on_help is not None:
                self._btn_help = self._chrome_button(right, "Help", self._fire_help)
                self._btn_help.pack(side=tk.LEFT, padx=(6, 0))

        # Subtle bottom edge for depth
        _edge = tk.Frame(self, bg="#03403B", height=1)
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
            bg=_BG_BAR,
            fg=_TEXT_BAR,
            font=("Segoe UI", 9),
            cursor="hand2",
            padx=8,
            pady=2,
        )
        btn.configure(takefocus=True)
        btn.bind("<Button-1>", lambda _e: command())
        btn.bind("<Return>", lambda _e: command())
        btn.bind("<space>", lambda _e: command())
        btn.bind("<Enter>", lambda _e: btn.configure(bg=_BG_BAR_LIGHT))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=_BG_BAR))
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
