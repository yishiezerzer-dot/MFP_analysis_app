"""Centralised ttkbootstrap styling helpers for the MFP lab analysis app.

All functions are safe no-ops on failure so they never break the app.
Import and call them where styling is needed – they only add visual polish,
they do **not** move widgets, change callbacks, or alter layout structure.
"""
from __future__ import annotations

import tkinter as tk
from typing import Any, cast

try:
    import ttkbootstrap as tb
    import tkinter.ttk as ttk_native
except ImportError:
    tb = None  # type: ignore[assignment]
    import tkinter.ttk as ttk_native

ttk: Any = tb if tb is not None else ttk_native

_SHELL_BG = "#E9EFEC"
_SURFACE_BG = "#F7F3EC"
_SURFACE_ALT = "#FCFAF6"
_INK = "#12201E"
_MUTED = "#60706C"
_BORDER = "#D3DBD7"
_BORDER_SOFT = "#E2E7E4"
_ACCENT = "#0E5A52"
_ACCENT_DEEP = "#0A443F"
_ACCENT_SOFT = "#D7E7E1"
_WARM = "#A96A3C"
_WARM_SOFT = "#F1E1D1"


# ---------------------------------------------------------------------------
# Global style configuration (call once after theme is applied)
# ---------------------------------------------------------------------------
def apply_global_styles(root: tk.Misc) -> None:
    """Apply default padding / font tweaks to Notebook, Treeview, buttons, etc.

    Safe to call multiple times; later calls just overwrite.
    """
    try:
        style = ttk.Style(root)
    except Exception:
        return

    try:
        cast(Any, root).configure(background=_SHELL_BG)
    except Exception:
        pass

    try:
        style.configure("TFrame", background=_SHELL_BG)
        style.configure("AppContent.TFrame", background=_SHELL_BG)
        style.configure("ModuleHost.TFrame", background=_SHELL_BG)
        style.configure("ShellPanel.TFrame", background=_SURFACE_BG)
        style.configure("Surface.TFrame", background=_SURFACE_ALT, borderwidth=0, relief="flat")
    except Exception:
        pass

    # ---- Notebook tabs: more padding, cleaner look ----
    try:
        style.configure("TNotebook", tabmargins=(6, 6, 6, 0), borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(14, 8),
        )
        style.map(
            "TNotebook.Tab",
            padding=[("selected", (14, 8))],
        )
    except Exception:
        pass

    try:
        style.configure(
            "Content.TNotebook",
            background=_SHELL_BG,
            borderwidth=0,
            tabmargins=(10, 8, 10, 0),
        )
        style.configure(
            "Content.TNotebook.Tab",
            padding=(20, 10),
            font=("Segoe UI Semibold", 10),
            foreground=_MUTED,
            background=_SURFACE_BG,
            borderwidth=0,
        )
        style.map(
            "Content.TNotebook.Tab",
            background=[("selected", _SURFACE_ALT), ("active", _SURFACE_ALT)],
            foreground=[("selected", _ACCENT_DEEP), ("active", _ACCENT_DEEP)],
            expand=[("selected", (0, 0, 0, 0))],
        )
    except Exception:
        pass

    # ---- Treeview: readable rows ----
    try:
        style.configure(
            "Treeview",
            rowheight=32,
            background=_SURFACE_ALT,
            fieldbackground=_SURFACE_ALT,
            foreground=_INK,
            borderwidth=0,
            relief="flat",
        )
        style.map(
            "Treeview",
            background=[("selected", _ACCENT_SOFT)],
            foreground=[("selected", _INK)],
        )
    except Exception:
        pass

    try:
        import tkinter.font as tkfont
        hf = tkfont.nametofont("TkDefaultFont").copy()
        hf.configure(weight="bold")
        style.configure(
            "Treeview.Heading",
            font=hf,
            padding=(10, 7),
            foreground=_ACCENT_DEEP,
            background=_SURFACE_BG,
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Treeview.Heading",
            background=[("active", _SURFACE_ALT)],
            foreground=[("active", _ACCENT_DEEP)],
        )
    except Exception:
        pass

    # ---- Buttons: generous padding to accentuate theme rounding ----
    try:
        style.configure("TButton", padding=(14, 8), font=("Segoe UI Semibold", 9))
        style.configure("TMenubutton", padding=(14, 8), font=("Segoe UI Semibold", 9))
        style.configure("TEntry", padding=(10, 7))
        style.configure("TCombobox", padding=(10, 7))
        style.configure("TCheckbutton", padding=(2, 2))
    except Exception:
        pass

    # ---- LabelFrame: lighter border, consistent padding ----
    try:
        style.configure("TLabelframe", relief="flat", borderwidth=1, background=_SURFACE_BG)
        style.configure("TLabelframe.Label", font=("Segoe UI Semibold", 9), foreground=_ACCENT_DEEP)
        style.configure("Card.TLabelframe", relief="flat", borderwidth=1, background=_SURFACE_BG)
        style.configure("Card.TLabelframe.Label", font=("Segoe UI Semibold", 9), foreground=_ACCENT_DEEP)
    except Exception:
        pass

    # ---- Separator: subtle ----
    try:
        style.configure("TSeparator", borderwidth=1)
    except Exception:
        pass

    # ---- Status bar custom style ----
    try:
        style.configure(
            "StatusBar.TLabel",
            relief="flat",
            padding=(14, 8),
            font=("Segoe UI", 9),
            foreground=_ACCENT_DEEP,
            background=_SURFACE_BG,
        )
    except Exception:
        pass

    # ---- Semantic label styles (replace inline foreground= colours) ----
    try:
        style.configure("Muted.TLabel", foreground=_MUTED, background=_SHELL_BG)
        style.configure("Danger.TLabel", foreground="#B42318", background=_SHELL_BG)
        style.configure("Info.TLabel", foreground=_ACCENT, background=_SHELL_BG)
        style.configure("Success.TLabel", foreground="#1D6A50", background=_SHELL_BG)
        style.configure("Warning.TLabel", foreground=_WARM, background=_SHELL_BG)
        style.configure("CardTitle.TLabel", foreground=_INK, background=_SURFACE_BG, font=("Segoe UI Semibold", 10))
        style.configure("CardHint.TLabel", foreground=_MUTED, background=_SURFACE_BG, font=("Segoe UI", 9))
        style.configure("CardMeta.TLabel", foreground=_MUTED, background=_SURFACE_BG, font=("Segoe UI", 9))
        style.configure("ToolbarLabel.TLabel", foreground=_ACCENT_DEEP, background=_SURFACE_ALT, font=("Segoe UI Semibold", 9))
        style.configure("ToolbarHint.TLabel", foreground=_MUTED, background=_SURFACE_ALT, font=("Segoe UI", 9))
        style.configure("ToolbarEyebrow.TLabel", foreground=_MUTED, background=_SURFACE_ALT, font=("Segoe UI", 8, "bold"))
        style.configure("WorkflowReady.TLabel", foreground="#1D6A50", background="#D9ECE3", padding=(12, 5), anchor="center")
        style.configure("WorkflowIdle.TLabel", foreground="#5C6865", background="#E7ECEA", padding=(12, 5), anchor="center")
        style.configure("AppEyebrow.TLabel", foreground=_MUTED, background=_SURFACE_BG, font=("Segoe UI", 8, "bold"))
        style.configure("CardStatus.TLabel", foreground=_WARM, background=_WARM_SOFT, padding=(10, 4), font=("Segoe UI Semibold", 8))
    except Exception:
        pass

    # ---- Section title styles (replaces scattered font=("TkDefaultFont", N, "bold")) ----
    try:
        import tkinter.font as tkfont
        _sec = tkfont.nametofont("TkDefaultFont").copy()
        _sec.configure(family="Segoe UI Semibold", size=12, weight="bold")
        style.configure("SectionTitle.TLabel", font=_sec)
        _sub = tkfont.nametofont("TkDefaultFont").copy()
        _sub.configure(family="Segoe UI Semibold", size=10, weight="bold")
        style.configure("SubSection.TLabel", font=_sub)
    except Exception:
        pass

    # ---- Raw Tk widget defaults (Listbox, Text) via option database ----
    try:
        root.option_add("*Listbox.font", "TkDefaultFont")
        root.option_add("*Listbox.relief", "flat")
        root.option_add("*Listbox.borderWidth", "1")
        root.option_add("*Listbox.highlightThickness", "0")
        root.option_add("*Listbox.background", _SURFACE_ALT)
        root.option_add("*Listbox.foreground", _INK)
        root.option_add("*Listbox.selectBackground", _ACCENT)
        root.option_add("*Listbox.selectForeground", "#ffffff")
        root.option_add("*Text.font", "TkDefaultFont")
        root.option_add("*Text.relief", "flat")
        root.option_add("*Text.borderWidth", "1")
        root.option_add("*Text.highlightThickness", "1")
        root.option_add("*Text.background", _SURFACE_ALT)
        root.option_add("*Text.foreground", _INK)
        root.option_add("*Text.highlightColor", _BORDER)
        root.option_add("*Text.highlightBackground", _BORDER_SOFT)
        root.option_add("*Text.selectBackground", _ACCENT)
        root.option_add("*Text.selectForeground", "#ffffff")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-widget bootstyle helpers (ttkbootstrap specific)
# ---------------------------------------------------------------------------
def style_primary(btn: Any) -> None:
    """Mark a button as a primary action (Open / Load / Add)."""
    _apply_bootstyle(btn, "primary")


def style_success(btn: Any) -> None:
    """Mark a button as a success/go action (Run / Apply / Export)."""
    _apply_bootstyle(btn, "success")


def style_danger(btn: Any) -> None:
    """Mark a button as a destructive action (Remove / Clear / Delete)."""
    _apply_bootstyle(btn, "danger-outline")


def style_secondary(btn: Any) -> None:
    """Mark a button as a secondary / settings action."""
    _apply_bootstyle(btn, "secondary-outline")


def style_toolbar(btn: Any) -> None:
    """Compact outline button for toolbar rows."""
    _apply_bootstyle(btn, "secondary-outline")


def style_card_frame(frame: Any) -> None:
    """Give a frame a subtle card-like appearance (relief + padding)."""
    try:
        frame.configure(relief="flat", borderwidth=1, padding=(12, 10), style="ShellPanel.TFrame")
    except Exception:
        try:
            frame.configure(relief="flat", bd=1)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
def _apply_bootstyle(widget: Any, bootstyle: str) -> None:
    """Best-effort: set ttkbootstrap bootstyle on *widget*."""
    try:
        widget.configure(bootstyle=bootstyle)  # type: ignore[call-overload]
    except Exception:
        pass
