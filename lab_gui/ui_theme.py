"""Centralised ttkbootstrap styling helpers for the MFP lab analysis app.

All functions are safe no-ops on failure so they never break the app.
Import and call them where styling is needed – they only add visual polish,
they do **not** move widgets, change callbacks, or alter layout structure.
"""
from __future__ import annotations

import tkinter as tk
from typing import Any

try:
    import ttkbootstrap as tb
    import tkinter.ttk as ttk_native
except ImportError:
    tb = None  # type: ignore[assignment]
    import tkinter.ttk as ttk_native

ttk: Any = tb if tb is not None else ttk_native


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

    # ---- Notebook tabs: more padding, cleaner look ----
    try:
        style.configure("TNotebook", tabmargins=(4, 4, 4, 0))
        style.configure(
            "TNotebook.Tab",
            padding=(14, 6),
        )
        style.map(
            "TNotebook.Tab",
            padding=[("selected", (14, 8))],
        )
    except Exception:
        pass

    # ---- Treeview: readable rows ----
    try:
        style.configure(
            "Treeview",
            rowheight=30,
            background="#FBFDFD",
            fieldbackground="#FBFDFD",
            foreground="#0F172A",
            borderwidth=0,
            relief="flat",
        )
        style.map(
            "Treeview",
            background=[("selected", "#DCEFEA")],
            foreground=[("selected", "#0F172A")],
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
            padding=(8, 6),
            foreground="#334155",
            background="#F3F7F8",
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Treeview.Heading",
            background=[("active", "#EAF2F1")],
            foreground=[("active", "#0F172A")],
        )
    except Exception:
        pass

    # ---- Buttons: generous padding to accentuate theme rounding ----
    try:
        style.configure("TButton", padding=(14, 8))
        style.configure("TMenubutton", padding=(14, 8))
        style.configure("TEntry", padding=(8, 6))
        style.configure("TCombobox", padding=(8, 6))
        style.configure("TCheckbutton", padding=(2, 2))
    except Exception:
        pass

    # ---- LabelFrame: lighter border, consistent padding ----
    try:
        style.configure("TLabelframe", relief="groove", borderwidth=1)
        style.configure("TLabelframe.Label", font=("Segoe UI Semibold", 9))
        style.configure("Card.TLabelframe", relief="flat", borderwidth=1)
        style.configure("Card.TLabelframe.Label", font=("Segoe UI Semibold", 9))
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
            padding=(10, 5),
            font=("Segoe UI", 9),
        )
        style.configure(
            "Surface.TFrame",
            relief="flat",
            borderwidth=0,
        )
    except Exception:
        pass

    # ---- Semantic label styles (replace inline foreground= colours) ----
    try:
        style.configure("Muted.TLabel", foreground="#6B7280")
        style.configure("Danger.TLabel", foreground="#DC2626")
        style.configure("Info.TLabel", foreground="#0b5394")
        style.configure("Success.TLabel", foreground="#047857")
        style.configure("Warning.TLabel", foreground="#B45309")
        style.configure("CardTitle.TLabel", foreground="#111827", font=("Segoe UI Semibold", 10))
        style.configure("CardHint.TLabel", foreground="#6B7280", font=("Segoe UI", 9))
        style.configure("ToolbarLabel.TLabel", foreground="#334155", font=("Segoe UI Semibold", 9))
        style.configure("ToolbarHint.TLabel", foreground="#64748B", font=("Segoe UI", 9))
        style.configure("WorkflowReady.TLabel", foreground="#065F46", background="#D1FAE5", padding=(10, 4), anchor="center")
        style.configure("WorkflowIdle.TLabel", foreground="#475569", background="#E2E8F0", padding=(10, 4), anchor="center")
    except Exception:
        pass

    # ---- Section title styles (replaces scattered font=("TkDefaultFont", N, "bold")) ----
    try:
        import tkinter.font as tkfont
        _sec = tkfont.nametofont("TkDefaultFont").copy()
        _sec.configure(size=12, weight="bold")
        style.configure("SectionTitle.TLabel", font=_sec)
        _sub = tkfont.nametofont("TkDefaultFont").copy()
        _sub.configure(size=10, weight="bold")
        style.configure("SubSection.TLabel", font=_sub)
    except Exception:
        pass

    # ---- Raw Tk widget defaults (Listbox, Text) via option database ----
    try:
        root.option_add("*Listbox.font", "TkDefaultFont")
        root.option_add("*Listbox.relief", "flat")
        root.option_add("*Listbox.borderWidth", "1")
        root.option_add("*Listbox.highlightThickness", "0")
        root.option_add("*Listbox.selectBackground", "#04504A")
        root.option_add("*Listbox.selectForeground", "#ffffff")
        root.option_add("*Text.font", "TkDefaultFont")
        root.option_add("*Text.relief", "flat")
        root.option_add("*Text.borderWidth", "1")
        root.option_add("*Text.highlightThickness", "1")
        root.option_add("*Text.highlightColor", "#d1d5db")
        root.option_add("*Text.selectBackground", "#04504A")
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
    _apply_bootstyle(btn, "outline")


def style_card_frame(frame: Any) -> None:
    """Give a frame a subtle card-like appearance (relief + padding)."""
    try:
        frame.configure(relief="groove", borderwidth=1, padding=(8, 6))
    except Exception:
        try:
            frame.configure(relief="groove", bd=1)
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
