from __future__ import annotations

import copy
import csv
import json
import math
import colorsys
import os
import queue
import time
import threading
import traceback
import uuid
import datetime
import warnings
import subprocess
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog

import ttkbootstrap as tb
import tkinter.ttk as ttk_native

ttk = tb
ttk.LabelFrame = ttk_native.LabelFrame

import numpy as np
import pandas as pd
from pyteomics import mzml

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from matplotlib import colors as mcolors
from matplotlib import cm

# Refactor step A: models extracted into lab_gui/*_model.py (UI-free).
from lab_gui.lcms_model import (
    SpectrumMeta,
    CustomLabel,
    LabelingSettings,
    UVLabelState,
    MzMLSession,
    UVSession,
    LCMSDataset,
    OverlaySession,
    _safe_float,
    _extract_ms_level,
    _extract_rt_minutes,
    _extract_polarity,
    _spectrum_id,
)
from lab_gui.ftir_model import FTIRBondAnnotation, FTIRDataset, FTIRWorkspace, FTIRDatasetKey, StyleState, OverlayGroup
from lab_gui.microscopy_model import MicroscopyDataset, MicroscopyWorkspace
from lab_gui.microscopy_tab import MicroscopyView
from lab_gui.plate_reader_model import PlateReaderDataset
from lab_gui.plate_reader_view import PlateReaderView
from lab_gui.data_studio_view import DataStudioView
from lab_gui.settings import load_settings, save_settings

# Refactor step B: IO extracted into lab_gui/lcms_io.py (UI-free).
from lab_gui.lcms_io import MzMLTICIndex, infer_uv_columns, preview_dataframe_rows, parse_uv_arrays

# Refactor step C: FTIR parsing extracted into lab_gui/ftir_io.py (UI-free).
from lab_gui.ftir_io import _try_parse_float_pair, _parse_ftir_xy_only, _parse_ftir_xy_numpy, _ftir_parse_for_executor

# Refactor step D: shared UI widgets extracted into lab_gui/ui_widgets.py.
from lab_gui.ui_widgets import ToolTip, MatplotlibNavigator

# Refactor step E: ExportEditor extracted into lab_gui/export_editor.py.
from lab_gui.export_editor import ExportEditor

# LCMS polymer matching engine (pure analysis helpers).
import lab_gui.lcms_polymer_match as poly_match


try:
    from lab_gui.ftir_analysis import FTIRPeak, format_peak_label, pick_peaks, preprocess_spectrum
except Exception:
    FTIRPeak = None  # type: ignore

    def preprocess_spectrum(*args, **kwargs):  # type: ignore
        raise ImportError("lab_gui/ftir_analysis.py is required for FTIR peak picking")

    def pick_peaks(*args, **kwargs):  # type: ignore
        raise ImportError("lab_gui/ftir_analysis.py is required for FTIR peak picking")

    def format_peak_label(*args, **kwargs):  # type: ignore
        raise ImportError("lab_gui/ftir_analysis.py is required for FTIR peak picking")


# Set True only when debugging FTIR performance/freezes.
FTIR_DEBUG = False


APP_NAME = "MFP lab analysis tool"
APP_VERSION = "0.0"

WORKSPACE_SCHEMA_VERSION = 1


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, set):
        return sorted(list(obj))
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        text = json.dumps(data, indent=2, ensure_ascii=False, default=json_default)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise


def build_lcms_workspace_dict(app: "App") -> Dict[str, Any]:
    try:
        app._save_active_session_state()
    except Exception:
        pass

    def _encode_annotations(
        custom_labels_by_spectrum: Dict[str, List[CustomLabel]],
        spec_label_overrides: Dict[str, Dict[Tuple[str, float], Optional[str]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for spec_id, items in (custom_labels_by_spectrum or {}).items():
            rows: List[Dict[str, Any]] = []
            for it in items or []:
                try:
                    rows.append({"mz": float(it.mz), "text": str(it.label), "kind": "custom"})
                except Exception:
                    continue
            if rows:
                out[str(spec_id)] = rows

        for spec_id, overrides in (spec_label_overrides or {}).items():
            rows = out.setdefault(str(spec_id), [])
            for (kind, mz_key), val in (overrides or {}).items():
                try:
                    row: Dict[str, Any] = {"mz": float(mz_key), "text": ("" if val is None else str(val)), "kind": str(kind)}
                    if val is None:
                        row["suppressed"] = True
                    rows.append(row)
                except Exception:
                    continue
        return out

    rt_unit = "minutes"
    try:
        rt_unit = str(app.rt_unit_var.get() or "minutes")
    except Exception:
        rt_unit = "minutes"
    if rt_unit not in ("minutes", "seconds"):
        rt_unit = "minutes"

    pol_default = "all"
    try:
        pol_default = str(app.polarity_var.get() or "all")
    except Exception:
        pol_default = "all"
    if pol_default not in ("all", "positive", "negative"):
        pol_default = "all"

    mzml_files: List[Dict[str, Any]] = []
    uv_files: List[Dict[str, Any]] = []
    annotations_by_mzml: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for uv_id in list(getattr(app, "_uv_order", []) or []):
        uv_sess = (getattr(app, "_uv_sessions", {}) or {}).get(str(uv_id))
        if uv_sess is None:
            continue
        uv_files.append(
            {
                "path": str(uv_sess.path),
                "load_order": int(getattr(uv_sess, "load_order", 0) or 0),
            }
        )

    for sid in list(getattr(app, "_session_order", []) or []):
        sess = (getattr(app, "_sessions", {}) or {}).get(str(sid))
        if sess is None:
            continue
        pol = sess.last_polarity_filter or pol_default
        if pol not in ("all", "positive", "negative"):
            pol = pol_default
        linked_uv_path = None
        try:
            if getattr(sess, "linked_uv_id", None) and str(sess.linked_uv_id) in getattr(app, "_uv_sessions", {}):
                linked_uv_path = str(app._uv_sessions[str(sess.linked_uv_id)].path)
        except Exception:
            linked_uv_path = None
        mzml_files.append(
            {
                "path": str(sess.path),
                "polarity": str(pol),
                "rt_unit": str(rt_unit),
                "display_name": str(getattr(sess, "display_name", "")),
                "load_order": int(getattr(sess, "load_order", 0) or 0),
                "last_scan_index": (None if sess.last_scan_index is None else int(sess.last_scan_index)),
                "last_selected_rt_min": (None if sess.last_selected_rt_min is None else float(sess.last_selected_rt_min)),
                "linked_uv_path": linked_uv_path,
            }
        )
        annotations_by_mzml[str(sess.path)] = _encode_annotations(
            getattr(sess, "custom_labels_by_spectrum", {}) or {},
            getattr(sess, "spec_label_overrides", {}) or {},
        )

    active_index = 0
    try:
        if app._active_session_id in (getattr(app, "_session_order", []) or []):
            active_index = int((getattr(app, "_session_order", []) or []).index(app._active_session_id))
    except Exception:
        active_index = 0

    linked_uv: Optional[Dict[str, Any]] = None
    try:
        if app._active_session_id and app._active_session_id in app._sessions:
            sess = app._sessions[app._active_session_id]
            if getattr(sess, "linked_uv_id", None) and str(sess.linked_uv_id) in app._uv_sessions:
                uv_sess = app._uv_sessions[str(sess.linked_uv_id)]
                linked_uv = {
                    "mzml_path": str(sess.path),
                    "uv_csv_path": str(uv_sess.path),
                    "uv_ms_offset": float(getattr(app, "_uv_ms_rt_offset_min", 0.0) or 0.0),
                }
    except Exception:
        linked_uv = None

    monomers_text = ""
    try:
        monomers_text = str(app.poly_monomers_text_var.get() or "")
    except Exception:
        monomers_text = ""
    monomers = [ln.strip() for ln in monomers_text.splitlines() if ln.strip()]

    poly_common: Dict[str, Any] = {
        "bond_delta": float(getattr(app, "poly_bond_delta_var", 0.0).get() if hasattr(app, "poly_bond_delta_var") else -18.010565),
        "extra_delta": float(getattr(app, "poly_extra_delta_var", 0.0).get() if hasattr(app, "poly_extra_delta_var") else 0.0),
        "adduct_mass": float(getattr(app, "poly_adduct_mass_var", 1.007276).get() if hasattr(app, "poly_adduct_mass_var") else 1.007276),
        "cluster_adduct_mass": float(getattr(app, "poly_cluster_adduct_mass_var", -1.007276).get() if hasattr(app, "poly_cluster_adduct_mass_var") else -1.007276),
        "adduct_na": bool(getattr(app, "poly_adduct_na_var", False).get() if hasattr(app, "poly_adduct_na_var") else False),
        "adduct_k": bool(getattr(app, "poly_adduct_k_var", False).get() if hasattr(app, "poly_adduct_k_var") else False),
        "adduct_cl": bool(getattr(app, "poly_adduct_cl_var", False).get() if hasattr(app, "poly_adduct_cl_var") else False),
        "adduct_formate": bool(getattr(app, "poly_adduct_formate_var", False).get() if hasattr(app, "poly_adduct_formate_var") else False),
        "adduct_acetate": bool(getattr(app, "poly_adduct_acetate_var", False).get() if hasattr(app, "poly_adduct_acetate_var") else False),
        "charges": str(getattr(app, "poly_charges_var", "1").get() if hasattr(app, "poly_charges_var") else "1"),
        "decarb": bool(getattr(app, "poly_decarb_enabled_var", False).get() if hasattr(app, "poly_decarb_enabled_var") else False),
        "oxid": bool(getattr(app, "poly_oxid_enabled_var", False).get() if hasattr(app, "poly_oxid_enabled_var") else False),
        "cluster": bool(getattr(app, "poly_cluster_enabled_var", False).get() if hasattr(app, "poly_cluster_enabled_var") else False),
        "min_rel_int": float(getattr(app, "poly_min_rel_int_var", 0.01).get() if hasattr(app, "poly_min_rel_int_var") else 0.01),
    }

    try:
        current_scan_index = None if app._current_scan_index is None else int(app._current_scan_index)
    except Exception:
        current_scan_index = None

    tic_settings = {
        "show_tic": bool(getattr(app, "show_tic_var", True).get() if hasattr(app, "show_tic_var") else True),
        "x_min": str(getattr(app, "tic_xlim_min_var", "").get() if hasattr(app, "tic_xlim_min_var") else ""),
        "x_max": str(getattr(app, "tic_xlim_max_var", "").get() if hasattr(app, "tic_xlim_max_var") else ""),
        "y_min": str(getattr(app, "tic_ylim_min_var", "").get() if hasattr(app, "tic_ylim_min_var") else ""),
        "y_max": str(getattr(app, "tic_ylim_max_var", "").get() if hasattr(app, "tic_ylim_max_var") else ""),
        "title": str(getattr(app, "tic_title_var", "").get() if hasattr(app, "tic_title_var") else ""),
    }

    annotations: Dict[str, List[Dict[str, Any]]] = {}
    try:
        if app._active_session_id and app._active_session_id in app._sessions:
            active_path = str(app._sessions[app._active_session_id].path)
            annotations = annotations_by_mzml.get(active_path, {})
    except Exception:
        annotations = {}

    return {
        "schema": "LCMS_WORKSPACE",
        "version": 1,
        "saved_at": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "mzml_files": mzml_files,
        "active_mzml_index": int(active_index),
        "linked_uv": linked_uv,
        "uv_files": uv_files,
        "annotations": annotations,
        "annotations_by_mzml": annotations_by_mzml,
        "current_scan_index": current_scan_index,
        "tic_settings": tic_settings,
        "polymer_settings": {
            "enabled": bool(getattr(app, "poly_enabled_var", False).get() if hasattr(app, "poly_enabled_var") else False),
            "monomers": list(monomers),
            "max_dp": int(getattr(app, "poly_max_dp_var", 12).get() if hasattr(app, "poly_max_dp_var") else 12),
            "tolerance": float(getattr(app, "poly_tol_value_var", 0.02).get() if hasattr(app, "poly_tol_value_var") else 0.02),
            "tolerance_unit": str(getattr(app, "poly_tol_unit_var", "Da").get() if hasattr(app, "poly_tol_unit_var") else "Da"),
            "positive_mode": dict(poly_common),
            "negative_mode": dict(poly_common),
        },
    }

# UI colors / lab palette
LIGHT_TEAL = "#5C7C75"
BG_LIGHT = LIGHT_TEAL
TEXT_DARK = "#111827"
TEXT_MUTED = "#6B7280"

# Brand accents (used across ttk + Matplotlib styling)
DIVIDER = "#E5E7EB"
PRIMARY_TEAL = "#04504A"
SECONDARY_TEAL = "#05312E"
DEEP_MAROON = "#7F1D1D"
ACCENT_ORANGE = "#F59E0B"
ACCENT_MAGENTA = "#C026D3"

# Keep tooltip styling consistent with the existing app theme.
ToolTip.set_style(background=BG_LIGHT, foreground=TEXT_DARK)


GUIDE: Dict[str, str] = {
        "Export & Plots": """\

Save TIC Plot… / Save Spectrum Plot… / Save UV Plot…
    What they do
        • Save the currently displayed plot to an image file.

Export Editor (for TIC/Spectrum/UV)
    What it does
        • Opens a dedicated export window where you can refine annotation placement.
        • Supports dragging labels and (optionally) replacing labels with numbers and inserting a table.
    Notes
        • Export Editor edits are export-only (they do not modify your underlying label rules/settings).
""",
        "Peak Annotation": """\
Annotate Peaks…  (Tools → Annotate Peaks…, Ctrl+P)

What it does
    • Controls the automatic peak labels drawn on the MS spectrum.
    • You can set:
            - Top N peaks (by intensity)
            - Minimum relative intensity (fraction of max peak intensity)
    • After applying, the spectrum is redrawn using those rules.

Under the hood (auto peak labeling)
    • The app computes the maximum intensity in the current spectrum.
    • It keeps peaks with intensity ≥ (min_rel × max_intensity).
    • From those, it keeps the top-N peaks by intensity (if Top N > 0).
    • Each selected peak is labeled by its m/z value (unless overridden/suppressed).

Editing labels
    • Right-click (or double-click) a label to edit or delete it.
    • Deleting a label suppresses it for that spectrum.
""",
        "Custom Labels": """\
Custom Labels…  (Tools → Custom Labels…, Ctrl+L)

What it does
    • Lets you add your own labels at specific m/z values for the current spectrum.
    • Options typically include:
            - exact m/z (label stays at that value), or
            - “snap to nearest peak” (the label attaches to the nearest observed peak).

Inputs / outputs
    • Input: label text + target m/z (+ optional snap behavior)
    • Output: label annotations on the spectrum plot.

Notes
    • Custom labels are stored per spectrum (by spectrum_id).
    • You can clear all custom labels for a spectrum from the dialog.
""",
        "Polymer / Reaction Match": """\
Polymer Match…  (Tools → Polymer Match…, Ctrl+M)

What it does
    • Searches for polymer/reaction series that match observed peaks.
    • You configure monomers, polymerization delta per bond, adducts, charges, tolerance, and max DP.
    • When enabled, matching labels are drawn on the spectrum.

    Inputs (what each option means in the math)
        Monomers
        • Enter one monomer per line.
        • Allowed formats:
            - name,mass
            - name mass
            - mass   (auto-named M1, M2, …)
        • The matcher supports any number of monomers (more monomers + higher DP can be a much larger search).

        Max total monomers (DP)
        • DP is the maximum total count across all monomers:
            total = c1 + c2 (+ c3)
            require 1 ≤ total ≤ DP
        • The search enumerates all integer compositions up to DP (so bigger DP = many more candidates).

        Per-bond delta (polymerization)
        • This is applied once per covalent bond formed.
        • For a chain with total monomers = total, the number of bonds is (total−1).

        Extra delta (once per chain)
        • A one-time mass shift applied to the whole chain after polymerization.
        • Use this for end-group chemistry or any fixed modification that should happen once per oligomer.

        H adduct mass (Da)
        • Used to convert neutral mass to m/z:
            m/z_pred = (M_neutral + adduct_mass) / z
        • Convenience behavior: if the entered adduct is close to |1.007276| and polarity is known,
          the app automatically chooses the sign (+ for positive mode, − for negative mode).

        Also match: +Na/+K (positive) or +Cl/+HCOO (negative)
        • Adds extra adduct options in addition to the H adduct above.
        • Each adduct produces its own predicted m/z values and can generate its own labels.

        Charges (comma-separated)
        • Charges are positive integers (e.g., “1” or “1,2,3”).
        • Every candidate mass is evaluated for each selected charge z.

        Tolerance (Da or ppm)
        • Da mode:
            accept if |m/z_obs − m/z_pred| ≤ tol_Da
        • ppm mode:
            tol_Da = |m/z_pred| × (ppm × 1e−6)
            accept if |m/z_obs − m/z_pred| ≤ tol_Da

        Min peak intensity (fraction of max)
        • Let max_int be the maximum intensity in the current spectrum.
        • Only peaks with intensity ≥ (min_rel × max_int) are allowed to become “hits”.

    How the masses are computed (core formula)
        1) Choose integer counts (c1, c2, c3) with total = sum(ci) between 1 and DP.
        2) Compute the “monomer sum” neutral mass:
            M_monomers = Σ(ci × monomer_mass_i)
        3) Apply polymerization + chain delta:
            M_poly = M_monomers + (total−1)×bond_delta + extra_delta
        4) Convert to m/z for each (charge z, adduct A):
            m/z_pred = (M_poly + A) / z

    Reaction/product options (what the checkboxes do)
        Also match oxidation products (+O)
        • Uses:
            M_ox = M_poly + 15.994915
            m/z_pred = (M_ox + A) / z
        • Labeled as “... +O ...”.

        Also match decarboxylation products (−CO2)
        • Uses:
            M_decarb = M_poly − 43.989829
            m/z_pred = (M_decarb + A) / z
        • Labeled as “… -CO2 …”.

        Enable noncovalent polymer dimers (2M−H)
        • Treats the covalent polymer mass as a monomeric unit M_poly and searches dimers:
            M_dimer = 2 × M_poly
            m/z_pred = (M_dimer + A_cluster) / z
        • “Cluster H adduct mass” is A_cluster (and can also auto-sign like the H adduct).
        • The same extra adduct toggles (+Na/+K or +Cl/+HCOO) are applied to the dimer search.

        How matching to peaks works
                • Each predicted m/z is matched by searching the full tolerance window and choosing the best peak
                    (smallest ppm error, then highest intensity) and accepted only if it passes both:
            - intensity threshold (min_rel × max_int)
            - tolerance (Da/ppm)
        • “Best-per-peak” selection
            - For each observed peak and each kind (poly / ox / decarb / 2M), if multiple candidates land
              on the same peak, the app keeps the one with the smallest |m/z_obs − m/z_pred|.
            - Multiple kinds can stack on the same peak (they don’t overwrite each other).

Important assumptions / limitations
    • This is a pattern-matching helper; it does not prove identity.
    • The search can be large if Max DP is high or many monomers/adducts are enabled.
        • Matching is “nearest peak” based; it does not do isotope modeling or full profile fitting.
""",
        "LCMS Overlay Mode": """\
Overlay Selected / Clear Overlay
    What it does
        • Overlay Selected plots multiple mzML datasets together (TIC + spectra).
        • Clear Overlay returns to single-file view (restores the previous active dataset).

How to select overlay datasets
    • In the LCMS Workspace list, click the leftmost checkbox column to mark files for overlay.
    • Each dataset can be assigned a color by clicking the color column.
    • Optional: choose an overlay color scheme from the toolbar.

Overlay Mode (Stacked / Normalized / Offset / Percent of max)
    • Stacked: raw TIC intensities (all datasets on the same scale).
    • Normalized: each TIC scaled to its own max.
    • Offset: each TIC shifted vertically for readability.
    • Percent of max: normalized and scaled to 0–100.

UV overlays
    • “Show UV overlays” plots linked UV traces for all overlay datasets.
    • If a dataset has no linked UV, it is omitted from the UV overlay.

Spectrum overlays
    • Clicking on any TIC/UV sets an RT and overlays spectra from each dataset.
    • The active dataset controls annotation/polymer settings; optionally show labels for all.

Notes / caveats
    • Overlay selection is ephemeral unless “Persist overlay” is enabled when starting overlay.
""",
        "UV↔MS label transfer + Confidence scoring": """\
What this feature is
    • When UV is linked, the app can transfer MS-derived labels to the UV chromatogram to help correlate features.

What you see on the UV plot
    • UV annotations appear near selected UV RT positions.
    • Each transferred label can show a confidence percentage like:
            label_text  [85%]
    • You can filter UV labels by minimum confidence in the Peak Annotations settings.

How labels are transferred (high-level)
    • The app takes a reference MS scan (current scan or a mapped anchor) and extracts the MS labels present there.
    • It chooses a UV anchor time corresponding to that MS scan:
            - if Auto-align is enabled: UV_RT = map_ms_to_uv(MS_RT)
            - else: UV_RT = MS_RT − offset
    • Labels are then displayed on UV near that UV_RT.

Confidence scoring (0–100%) — what it means
    Confidence is a heuristic score that increases when:
        • the mapped MS time matches the chosen MS scan time closely,
        • the UV trace has a clear peak apex near the UV anchor time,
        • the TIC has a peak near the MS scan time,
        • and the UV region is not too crowded with nearby peaks.

Under the hood (high-level math/logic)
    1) RT delta score (alignment consistency)
         • Compute predicted MS time from UV anchor:
                 MS_pred = map_uv_to_ms(UV_anchor)   (auto-align) OR  UV_anchor + offset
         • RT error:  Δt = |MS_pred − MS_scan|
         • Exponential decay score:
                 score_rt = exp(−Δt / τ)   with τ≈0.08 min
    2) UV peakiness score (apex proximity)
         • Find local maxima in the UV trace within a small time window.
         • Let d_uv be the distance to the nearest UV apex.
         • score_uv = exp(−d_uv / 0.08)
    3) TIC peakiness score
         • Similar local-max test in the TIC near the MS scan time.
         • score_tic = exp(−d_tic / 0.08)
    4) Crowding penalty
         • If multiple UV peaks are within a small neighborhood, apply a penalty factor:
                 crowd_factor = 0.85^(count−1)
    5) Weighted combination
         • raw = 0.50*score_rt + 0.35*score_uv + 0.15*score_tic
         • confidence_percent = clamp(100 * raw * crowd_factor, 0, 100)

Limitations
    • Confidence is not an identification probability; it is a practical “alignment plausibility” heuristic.
    • Auto-align (and therefore confidence) can be unreliable if UV and TIC do not share peak structure.
""",
        "Export tools": """\
Export Spectrum CSV…  (File → Export Spectrum CSV…, Ctrl+E)
    What it does
        • Exports the currently displayed spectrum as a CSV with:
                mz, intensity, labels, rt_min, spectrum_id, polarity
    Label column
        • “labels” is a combined text field including auto/custom/poly labels present at each m/z (when applicable).

Export All Labels (Excel)…  (File → Export All Labels…)
    What it does
        • Iterates through the filtered MS1 scans and exports every label occurrence.
    Output columns
        • file_name, spectrum_id, rt_min, polarity, label_kind, label_text, mz, intensity, mz_key
    Requirements
        • Excel export requires the `openpyxl` package (pandas uses it as the Excel engine).
    Notes
        • This export reflects the same labeling rules as the GUI at the time you start export.
        • UV confidence is not currently included in the Excel export (it exports MS-spectrum labels).
""",
        "FTIR (Workstation)": """\
FTIR tab overview
    • The FTIR tab is a multi-workspace workstation for working with many FTIR spectra in one session.
    • Each workspace can contain multiple FTIR datasets, and you can switch/duplicate/delete workspaces.

Loading FTIR data
    Load FTIR…
        • Loads one or more FTIR CSV/TXT files.
        • Each file becomes an FTIR dataset with its own name, peaks, and peak label edits.
    Remove / Clear All
        • Remove deletes the selected dataset from the current workspace.
        • Clear All clears only the current workspace (it does not delete other workspaces).

Workspaces (FTIR)
    New Workspace
        • Creates an empty workspace.
    Rename
        • Renames the current workspace.
    Duplicate
        • Copies the current workspace (including datasets and peak edits).
    Delete
        • Deletes the current workspace (if it’s the last one, it clears it instead).

Overlays (saved overlay groups)
    • Overlay Groups are saved overlay “sets” that can include spectra from different workspaces.
    New Overlay from Selection
        • Uses the Selection list to create a new overlay group.
    Activate Overlay
        • Activates the selected overlay group.
        • When active, the plot overlays all group members.
    Members
        • Shows which spectra belong to the selected overlay group.
        • “Set Active = selected member” makes that member the active dataset.
    Clear Active Overlay
        • Turns off overlay mode (does not delete the overlay group).

Peak picking + labels
    Peaks…
        • Opens FTIR Peaks settings.
        • Apply applies settings to the active dataset.
        • Apply to All applies settings to every dataset in every FTIR workspace.
        • You can right-click a peak label to edit/hide it.

Exporting
    Save FTIR Plot…
        • Opens the FTIR Export Editor.
        • Export is isolated (changes do not affect the live FTIR plot).
    Export Peaks…
        • Export peaks from the active dataset to CSV, or all datasets to Excel.

Persistence (save/load)
    File → Save FTIR Workspace…
        • Saves the FTIR session (workspaces, overlays, peaks, and settings) to a JSON file.
        • Does not affect mzML/UV sessions.
    File → Load FTIR Workspace…
        • Loads a previously saved FTIR session JSON.
        • Replaces the current FTIR state only (mzML/UV are unchanged).
""",
        "Tips & Troubleshooting": """\
General tips
    • If the spectrum panel says “Click TIC/UV to load a spectrum”, pick a scan by clicking the TIC or using Navigate.
    • If UV says “No UV linked”, link a UV file to the active mzML session (UV Workspace).

Auto-align troubleshooting
    • “Not enough peaks detected”
            - Try lowering noise (smoother UV), ensure the chromatograms have clear peaks.
            - Ensure the mzML loaded has MS1 scans and the polarity filter isn’t excluding them.
    • “Failed to find a stable peak alignment”
            - UV and TIC may not share similar peak structure (e.g., detector differences, missing peaks).
            - Try adjusting the fixed offset to a reasonable estimate before running Auto-align.
    • Mapping outside anchors
            - Auto-align is only trusted between the earliest and latest anchors.
            - Outside that region, the app falls back to fixed offset.

Excel export troubleshooting
    • If Excel export errors mention openpyxl:
            - Install it in your environment and retry.

          "pm_match_region": "Recompute the selected TIC region spectrum and apply polymer matching to it.",
Label editing
    • If a label keeps reappearing, check whether it is an auto/poly label (controlled by settings), not a custom label.
    • Use right-click/double-click on labels to edit or delete (delete suppresses for that spectrum).
""",
        "Microscopy": """\
Microscopy tab (ImageJ/Fiji integration)

What this feature is
    • A workspace manager for microscopy datasets (the app stores only file paths, not image bytes).
    • Image viewing/processing is delegated to an external ImageJ/Fiji installation.

One-time setup (Windows)
    • Click “Set Fiji/ImageJ Path…” and select the ImageJ/Fiji executable (.exe).
    • The app stores this persistently in %APPDATA%\\MFP lab analysis tool\\settings.json.

Opening files
    • “Open Selected in Fiji” launches ImageJ/Fiji with each selected dataset path.

Active workspace
    • Use the “Active workspace” selector at the top of the Microscopy tab.
    • “Load Files…” adds new datasets into the currently active workspace.

Outputs discovery
    • The Outputs panel scans the dataset output folder for CSV/PNG/JPG/TXT/PDF/XLSX/TIF and lists them.
    • “Import Outputs…” can copy outputs from another folder into the dataset output folder.

Output folder location
    • Each dataset has its own output folder.
    • You can change it via “Set…” next to “Output folder” in the dataset details panel.
""",
}

GUIDE_SECTIONS: List[str] = list(GUIDE.keys())


# --- Tooltip library ---
TOOLTIP_TEXT: Dict[str, str] = {
    # ===== Main toolbar (top row) =====
    "open_mzml": (
        "Open an mzML file and build the MS1 TIC index.\n"
        "• Use this first to load MS data.\n"
        "• The app scans the file for MS1 spectra, computes TIC, and populates navigation.\n"
        "• Tip: RT units follow the current RT unit setting (min/sec)."
    ),
    "load_uv_csv": (
        "Load a UV chromatogram from a CSV file.\n"
        "• The app auto-detects the time column (RT/time/min/sec) and a signal column (UV/Abs/AU).\n"
        "• UV time is converted to minutes internally to match mzML.\n"
        "• Loading a new UV file clears previously transferred MS→UV labels (they depend on UV RT axis)."
    ),
    "edit_graph": (
        "Edit titles, axis labels, font sizes, and axis ranges.\n"
        "• Use this to make plots publication-ready (TIC, Spectrum, UV).\n"
        "• Leave axis limits blank to auto-scale.\n"
        "• Apply updates to redraw the plots immediately."
    ),
    "annotate_peaks": (
        "Configure automatic peak labeling on the Spectrum plot.\n"
        "• Enable/disable labeling, pick Top N peaks, and set a minimum relative intensity threshold.\n"
        "• Also controls transferring top MS labels onto the UV chromatogram at selected RT.\n"
        "• Tip: Right-click labels to edit or delete them."
    ),
    "custom_labels": (
        "Add your own labels at specific m/z values on the currently displayed spectrum.\n"
        "• Choose whether the label should snap to the nearest peak or stay at the exact m/z.\n"
        "• Labels are stored per-spectrum (per scan) and will reappear when you return to that scan.\n"
        "• Right-click a label to edit/delete later."
    ),
    "polymer_match": (
        "Match polymer/reaction series to observed peaks in the current spectrum.\n"
        "• Define monomers (name,mass), polymerization delta (e.g., dehydration), adducts, charges, tolerance, and max DP.\n"
        "• Adds labels like poly / ox / decarb / 2M when matches are found.\n"
        "• Tip: Polarity affects the sign of H adduct if it is ~1.007276."
    ),
    "save_tic": (
        "Open the Export Editor for the TIC plot.\n"
        "• Lets you adjust titles/labels, fonts, axis limits, colors, and optionally number annotations.\n"
        "• Save as PNG/PDF/SVG without affecting the main view."
    ),
    "save_spectrum": (
        "Open the Export Editor for the Spectrum plot.\n"
        "• Includes your current auto labels, custom labels, and polymer labels.\n"
        "• You can drag labels in the export window and optionally convert labels to numbers + add an in-plot table.\n"
        "• Save as PNG/PDF/SVG."
    ),
    "save_uv": (
        "Open the Export Editor for the UV chromatogram.\n"
        "• Includes stored MS→UV transferred labels (if enabled) and the selected RT marker.\n"
        "• Adjust styling and export as PNG/PDF/SVG."
    ),
    "export_spectrum_csv": (
        "Export the currently displayed spectrum to a CSV file.\n"
        "• Outputs m/z, intensity, and a ‘labels’ column summarizing all active labels on each peak.\n"
        "• Also includes RT, spectrum_id, and polarity.\n"
        "• Use this for downstream processing in Python/Excel."
    ),
    "overlay_selected": (
        "Start overlay mode for the datasets checked in the Workspace list.\n"
        "• Overlays TICs, UV traces (optional), and spectra at the selected RT."
    ),
    "overlay_clear": (
        "Exit overlay mode and restore single-file view.\n"
        "• Restores the previously active dataset if possible."
    ),
    "overlay_mode": (
        "Overlay display mode for TICs (and normalized spectrum display).\n"
        "• Stacked: raw intensities\n"
        "• Normalized: each dataset scaled to its max\n"
        "• Offset: vertical offsets per dataset\n"
        "• Percent of max: 0–100 scale"
    ),
    "overlay_colors": "Color scheme used for overlay datasets (scientific palettes and single-hue gradients).",
    "overlay_pick_hue": "Pick the base color for the Single hue overlay scheme.",
    "overlay_uv": "Show UV overlays for linked UV files (if available).",
    "overlay_stack": "Stack spectra vertically to reduce overlap.",
    "overlay_persist": "Persist overlay selection when saving a workspace (optional).",
    "overlay_labels_all": "Show custom/polymer labels for all overlayed datasets.",
    "overlay_multi_drag": "Drag same-named labels together across overlayed spectra.",

    # ===== Main controls (RT unit, polarity) =====
    "rt_unit": (
        "Retention time (RT) unit for mzML scan times.\n"
        "• Choose minutes or seconds depending on how your mzML encodes scan start time.\n"
        "• Changing this affects how the TIC index is interpreted when loading mzML."
    ),
    "polarity_all": (
        "Show all MS1 spectra regardless of polarity.\n"
        "• Useful if your file contains mixed polarity scans.\n"
        "• Navigation and searches operate on the currently filtered set."
    ),
    "polarity_pos": (
        "Show only positive-mode MS1 spectra.\n"
        "• Filters the TIC and navigation list.\n"
        "• Polymer matching adduct defaults may behave differently in positive mode."
    ),
    "polarity_neg": (
        "Show only negative-mode MS1 spectra.\n"
        "• Filters the TIC and navigation list.\n"
        "• Enables matching of negative-mode extra adducts such as +Cl or +HCOO (if selected)."
    ),

    # ===== Navigate section (row2) =====
    "nav_prev": (
        "Go to the previous MS1 spectrum (previous scan in the filtered TIC list).\n"
        "• Updates the TIC marker, spectrum plot, and the status bar.\n"
        "• Shortcut: Left Arrow."
    ),
    "nav_next": (
        "Go to the next MS1 spectrum (next scan in the filtered TIC list).\n"
        "• Updates the TIC marker, spectrum plot, and the status bar.\n"
        "• Shortcut: Right Arrow."
    ),
    "nav_first": (
        "Jump to the first MS1 spectrum in the filtered list.\n"
        "• Useful after changing polarity filter.\n"
        "• Shortcut: Home."
    ),
    "nav_last": (
        "Jump to the last MS1 spectrum in the filtered list.\n"
        "• Shortcut: End."
    ),
    "jump_rt_entry": (
        "Jump to an RT (minutes) in the mzML TIC.\n"
        "• Type a retention time in minutes and press Go.\n"
        "• The app selects the nearest available scan (it does not snap to peak apex by default)."
    ),
    "jump_rt_go": (
        "Go to the scan closest to the RT you typed.\n"
        "• Uses the filtered TIC list.\n"
        "• Tip: if you loaded UV, the UV marker updates using the UV↔MS offset."
    ),
    "uv_ms_offset_entry": (
        "UV↔MS RT offset in minutes.\n"
        "• Used to align UV time with MS time: MS_RT ≈ UV_RT + offset.\n"
        "• Example: if UV peaks occur 0.125 min earlier than MS, set offset to 0.125.\n"
        "• If Auto-align is enabled, the offset is used as a fallback outside the aligned region."
    ),
    "uv_ms_offset_apply": (
        "Apply the UV↔MS offset and redraw the UV plot.\n"
        "• Shifts where the selected RT marker appears in UV.\n"
        "• Helps correlate UV features with MS scans.\n"
        "• Tip: keep offset reasonable (small minutes), large offsets are blocked."
    ),
    "uv_ms_align_enable": (
        "Enable/disable Auto-align UV↔MS.\n"
        "• When enabled, UV↔MS mapping uses a peak-based alignment between UV chromatogram and MS TIC.\n"
        "• When disabled, mapping uses the fixed offset only."
    ),
    "uv_ms_align_button": (
        "Compute an automatic UV↔MS alignment from peaks.\n"
        "• Uses peak-based correlation between the UV chromatogram and the MS TIC.\n"
        "• Runs in the background.\n"
        "• On success, Auto-align is enabled and clicks/navigations map via the computed curve."
    ),
    "uv_ms_align_diag_button": (
        "Open Alignment Diagnostics.\n"
        "• Shows the UV→MS anchor mapping curve and residuals.\n"
        "• Lets you click an anchor to jump the UV marker and MS scan.\n"
        "• Optional: remove anchors and export anchors as CSV."
    ),

    "sim_button": (
        "EIC chromatogram (Extracted Ion Chromatogram) from MS1 scans.\n"
        "• Pick a target m/z and tolerance, then the app sums intensities of all peaks within tolerance at each MS1 scan.\n"
        "• Displays intensity vs RT for only that selected m/z window (EIC trace).\n"
        "• Click the EIC trace to jump the main app to the nearest scan at that RT."
    ),
    "sim_target_mz": "Target m/z for EIC extraction (float).",
    "sim_tol_value": "Tolerance around target m/z. In ppm (relative) or Da (absolute).",
    "sim_tol_unit": "Tolerance unit: ppm (relative) or Da (absolute).",
    "sim_use_polarity": "If enabled, use the current polarity filter; otherwise use all MS1 scans regardless of polarity.",

    # ===== Feature buttons you added earlier =====
    "find_mz": (
        "Find the nearest scan containing a peak near a target m/z.\n"
        "• Enter m/z and tolerance (ppm or Da) to jump to the closest matching RT.\n"
        "• Press again to jump to the next matching RT if the current scan already matches.\n"
        "• Works within the active mzML and current polarity filter."
    ),
    "export_all_labels_excel": (
        "Scan all MS1 spectra and export all detected labels to an Excel file.\n"
        "• Iterates through the TIC scan list and collects every label found per scan.\n"
        "• Exports columns: label, RT (min), m/z, intensity, spectrum_id, polarity.\n"
        "• Use this to summarize polymer/auto/custom label hits across the whole run."
    ),

    # ===== Menu items (optional if implemented) =====
    "menu_open": (
        "Open an mzML file (Ctrl+O).\n"
        "• Same as the Open mzML button."
    ),
    "menu_uv": (
        "Load a UV CSV file (Ctrl+U).\n"
        "• Same as the Load UV CSV button."
    ),
    "menu_export": (
        "Export the currently displayed spectrum to CSV (Ctrl+E).\n"
        "• Same as the Export Spectrum button."
    ),
    "menu_exit": (
        "Close the application.\n"
        "• If an mzML file is open, the reader is closed safely."
    ),

    # ===== Additional app/tooling keys (added) =====
    "reset_view": (
        "Reset plot zoom and axis ranges.\n"
        "• Restores auto-scaling for TIC, Spectrum, and UV.\n"
        "• Does not delete labels or change filters."
    ),
    "export_dropdown": (
        "Export tools.\n"
        "• Export spectrum CSV, or save TIC/Spectrum/UV plots.\n"
        "• Plot exports open the Export Editor (export-only styling)."
    ),

    "workspace_tree": "Loaded mzML files. Double-click to activate the selected session.",
    "workspace_add": "Add a single mzML file to the workspace.",
    "workspace_add_many": "Add multiple mzML files to the workspace at once.",
    "workspace_set_active": "Make the selected workspace mzML the active session.",
    "workspace_remove": "Remove the selected mzML session from the workspace.",

    "uv_tree": "Loaded UV CSV files. Select one and link it to the active mzML session.",
    "uv_add": "Add a single UV CSV to the UV workspace.",
    "uv_add_many": "Add multiple UV CSV files to the UV workspace.",
    "uv_remove": "Remove the selected UV file from the UV workspace.",
    "uv_link": "Link the selected UV file to the active mzML session.",
    "uv_autolink": "Try to auto-link UV files to mzML sessions by matching names.",

    "graph_settings": "Open Graph Settings to edit titles/axes and styling.",
    "panels_show_tic": "Toggle TIC panel visibility.",
    "panels_show_spectrum": "Toggle Spectrum panel visibility.",
    "panels_show_uv": "Toggle UV chromatogram panel visibility.",
    "panels_show_diag": "Show or hide the diagnostics (log + current context) panel.",

    "annotate_enable": "Enable/disable automatic m/z labels for the spectrum plot.",
    "annotate_top_n": "Label only the Top N peaks by intensity (0 disables Top-N filtering).",
    "annotate_min_rel": "Minimum relative intensity for labeling (0..1 fraction of max).",
    "annotate_drag": "Allow dragging labels with the mouse on the main spectrum plot.",
    "uv_transfer_labels": "Transfer top MS peaks as labels onto the UV plot at the selected RT.",
    "uv_transfer_howmany": "Choose how many top MS peaks to transfer to UV labels.",

    "poly_enable": "Enable polymer/reaction matching labels on the spectrum plot.",

    # ===== Find m/z dialog =====
    "find_history": "Pick a previous Find m/z query (last 10). Selecting fills the fields below.",
    "find_target_mz": "Target m/z (e.g., 123.4567).",
    "find_tol": "Tolerance around target m/z (in ppm or Da).",
    "find_tol_unit": "Tolerance unit: ppm (relative) or Da (absolute).",
    "find_min_int": "Only count peaks with intensity ≥ this value (0 disables).",
    "find_mode": "Search mode: Nearest (closest RT), Forward (next RT), Backward (previous RT).",
    "find_action": "Run the search. Press again to jump to the next match when parameters are unchanged.",
    "find_close": "Close the Find m/z dialog.",

    # ===== Graph Settings dialog =====
    "gs_tic_title": "Edit the TIC plot title.",
    "gs_tic_xlabel": "Edit the TIC x-axis label.",
    "gs_tic_ylabel": "Edit the TIC y-axis label.",
    "gs_spec_title": "Edit the Spectrum plot title.",
    "gs_spec_xlabel": "Edit the Spectrum x-axis label.",
    "gs_spec_ylabel": "Edit the Spectrum y-axis label.",
    "gs_uv_title": "Edit the UV plot title.",
    "gs_uv_xlabel": "Edit the UV x-axis label.",
    "gs_uv_ylabel": "Edit the UV y-axis label.",
    "gs_title_fs": "Title font size used for all plots.",
    "gs_label_fs": "Axis label font size used for all plots.",
    "gs_tick_fs": "Tick label font size used for all plots.",
    "gs_limits": "Axis range limits (blank = auto).",
    "gs_apply": "Apply changes and redraw plots.",
    "gs_reset": "Reset graph settings to defaults.",
    "gs_close": "Close Graph Settings.",

    # ===== Peak Annotations dialog =====
    "pa_apply": "Apply annotation settings and redraw Spectrum/UV.",
    "pa_close": "Close Peak Annotations.",

    # ===== Custom Labels dialog =====
    "cl_list": "Custom labels for the currently displayed spectrum.",
    "cl_label": "Text to display as a label.",
    "cl_mz": "m/z position for the label.",
    "cl_snap": "If enabled, the label snaps to the nearest detected peak.",
    "cl_add": "Add the custom label to this spectrum.",
    "cl_remove": "Remove the selected custom label.",
    "cl_clear": "Remove all custom labels for this spectrum.",
    "cl_close": "Close Custom Labels.",

    # ===== Polymer Match dialog =====
    "pm_enable": "Enable polymer/reaction matching labels on the spectrum.",
    "pm_monomers": "Enter monomers: one per line as name,mass (Da) or just mass.",
    "pm_bond_combo": "Choose a preset polymerization delta per bond, or Custom.",
    "pm_bond_custom": "Custom polymerization delta per bond (Da).",
    "pm_extra_combo": "Choose a preset extra delta applied once per chain, or Custom.",
    "pm_extra_custom": "Custom extra delta (Da).",
    "pm_adduct": "Adduct mass used for matching (H by default).",
    "pm_extra_adducts": "Also match additional adducts appropriate for current polarity.",
    "pm_charges": "Charge states to consider (comma-separated, e.g., 1,2).",
    "pm_dp": "Maximum degree of polymerization (max total monomers) to search.",
    "pm_tol": "Mass tolerance value.",
    "pm_tol_unit": "Tolerance unit: Da or ppm.",
    "pm_minrel": "Minimum peak intensity (fraction of max) used for matching.",
    "pm_apply": "Apply polymer settings and redraw Spectrum.",
    "pm_match_region": "Recompute the selected TIC region spectrum and apply polymer matching to it.",
    "pm_reset": "Reset polymer settings to defaults.",
    "pm_close": "Close Polymer / Reaction Match.",

    # ===== Export Editor =====
    "exp_controls": "Open the export-only Controls window for styling/limits/table/colors.",
    "exp_saveas": "Save the export figure to a file (PNG/PDF/SVG).",
    "exp_close": "Close the Export Editor.",
    "exp_title": "Export-only plot title.",
    "exp_xlabel": "Export-only x-axis label.",
    "exp_ylabel": "Export-only y-axis label.",
    "exp_apply": "Apply export controls to the export figure.",
    "exp_colors": "Pick plot/label/background/table colors for the export figure.",
    "exp_number": "Replace labels with numbers and add an in-plot table.",

    # ===== FTIR (Workstation) =====
    "ftir_ws_combo": "Choose the active FTIR workspace.",
    "ftir_ws_new": "Create a new FTIR workspace (an independent collection of datasets).",
    "ftir_ws_rename": "Rename the active FTIR workspace.",
    "ftir_ws_duplicate": "Duplicate the active FTIR workspace (copies datasets + peak edits).",
    "ftir_ws_delete": "Delete the active FTIR workspace (if it is the last, it will be cleared).",

    "ftir_ws_graph_color": "Set the line color used for FTIR plots in this workspace (applies to all datasets in the workspace).",

    "ftir_tree": "FTIR datasets in the current workspace. Select one to make it active; double-click/F2 to rename.",
    "ftir_load": "Load FTIR CSV/TXT file(s) into the current workspace.",
    "ftir_remove": "Remove the selected FTIR dataset from the current workspace.",
    "ftir_clear": "Clear all datasets from the current workspace (other workspaces are unaffected).",

    "ftir_overlay_groups": "Saved overlay groups. Double-click to activate the selected group.",
    "ftir_overlay_new": "Create a new overlay group from the Selection list below.",
    "ftir_overlay_activate": "Activate the selected overlay group (plot will overlay all members).",
    "ftir_overlay_rename": "Rename the selected overlay group.",
    "ftir_overlay_duplicate": "Duplicate the selected overlay group.",
    "ftir_overlay_delete": "Delete the selected overlay group.",
    "ftir_overlay_clear_active": "Turn off overlay mode (does not delete overlay groups).",
    "ftir_overlay_members": "Members of the selected overlay group.",
    "ftir_overlay_set_active_member": "Make the selected member the active dataset.",
    "ftir_overlay_colors": "Color scheme for FTIR overlay lines (scientific palettes or single-hue gradients).",
    "ftir_overlay_pick_hue": "Pick the base color for the Single hue overlay scheme.",
    "ftir_overlay_filter": "Filter the Selection list by workspace name, dataset name, or file path.",
    "ftir_overlay_selection": "Pick one or more datasets here, then create a new overlay group.",

    "ftir_save_plot": "Open the FTIR Export Editor (export-only styling; does not change the live plot).",
    "ftir_peaks": "Configure FTIR peak picking / labeling for the active dataset (or Apply to All).",
    "ftir_export_peaks": "Export peaks (active dataset → CSV, or all datasets → Excel).",
    "ftir_reverse_x": "Reverse the x-axis direction (common for FTIR wavenumber plots).",
    "ftir_show_peaks_all": "When in overlay mode, also draw peak markers/labels for all overlayed spectra.",

    "ftir_peaks_enable": "Enable/disable drawing peaks (markers + labels).",
    "ftir_peaks_min_prom": "Minimum prominence threshold for peak detection.",
    "ftir_peaks_min_dist": "Minimum peak spacing (in cm⁻¹) for peak detection.",
    "ftir_peaks_label_fmt": "Label format string for FTIR peaks (e.g., {wn:.1f}).",
    "ftir_peaks_unhide": "Unhide all suppressed peak labels in the active dataset.",
    "ftir_peaks_apply": "Apply these peak settings to the active dataset.",
    "ftir_peaks_apply_all": "Apply these peak settings to ALL datasets in ALL FTIR workspaces (runs in background).",
    "ftir_peaks_close": "Close the FTIR Peaks dialog.",
}


def _clamp_u8(v: float) -> int:
    return int(max(0, min(255, round(v))))


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    s = (hex_color or "").strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return (0, 0, 0)
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except Exception:
        return (0, 0, 0)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{_clamp_u8(r):02x}{_clamp_u8(g):02x}{_clamp_u8(b):02x}"


def _adjust_color(hex_color: str, factor: float) -> str:
    """factor>1 brightens, factor<1 darkens."""
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex((r * factor, g * factor, b * factor))


def _default_logo_path() -> Path:
    try:
        here = Path(__file__).resolve().parent
    except Exception:
        here = Path.cwd()
    # When packaged under lab_gui/, assets live one directory up.
    p1 = here / "assets" / "lab_logo.png"
    p2 = here.parent / "assets" / "lab_logo.png"
    return p1 if p1.exists() else p2


def _resolve_logo_path() -> Optional[Path]:
    """Best-effort logo lookup; returns first existing path or None."""
    try:
        here = Path(__file__).resolve().parent
    except Exception:
        here = Path.cwd()

    root = here.parent

    # Preferred: assets/lab_logo.png (support both package dir and project root).
    candidates: List[Path] = [here / "assets" / "lab_logo.png", root / "assets" / "lab_logo.png"]

    # Next: any common image in assets/
    assets_dir = here / "assets"
    assets_dir_root = root / "assets"
    try:
        if assets_dir.exists():
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                candidates.extend(sorted(assets_dir.glob(f"*{ext}")))
    except Exception:
        pass

    try:
        if assets_dir_root.exists():
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                candidates.extend(sorted(assets_dir_root.glob(f"*{ext}")))
    except Exception:
        pass

    # Also allow a single file dropped next to the script (your case)
    candidates.extend(
        [
            here / "logo FIN-1.png",
            here / "logo FIN-1.PNG",
            here / "logo FIN-1.jpg",
            here / "logo FIN-1.jpeg",
            root / "logo FIN-1.png",
            root / "logo FIN-1.PNG",
            root / "logo FIN-1.jpg",
            root / "logo FIN-1.jpeg",
        ]
    )

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


@dataclass
class Workspace:
    lcms_datasets: List[LCMSDataset] = field(default_factory=list)
    active_lcms: Optional[str] = None  # session_id

    ftir_datasets: List[FTIRDataset] = field(default_factory=list)
    active_ftir_id: Optional[str] = None  # FTIRDataset.id (mirrors active FTIR workspace)

    microscopy_workspaces: List[MicroscopyWorkspace] = field(default_factory=list)
    active_microscopy_workspace_id: Optional[str] = None

    plate_reader_datasets: List[PlateReaderDataset] = field(default_factory=list)
    active_plate_reader_id: Optional[str] = None


class FTIRExportEditor(tk.Toplevel):
    def __init__(
        self,
        app: "App",
        *,
        snapshot: Dict[str, Any],
        default_stem: str,
    ) -> None:
        super().__init__(app)
        self.app = app
        self.snapshot = dict(snapshot or {})
        self.default_stem = str(default_stem or "ftir")

        try:
            self._init_ui()
        except Exception:
            msg = traceback.format_exc()
            try:
                messagebox.showerror(
                    "FTIR Export Editor",
                    "FTIR Export Editor failed to open.\n\n" + msg,
                    parent=app,
                )
            except Exception:
                pass
            try:
                self.destroy()
            except Exception:
                pass

    def _init_ui(self) -> None:
        self._controls_win: Optional[tk.Toplevel] = None
        self._controls_scroll_canvas: Optional[tk.Canvas] = None

        self.title("FTIR Export Editor")
        try:
            sw = int(self.winfo_screenwidth())
            sh = int(self.winfo_screenheight())
            self.geometry(f"{max(1100, int(sw * 0.92))}x{max(700, int(sh * 0.82))}")
        except Exception:
            self.geometry("1500x900")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=6)
        top.grid(row=0, column=0, sticky="ew")
        controls_btn = ttk.Button(top, text="Controls…", command=self._open_controls_window)
        controls_btn.pack(side=tk.LEFT)
        saveas_btn = ttk.Button(top, text="Save As…", command=self._save_as)
        saveas_btn.pack(side=tk.LEFT, padx=(8, 0))
        close_btn = ttk.Button(top, text="Close", command=self._on_close)
        close_btn.pack(side=tk.RIGHT)

        plot = ttk.Frame(self)
        plot.grid(row=1, column=0, sticky="nsew")
        plot.columnconfigure(0, weight=1)
        plot.rowconfigure(0, weight=1)
        plot.rowconfigure(1, weight=0)
        plot.rowconfigure(2, weight=0)

        self._fig = Figure(figsize=(14.0, 7.5), dpi=110)
        self._ax = self._fig.add_subplot(1, 1, 1)

        self._canvas = FigureCanvasTkAgg(self._fig, master=plot)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

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

        # Snapshot payload
        self._displayed: List[Dict[str, Any]] = list(self.snapshot.get("displayed") or [])
        self._overlay_on = bool(self.snapshot.get("overlay_on", False))
        self._active_key = (None if not self.snapshot.get("active_key") else tuple(self.snapshot.get("active_key")))

        # Per-spectrum style state (editor-local)
        self._style_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._line_by_key: Dict[Tuple[str, str], Any] = {}
        self._base_xy_by_key: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._peak_markers_by_key: Dict[Tuple[str, str], Any] = {}
        self._ann_by_key: Dict[Tuple[str, str], List[Any]] = {}
        self._ann_to_info: Dict[Any, Tuple[str, str, str]] = {}  # ann -> (ws, ds, peak_id)
        self._label_pos_by_key_pid: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
        self._label_override_by_key_pid: Dict[Tuple[str, str], Dict[str, str]] = {}

        # Bond labels (editor-local)
        self._bond_rows: List[Dict[str, Any]] = [dict(r) for r in (self.snapshot.get("bond_annotations") or []) if isinstance(r, dict)]
        self._bond_texts: List[Any] = []
        self._bond_vlines: List[Any] = []
        self._bond_artist_to_idx: Dict[Any, int] = {}
        self._drag_bond_artist: Any = None
        self._drag_bond_idx: Optional[int] = None
        self._drag_bond_dx = 0.0
        self._drag_bond_dy = 0.0

        # Global controls
        self.title_var = tk.StringVar(value=str(self.snapshot.get("title") or "FTIR"))
        self.xlabel_var = tk.StringVar(value=str(self.snapshot.get("xlabel") or "Wavenumber"))
        self.ylabel_var = tk.StringVar(value=str(self.snapshot.get("ylabel") or "Absorbance"))

        self.title_fs_var = tk.IntVar(value=int(getattr(self.app, "title_fontsize_var", tk.IntVar(value=12)).get()))
        self.label_fs_var = tk.IntVar(value=int(getattr(self.app, "label_fontsize_var", tk.IntVar(value=10)).get()))
        self.tick_fs_var = tk.IntVar(value=int(getattr(self.app, "tick_fontsize_var", tk.IntVar(value=9)).get()))
        self.peak_label_fs_var = tk.IntVar(value=max(6, int(getattr(self.app, "tick_fontsize_var", tk.IntVar(value=9)).get()) - 1))

        self.xmin_var = tk.StringVar(value="")
        self.xmax_var = tk.StringVar(value="")
        self.ymin_var = tk.StringVar(value="")
        self.ymax_var = tk.StringVar(value="")
        self.reverse_x_var = tk.BooleanVar(value=bool(self.snapshot.get("reverse_x", False)))

        self.axes_facecolor_var = tk.StringVar(value=str(self.snapshot.get("axes_bg") or "#ffffff"))
        self.grid_var = tk.BooleanVar(value=bool(self.snapshot.get("grid_on", False)))

        self.legend_on_var = tk.BooleanVar(value=bool(self.snapshot.get("legend_on", True)))
        self.legend_fs_var = tk.IntVar(value=int(self.snapshot.get("legend_fontsize", 8) or 8))

        self.peak_marker_size_var = tk.IntVar(value=int(self.snapshot.get("peak_marker_size", 4) or 4))

        self.highlight_enabled_var = tk.BooleanVar(value=True)
        self.highlight_factor_var = tk.DoubleVar(value=1.8)

        # Overlay styling controls
        self._overlay_tree: Optional[ttk.Treeview] = None
        self._style_prop_var = tk.StringVar(value="Line color")
        self._overlay_offset_mode_var = tk.StringVar(value=str(self.snapshot.get("overlay_offset_mode") or "Normal"))
        try:
            self._overlay_offset_var = tk.DoubleVar(value=float(self.snapshot.get("overlay_offset") or 0.0))
        except Exception:
            self._overlay_offset_var = tk.DoubleVar(value=0.0)

        # Dragging annotations (editor-local)
        self._drag_ann: Any = None
        self._drag_info: Optional[Tuple[str, str, str]] = None
        self._drag_dx = 0.0
        self._drag_dy = 0.0

        self._build_initial_plot()

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
            self.transient(self.app)
        except Exception:
            pass
        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass
        try:
            self.after(0, self._open_controls_window)
        except Exception:
            pass

    def _on_close(self) -> None:
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

    def _parse_optional_float(self, raw: str) -> Optional[float]:
        raw = (raw or "").strip()
        if not raw:
            return None
        return float(raw)

    def _open_controls_window(self) -> None:
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
        win.title("FTIR Export Controls")
        try:
            win.geometry("620x900")
        except Exception:
            pass

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
            w = 620
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
            canvas.bind("<Configure>", _on_canvas_config, add=True)
        except Exception:
            pass

        def _pick_one(var: tk.StringVar, title: str) -> None:
            try:
                c = colorchooser.askcolor(color=(var.get() or None), title=title, parent=win)[1]
                if c:
                    var.set(str(c))
            except Exception:
                return

        # --- Controls UI ---
        row = 0

        # Text
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

        # Fonts
        fonts_group = ttk.Labelframe(inner, text="Fonts", padding=(8, 6))
        fonts_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        fonts_group.columnconfigure(1, weight=1)
        row += 1

        def _add_slider(parent: tk.Widget, *, variable: Union[tk.IntVar, tk.DoubleVar], from_: float, to: float, step: float = 1.0) -> ttk.Frame:
            holder = ttk.Frame(parent)
            holder.columnconfigure(0, weight=1)
            scale_var = tk.DoubleVar(value=float(variable.get()))
            lbl = ttk.Label(holder, text="")

            def _apply(v: Any = None) -> None:
                try:
                    fv = float(v if v is not None else scale_var.get())
                except Exception:
                    return
                try:
                    fv = round(fv / float(step)) * float(step)
                except Exception:
                    pass
                if isinstance(variable, tk.IntVar):
                    iv = int(round(fv))
                    variable.set(iv)
                    scale_var.set(float(iv))
                    lbl.configure(text=str(iv))
                else:
                    variable.set(float(fv))
                    lbl.configure(text=f"{float(variable.get()):.2f}")

            s = ttk.Scale(holder, from_=float(from_), to=float(to), variable=scale_var, command=_apply)
            s.grid(row=0, column=0, sticky="ew")
            lbl.grid(row=0, column=1, sticky="e", padx=(8, 0))
            _apply(scale_var.get())
            return holder

        ttk.Label(fonts_group, text="Title font size").grid(row=0, column=0, sticky="w")
        _add_slider(fonts_group, variable=self.title_fs_var, from_=6, to=48).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(fonts_group, text="Axis label font size").grid(row=1, column=0, sticky="w", pady=(6, 0))
        _add_slider(fonts_group, variable=self.label_fs_var, from_=6, to=48).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Label(fonts_group, text="Tick font size").grid(row=2, column=0, sticky="w", pady=(6, 0))
        _add_slider(fonts_group, variable=self.tick_fs_var, from_=6, to=48).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Label(fonts_group, text="Peak label font size").grid(row=3, column=0, sticky="w", pady=(6, 0))
        _add_slider(fonts_group, variable=self.peak_label_fs_var, from_=6, to=48).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        # Legend
        leg_group = ttk.Labelframe(inner, text="Legend", padding=(8, 6))
        leg_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        leg_group.columnconfigure(1, weight=1)
        row += 1
        ttk.Checkbutton(leg_group, text="Show legend", variable=self.legend_on_var, command=self._apply_style).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(leg_group, text="Legend font size").grid(row=1, column=0, sticky="w", pady=(6, 0))
        _add_slider(leg_group, variable=self.legend_fs_var, from_=6, to=48).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        # Limits
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
        ttk.Checkbutton(lim_group, text="Reverse X axis", variable=self.reverse_x_var, command=self._apply_style).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # Global colors
        col_group = ttk.Labelframe(inner, text="Global Colors", padding=(8, 6))
        col_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        col_group.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(col_group, text="Axes background").grid(row=0, column=0, sticky="w")
        ttk.Entry(col_group, textvariable=self.axes_facecolor_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(col_group, text="Pick…", command=lambda: _pick_one(self.axes_facecolor_var, "Axes background")).grid(row=0, column=2, sticky="e", padx=(8, 0))
        ttk.Checkbutton(col_group, text="Grid", variable=self.grid_var, command=self._apply_style).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        # Peak specifics
        peak_group = ttk.Labelframe(inner, text="Peak Styling", padding=(8, 6))
        peak_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        peak_group.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(peak_group, text="Marker size").grid(row=0, column=0, sticky="w")
        _add_slider(peak_group, variable=self.peak_marker_size_var, from_=1, to=12).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # Highlight controls
        hi_group = ttk.Labelframe(inner, text="Active Spectrum Highlight", padding=(8, 6))
        hi_group.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        hi_group.columnconfigure(1, weight=1)
        row += 1

        ttk.Checkbutton(hi_group, text="Enable highlight", variable=self.highlight_enabled_var, command=self._apply_style).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(hi_group, text="Linewidth factor").grid(row=1, column=0, sticky="w", pady=(6, 0))
        _add_slider(hi_group, variable=self.highlight_factor_var, from_=1.0, to=3.0, step=0.05).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        # Overlay styling
        if self._overlay_on and len(self._displayed) > 1:
            ov_group = ttk.Labelframe(inner, text="Overlay Styling (per spectrum)", padding=(8, 6))
            ov_group.grid(row=row, column=0, sticky="nsew", pady=(10, 0))
            ov_group.columnconfigure(0, weight=1)
            ov_group.rowconfigure(2, weight=1)
            row += 1

            ov_ctrls = ttk.Frame(ov_group)
            ov_ctrls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
            ttk.Label(ov_ctrls, text="Overlay offset").pack(side=tk.LEFT)
            ov_mode = ttk.Combobox(
                ov_ctrls,
                textvariable=self._overlay_offset_mode_var,
                values=["Normal", "Offset Y", "Offset X"],
                state="readonly",
                width=12,
            )
            ov_mode.pack(side=tk.LEFT, padx=(8, 0))
            ov_mode.bind("<<ComboboxSelected>>", lambda _e: self._apply_style(rebuild_peaks=True))
            ttk.Label(ov_ctrls, text="Value").pack(side=tk.LEFT, padx=(8, 0))
            ov_val = ttk.Entry(ov_ctrls, textvariable=self._overlay_offset_var, width=10)
            ov_val.pack(side=tk.LEFT, padx=(4, 0))
            ov_val.bind("<KeyRelease>", lambda _e: self._apply_style(rebuild_peaks=True))

            ttk.Separator(ov_group).grid(row=1, column=0, sticky="ew", pady=(0, 6))

            tv = ttk.Treeview(
                ov_group,
                columns=("vis", "name", "line", "lw", "pcol", "lcol", "p", "l"),
                show="headings",
                selectmode="browse",
                height=8,
            )
            tv.heading("vis", text="Vis")
            tv.heading("name", text="Name")
            tv.heading("line", text="Line")
            tv.heading("lw", text="LW")
            tv.heading("pcol", text="Peak")
            tv.heading("lcol", text="Label")
            tv.heading("p", text="Peaks")
            tv.heading("l", text="Labels")
            tv.column("vis", width=46, stretch=False, anchor="center")
            tv.column("name", width=260, stretch=True)
            tv.column("line", width=90, stretch=False)
            tv.column("lw", width=50, stretch=False, anchor="e")
            tv.column("pcol", width=90, stretch=False)
            tv.column("lcol", width=90, stretch=False)
            tv.column("p", width=60, stretch=False, anchor="center")
            tv.column("l", width=60, stretch=False, anchor="center")
            tv.grid(row=2, column=0, sticky="nsew")
            self._overlay_tree = tv
            sb = ttk.Scrollbar(ov_group, orient="vertical", command=tv.yview)
            sb.grid(row=1, column=1, sticky="ns")
            tv.configure(yscrollcommand=sb.set)

            bar = ttk.Frame(ov_group)
            bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
            ttk.Label(bar, text="Property").pack(side=tk.LEFT)
            ttk.Combobox(bar, textvariable=self._style_prop_var, state="readonly", width=18, values=["Line color", "Peak marker color", "Peak label color"]).pack(
                side=tk.LEFT, padx=(8, 0)
            )
            ttk.Button(bar, text="Pick color…", command=self._pick_color_for_selected).pack(side=tk.LEFT, padx=(8, 0))
            ttk.Button(bar, text="Apply to all", command=self._apply_selected_style_to_all).pack(side=tk.LEFT, padx=(8, 0))

            def _on_double(evt=None):
                self._toggle_overlay_tree_cell(evt)

            try:
                tv.bind("<Double-1>", _on_double, add=True)
            except Exception:
                pass

            def _on_click(evt=None):
                if evt is None:
                    return
                try:
                    rid = tv.identify_row(evt.y)
                except Exception:
                    rid = ""
                if rid:
                    try:
                        tv.selection_set(rid)
                    except Exception:
                        pass

            try:
                tv.bind("<Button-1>", _on_click, add=True)
            except Exception:
                pass

            # The controls window is opened after the initial plot build, so we must
            # populate the overlay table now (otherwise it stays empty).
            try:
                self._refresh_overlay_tree()
            except Exception:
                pass

        btns = ttk.Frame(inner)
        btns.grid(row=row, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(btns, text="Apply", command=self._apply_style).pack(side=tk.LEFT)
        ttk.Button(btns, text="Close", command=lambda: (self._on_controls_closed(), win.destroy())).pack(side=tk.RIGHT)

        inner.columnconfigure(0, weight=1)

        # Live-ish traces (coalesced)
        for v in (
            self.title_var,
            self.xlabel_var,
            self.ylabel_var,
            self.axes_facecolor_var,
        ):
            try:
                v.trace_add("write", lambda *_a: self._schedule_apply())
            except Exception:
                pass
        for v in (
            self.title_fs_var,
            self.label_fs_var,
            self.tick_fs_var,
            self.peak_label_fs_var,
            self.grid_var,
            self.peak_marker_size_var,
            self.legend_on_var,
            self.legend_fs_var,
        ):
            try:
                v.trace_add("write", lambda *_a: self._schedule_apply())
            except Exception:
                pass

    def _on_controls_closed(self) -> None:
        self._controls_win = None
        self._controls_scroll_canvas = None

    def _schedule_apply(self) -> None:
        if getattr(self, "_apply_after", None) is not None:
            return

        def _do():
            self._apply_after = None
            self._apply_style()

        try:
            self._apply_after = self.after(80, _do)
        except Exception:
            self._apply_after = None
            _do()

    def _build_initial_plot(self) -> None:
        self._ax.clear()
        self._base_xy_by_key = {}

        # Initialize per-spectrum style defaults from snapshot (or from displayed line colors in the main view, if provided)
        for item in self._displayed:
            key = tuple(item.get("key") or ())
            if len(key) != 2:
                continue
            ws_id, ds_id = str(key[0]), str(key[1])
            k = (ws_id, ds_id)

            base_line_color = str(item.get("line_color") or "#1f77b4")
            base_lw = float(item.get("line_width") or 1.2)
            peak_c = str(item.get("peak_color") or base_line_color)
            label_c = str(item.get("peak_label_color") or peak_c)

            show_all_overlay = bool(self.snapshot.get("show_peaks_all_overlay", False))
            is_active = (self._active_key is not None and tuple(self._active_key) == k)
            show_peaks = bool(item.get("show_peaks")) if ("show_peaks" in item) else (bool(is_active) or bool(show_all_overlay))
            # If overlay "show all peaks" is enabled, default labels to on as well.
            show_labels = bool(item.get("show_labels")) if ("show_labels" in item) else (bool(is_active) or bool(show_all_overlay))

            self._style_by_key[k] = {
                "visible": True,
                "line_color": base_line_color,
                "line_width": float(base_lw),
                "peak_color": peak_c,
                "label_color": label_c,
                "show_peaks": bool(show_peaks),
                "show_labels": bool(show_labels),
                "name": str(item.get("name") or f"{ws_id}:{ds_id}"),
            }

            # Start with existing manual positions / overrides
            pos = dict(item.get("peak_label_positions") or {})
            self._label_pos_by_key_pid[k] = {}
            for pid, xy in pos.items():
                try:
                    if isinstance(xy, (list, tuple)) and len(xy) == 2:
                        self._label_pos_by_key_pid[k][str(pid)] = (float(xy[0]), float(xy[1]))
                except Exception:
                    continue
            self._label_override_by_key_pid[k] = dict(item.get("peak_label_overrides") or {})

        # Draw lines
        for item in self._displayed:
            key = tuple(item.get("key") or ())
            if len(key) != 2:
                continue
            k = (str(key[0]), str(key[1]))
            st = self._style_by_key.get(k)
            if st is None:
                continue
            x_raw = item.get("x_disp")
            y_raw = item.get("y_disp")
            x = np.asarray(([] if x_raw is None else x_raw), dtype=float)
            y = np.asarray(([] if y_raw is None else y_raw), dtype=float)
            if int(x.size) < 2 or int(y.size) < 2:
                continue
            self._base_xy_by_key[k] = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            try:
                (ln,) = self._ax.plot(x, y, lw=float(st["line_width"]), color=str(st["line_color"]))
                ln.set_label(str(st.get("name") or ""))
            except Exception:
                continue
            self._line_by_key[k] = ln

        self._apply_style(rebuild_peaks=True)

        # Bond labels
        self._rebuild_all_bonds()

        if self._overlay_tree is not None:
            self._refresh_overlay_tree()

    def _apply_style(self, *, rebuild_peaks: bool = False) -> None:
        # Global titles/labels
        try:
            self._ax.set_title(str(self.title_var.get() or ""), fontsize=int(self.title_fs_var.get()))
        except Exception:
            pass
        try:
            self._ax.set_xlabel(str(self.xlabel_var.get() or ""), fontsize=int(self.label_fs_var.get()))
        except Exception:
            pass
        try:
            self._ax.set_ylabel(str(self.ylabel_var.get() or ""), fontsize=int(self.label_fs_var.get()))
        except Exception:
            pass
        try:
            self._ax.tick_params(axis="both", which="major", labelsize=int(self.tick_fs_var.get()))
        except Exception:
            pass

        # Background + grid
        try:
            bg = (self.axes_facecolor_var.get() or "").strip()
            if bg:
                self._ax.set_facecolor(bg)
        except Exception:
            pass
        try:
            self._ax.grid(bool(self.grid_var.get()))
        except Exception:
            pass

        # Overlay offset (export)
        try:
            offset_mode = str(self._overlay_offset_mode_var.get() or "Normal")
        except Exception:
            offset_mode = "Normal"
        try:
            offset_val = float(self._overlay_offset_var.get() or 0.0)
        except Exception:
            offset_val = 0.0
        try:
            if not self._overlay_on or offset_mode == "Normal" or offset_val == 0.0:
                for k, ln in (self._line_by_key or {}).items():
                    base = self._base_xy_by_key.get(k)
                    if base is None:
                        continue
                    ln.set_data(base[0], base[1])
            else:
                try:
                    order = [tuple(item.get("key") or ()) for item in (self._displayed or []) if isinstance(item, dict)]
                    order = [(str(k[0]), str(k[1])) for k in order if len(k) == 2]
                except Exception:
                    order = list(self._line_by_key.keys())
                idx_map = {k: i for i, k in enumerate(order)}
                for k, ln in (self._line_by_key or {}).items():
                    base = self._base_xy_by_key.get(k)
                    if base is None:
                        continue
                    x = np.asarray(base[0], dtype=float)
                    y = np.asarray(base[1], dtype=float)
                    idx = idx_map.get(k, 0)
                    if offset_mode == "Offset Y":
                        y = y + (float(idx) * float(offset_val))
                    elif offset_mode == "Offset X":
                        x = x + (float(idx) * float(offset_val))
                    ln.set_data(x, y)
        except Exception:
            pass

        # Per-spectrum lines
        for k, st in (self._style_by_key or {}).items():
            ln = self._line_by_key.get(k)
            if ln is None:
                continue
            try:
                ln.set_visible(bool(st.get("visible", True)))
            except Exception:
                pass
            try:
                ln.set_color(str(st.get("line_color") or "#1f77b4"))
            except Exception:
                pass
            try:
                ln.set_linewidth(float(st.get("line_width") or 1.2))
            except Exception:
                pass

        # Highlight active spectrum
        try:
            if bool(self.highlight_enabled_var.get()) and self._active_key is not None:
                factor = float(self.highlight_factor_var.get() or 1.0)
                for k, ln in (self._line_by_key or {}).items():
                    st = self._style_by_key.get(k) or {}
                    if not bool(st.get("visible", True)):
                        continue
                    if tuple(k) == tuple(self._active_key):
                        ln.set_linewidth(float(st.get("line_width") or 1.2) * float(factor))
                        ln.set_alpha(1.0)
                        ln.set_zorder(4)
                    else:
                        ln.set_alpha(0.75)
                        ln.set_zorder(2)
            else:
                for k, ln in (self._line_by_key or {}).items():
                    try:
                        ln.set_alpha(1.0)
                        ln.set_zorder(2)
                    except Exception:
                        continue
        except Exception:
            pass

        # Axis limits
        try:
            xmn = self._parse_optional_float(self.xmin_var.get())
            xmx = self._parse_optional_float(self.xmax_var.get())
            if xmn is not None or xmx is not None:
                cur = self._ax.get_xlim()
                a = float(cur[0]) if xmn is None else float(xmn)
                b = float(cur[1]) if xmx is None else float(xmx)
                self._ax.set_xlim(a, b)
        except Exception:
            pass
        try:
            ymn = self._parse_optional_float(self.ymin_var.get())
            ymx = self._parse_optional_float(self.ymax_var.get())
            if ymn is not None or ymx is not None:
                cur = self._ax.get_ylim()
                a = float(cur[0]) if ymn is None else float(ymn)
                b = float(cur[1]) if ymx is None else float(ymx)
                self._ax.set_ylim(a, b)
        except Exception:
            pass
        try:
            if bool(self.reverse_x_var.get()):
                a, b = self._ax.get_xlim()
                if a < b:
                    self._ax.set_xlim(b, a)
            else:
                a, b = self._ax.get_xlim()
                if a > b:
                    self._ax.set_xlim(b, a)
        except Exception:
            pass

        # Legend
        try:
            vis = [k for k, st in (self._style_by_key or {}).items() if bool(st.get("visible", True))]
            want = bool(self.legend_on_var.get()) and (len(vis) > 1)
            if want:
                self._ax.legend(loc="best", fontsize=int(self.legend_fs_var.get()))
            else:
                leg = self._ax.get_legend()
                if leg is not None:
                    leg.remove()
        except Exception:
            pass

        if rebuild_peaks:
            self._rebuild_all_peaks()
            self._rebuild_all_bonds()
        else:
            # Update existing peak artists styles
            self._apply_peak_styles_only()
            self._apply_bond_visibility_only()

        try:
            self._canvas.draw_idle()
        except Exception:
            pass
        try:
            if getattr(self, "_mpl_nav", None) is not None:
                self._mpl_nav.update_home_from_artists()
        except Exception:
            pass

    def _overlay_offset_for_key(self, key: Tuple[str, str]) -> Tuple[float, float]:
        try:
            if not self._overlay_on:
                return 0.0, 0.0
        except Exception:
            return 0.0, 0.0
        try:
            offset_mode = str(self._overlay_offset_mode_var.get() or "Normal")
        except Exception:
            offset_mode = "Normal"
        try:
            offset_val = float(self._overlay_offset_var.get() or 0.0)
        except Exception:
            offset_val = 0.0
        if offset_mode == "Normal" or offset_val == 0.0:
            return 0.0, 0.0

        try:
            order = [tuple(item.get("key") or ()) for item in (self._displayed or []) if isinstance(item, dict)]
            order = [(str(k[0]), str(k[1])) for k in order if len(k) == 2]
        except Exception:
            order = []
        try:
            idx = order.index((str(key[0]), str(key[1])))
        except Exception:
            idx = 0

        if offset_mode == "Offset Y":
            return 0.0, float(idx) * float(offset_val)
        if offset_mode == "Offset X":
            return float(idx) * float(offset_val), 0.0
        return 0.0, 0.0

    def _interp_y_on_line(self, key: Tuple[str, str], wn: float) -> Optional[float]:
        base = self._base_xy_by_key.get(key)
        if base is None:
            return None
        try:
            x = np.asarray(base[0], dtype=float)
            y = np.asarray(base[1], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size < 2:
                return None
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            return float(np.interp(float(wn), x, y))
        except Exception:
            return None

    def _apply_peak_styles_only(self) -> None:
        fs = int(self.peak_label_fs_var.get())
        ms = int(self.peak_marker_size_var.get())
        for k, st in (self._style_by_key or {}).items():
            mk = self._peak_markers_by_key.get(k)
            if mk is not None:
                try:
                    mk.set_markersize(float(ms))
                except Exception:
                    pass
                try:
                    mk.set_color(str(st.get("peak_color") or "#111111"))
                    mk.set_markerfacecolor(str(st.get("peak_color") or "#111111"))
                    mk.set_markeredgecolor(str(st.get("peak_color") or "#111111"))
                except Exception:
                    pass
                try:
                    mk.set_visible(bool(st.get("visible", True)) and bool(st.get("show_peaks", True)))
                except Exception:
                    pass

            for ann in list(self._ann_by_key.get(k) or []):
                try:
                    ann.set_fontsize(int(fs))
                except Exception:
                    pass
                try:
                    ann.set_color(str(st.get("label_color") or "#111111"))
                except Exception:
                    pass
                try:
                    if ann.arrow_patch is not None:
                        ann.arrow_patch.set_color(str(st.get("label_color") or "#111111"))
                except Exception:
                    pass
                try:
                    ann.set_visible(bool(st.get("visible", True)) and bool(st.get("show_labels", True)) and bool(st.get("show_peaks", True)))
                except Exception:
                    pass

    def _rebuild_all_peaks(self) -> None:
        # Remove old
        for mk in list((self._peak_markers_by_key or {}).values()):
            try:
                mk.remove()
            except Exception:
                pass
        self._peak_markers_by_key = {}
        for anns in list((self._ann_by_key or {}).values()):
            for ann in list(anns or []):
                try:
                    ann.remove()
                except Exception:
                    pass
        self._ann_by_key = {}
        self._ann_to_info = {}

        # Recreate
        overlay_on = bool(self._overlay_on and len(self._displayed) > 1)
        try:
            order = [tuple(item.get("key") or ()) for item in (self._displayed or []) if isinstance(item, dict)]
            order = [(str(k[0]), str(k[1])) for k in order if len(k) == 2]
        except Exception:
            order = []
        try:
            offset_mode = str(self._overlay_offset_mode_var.get() or "Normal")
        except Exception:
            offset_mode = "Normal"
        try:
            offset_val = float(self._overlay_offset_var.get() or 0.0)
        except Exception:
            offset_val = 0.0
        for item in self._displayed:
            key = tuple(item.get("key") or ())
            if len(key) != 2:
                continue
            k = (str(key[0]), str(key[1]))
            st = self._style_by_key.get(k)
            if st is None:
                continue
            if not (bool(st.get("visible", True)) and bool(st.get("show_peaks", True))):
                continue

            peaks = list(item.get("peaks") or [])
            suppressed = set(item.get("peak_suppressed") or [])
            peaks = [p for p in peaks if isinstance(p, dict) and str(p.get("id") or "") and (str(p.get("id")) not in suppressed)]

            if not peaks:
                continue

            dx = 0.0
            dy = 0.0
            if overlay_on and offset_mode != "Normal" and offset_val != 0.0:
                try:
                    idx = order.index(k)
                except Exception:
                    idx = 0
                if offset_mode == "Offset Y":
                    dy = float(idx) * float(offset_val)
                elif offset_mode == "Offset X":
                    dx = float(idx) * float(offset_val)

            xs: List[float] = []
            ys: List[float] = []
            for p in peaks:
                try:
                    wn0 = float(p.get("wn"))
                    y0 = self._interp_y_on_line(k, wn0)
                    if y0 is None:
                        y0 = float(p.get("y_display", p.get("y", 0.0)))
                    xs.append(float(wn0) + float(dx))
                    ys.append(float(y0) + float(dy))
                except Exception:
                    continue

            if xs and ys:
                try:
                    (mk,) = self._ax.plot(
                        xs,
                        ys,
                        linestyle="none",
                        marker="o",
                        markersize=float(self.peak_marker_size_var.get()),
                        color=str(st.get("peak_color") or "#111111"),
                        markerfacecolor=str(st.get("peak_color") or "#111111"),
                        markeredgecolor=str(st.get("peak_color") or "#111111"),
                    )
                    self._peak_markers_by_key[k] = mk
                except Exception:
                    pass

            if not bool(st.get("show_labels", True)):
                continue

            fmt = str((item.get("peak_settings") or {}).get("label_fmt") or "{wn:.1f}")
            anns: List[Any] = []
            for p in peaks:
                pid = str(p.get("id") or "").strip()
                if not pid:
                    continue
                try:
                    base_wn = float(p.get("wn"))
                    y0 = self._interp_y_on_line(k, base_wn)
                    if y0 is None:
                        y0 = float(p.get("y_display", p.get("y", 0.0)))
                    peak_wn = float(base_wn) + float(dx)
                    peak_y = float(y0) + float(dy)
                    prom0 = float(p.get("prominence", 0.0) or 0.0)
                except Exception:
                    continue

                pos_x = float(peak_wn)
                pos_y = float(peak_y)
                try:
                    pos = (self._label_pos_by_key_pid.get(k) or {}).get(pid)
                    if pos is not None:
                        pos_x, pos_y = float(pos[0]) + float(dx), float(pos[1]) + float(dy)
                except Exception:
                    pass

                label = str((self._label_override_by_key_pid.get(k) or {}).get(pid, "") or "").strip()
                if not label:
                    try:
                        if FTIRPeak is not None:
                            label = format_peak_label(FTIRPeak(wn=peak_wn - float(dx), y=peak_y - float(dy), prominence=prom0), fmt=fmt)
                        else:
                            label = f"{(peak_wn - float(dx)):.1f}"
                    except Exception:
                        label = f"{(peak_wn - float(dx)):.1f}"

                try:
                    ann = self._ax.annotate(
                        str(label),
                        xy=(float(peak_wn), float(peak_y)),
                        xytext=(float(pos_x), float(pos_y)),
                        textcoords="data",
                        xycoords="data",
                        va="bottom",
                        ha="left",
                        fontsize=int(self.peak_label_fs_var.get()),
                        color=str(st.get("label_color") or "#111111"),
                        clip_on=True,
                        arrowprops={
                            "arrowstyle": "-",
                            "color": str(st.get("label_color") or "#111111"),
                            "lw": 0.8,
                            "shrinkA": 0.0,
                            "shrinkB": 0.0,
                        },
                    )
                    ann.set_picker(True)
                    anns.append(ann)
                    self._ann_to_info[ann] = (str(k[0]), str(k[1]), str(pid))
                except Exception:
                    continue

            self._ann_by_key[k] = anns

        self._apply_peak_styles_only()

    def _bond_should_show(self, dataset_id: str) -> bool:
        dsid = str(dataset_id or "")
        if not dsid:
            return False
        if dsid == "__ALL_OVERLAY__":
            # Show if any spectrum is visible
            for _k, st in (self._style_by_key or {}).items():
                if bool((st or {}).get("visible", True)):
                    return True
            return False
        for (ws_id, ds_id), st in (self._style_by_key or {}).items():
            if str(ds_id) == dsid and bool((st or {}).get("visible", True)):
                return True
        return False

    def _clear_bond_artists(self) -> None:
        for t in list(getattr(self, "_bond_texts", []) or []):
            try:
                t.remove()
            except Exception:
                pass
        for ln in list(getattr(self, "_bond_vlines", []) or []):
            try:
                ln.remove()
            except Exception:
                pass
        self._bond_texts = []
        self._bond_vlines = []
        self._bond_artist_to_idx = {}

    def _apply_bond_visibility_only(self) -> None:
        for artist, idx in list((getattr(self, "_bond_artist_to_idx", {}) or {}).items()):
            try:
                row = (self._bond_rows or [])[int(idx)]
            except Exception:
                continue
            did = str(row.get("dataset_id") or "")
            vis = bool(self._bond_should_show(did))
            try:
                artist.set_visible(vis)
            except Exception:
                pass

    def _rebuild_all_bonds(self) -> None:
        self._clear_bond_artists()
        rows = list(getattr(self, "_bond_rows", []) or [])
        if not rows:
            return

        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue

            try:
                x_cm1 = float(row.get("x_cm1"))
                y_val = float(row.get("y_value"))
            except Exception:
                continue

            xy = row.get("xytext")
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                try:
                    tx, ty = float(xy[0]), float(xy[1])
                except Exception:
                    tx, ty = float(x_cm1), float(y_val)
            else:
                tx, ty = float(x_cm1), float(y_val)

            show_vline = bool(row.get("show_vline", False))
            line_color = str(row.get("line_color") or "#444444")
            text_color = str(row.get("text_color") or "#111111")
            try:
                fontsize = int(row.get("fontsize") or 9)
            except Exception:
                fontsize = 9
            try:
                rotation = int(row.get("rotation") or 0)
            except Exception:
                rotation = 0

            vis = bool(self._bond_should_show(str(row.get("dataset_id") or "")))

            if show_vline:
                try:
                    ln = self._ax.axvline(float(x_cm1), color=str(line_color), lw=0.8, alpha=0.7)
                    ln.set_visible(bool(vis))
                    ln.set_picker(True)
                    self._bond_vlines.append(ln)
                    self._bond_artist_to_idx[ln] = int(i)
                except Exception:
                    pass

            try:
                t = self._ax.text(
                    float(tx),
                    float(ty),
                    str(text),
                    color=str(text_color),
                    fontsize=int(fontsize),
                    rotation=int(rotation),
                    va="center",
                    ha="center",
                    clip_on=True,
                )
                t.set_visible(bool(vis))
                t.set_picker(True)
                self._bond_texts.append(t)
                self._bond_artist_to_idx[t] = int(i)
            except Exception:
                continue

    def _refresh_overlay_tree(self) -> None:
        tv = self._overlay_tree
        if tv is None:
            return
        try:
            for iid in list(tv.get_children("")):
                tv.delete(iid)
        except Exception:
            pass

        for item in self._displayed:
            key = tuple(item.get("key") or ())
            if len(key) != 2:
                continue
            k = (str(key[0]), str(key[1]))
            st = self._style_by_key.get(k)
            if st is None:
                continue
            iid = f"{k[0]}::{k[1]}"
            tv.insert(
                "",
                "end",
                iid=str(iid),
                values=(
                    "✓" if bool(st.get("visible", True)) else "",
                    str(st.get("name", "")),
                    str(st.get("line_color", "")),
                    f"{float(st.get('line_width', 1.2)):g}",
                    str(st.get("peak_color", "")),
                    str(st.get("label_color", "")),
                    "✓" if bool(st.get("show_peaks", True)) else "",
                    "✓" if bool(st.get("show_labels", True)) else "",
                ),
            )

    def _toggle_overlay_tree_cell(self, evt) -> None:
        tv = self._overlay_tree
        if tv is None or evt is None:
            return
        try:
            iid = tv.identify_row(evt.y)
            col = tv.identify_column(evt.x)
        except Exception:
            return
        if not iid:
            return
        if "::" not in str(iid):
            return
        a, b = str(iid).split("::", 1)
        k = (str(a), str(b))
        st = self._style_by_key.get(k)
        if st is None:
            return

        # Columns: 1=vis, 2=name, 3=line color, 4=line width, 5=peak color, 6=label color, 7=peaks, 8=labels
        if col == "#1":
            st["visible"] = not bool(st.get("visible", True))
            self._apply_style(rebuild_peaks=True)
            self._refresh_overlay_tree()
            return
        if col == "#2":
            cur = str(st.get("name", ""))
            new_name = simpledialog.askstring("Legend label", "Label:", initialvalue=cur, parent=self)
            if new_name is None:
                return
            new_name = str(new_name).strip()
            st["name"] = new_name
            try:
                ln = self._line_by_key.get(k)
                if ln is not None:
                    ln.set_label(str(new_name))
            except Exception:
                pass
            self._apply_style(rebuild_peaks=False)
            self._refresh_overlay_tree()
            return
        if col in ("#3", "#5", "#6"):
            prop = "line_color" if col == "#3" else ("peak_color" if col == "#5" else "label_color")
            title = "Pick line color" if prop == "line_color" else ("Pick peak marker color" if prop == "peak_color" else "Pick peak label color")
            try:
                c = colorchooser.askcolor(title=title, parent=self)[1]
            except Exception:
                c = None
            if not c:
                return
            st[prop] = str(c)
            self._apply_style(rebuild_peaks=True)
            self._refresh_overlay_tree()
            return
        if col == "#4":
            try:
                cur = float(st.get("line_width", 1.2))
            except Exception:
                cur = 1.2
            new_lw = simpledialog.askfloat("Line width", "Line width:", initialvalue=float(cur), minvalue=0.1, maxvalue=20.0, parent=self)
            if new_lw is None:
                return
            try:
                st["line_width"] = float(new_lw)
            except Exception:
                return
            self._apply_style(rebuild_peaks=False)
            self._refresh_overlay_tree()
            return
        if col == "#7":
            st["show_peaks"] = not bool(st.get("show_peaks", True))
            self._apply_style(rebuild_peaks=True)
            self._refresh_overlay_tree()
            return
        if col == "#8":
            st["show_labels"] = not bool(st.get("show_labels", True))
            self._apply_style(rebuild_peaks=True)
            self._refresh_overlay_tree()
            return

    def _selected_overlay_key(self) -> Optional[Tuple[str, str]]:
        tv = self._overlay_tree
        if tv is None:
            return None
        try:
            sel = list(tv.selection() or [])
        except Exception:
            sel = []
        if not sel:
            return None
        iid = str(sel[0])
        if "::" not in iid:
            return None
        a, b = iid.split("::", 1)
        return (str(a), str(b))

    def _pick_color_for_selected(self) -> None:
        k = self._selected_overlay_key()
        if k is None:
            return
        st = self._style_by_key.get(k)
        if st is None:
            return
        prop = str(self._style_prop_var.get() or "").strip()
        if prop not in ("Line color", "Peak marker color", "Peak label color"):
            return
        title = f"Pick {prop}"
        try:
            c = colorchooser.askcolor(title=title, parent=self)[1]
        except Exception:
            c = None
        if not c:
            return

        if prop == "Line color":
            st["line_color"] = str(c)
        elif prop == "Peak marker color":
            st["peak_color"] = str(c)
        elif prop == "Peak label color":
            st["label_color"] = str(c)

        self._apply_style(rebuild_peaks=True)
        self._refresh_overlay_tree()

    def _apply_selected_style_to_all(self) -> None:
        k = self._selected_overlay_key()
        if k is None:
            return
        src = self._style_by_key.get(k)
        if src is None:
            return
        prop = str(self._style_prop_var.get() or "").strip()
        for kk, st in (self._style_by_key or {}).items():
            if prop == "Line color":
                st["line_color"] = src.get("line_color")
            elif prop == "Peak marker color":
                st["peak_color"] = src.get("peak_color")
            elif prop == "Peak label color":
                st["label_color"] = src.get("label_color")
        self._apply_style(rebuild_peaks=True)
        self._refresh_overlay_tree()

    def _on_press(self, evt) -> None:
        # Left click drag on labels; double click to edit text
        try:
            if evt is None or getattr(evt, "inaxes", None) is None:
                return
            if evt.xdata is None or evt.ydata is None:
                return
        except Exception:
            return

        # Double click to edit (bond labels or peak labels)
        try:
            if int(getattr(evt, "dblclick", 0) or 0) == 1:
                # Bond labels first
                for artist, idx in list((self._bond_artist_to_idx or {}).items()):
                    try:
                        # Only edit the text object on double-click (avoid editing vline)
                        if artist not in (self._bond_texts or []):
                            continue
                        contains, _ = artist.contains(evt)
                        if not contains:
                            continue
                        try:
                            row = (self._bond_rows or [])[int(idx)]
                        except Exception:
                            return
                        cur = str(artist.get_text() or "")
                        new_txt = simpledialog.askstring("Edit bond label", "Label:", initialvalue=cur, parent=self)
                        if new_txt is None:
                            return
                        row["text"] = str(new_txt)
                        try:
                            artist.set_text(str(new_txt))
                            self._canvas.draw_idle()
                        except Exception:
                            self._rebuild_all_bonds()
                            self._canvas.draw_idle()
                        return
                    except Exception:
                        continue

                for ann in list(self._ann_to_info.keys()):
                    try:
                        contains, _ = ann.contains(evt)
                        if contains:
                            info = self._ann_to_info.get(ann)
                            if info is None:
                                return
                            ws_id, ds_id, pid = info
                            k = (str(ws_id), str(ds_id))
                            cur = str(ann.get_text() or "")
                            new_txt = simpledialog.askstring("Edit peak label", "Label:", initialvalue=cur, parent=self)
                            if new_txt is None:
                                return
                            self._label_override_by_key_pid.setdefault(k, {})[str(pid)] = str(new_txt)
                            self._apply_style(rebuild_peaks=True)
                            if self._overlay_tree is not None:
                                self._refresh_overlay_tree()
                            return
                    except Exception:
                        continue
        except Exception:
            pass

        # Drag
        try:
            if int(getattr(evt, "button", 0) or 0) != 1:
                return
        except Exception:
            return

        # Drag bond label
        for artist, idx in list((self._bond_artist_to_idx or {}).items()):
            try:
                if artist not in (self._bond_texts or []):
                    continue
                contains, _ = artist.contains(evt)
                if not contains:
                    continue
                x0, y0 = artist.get_position()
                self._drag_bond_artist = artist
                self._drag_bond_idx = int(idx)
                self._drag_bond_dx = float(x0) - float(evt.xdata)
                self._drag_bond_dy = float(y0) - float(evt.ydata)
                return
            except Exception:
                continue

        for ann in list(self._ann_to_info.keys()):
            try:
                contains, _ = ann.contains(evt)
                if contains:
                    info = self._ann_to_info.get(ann)
                    if info is None:
                        return
                    x0, y0 = ann.get_position()
                    self._drag_ann = ann
                    self._drag_info = info
                    self._drag_dx = float(x0) - float(evt.xdata)
                    self._drag_dy = float(y0) - float(evt.ydata)
                    return
            except Exception:
                continue

    def _on_motion(self, evt) -> None:
        if self._drag_bond_artist is not None and self._drag_bond_idx is not None:
            try:
                if evt is None or getattr(evt, "inaxes", None) is None:
                    return
                if evt.xdata is None or evt.ydata is None:
                    return
            except Exception:
                return

            try:
                new_x = float(evt.xdata) + float(self._drag_bond_dx)
                new_y = float(evt.ydata) + float(self._drag_bond_dy)
                self._drag_bond_artist.set_position((new_x, new_y))
            except Exception:
                return

            try:
                row = (self._bond_rows or [])[int(self._drag_bond_idx)]
                row["xytext"] = [float(new_x), float(new_y)]
                # Keep vline/x anchor in sync with dragged x
                row["x_cm1"] = float(new_x)
                for ln in list(self._bond_vlines or []):
                    try:
                        if self._bond_artist_to_idx.get(ln) == int(self._drag_bond_idx):
                            ln.set_xdata([float(new_x), float(new_x)])
                    except Exception:
                        continue
            except Exception:
                pass

            try:
                self._canvas.draw_idle()
            except Exception:
                pass
            return

        if self._drag_ann is None or self._drag_info is None:
            return
        try:
            if evt is None or getattr(evt, "inaxes", None) is None:
                return
            if evt.xdata is None or evt.ydata is None:
                return
        except Exception:
            return

        try:
            new_x = float(evt.xdata) + float(self._drag_dx)
            new_y = float(evt.ydata) + float(self._drag_dy)
            self._drag_ann.set_position((new_x, new_y))
        except Exception:
            return

        try:
            ws_id, ds_id, pid = self._drag_info
            k = (str(ws_id), str(ds_id))
            try:
                dx, dy = self._overlay_offset_for_key(k)
            except Exception:
                dx, dy = 0.0, 0.0
            # Store base (un-offset) position so offsets don't double-shift on redraw/export.
            self._label_pos_by_key_pid.setdefault(k, {})[str(pid)] = (float(new_x) - float(dx), float(new_y) - float(dy))
        except Exception:
            pass

        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _on_release(self, evt) -> None:
        self._drag_bond_artist = None
        self._drag_bond_idx = None
        self._drag_bond_dx = 0.0
        self._drag_bond_dy = 0.0
        self._drag_ann = None
        self._drag_info = None
        self._drag_dx = 0.0
        self._drag_dy = 0.0

    def _save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save FTIR export",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All files", "*.*")],
            initialfile=f"{self.default_stem}.png",
        )
        if not path:
            return
        try:
            # Avoid tight_layout/bbox_inches='tight' to prevent UI freezes.
            self._fig.savefig(path, dpi=200)
            try:
                self.app._log("INFO", f"Saved FTIR export: {Path(path).name}")
            except Exception:
                pass
        except Exception as exc:
            messagebox.showerror("FTIR Export", f"Failed to save:\n\n{exc}", parent=self)


class AlignmentDiagnostics(tk.Toplevel):
    def __init__(self, app: "App") -> None:
        super().__init__(app)
        self.app = app
        self._selected_anchor_idx: Optional[int] = None
        self._selected_anchor_uv_rt: Optional[float] = None
        self._selected_anchor_ms_rt: Optional[float] = None

        self.title("Alignment Diagnostics")
        try:
            self.geometry("1050x720")
        except Exception:
            pass

        outer = ttk.Frame(self, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        # Plot area
        plotf = ttk.Frame(outer)
        plotf.grid(row=0, column=0, sticky="nsew")
        plotf.rowconfigure(0, weight=1)
        plotf.columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(9, 5), dpi=100)
        try:
            self._ax_map, self._ax_res = self._fig.subplots(2, 1, sharex=True)
        except Exception:
            self._ax_map = self._fig.add_subplot(2, 1, 1)
            self._ax_res = self._fig.add_subplot(2, 1, 2)

        self._canvas = FigureCanvasTkAgg(self._fig, master=plotf)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        try:
            toolbar = NavigationToolbar2Tk(self._canvas, plotf)
            toolbar.update()
        except Exception:
            toolbar = None

        # Bottom area: stats + table + buttons
        bottom = ttk.Frame(outer)
        bottom.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=3)

        statsf = ttk.LabelFrame(bottom, text="Stats", padding=10)
        statsf.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        for r in range(5):
            statsf.rowconfigure(r, weight=0)
        statsf.columnconfigure(0, weight=1)

        self._stat_n = tk.StringVar(value="N: -")
        self._stat_mae = tk.StringVar(value="MAE: -")
        self._stat_med = tk.StringVar(value="Median |res|: -")
        self._stat_max = tk.StringVar(value="Max |res|: -")
        self._stat_p90 = tk.StringVar(value="P90 |res|: -")

        ttk.Label(statsf, textvariable=self._stat_n).grid(row=0, column=0, sticky="w")
        ttk.Label(statsf, textvariable=self._stat_mae).grid(row=1, column=0, sticky="w")
        ttk.Label(statsf, textvariable=self._stat_med).grid(row=2, column=0, sticky="w")
        ttk.Label(statsf, textvariable=self._stat_max).grid(row=3, column=0, sticky="w")
        ttk.Label(statsf, textvariable=self._stat_p90).grid(row=4, column=0, sticky="w")

        tablef = ttk.LabelFrame(bottom, text="Anchors", padding=10)
        tablef.grid(row=0, column=1, sticky="nsew")
        tablef.rowconfigure(0, weight=1)
        tablef.columnconfigure(0, weight=1)

        cols = ("idx", "uv", "ms", "res")
        self._tree = ttk.Treeview(tablef, columns=cols, show="headings", height=7, selectmode="browse")
        self._tree.heading("idx", text="#")
        self._tree.heading("uv", text="UV_RT")
        self._tree.heading("ms", text="MS_RT")
        self._tree.heading("res", text="Residual (min)")
        self._tree.column("idx", width=40, anchor="e", stretch=False)
        self._tree.column("uv", width=120, anchor="e", stretch=False)
        self._tree.column("ms", width=120, anchor="e", stretch=False)
        self._tree.column("res", width=140, anchor="e", stretch=False)
        self._tree.grid(row=0, column=0, sticky="nsew")

        ysb = ttk.Scrollbar(tablef, orient="vertical", command=self._tree.yview)
        ysb.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=ysb.set)

        btns = ttk.Frame(outer)
        btns.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        btns.columnconfigure(0, weight=1)

        self._btn_refresh = ttk.Button(btns, text="Refresh", command=self.refresh)
        self._btn_remove = ttk.Button(btns, text="Remove selected anchor", command=self._remove_selected)
        self._btn_export = ttk.Button(btns, text="Export anchors CSV…", command=self._export_csv)
        close_btn = ttk.Button(btns, text="Close", command=self._on_close)

        self._btn_refresh.pack(side=tk.LEFT)
        self._btn_remove.pack(side=tk.LEFT, padx=(10, 0))
        self._btn_export.pack(side=tk.LEFT, padx=(10, 0))
        close_btn.pack(side=tk.RIGHT)

        try:
            self._tree.bind("<<TreeviewSelect>>", self._on_select, add=True)
        except Exception:
            pass

        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

        self.refresh()

    def _on_close(self) -> None:
        try:
            if getattr(self.app, "_alignment_diag_win", None) is self:
                self.app._alignment_diag_win = None
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def _get_anchors(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        uv = getattr(self.app, "_uv_ms_align_uv_rts", None)
        ms = getattr(self.app, "_uv_ms_align_ms_rts", None)
        if uv is None or ms is None:
            return None, None
        uv = np.asarray(uv, dtype=float)
        ms = np.asarray(ms, dtype=float)
        if uv.size < 1 or ms.size < 1 or uv.size != ms.size:
            return None, None
        return uv, ms

    def _compute_residuals(self, uv: np.ndarray, ms: np.ndarray) -> np.ndarray:
        try:
            ms_pred = np.asarray([float(self.app._map_uv_to_ms_rt(float(u))) for u in uv.tolist()], dtype=float)
        except Exception:
            ms_pred = uv + float(getattr(self.app, "_uv_ms_rt_offset_min", 0.0))
        return np.asarray(ms, dtype=float) - np.asarray(ms_pred, dtype=float)

    def _update_stats(self, residuals: np.ndarray) -> None:
        res = np.asarray(residuals, dtype=float)
        if res.size == 0:
            self._stat_n.set("N: -")
            self._stat_mae.set("MAE: -")
            self._stat_med.set("Median |res|: -")
            self._stat_max.set("Max |res|: -")
            self._stat_p90.set("P90 |res|: -")
            return
        a = np.abs(res)
        self._stat_n.set(f"N: {int(res.size)}")
        self._stat_mae.set(f"MAE: {float(np.mean(a)):.5f} min")
        self._stat_med.set(f"Median |res|: {float(np.median(a)):.5f} min")
        self._stat_max.set(f"Max |res|: {float(np.max(a)):.5f} min")
        try:
            p90 = float(np.percentile(a, 90))
            self._stat_p90.set(f"P90 |res|: {p90:.5f} min")
        except Exception:
            self._stat_p90.set("P90 |res|: -")

    def _refresh_plots(self) -> None:
        uv, ms = self._get_anchors()
        self._ax_map.clear()
        self._ax_res.clear()

        if uv is None or ms is None or uv.size < 1:
            self._ax_map.set_title("Anchor mapping (UV→MS)")
            self._ax_map.set_xlabel("UV RT (min)")
            self._ax_map.set_ylabel("MS RT (min)")
            self._ax_map.text(0.5, 0.5, "No anchors available", ha="center", va="center", transform=self._ax_map.transAxes)
            self._ax_res.set_title("Residuals")
            self._ax_res.set_xlabel("UV RT (min)")
            self._ax_res.set_ylabel("Residual (min)")
            self._update_stats(np.asarray([], dtype=float))
            try:
                self._canvas.draw_idle()
            except Exception:
                pass
            return

        order = np.argsort(uv)
        uv = uv[order]
        ms = ms[order]
        residuals = self._compute_residuals(uv, ms)
        self._update_stats(residuals)

        # A) mapping scatter + curve
        self._ax_map.set_title("Anchor mapping (UV→MS)")
        self._ax_map.set_xlabel("UV RT (min)")
        self._ax_map.set_ylabel("MS RT (min)")

        self._ax_map.scatter(uv, ms, s=22, color=PRIMARY_TEAL, alpha=0.85, label="anchors")
        if self._selected_anchor_idx is not None and 0 <= int(self._selected_anchor_idx) < int(uv.size):
            i = int(self._selected_anchor_idx)
            self._ax_map.scatter([float(uv[i])], [float(ms[i])], s=70, color="red", alpha=0.9, label="selected")

        if uv.size >= 2:
            u0 = float(np.min(uv))
            u1 = float(np.max(uv))
            u = np.linspace(u0, u1, 200)
            try:
                m = np.asarray([float(self.app._map_uv_to_ms_rt(float(x))) for x in u.tolist()], dtype=float)
            except Exception:
                m = u + float(getattr(self.app, "_uv_ms_rt_offset_min", 0.0))
            self._ax_map.plot(u, m, color="black", linewidth=1.3, alpha=0.8, label="mapping")

            # Optional reference line y = x + offset
            try:
                off = float(getattr(self.app, "_uv_ms_rt_offset_min", 0.0))
                self._ax_map.plot(u, u + off, color="gray", linewidth=1.0, alpha=0.35, linestyle="--", label="UV+offset")
            except Exception:
                pass

        try:
            self._ax_map.legend(loc="best", fontsize=8, frameon=False)
        except Exception:
            pass

        # B) residuals
        self._ax_res.set_title("Residuals (anchor MS − mapped MS)")
        self._ax_res.set_xlabel("UV RT (min)")
        self._ax_res.set_ylabel("Residual (min)")
        self._ax_res.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        self._ax_res.scatter(uv, residuals, s=22, color=PRIMARY_TEAL, alpha=0.85)
        if self._selected_anchor_idx is not None and 0 <= int(self._selected_anchor_idx) < int(uv.size):
            i = int(self._selected_anchor_idx)
            self._ax_res.scatter([float(uv[i])], [float(residuals[i])], s=70, color="red", alpha=0.9)

        try:
            self._fig.tight_layout()
        except Exception:
            pass
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

        # Buttons enable/disable
        try:
            can_remove = bool(self._selected_anchor_idx is not None) and (uv.size >= 4)
            self._btn_remove.configure(state=("normal" if can_remove else "disabled"))
        except Exception:
            pass

    def _refresh_table(self) -> None:
        try:
            for iid in self._tree.get_children(""):
                self._tree.delete(iid)
        except Exception:
            pass

        uv, ms = self._get_anchors()
        if uv is None or ms is None or uv.size < 1:
            return
        order = np.argsort(uv)
        uv = uv[order]
        ms = ms[order]
        residuals = self._compute_residuals(uv, ms)
        for i in range(int(uv.size)):
            self._tree.insert(
                "",
                "end",
                values=(
                    f"{i + 1:d}",
                    f"{float(uv[i]):.5f}",
                    f"{float(ms[i]):.5f}",
                    f"{float(residuals[i]):+.6f}",
                ),
            )

    def refresh(self) -> None:
        self._refresh_table()
        self._refresh_plots()

    def _on_select(self, event=None) -> None:
        sel = None
        try:
            s = self._tree.selection()
            if s:
                sel = s[0]
        except Exception:
            sel = None
        if not sel:
            self._selected_anchor_idx = None
            self._refresh_plots()
            return
        try:
            vals = self._tree.item(sel, "values")
        except Exception:
            return
        if not vals or len(vals) < 3:
            return
        try:
            idx = int(str(vals[0])) - 1
        except Exception:
            idx = None
        try:
            uv_rt = float(str(vals[1]))
            ms_rt = float(str(vals[2]))
        except Exception:
            return
        self._selected_anchor_idx = idx
        self._selected_anchor_uv_rt = float(uv_rt)
        self._selected_anchor_ms_rt = float(ms_rt)

        # Jump app selection
        try:
            self.app._selected_rt_min = float(uv_rt)
        except Exception:
            pass
        try:
            rts = getattr(self.app, "_filtered_rts", None)
            if rts is not None and np.asarray(rts).size > 0:
                rts = np.asarray(rts, dtype=float)
                nearest_idx = int(np.argmin(np.abs(rts - float(ms_rt))))
                self.app._show_spectrum_for_index(nearest_idx)
        except Exception:
            pass
        try:
            if self.app._active_uv_session() is not None:
                self.app._plot_uv()
        except Exception:
            pass

        self._refresh_plots()

    def _remove_selected(self) -> None:
        uv, ms = self._get_anchors()
        if uv is None or ms is None or uv.size < 1:
            messagebox.showinfo("Remove anchor", "No anchors available.", parent=self)
            return
        if self._selected_anchor_uv_rt is None or self._selected_anchor_ms_rt is None:
            messagebox.showinfo("Remove anchor", "Select an anchor first.", parent=self)
            return
        if uv.size <= 3:
            messagebox.showinfo("Remove anchor", "Need at least 3 anchors to keep alignment. Removal disabled.", parent=self)
            return
        # Remove the closest matching anchor in the app arrays (robust vs sorting).
        try:
            target_uv = float(self._selected_anchor_uv_rt)
            target_ms = float(self._selected_anchor_ms_rt)
            i = int(np.argmin(np.abs(np.asarray(uv, dtype=float) - target_uv)))
            # If multiple anchors share similar UV RT, also sanity-check MS RT.
            if not np.isfinite(float(ms[i])) or abs(float(ms[i]) - target_ms) > 0.25:
                j = int(np.argmin(np.abs(np.asarray(ms, dtype=float) - target_ms)))
                if 0 <= j < int(uv.size):
                    i = j
        except Exception:
            return
        if not (0 <= int(i) < int(uv.size)):
            return
        try:
            uv2 = np.delete(uv, i)
            ms2 = np.delete(ms, i)
        except Exception:
            return
        if uv2.size < 3 or ms2.size < 3:
            messagebox.showerror("Remove anchor", "Removal would leave fewer than 3 anchors. Aborting.", parent=self)
            return

        self.app._uv_ms_align_uv_rts = np.asarray(uv2, dtype=float)
        self.app._uv_ms_align_ms_rts = np.asarray(ms2, dtype=float)

        # If auto-align was enabled, keep it enabled; otherwise leave it off.
        try:
            if bool(self.app.uv_ms_align_enabled_var.get()):
                self.app.uv_ms_align_enabled_var.set(True)
        except Exception:
            pass

        self._selected_anchor_idx = None
        self._selected_anchor_uv_rt = None
        self._selected_anchor_ms_rt = None
        try:
            if self.app._active_uv_session() is not None:
                self.app._plot_uv()
        except Exception:
            pass
        self.refresh()

    def _export_csv(self) -> None:
        uv, ms = self._get_anchors()
        if uv is None or ms is None or uv.size < 1:
            messagebox.showinfo("Export", "No anchors available to export.", parent=self)
            return
        residuals = self._compute_residuals(uv, ms)
        abs_res = np.abs(np.asarray(residuals, dtype=float))

        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export anchors CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="uv_ms_alignment_anchors.csv",
        )
        if not path:
            return
        df = pd.DataFrame(
            {
                "uv_rt_min": np.asarray(uv, dtype=float),
                "ms_rt_min": np.asarray(ms, dtype=float),
                "residual_min": np.asarray(residuals, dtype=float),
            }
        )
        try:
            df = df.sort_values("uv_rt_min", ascending=True)
        except Exception:
            pass

        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("# UV↔MS alignment anchors\n")
                try:
                    f.write(f"# N={int(abs_res.size)}\n")
                    f.write(f"# MAE_min={float(np.mean(abs_res)):.8f}\n")
                    f.write(f"# MedianAbs_min={float(np.median(abs_res)):.8f}\n")
                    f.write(f"# MaxAbs_min={float(np.max(abs_res)):.8f}\n")
                    f.write(f"# P90Abs_min={float(np.percentile(abs_res, 90)):.8f}\n")
                except Exception:
                    pass
                df.to_csv(f, index=False)
        except Exception as e:
            messagebox.showerror("Export", f"Failed to export CSV:\n\n{e}", parent=self)
            return
        messagebox.showinfo("Export", f"Saved anchors CSV:\n\n{path}", parent=self)


class SIMWindow(tk.Toplevel):
    def __init__(
        self,
        app: "App",
        *,
        target_mz: float,
        tol_value: float,
        tol_unit: str,
        use_current_polarity: bool,
    ) -> None:
        super().__init__(app)
        self.app = app

        try:
            self.app._register_nonmodal_dialog(self)
        except Exception:
            pass

        self._target_mz = float(target_mz)
        self._tol_value = float(tol_value)
        self._tol_unit = str(tol_unit or "ppm")
        self._use_current_polarity = bool(use_current_polarity)

        self._rts: Optional[np.ndarray] = None
        self._ints: Optional[np.ndarray] = None

        self._mzml_path: Optional[Path] = None
        self._polarity_filter: str = "all"

        self._marker_artist = None
        self._line_artist = None
        self._no_signal_artist = None

        self.title("EIC chromatogram")
        try:
            self.geometry("980x560")
        except Exception:
            pass

        outer = ttk.Frame(self, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        self._header_var = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self._header_var, foreground=TEXT_MUTED).grid(row=0, column=0, sticky="w")

        plotf = ttk.Frame(outer)
        plotf.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        plotf.rowconfigure(0, weight=1)
        plotf.columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(9, 4), dpi=100)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plotf)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        try:
            toolbar = NavigationToolbar2Tk(self._canvas, plotf)
            toolbar.update()
        except Exception:
            toolbar = None

        btns = ttk.Frame(outer)
        btns.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        export_btn = ttk.Button(btns, text="Export EIC CSV…", command=self._export_csv)
        close_btn = ttk.Button(btns, text="Close", command=self._on_close)
        export_btn.pack(side=tk.LEFT)
        close_btn.pack(side=tk.RIGHT)

        try:
            self._cid_click = self._canvas.mpl_connect("button_press_event", self._on_click)
        except Exception:
            self._cid_click = None

        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

        # Subscribe to app navigation + mzML switching
        try:
            self.app._register_ms_position_listener(self._on_app_ms_position_changed)
        except Exception:
            pass
        try:
            self.app._register_active_session_listener(self._on_app_active_session_changed)
        except Exception:
            pass

        # Initial data load (cached or threaded)
        self.app._run_sim_for_window(self)

    @property
    def sim_params(self) -> Tuple[float, float, str, bool]:
        return (float(self._target_mz), float(self._tol_value), str(self._tol_unit), bool(self._use_current_polarity))

    def _on_close(self) -> None:
        try:
            self.app._unregister_ms_position_listener(self._on_app_ms_position_changed)
        except Exception:
            pass
        try:
            self.app._unregister_active_session_listener(self._on_app_active_session_changed)
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def _fmt_tol(self) -> str:
        unit = (self._tol_unit or "ppm").strip()
        if unit.lower() == "ppm":
            return f"{float(self._tol_value):g} ppm"
        return f"{float(self._tol_value):g} Da"

    def set_data(self, *, mzml_path: Path, rts: np.ndarray, intensities: np.ndarray, polarity_filter: str) -> None:
        self._mzml_path = Path(mzml_path) if mzml_path is not None else None
        self._polarity_filter = str(polarity_filter or "all")
        self._rts = np.asarray(rts, dtype=float)
        self._ints = np.asarray(intensities, dtype=float)
        self._redraw_plot()
        self._update_marker_from_app()

    def _header_text(self) -> str:
        mzml_name = self._mzml_path.name if self._mzml_path is not None else "(no mzML)"
        pol_txt = (self._polarity_filter if bool(self._use_current_polarity) else "all")
        return (
            f"{mzml_name}   |   m/z {float(self._target_mz):.6f}"
            f"   |   tol {self._fmt_tol()}   |   polarity {pol_txt}"
        )

    def _redraw_plot(self) -> None:
        self._ax.clear()
        self._header_var.set(self._header_text())

        rts = self._rts
        ints = self._ints
        if rts is None or ints is None or rts.size == 0 or ints.size == 0:
            self._ax.set_title("EIC chromatogram")
            self._ax.set_xlabel("Retention time (min)")
            self._ax.set_ylabel("Intensity")
            self._ax.text(0.5, 0.5, "No EIC data", ha="center", va="center", transform=self._ax.transAxes)
            self._line_artist = None
            self._marker_artist = None
            self._no_signal_artist = None
            self.app._apply_plot_style_to_axes(self._fig, [self._ax])
            try:
                self._canvas.draw_idle()
            except Exception:
                pass
            return

        label = f"EIC m/z {float(self._target_mz):.4f} ± {self._fmt_tol()}"
        (self._line_artist,) = self._ax.plot(rts, ints, linewidth=1.2, color=PRIMARY_TEAL, label=label)
        self._ax.set_title("EIC chromatogram")
        self._ax.set_xlabel("Retention time (min)")
        self._ax.set_ylabel("Intensity")
        try:
            self._ax.legend(loc="best", fontsize=9, frameon=False)
        except Exception:
            pass

        try:
            if float(np.nanmax(ints)) <= 0.0:
                self._no_signal_artist = self._ax.text(
                    0.5,
                    0.9,
                    "No signal detected",
                    ha="center",
                    va="center",
                    transform=self._ax.transAxes,
                    color=ACCENT_MAGENTA,
                )
            else:
                self._no_signal_artist = None
        except Exception:
            self._no_signal_artist = None

        self.app._apply_plot_style_to_axes(self._fig, [self._ax])
        try:
            self._fig.tight_layout()
        except Exception:
            pass
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _update_marker_from_app(self) -> None:
        meta = getattr(self.app, "_current_spectrum_meta", None)
        if meta is None:
            self._set_marker_rt(None)
            return
        try:
            self._set_marker_rt(float(meta.rt_min))
        except Exception:
            self._set_marker_rt(None)

    def _set_marker_rt(self, rt: Optional[float]) -> None:
        rts = self._rts
        ints = self._ints
        if rts is None or ints is None or rts.size == 0 or ints.size == 0:
            return

        if rt is None or not np.isfinite(float(rt)):
            try:
                if self._marker_artist is not None:
                    self._marker_artist.remove()
            except Exception:
                pass
            self._marker_artist = None
            try:
                self._canvas.draw_idle()
            except Exception:
                pass
            return

        i = int(np.argmin(np.abs(rts - float(rt))))
        x = float(rts[i])
        y = float(ints[i])
        try:
            if self._marker_artist is not None:
                self._marker_artist.remove()
        except Exception:
            pass
        self._marker_artist = self._ax.scatter(
            [x],
            [y],
            s=50,
            color=ACCENT_ORANGE,
            edgecolors=TEXT_DARK,
            linewidths=0.4,
        )
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _on_click(self, event) -> None:
        if event is None or getattr(event, "inaxes", None) != self._ax:
            return
        if getattr(event, "xdata", None) is None:
            return
        if self._rts is None or self._rts.size == 0:
            return
        if self.app._filtered_rts is None or self.app._filtered_rts.size == 0 or not self.app._filtered_meta:
            messagebox.showinfo("EIC", "No mzML scans available for navigation.", parent=self)
            return
        rt_clicked = float(event.xdata)
        rt_sim = float(self._rts[int(np.argmin(np.abs(np.asarray(self._rts, dtype=float) - rt_clicked)))])
        nearest_idx = int(np.argmin(np.abs(np.asarray(self.app._filtered_rts, dtype=float) - rt_sim)))
        self.app._show_spectrum_for_index(nearest_idx)

    def _on_app_ms_position_changed(self, rt_min: Optional[float]) -> None:
        self._set_marker_rt(rt_min)

    def _on_app_active_session_changed(self) -> None:
        self.app._run_sim_for_window(self)

    def _export_csv(self) -> None:
        if self._rts is None or self._ints is None or self._rts.size == 0 or self._ints.size == 0:
            messagebox.showinfo("Export", "No EIC data to export.", parent=self)
            return
        mzml_path = self._mzml_path
        stem = (mzml_path.stem if mzml_path is not None else "mzML")
        unit = (self._tol_unit or "ppm").strip()
        mz_s = f"{float(self._target_mz):.4f}".rstrip("0").rstrip(".")
        tol_s = f"{float(self._tol_value):g}{unit}"
        default_name = f"{stem}_EIC_mz{mz_s}_{tol_s}.csv"
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export EIC CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=default_name,
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["rt_min", "intensity"])
                for rt_v, inten_v in zip(self._rts.tolist(), self._ints.tolist()):
                    w.writerow([float(rt_v), float(inten_v)])
        except Exception as exc:
            messagebox.showerror("Export", f"Failed to export EIC CSV:\n\n{exc}", parent=self)
            return
        messagebox.showinfo("Export", f"Saved EIC CSV:\n\n{path}", parent=self)


class InstructionWindow(tk.Toplevel):
    def __init__(self, app: "App") -> None:
        super().__init__(app)
        self.app = app

        try:
            self.app._register_nonmodal_dialog(self)
        except Exception:
            pass

        self.title("Instructions / User Guide")
        try:
            self.geometry("1100x760")
        except Exception:
            pass

        self._active_section: Optional[str] = None
        self._match_ranges: List[Tuple[str, str]] = []
        self._match_pos: int = 0

        outer = ttk.Frame(self, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        # Top: search
        top = ttk.Frame(outer)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Search").grid(row=0, column=0, sticky="w")
        self._search_var = tk.StringVar(value="")
        ent = ttk.Entry(top, textvariable=self._search_var)
        ent.grid(row=0, column=1, sticky="ew", padx=(10, 10))
        find_btn = ttk.Button(top, text="Find", command=self._apply_search)
        next_btn = ttk.Button(top, text="Next", command=lambda: self._jump_match(+1))
        prev_btn = ttk.Button(top, text="Prev", command=lambda: self._jump_match(-1))
        copy_btn = ttk.Button(top, text="Copy section", command=self._copy_section)

        find_btn.grid(row=0, column=2, sticky="e")
        prev_btn.grid(row=0, column=3, sticky="e", padx=(6, 0))
        next_btn.grid(row=0, column=4, sticky="e", padx=(6, 0))
        copy_btn.grid(row=0, column=5, sticky="e", padx=(12, 0))

        try:
            ent.bind("<Return>", lambda e: (self._apply_search(), "break"))
        except Exception:
            pass

        # Middle: left nav + right text
        mid = ttk.Frame(outer)
        mid.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=0)
        mid.columnconfigure(1, weight=1)

        navf = ttk.LabelFrame(mid, text="Sections", padding=8)
        navf.grid(row=0, column=0, sticky="ns")
        navf.rowconfigure(0, weight=1)

        self._nav = ttk.Treeview(navf, show="tree", selectmode="browse", height=20)
        self._nav.grid(row=0, column=0, sticky="ns")
        nav_scroll = ttk.Scrollbar(navf, orient="vertical", command=self._nav.yview)
        nav_scroll.grid(row=0, column=1, sticky="ns")
        self._nav.configure(yscrollcommand=nav_scroll.set)

        for sec in GUIDE_SECTIONS:
            self._nav.insert("", "end", iid=sec, text=sec)

        bodyf = ttk.LabelFrame(mid, text="Guide", padding=8)
        bodyf.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        bodyf.rowconfigure(0, weight=1)
        bodyf.columnconfigure(0, weight=1)

        self._text = tk.Text(bodyf, wrap="word", height=28)
        self._text.grid(row=0, column=0, sticky="nsew")
        txt_scroll = ttk.Scrollbar(bodyf, orient="vertical", command=self._text.yview)
        txt_scroll.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=txt_scroll.set)

        try:
            self._text.tag_configure("search_hit", background=BG_LIGHT)
            self._text.tag_configure("search_active", background=LIGHT_TEAL)
        except Exception:
            pass

        # Bottom: build info
        bottom = ttk.Frame(outer)
        bottom.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(0, weight=1)

        try:
            built = datetime.date.today().isoformat()
        except Exception:
            built = ""
        info = f"{APP_NAME} • v{APP_VERSION} • {built}".strip(" •")
        ttk.Label(bottom, text=info).grid(row=0, column=0, sticky="w")
        ttk.Button(bottom, text="Close", command=self._on_close).grid(row=0, column=1, sticky="e")

        try:
            self._nav.bind("<<TreeviewSelect>>", self._on_select_section, add=True)
        except Exception:
            pass

        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

        # Default section
        try:
            self._nav.selection_set(GUIDE_SECTIONS[0])
            self._nav.see(GUIDE_SECTIONS[0])
        except Exception:
            pass
        self._set_section(GUIDE_SECTIONS[0])

    def _on_close(self) -> None:
        try:
            if getattr(self.app, "_instructions_win", None) is self:
                self.app._instructions_win = None
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def _set_section(self, section: str) -> None:
        sec = str(section)
        self._active_section = sec
        content = GUIDE.get(sec, "")
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", content)
        self._text.configure(state="disabled")
        self._clear_search_highlights()

    def _on_select_section(self, _evt=None) -> None:
        try:
            sel = self._nav.selection()
        except Exception:
            sel = ()
        if not sel:
            return
        sec = str(sel[0])
        self._set_section(sec)
        # Re-apply search term to new section (if any)
        try:
            term = (self._search_var.get() or "").strip()
        except Exception:
            term = ""
        if term:
            self._apply_search()

    def _copy_section(self) -> None:
        sec = (self._active_section or "").strip()
        if not sec:
            return
        text = GUIDE.get(sec, "")
        if not text:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update_idletasks()
        except Exception:
            pass

    def _clear_search_highlights(self) -> None:
        self._match_ranges = []
        self._match_pos = 0
        try:
            self._text.configure(state="normal")
            self._text.tag_remove("search_hit", "1.0", "end")
            self._text.tag_remove("search_active", "1.0", "end")
        except Exception:
            pass
        finally:
            try:
                self._text.configure(state="disabled")
            except Exception:
                pass

    def _apply_search(self) -> None:
        try:
            term = (self._search_var.get() or "").strip()
        except Exception:
            term = ""
        self._clear_search_highlights()
        if not term:
            return

        # Find all matches (case-insensitive)
        try:
            self._text.configure(state="normal")
            start = "1.0"
            while True:
                idx = self._text.search(term, start, nocase=True, stopindex="end")
                if not idx:
                    break
                end = f"{idx}+{len(term)}c"
                self._match_ranges.append((idx, end))
                self._text.tag_add("search_hit", idx, end)
                start = end
        except Exception:
            self._match_ranges = []
        finally:
            try:
                self._text.configure(state="disabled")
            except Exception:
                pass

        if not self._match_ranges:
            return
        self._match_pos = 0
        self._scroll_to_match(self._match_pos)

    def _scroll_to_match(self, pos: int) -> None:
        if not self._match_ranges:
            return
        p = int(max(0, min(int(pos), len(self._match_ranges) - 1)))
        self._match_pos = p
        try:
            self._text.configure(state="normal")
            self._text.tag_remove("search_active", "1.0", "end")
            a, b = self._match_ranges[p]
            self._text.tag_add("search_active", a, b)
            self._text.see(a)
        except Exception:
            pass
        finally:
            try:
                self._text.configure(state="disabled")
            except Exception:
                pass

    def _jump_match(self, step: int) -> None:
        if not self._match_ranges:
            return
        n = int(len(self._match_ranges))
        if n <= 0:
            return
        self._scroll_to_match((int(self._match_pos) + int(step)) % n)


class LabelExplanationWindow(tk.Toplevel):
    def __init__(self, app: "App", *, title: str = "Explain Label") -> None:
        super().__init__(app)
        self.app = app

        try:
            self.app._register_nonmodal_dialog(self)
        except Exception:
            pass

        self.title(str(title))
        try:
            self.geometry("720x640")
        except Exception:
            pass

        outer = ttk.Frame(self, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self._text = tk.Text(outer, wrap="word")
        self._text.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(outer, orient="vertical", command=self._text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=sb.set)
        try:
            self._text.configure(state="disabled")
        except Exception:
            pass

        bottom = ttk.Frame(outer)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(0, weight=1)
        ttk.Button(bottom, text="Close", command=self._on_close).grid(row=0, column=1, sticky="e")

        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

    def set_content(self, text: str) -> None:
        try:
            self._text.configure(state="normal")
            self._text.delete("1.0", "end")
            self._text.insert("1.0", str(text))
        except Exception:
            pass
        finally:
            try:
                self._text.configure(state="disabled")
            except Exception:
                pass

    def _on_close(self) -> None:
        try:
            if getattr(self.app, "_label_explain_win", None) is self:
                self.app._label_explain_win = None
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass


class LCMSView(ttk.Frame):
    """LCMS module UI.

    Builds the existing LCMS UI inside a tab, while reusing the App's existing LCMS
    callbacks/state to avoid behavior changes.
    """

    def __init__(self, parent: tk.Widget, app: "App", workspace: Workspace) -> None:
        super().__init__(parent)
        self.app = app
        self.workspace = workspace
        self._build()

    def _build(self) -> None:
        a = self.app

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=(6, 6, 6, 6))
        toolbar.grid(row=0, column=0, sticky="ew")

        content = ttk.Frame(self)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        # Let the user resize left controls vs plots
        paned = ttk.Panedwindow(content, orient="horizontal")
        paned.grid(row=0, column=0, sticky="nsew")

        # --- Toolbar ---
        btn_w = 14
        b_open = ttk.Button(toolbar, text="Open mzML…", command=a._open_mzml, width=btn_w)
        b_open.pack(side=tk.LEFT)
        ToolTip.attach(b_open, TOOLTIP_TEXT["open_mzml"])

        b_uv = ttk.Button(toolbar, text="Add UV…", command=a._open_uv_csv_single, width=btn_w)
        b_uv.pack(side=tk.LEFT, padx=(8, 0))
        ToolTip.attach(b_uv, TOOLTIP_TEXT["load_uv_csv"])

        export_btn = ttk.Menubutton(toolbar, text="Export…", width=btn_w)
        export_btn.pack(side=tk.LEFT, padx=(18, 0))
        export_menu = tk.Menu(export_btn, tearoff=0)
        export_menu.add_command(label="Export Spectrum CSV…\tCtrl+E", command=a._export_spectrum_csv)
        export_menu.add_command(label="Export All Labels (Excel)…", command=a._export_all_labels_xlsx)
        export_menu.add_separator()
        export_menu.add_command(label="Export Overlay TIC CSV…", command=a._export_overlay_tic_csv)
        export_menu.add_command(label="Export Overlay Spectra…", command=a._export_overlay_spectra)
        export_menu.add_separator()
        export_menu.add_command(label="Open TIC Window…", command=a._open_tic_window)
        export_menu.add_command(label="Open Spectrum Window…", command=a._open_spectrum_window)
        export_menu.add_command(label="Open UV Window…", command=a._open_uv_window)
        export_menu.add_separator()
        export_menu.add_command(label="Save TIC Plot…", command=a._save_tic_plot)
        export_menu.add_command(label="Save Spectrum Plot…", command=a._save_spectrum_plot)
        export_menu.add_command(label="Save UV Plot…", command=a._save_uv_plot)
        export_btn.configure(menu=export_menu)
        ToolTip.attach(export_btn, TOOLTIP_TEXT["export_dropdown"])

        b_reset = ttk.Button(toolbar, text="Reset View", command=a._reset_view_all, width=btn_w)
        b_reset.pack(side=tk.LEFT, padx=(8, 0))
        ToolTip.attach(b_reset, TOOLTIP_TEXT["reset_view"])

        overlay_frame = ttk.Frame(toolbar)
        overlay_frame.pack(side=tk.LEFT, padx=(18, 0))
        ov_start = ttk.Button(overlay_frame, text="Overlay Selected", command=a._start_overlay_selected)
        ov_start.pack(side=tk.LEFT)
        ov_clear = ttk.Button(overlay_frame, text="Clear Overlay", command=a._clear_overlay)
        ov_clear.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(overlay_frame, text="Mode").pack(side=tk.LEFT, padx=(12, 0))
        overlay_mode = ttk.Combobox(
            overlay_frame,
            textvariable=a._overlay_mode_var,
            values=["Stacked", "Normalized", "Offset", "Percent of max"],
            state="readonly",
            width=14,
        )
        overlay_mode.pack(side=tk.LEFT, padx=(6, 0))
        overlay_mode.bind("<<ComboboxSelected>>", lambda _e: a._refresh_overlay_view())
        ttk.Label(overlay_frame, text="Colors").pack(side=tk.LEFT, padx=(12, 0))
        overlay_colors = ttk.Combobox(
            overlay_frame,
            textvariable=a._overlay_scheme_var,
            values=a._overlay_scheme_options(),
            state="readonly",
            width=18,
        )
        overlay_colors.pack(side=tk.LEFT, padx=(6, 0))
        overlay_colors.bind("<<ComboboxSelected>>", lambda _e: a._on_overlay_scheme_changed())
        ov_pick = ttk.Button(overlay_frame, text="Pick hue…", command=a._pick_overlay_single_hue_color)
        ov_pick.pack(side=tk.LEFT, padx=(6, 0))
        ov_uv = ttk.Checkbutton(overlay_frame, text="Show UV overlays", variable=a._overlay_show_uv_var, command=a._refresh_overlay_view)
        ov_uv.pack(
            side=tk.LEFT, padx=(10, 0)
        )
        ov_stack = ttk.Checkbutton(overlay_frame, text="Stack spectra", variable=a._overlay_stack_spectra_var, command=a._refresh_overlay_view)
        ov_stack.pack(
            side=tk.LEFT, padx=(10, 0)
        )
        ov_persist = ttk.Checkbutton(overlay_frame, text="Persist overlay", variable=a._overlay_persist_var)
        ov_persist.pack(side=tk.LEFT, padx=(10, 0))
        ToolTip.attach(ov_start, TOOLTIP_TEXT["overlay_selected"])
        ToolTip.attach(ov_clear, TOOLTIP_TEXT["overlay_clear"])
        ToolTip.attach(overlay_mode, TOOLTIP_TEXT["overlay_mode"])
        ToolTip.attach(overlay_colors, TOOLTIP_TEXT["overlay_colors"])
        ToolTip.attach(ov_pick, TOOLTIP_TEXT["overlay_pick_hue"])
        ToolTip.attach(ov_uv, TOOLTIP_TEXT["overlay_uv"])
        ToolTip.attach(ov_stack, TOOLTIP_TEXT["overlay_stack"])
        ToolTip.attach(ov_persist, TOOLTIP_TEXT["overlay_persist"])

        hint = ttk.Label(toolbar, text="Tip: Click TIC/UV to select RT • Right-click label to edit")
        hint.pack(side=tk.RIGHT)

        # --- Left Control Panel (scrollable) ---
        left_outer = ttk.Frame(paned)
        left_outer.columnconfigure(0, weight=1)
        left_outer.rowconfigure(0, weight=1)

        left_canvas = tk.Canvas(left_outer, highlightthickness=0, bd=0, background=BG_LIGHT)
        left_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scroll.grid(row=0, column=1, sticky="ns")

        left = ttk.Frame(left_canvas)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        left_win = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _sync_left_scrollregion(_evt=None) -> None:
            try:
                left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            except Exception:
                pass

        def _sync_left_width(_evt=None) -> None:
            try:
                left_canvas.itemconfigure(left_win, width=left_canvas.winfo_width())
            except Exception:
                pass
            _sync_left_scrollregion()

        try:
            left.bind("<Configure>", _sync_left_scrollregion, add=True)
            left_canvas.bind("<Configure>", _sync_left_width, add=True)
        except Exception:
            pass

        # Mouse wheel scrolling when cursor is over the left panel
        def _scroll_target_for_event(evt) -> Optional[Any]:
            try:
                w = a.winfo_containing(evt.x_root, evt.y_root)
            except Exception:
                w = None
            while w is not None:
                try:
                    if isinstance(w, (ttk.Treeview, tk.Listbox, tk.Text)):
                        return w
                except Exception:
                    pass
                try:
                    w = w.master
                except Exception:
                    w = None
            return None

        def _on_mousewheel(evt) -> str:
            try:
                delta = int(getattr(evt, "delta", 0) or 0)
                if delta == 0:
                    return "break"
                target = _scroll_target_for_event(evt)
                if target is not None:
                    try:
                        target.yview_scroll(int(-1 * (delta / 120)), "units")
                        return "break"
                    except Exception:
                        pass
                left_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
            except Exception:
                pass
            return "break"

        def _bind_wheel(_evt=None) -> None:
            try:
                a.bind_all("<MouseWheel>", _on_mousewheel)
            except Exception:
                pass

        def _unbind_wheel(_evt=None) -> None:
            try:
                a.unbind_all("<MouseWheel>")
            except Exception:
                pass

        try:
            left_canvas.bind("<Enter>", _bind_wheel, add=True)
            left_canvas.bind("<Leave>", _unbind_wheel, add=True)
        except Exception:
            pass

        # --- Plot area ---
        plot = ttk.Frame(paned)
        plot.columnconfigure(0, weight=1)
        plot.rowconfigure(2, weight=1)

        try:
            paned.add(left_outer)
            paned.add(plot)
        except Exception:
            left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
            plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Workspace sidebar (existing LCMS sidebar UI) ---
        # NOTE: this assigns widget references back onto the App (`a`) to preserve
        # behavior without refactoring all callbacks.
        ws = ttk.LabelFrame(left, text="Workspace", padding=8)
        ws.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ws.columnconfigure(0, weight=1)

        tree = ttk.Treeview(
            ws,
            columns=("overlay", "active", "color", "name", "ms1", "pol"),
            show="headings",
            height=7,
            selectmode="browse",
        )
        tree.heading("overlay", text="Overlay")
        tree.heading("active", text="")
        tree.heading("color", text="Color")
        tree.heading("name", text="mzML")
        tree.heading("ms1", text="MS1")
        tree.heading("pol", text="Pol")
        tree.column("overlay", width=60, stretch=False, anchor="center")
        tree.column("active", width=26, stretch=False, anchor="center")
        tree.column("color", width=54, stretch=False, anchor="center")
        tree.column("name", width=200, stretch=True)
        tree.column("ms1", width=70, stretch=False, anchor="e")
        tree.column("pol", width=70, stretch=False)
        tree.grid(row=0, column=0, sticky="ew")
        a._ws_tree = tree

        ws_btns = ttk.Frame(ws)
        ws_btns.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(ws_btns, text="Add mzML…", command=a._open_mzml_many).grid(row=0, column=0, sticky="w")
        ttk.Button(ws_btns, text="Remove mzML", command=a._remove_selected_session).grid(row=0, column=1, sticky="w", padx=(8, 0))

        uv_tree = ttk.Treeview(ws, columns=("linked", "name", "rt", "sig"), show="headings", height=6, selectmode="browse")
        uv_tree.heading("linked", text="Linked to")
        uv_tree.heading("name", text="UV CSV")
        uv_tree.heading("rt", text="RT range")
        uv_tree.heading("sig", text="Signal")
        uv_tree.column("linked", width=110, stretch=False)
        uv_tree.column("name", width=180, stretch=True)
        uv_tree.column("rt", width=120, stretch=False)
        uv_tree.column("sig", width=110, stretch=False)
        uv_tree.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        a._uv_ws_tree = uv_tree

        uv_btns = ttk.Frame(ws)
        uv_btns.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(uv_btns, text="Add UV CSV…", command=a._open_uv_csv_many).grid(row=0, column=0, sticky="w")
        ttk.Button(uv_btns, text="Remove UV", command=a._remove_selected_uv).grid(row=0, column=1, sticky="w", padx=(8, 0))
        link_btn = ttk.Button(uv_btns, text="Link UV to mzML", command=a._link_selected_uv_to_selected_mzml, state="disabled")
        link_btn.grid(row=0, column=2, sticky="w", padx=(8, 0))
        a._uv_ws_link_btn = link_btn
        ttk.Button(uv_btns, text="Auto-link by name", command=a._auto_link_uv_by_name).grid(row=0, column=3, sticky="w", padx=(12, 0))

        try:
            tree.bind("<<TreeviewSelect>>", a._on_ws_select, add=True)
            tree.bind("<Button-3>", a._on_ws_right_click, add=True)
            tree.bind("<Button-1>", a._on_ws_left_click, add=True)
        except Exception:
            pass
        try:
            uv_tree.bind("<<TreeviewSelect>>", lambda e: a._update_uv_ws_controls(), add=True)
            uv_tree.bind("<Double-1>", lambda e: (a._link_selected_uv_to_selected_mzml(), "break"), add=True)
            uv_tree.bind("<Button-3>", a._on_uv_ws_right_click, add=True)
        except Exception:
            pass

        quick = ttk.LabelFrame(left, text="Quick Actions", padding=8)
        quick.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        quick.columnconfigure(0, weight=1)
        ttk.Button(quick, text="EIC (new chromatogram)…", command=a._open_sim_dialog).grid(row=0, column=0, sticky="ew")
        ttk.Button(quick, text="Jump to m/z…", command=a._open_jump_to_mz_dialog).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(quick, text="Export labels (all scans)…", command=a._export_all_labels_xlsx).grid(row=2, column=0, sticky="ew", pady=(6, 0))

        adv = ttk.LabelFrame(left, text="Advanced", padding=8)
        adv.grid(row=2, column=0, sticky="nsew")
        adv.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        adv_hdr = ttk.Frame(adv)
        adv_hdr.grid(row=0, column=0, sticky="ew")
        adv_hdr.columnconfigure(0, weight=1)
        adv_btn = ttk.Button(adv_hdr, text="Show ▼", command=a._toggle_advanced)
        adv_btn.grid(row=0, column=1, sticky="e")
        a._advanced_toggle_btn = adv_btn

        adv_body = ttk.Frame(adv)
        adv_body.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        adv_body.columnconfigure(0, weight=1)
        adv.rowconfigure(1, weight=1)
        a._advanced_body = adv_body

        ttk.Checkbutton(adv_body, text="Show polymer matching controls", variable=a._adv_show_polymer_var, command=a._apply_advanced_visibility).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Checkbutton(adv_body, text="Show confidence controls", variable=a._adv_show_confidence_var, command=a._apply_advanced_visibility).grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Checkbutton(adv_body, text="Show alignment diagnostics controls", variable=a._adv_show_alignment_diag_var, command=a._apply_advanced_visibility).grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )

        nb = ttk.Notebook(adv_body)
        nb.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        adv_body.rowconfigure(3, weight=1)
        a._sidebar_notebook = nb

        tab_nav = ttk.Frame(nb, padding=10)
        tab_view = ttk.Frame(nb, padding=10)
        tab_ann = ttk.Frame(nb, padding=10)
        tab_poly = ttk.Frame(nb, padding=10)
        nb.add(tab_nav, text="Navigate")
        nb.add(tab_view, text="View")
        nb.add(tab_ann, text="Annotate")
        a._tab_poly_frame = tab_poly  # type: ignore[attr-defined]

        # Navigate tab
        navf = ttk.LabelFrame(tab_nav, text="Spectrum", padding=10)
        navf.grid(row=0, column=0, sticky="ew")
        for c in range(4):
            navf.columnconfigure(c, weight=1)
        nav_prev = ttk.Button(navf, text="◀ Prev", command=lambda: a._step_spectrum(-1))
        nav_prev.grid(row=0, column=0, sticky="ew")
        nav_next = ttk.Button(navf, text="Next ▶", command=lambda: a._step_spectrum(+1))
        nav_next.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        nav_first = ttk.Button(navf, text="First", command=lambda: a._go_to_index(0))
        nav_first.grid(row=0, column=2, sticky="ew", padx=(12, 0))
        nav_last = ttk.Button(navf, text="Last", command=a._go_last)
        nav_last.grid(row=0, column=3, sticky="ew", padx=(6, 0))
        nav_find = ttk.Button(navf, text="Find m/z…", command=a._open_find_mz_dialog)
        nav_find.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(10, 0))

        nav_align = ttk.Button(navf, text="Auto-align UV↔MS", command=a._auto_align_uv_ms)
        nav_align.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0))

        nav_diag = ttk.Button(navf, text="Alignment Diagnostics…", command=a._open_alignment_diagnostics)
        nav_diag.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        a._nav_diag_btn = nav_diag  # type: ignore[attr-defined]

        nav_sim = ttk.Button(navf, text="EIC…", command=a._open_sim_dialog)
        nav_sim.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 0))

        ToolTip.attach(nav_prev, TOOLTIP_TEXT["nav_prev"])
        ToolTip.attach(nav_next, TOOLTIP_TEXT["nav_next"])
        ToolTip.attach(nav_first, TOOLTIP_TEXT["nav_first"])
        ToolTip.attach(nav_last, TOOLTIP_TEXT["nav_last"])
        ToolTip.attach(nav_find, TOOLTIP_TEXT["find_mz"])
        ToolTip.attach(nav_align, TOOLTIP_TEXT["uv_ms_align_button"])
        ToolTip.attach(nav_diag, TOOLTIP_TEXT["uv_ms_align_diag_button"])
        ToolTip.attach(nav_sim, TOOLTIP_TEXT["sim_button"])

        jmp = ttk.LabelFrame(tab_nav, text="Jump", padding=10)
        jmp.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(jmp, text="RT (min):").grid(row=0, column=0, sticky="w")
        rt_ent = ttk.Entry(jmp, textvariable=a._rt_jump_var, width=12)
        rt_ent.grid(row=0, column=1, sticky="w", padx=(8, 0))
        a._rt_jump_entry = rt_ent
        go_btn = ttk.Button(jmp, text="Go", command=a._jump_to_rt)
        go_btn.grid(row=0, column=2, sticky="w", padx=(8, 0))
        ToolTip.attach(rt_ent, TOOLTIP_TEXT["jump_rt_entry"])
        ToolTip.attach(go_btn, TOOLTIP_TEXT["jump_rt_go"])
        try:
            rt_ent.bind("<Return>", lambda e: (a._jump_to_rt(), "break"))
        except Exception:
            pass

        # View tab
        vf = ttk.LabelFrame(tab_view, text="Filters", padding=10)
        vf.grid(row=0, column=0, sticky="ew")
        ttk.Label(vf, text="RT unit").grid(row=0, column=0, sticky="w")
        rt_unit = ttk.Combobox(vf, textvariable=a.rt_unit_var, values=["minutes", "seconds"], state="readonly", width=10)
        rt_unit.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ToolTip.attach(rt_unit, TOOLTIP_TEXT["rt_unit"])

        polf = ttk.LabelFrame(tab_view, text="Polarity", padding=10)
        polf.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for i, (label, value, tt_key) in enumerate(
            [("All", "all", "polarity_all"), ("Positive", "positive", "polarity_pos"), ("Negative", "negative", "polarity_neg")]
        ):
            rb = ttk.Radiobutton(polf, text=label, value=value, variable=a.polarity_var, command=a._refresh_tic)
            rb.grid(row=0, column=i, sticky="w", padx=(0, 10))
            ToolTip.attach(rb, TOOLTIP_TEXT[tt_key])

        off = ttk.LabelFrame(tab_view, text="UV↔MS alignment", padding=10)
        off.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(off, text="Offset (min)").grid(row=0, column=0, sticky="w")
        off_ent = ttk.Entry(off, textvariable=a.uv_ms_rt_offset_var, width=10)
        off_ent.grid(row=0, column=1, sticky="w", padx=(8, 0))
        off_apply = ttk.Button(off, text="Apply", command=a._apply_uv_ms_offset)
        off_apply.grid(row=0, column=2, sticky="w", padx=(8, 0))

        align_cb = ttk.Checkbutton(
            off,
            text="Enable auto-align",
            variable=a.uv_ms_align_enabled_var,
            command=a._on_uv_ms_align_enabled_changed,
        )
        align_cb.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ToolTip.attach(off_ent, TOOLTIP_TEXT["uv_ms_offset_entry"])
        ToolTip.attach(off_apply, TOOLTIP_TEXT["uv_ms_offset_apply"])
        ToolTip.attach(align_cb, TOOLTIP_TEXT["uv_ms_align_enable"])
        try:
            off_ent.bind("<Return>", lambda e: (a._apply_uv_ms_offset(), "break"))
            off_ent.bind("<FocusOut>", lambda e: a._apply_uv_ms_offset())
        except Exception:
            pass

        more_view = ttk.Frame(tab_view)
        more_view.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        graph_btn = ttk.Button(more_view, text="Graph Settings…", command=a._open_graph_settings)
        graph_btn.pack(side=tk.LEFT)
        ToolTip.attach(graph_btn, TOOLTIP_TEXT["edit_graph"])

        panels = ttk.LabelFrame(tab_view, text="Panels", padding=10)
        panels.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        cb_tic = ttk.Checkbutton(panels, text="Show TIC", variable=a.show_tic_var, command=a._on_panels_changed)
        cb_spec = ttk.Checkbutton(panels, text="Show Spectrum", variable=a.show_spectrum_var, command=a._on_panels_changed)
        cb_uv = ttk.Checkbutton(panels, text="Show UV", variable=a.show_uv_var, command=a._on_panels_changed)
        cb_tic.grid(row=0, column=0, sticky="w")
        cb_spec.grid(row=1, column=0, sticky="w", pady=(6, 0))
        cb_uv.grid(row=2, column=0, sticky="w", pady=(6, 0))
        ToolTip.attach(cb_tic, TOOLTIP_TEXT["panels_show_tic"])
        ToolTip.attach(cb_spec, TOOLTIP_TEXT["panels_show_spectrum"])
        ToolTip.attach(cb_uv, TOOLTIP_TEXT["panels_show_uv"])

        region = ttk.LabelFrame(tab_view, text="TIC region", padding=10)
        region.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        cb_region = ttk.Checkbutton(
            region,
            text="Region Select (drag on TIC)",
            variable=a.tic_region_select_var,
            command=a._on_tic_region_select_changed,
        )
        cb_region.grid(row=0, column=0, sticky="w")
        clear_btn = ttk.Button(region, text="Clear Region", command=a._clear_tic_region_selection, state="disabled")
        clear_btn.grid(row=1, column=0, sticky="w", pady=(8, 0))
        a._tic_region_clear_btn = clear_btn

        # Annotate tab
        af = ttk.LabelFrame(tab_ann, text="Spectrum labels", padding=10)
        af.grid(row=0, column=0, sticky="ew")
        cb_auto = ttk.Checkbutton(
            af, text="Annotate spectrum peaks with m/z", variable=a.annotate_peaks_var, command=a._apply_quick_annotate_settings
        )
        cb_auto.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(af, text="Top N").grid(row=1, column=0, sticky="w", pady=(8, 0))
        topn = ttk.Spinbox(af, from_=0, to=200, textvariable=a.annotate_top_n_var, width=8, command=a._apply_quick_annotate_settings)
        topn.grid(row=1, column=1, sticky="w", pady=(8, 0), padx=(8, 0))
        ttk.Label(af, text="Min rel").grid(row=2, column=0, sticky="w", pady=(6, 0))
        minrel = ttk.Entry(af, textvariable=a.annotate_min_rel_var, width=10)
        minrel.grid(row=2, column=1, sticky="w", pady=(6, 0), padx=(8, 0))
        cb_drag = ttk.Checkbutton(af, text="Enable dragging labels with mouse", variable=a.drag_annotations_var)
        cb_drag.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ToolTip.attach(cb_auto, TOOLTIP_TEXT["annotate_enable"])
        ToolTip.attach(topn, TOOLTIP_TEXT["annotate_top_n"])
        ToolTip.attach(minrel, TOOLTIP_TEXT["annotate_min_rel"])
        ToolTip.attach(cb_drag, TOOLTIP_TEXT["annotate_drag"])
        try:
            minrel.bind("<Return>", lambda e: (a._apply_quick_annotate_settings(), "break"))
            minrel.bind("<FocusOut>", lambda e: a._apply_quick_annotate_settings())
        except Exception:
            pass

        uvf = ttk.LabelFrame(tab_ann, text="UV labels", padding=10)
        uvf.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        cb_uv = ttk.Checkbutton(
            uvf,
            text="Transfer top MS peaks to UV labels at selected RT",
            variable=a.uv_label_from_ms_var,
            command=a._apply_quick_annotate_settings,
        )
        cb_uv.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(uvf, text="How many peaks").grid(row=1, column=0, sticky="w", pady=(8, 0))
        uvn = ttk.Combobox(uvf, textvariable=a.uv_label_from_ms_top_n_var, values=[2, 3], state="readonly", width=6)
        uvn.grid(row=1, column=1, sticky="w", pady=(8, 0), padx=(8, 0))
        ToolTip.attach(cb_uv, TOOLTIP_TEXT["uv_transfer_labels"])
        ToolTip.attach(uvn, TOOLTIP_TEXT["uv_transfer_howmany"])
        try:
            uvn.bind("<<ComboboxSelected>>", lambda e: a._apply_quick_annotate_settings())
        except Exception:
            pass

        more_ann = ttk.Frame(tab_ann)
        more_ann.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ann_btn = ttk.Button(more_ann, text="Annotate Peaks…", command=a._open_annotation_settings)
        ann_btn.pack(side=tk.LEFT)
        custom_btn = ttk.Button(more_ann, text="Custom Labels…", command=a._open_custom_labels)
        custom_btn.pack(side=tk.LEFT, padx=(8, 0))
        ToolTip.attach(ann_btn, TOOLTIP_TEXT["annotate_peaks"])
        ToolTip.attach(custom_btn, TOOLTIP_TEXT["custom_labels"])

        overlay_labels = ttk.LabelFrame(tab_ann, text="Overlay labels", padding=10)
        overlay_labels.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        cb_all = ttk.Checkbutton(
            overlay_labels,
            text="Show labels for all overlayed spectra",
            variable=a._overlay_show_labels_all_var,
            command=a._refresh_overlay_view,
        )
        cb_all.grid(row=0, column=0, sticky="w")
        cb_multi = ttk.Checkbutton(
            overlay_labels,
            text="Multi-drag labels across overlay",
            variable=a._overlay_multi_drag_var,
        )
        cb_multi.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ToolTip.attach(cb_all, TOOLTIP_TEXT["overlay_labels_all"])
        ToolTip.attach(cb_multi, TOOLTIP_TEXT["overlay_multi_drag"])

        # Polymer tab
        pf = ttk.LabelFrame(tab_poly, text="Polymer matching", padding=10)
        pf.grid(row=0, column=0, sticky="ew")
        poly_enable = ttk.Checkbutton(
            pf, text="Enable polymer/reaction matching", variable=a.poly_enabled_var, command=a._redraw_spectrum_only
        )
        poly_enable.grid(row=0, column=0, sticky="w")
        poly_summary = ttk.Label(pf, text="Use Polymer Match… for full settings", wraplength=280, justify="left")
        poly_summary.grid(row=1, column=0, sticky="w", pady=(8, 0))
        poly_btn = ttk.Button(pf, text="Polymer Match…", command=a._open_polymer_match)
        poly_btn.grid(row=2, column=0, sticky="w", pady=(10, 0))
        ToolTip.attach(poly_enable, TOOLTIP_TEXT["poly_enable"])
        ToolTip.attach(poly_btn, TOOLTIP_TEXT["polymer_match"])

        try:
            if a._advanced_body is not None:
                a._advanced_body.grid_remove()
        except Exception:
            pass
        a._apply_advanced_visibility()

        a._now_view_var = tk.StringVar(value="")
        now_lbl = ttk.Label(plot, textvariable=a._now_view_var, padding=(6, 6))
        now_lbl.grid(row=0, column=0, sticky="ew")

        # Overlay legend (colors + dataset names)
        ov_leg = ttk.LabelFrame(plot, text="Overlay legend", padding=(6, 4))
        ov_leg.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ov_leg.columnconfigure(0, weight=1)

        ov_tree = ttk.Treeview(
            ov_leg,
            columns=("color", "name", "ms1", "pol", "status"),
            show="headings",
            height=4,
            selectmode="browse",
        )
        ov_tree.heading("color", text="Color")
        ov_tree.heading("name", text="mzML")
        ov_tree.heading("ms1", text="MS1")
        ov_tree.heading("pol", text="Pol")
        ov_tree.heading("status", text="Status")
        ov_tree.column("color", width=60, stretch=False, anchor="center")
        ov_tree.column("name", width=240, stretch=True)
        ov_tree.column("ms1", width=70, stretch=False, anchor="e")
        ov_tree.column("pol", width=70, stretch=False)
        ov_tree.column("status", width=120, stretch=True)
        ov_tree.grid(row=0, column=0, sticky="ew")
        ov_sb = ttk.Scrollbar(ov_leg, orient="vertical", command=ov_tree.yview)
        ov_sb.grid(row=0, column=1, sticky="ns")
        ov_tree.configure(yscrollcommand=ov_sb.set)
        a._overlay_legend_tree = ov_tree
        a._overlay_legend_frame = ov_leg
        try:
            ov_leg.grid_remove()
        except Exception:
            pass

        # Plot + diagnostics panel (resizable)
        plot_paned = ttk.Panedwindow(plot, orient=tk.VERTICAL)
        plot_paned.grid(row=2, column=0, sticky="nsew")

        fig_frame = ttk.Frame(plot_paned)
        fig_frame.columnconfigure(0, weight=1)
        fig_frame.rowconfigure(0, weight=1)
        fig_frame.rowconfigure(1, weight=0)
        fig_frame.rowconfigure(2, weight=0)

        diag_frame = ttk.Frame(plot_paned)
        a._build_diagnostics_panel(diag_frame)

        a._plot_paned = plot_paned
        a._diag_frame = diag_frame

        try:
            plot_paned.add(fig_frame, weight=4)
            plot_paned.add(diag_frame, weight=1)
        except Exception:
            plot_paned.add(fig_frame)
            plot_paned.add(diag_frame)

        try:
            a._apply_diag_panel_visibility()
        except Exception:
            pass

        # Figure (axes are created dynamically based on panel visibility)
        fig = Figure(figsize=(10.5, 8.5), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        try:
            toolbar_mpl = NavigationToolbar2Tk(canvas, fig_frame, pack_toolbar=False)
            toolbar_mpl.update()
            toolbar_mpl.grid(row=1, column=0, sticky="ew")
        except Exception:
            toolbar_mpl = None

        coord_var = tk.StringVar(value="")
        coord_lbl = ttk.Label(fig_frame, textvariable=coord_var, anchor="w")
        coord_lbl.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        a._fig = fig
        a._ax_tic = None
        a._ax_spec = None
        a._ax_uv = None
        a._canvas = canvas
        a._toolbar = toolbar_mpl
        a._plot_coord_var = coord_var
        a._plot_coord_label = coord_lbl

        try:
            a._mpl_nav = MatplotlibNavigator(
                canvas=canvas,
                axes_provider=lambda: [ax for ax in (a._ax_tic, a._ax_uv) if ax is not None],
                status_label=coord_var,
                box_zoom_enabled_cb=lambda: (not bool(a.tic_region_select_var.get())),
                box_click_callback=a._on_plot_click,
            )
            a._mpl_nav.attach()
        except Exception:
            a._mpl_nav = None

        a._mpl_cid = canvas.mpl_connect("button_press_event", a._on_plot_click)
        canvas.mpl_connect("motion_notify_event", a._on_plot_motion)
        canvas.mpl_connect("button_release_event", a._on_plot_release)

        try:
            w = canvas.get_tk_widget()
            w.bind("<KeyPress-r>", lambda _e: a._reset_spectrum_view(), add=True)
            w.bind("<KeyPress-R>", lambda _e: a._reset_spectrum_view(), add=True)
        except Exception:
            pass

        a._rebuild_plot_axes()
        a._update_now_viewing_header()
        a._update_current_context_panel()


class FTIRView(ttk.Frame):
    """FTIR module UI: multi-file workstation (single active spectrum)."""

    def __init__(self, parent: tk.Widget, app: "App", workspace: Workspace) -> None:
        super().__init__(parent)
        self.app = app
        self.workspace = workspace

        self._fig: Optional[Figure] = None
        self._ax: Any = None
        self._canvas: Optional[FigureCanvasTkAgg] = None
        self._toolbar: Any = None

        # Cached plotted line artists (for overlay mode)
        self._line_artists: Dict[Tuple[str, str], Any] = {}
        self._legend_artist: Any = None
        self._legend_keys: Tuple[Tuple[str, str], ...] = tuple()

        # Matplotlib draw instrumentation (FTIR only)
        self._mpl_draw_pending: bool = False
        self._mpl_draw_req_t: Optional[float] = None
        self._mpl_draw_watchdog_after: Optional[str] = None

        self._status_var = tk.StringVar(value="")
        self._reverse_x_var = tk.BooleanVar(value=True)
        self._reverse_pref_by_id: Dict[str, bool] = {}

        # Multi-workspace + overlay groups state (in-memory; can be serialized via workspace save)
        self.workspaces: Dict[str, FTIRWorkspace] = {}
        self.active_workspace_id: str = ""
        self._overlay_groups: Dict[str, OverlayGroup] = {}
        self._active_overlay_group_id: Optional[str] = None
        self._show_peaks_all_overlay_var = tk.BooleanVar(value=False)
        self._overlay_filter_var = tk.StringVar(value="")
        self._workspace_select_var = tk.StringVar(value="")
        self._overlay_color_scheme_var = tk.StringVar(value="Manual (workspace)")
        self._overlay_single_hue_color: str = "#1f77b4"
        self._overlay_offset_mode_var = tk.StringVar(value="Normal")
        self._overlay_offset_var = tk.DoubleVar(value=0.0)

        # FTIR-only perf + threading helpers
        self._ftir_event_q: "queue.Queue[Tuple[Any, ...]]" = queue.Queue()
        self._ftir_poll_after: Optional[str] = None
        self._ftir_busy: bool = False
        self._peaks_busy: bool = False

        self._btn_load: Optional[ttk.Button] = None
        self._btn_remove: Optional[ttk.Button] = None
        self._btn_clear: Optional[ttk.Button] = None
        self._btn_peaks: Optional[ttk.Button] = None
        self._btn_export_peaks: Optional[ttk.Button] = None
        self._btn_add_bond_label: Optional[ttk.Button] = None

        # Peak picking UI + state
        self._peaks_dialog: Optional[tk.Toplevel] = None
        self._peaks_enabled_var = tk.BooleanVar(value=False)
        self._peaks_min_prom_var = tk.DoubleVar(value=1)
        self._peaks_min_dist_var = tk.DoubleVar(value=5.0)
        self._peaks_topn_var = tk.IntVar(value=0)
        self._peaks_label_fmt_var = tk.StringVar(value="{wn:.1f}")

        self._peak_texts: List[Any] = []
        self._peak_texts_by_key: Dict[Tuple[str, str], List[Any]] = {}
        self._peak_markers: List[Any] = []
        self._peak_summary_text: Any = None
        self._peak_artist_to_info: Dict[Any, Tuple[str, str, str]] = {}  # artist -> (ws_id, ds_id, peak_id)
        self._peak_menu: Optional[tk.Menu] = None
        self._peak_menu_last_info: Optional[Tuple[str, str, str]] = None
        self._peak_color: Optional[str] = None

        # Bond label annotations (persistent, per-dataset)
        self._bond_texts: List[Any] = []
        self._bond_vlines: List[Any] = []
        self._bond_artist_to_info: Dict[Any, Tuple[str, str, int]] = {}  # artist -> (ws_id, ds_id, ann_index)
        self._bond_menu: Optional[tk.Menu] = None
        self._bond_menu_last_info: Optional[Tuple[str, str, int]] = None

        self._drag_bond_key: Optional[Tuple[str, str]] = None
        self._drag_bond_idx: Optional[int] = None
        self._drag_bond_artist: Any = None
        self._drag_bond_dx: float = 0.0
        self._drag_bond_dy: float = 0.0

        self._bond_dialog: Optional[tk.Toplevel] = None
        self._bond_placement_active: bool = False
        self._bond_place_opts: Dict[str, Any] = {}
        self._bond_place_cid_click: Optional[int] = None
        self._bond_place_cid_key: Optional[int] = None

        self._export_peaks_menu: Optional[tk.Menu] = None
        self._export_peaks_dialog_win: Optional[tk.Toplevel] = None
        self._export_include_candidates_var = tk.BooleanVar(value=True)

        # Dragging peak labels
        self._drag_peak_id: Optional[str] = None
        self._drag_peak_key: Optional[Tuple[str, str]] = None
        self._drag_peak_artist: Any = None
        self._drag_dx: float = 0.0
        self._drag_dy: float = 0.0

        # Overlay peak display selection cache (stable across active switching)
        self._overlay_peak_display_ids_by_key: Dict[Tuple[str, str], List[str]] = {}

        self._tree: Optional[ttk.Treeview] = None
        self._tree_menu: Optional[tk.Menu] = None
        self._ignore_tree_select: bool = False

        # Workspace selector + overlay panel widgets
        self._ws_combo: Optional[ttk.Combobox] = None
        self._overlay_groups_tree: Optional[ttk.Treeview] = None
        self._overlay_members_tree: Optional[ttk.Treeview] = None
        self._overlay_selection_tree: Optional[ttk.Treeview] = None

        # Coalesce redraws to avoid event storms
        self._redraw_after: Optional[str] = None

        self._init_ftir_workspaces_from_app_workspace()

        self._build()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        outer = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        outer.grid(row=0, column=0, sticky="nsew")

        # Left: dataset panel (scrollable)
        left_outer = ttk.Frame(outer)
        left_outer.columnconfigure(0, weight=1)
        left_outer.rowconfigure(0, weight=1)

        left_canvas = tk.Canvas(left_outer, highlightthickness=0)
        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_vsb = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left_vsb.grid(row=0, column=1, sticky="ns")
        try:
            left_canvas.configure(yscrollcommand=left_vsb.set)
        except Exception:
            pass

        left = ttk.Frame(left_canvas, padding=(10, 10, 8, 10))
        left.columnconfigure(0, weight=1)
        try:
            left_win = left_canvas.create_window((0, 0), window=left, anchor="nw")
        except Exception:
            left_win = None

        def _sync_left_scrollregion(_evt=None) -> None:
            try:
                left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            except Exception:
                pass

        def _sync_left_width(_evt=None) -> None:
            try:
                if left_win is not None:
                    left_canvas.itemconfigure(left_win, width=left_canvas.winfo_width())
            except Exception:
                pass
            _sync_left_scrollregion()

        try:
            left.bind("<Configure>", _sync_left_scrollregion, add=True)
            left_canvas.bind("<Configure>", _sync_left_width, add=True)
        except Exception:
            pass

        # Mouse wheel scrolling when cursor is over the left panel
        def _scroll_target_for_event(evt) -> Optional[Any]:
            try:
                w = self.app.winfo_containing(evt.x_root, evt.y_root)
            except Exception:
                w = None
            while w is not None:
                try:
                    if isinstance(w, (ttk.Treeview, tk.Listbox, tk.Text)):
                        return w
                except Exception:
                    pass
                try:
                    w = w.master
                except Exception:
                    w = None
            return None

        def _on_mousewheel(evt) -> str:
            try:
                delta = int(getattr(evt, "delta", 0) or 0)
                if delta == 0:
                    return "break"
                target = _scroll_target_for_event(evt)
                if target is not None:
                    try:
                        target.yview_scroll(int(-1 * (delta / 120)), "units")
                        return "break"
                    except Exception:
                        pass
                left_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
            except Exception:
                pass
            return "break"

        def _bind_wheel(_evt=None) -> None:
            try:
                self.app.bind_all("<MouseWheel>", _on_mousewheel)
            except Exception:
                pass

        def _unbind_wheel(_evt=None) -> None:
            try:
                self.app.unbind_all("<MouseWheel>")
            except Exception:
                pass

        try:
            left_canvas.bind("<Enter>", _bind_wheel, add=True)
            left_canvas.bind("<Leave>", _unbind_wheel, add=True)
        except Exception:
            pass

        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        left.rowconfigure(4, weight=1)

        wsblk = ttk.LabelFrame(left, text="Workspaces", padding=(8, 6, 8, 8))
        wsblk.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        wsblk.columnconfigure(1, weight=1)
        ttk.Label(wsblk, text="Workspace").grid(row=0, column=0, sticky="w")
        ws_combo = ttk.Combobox(wsblk, textvariable=self._workspace_select_var, state="readonly")
        ws_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._ws_combo = ws_combo
        try:
            ToolTip.attach(ws_combo, TOOLTIP_TEXT.get("ftir_ws_combo", ""))
        except Exception:
            pass
        try:
            ws_combo.bind("<<ComboboxSelected>>", lambda e: self._on_workspace_selected(), add=True)
        except Exception:
            pass

        ws_btns = ttk.Frame(wsblk)
        ws_btns.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        b_ws_new = ttk.Button(ws_btns, text="New Workspace", command=self._new_workspace)
        b_ws_ren = ttk.Button(ws_btns, text="Rename", command=self._rename_workspace)
        b_ws_dup = ttk.Button(ws_btns, text="Duplicate", command=self._duplicate_workspace)
        b_ws_col = ttk.Button(ws_btns, text="Graph Color…", command=self._edit_active_workspace_graph_color)
        b_ws_del = ttk.Button(ws_btns, text="Delete", command=self._delete_workspace)
        b_ws_new.pack(side=tk.LEFT)
        b_ws_ren.pack(side=tk.LEFT, padx=(8, 0))
        b_ws_dup.pack(side=tk.LEFT, padx=(8, 0))
        b_ws_col.pack(side=tk.LEFT, padx=(8, 0))
        b_ws_del.pack(side=tk.LEFT, padx=(8, 0))

        try:
            ToolTip.attach(b_ws_new, TOOLTIP_TEXT.get("ftir_ws_new", ""))
            ToolTip.attach(b_ws_ren, TOOLTIP_TEXT.get("ftir_ws_rename", ""))
            ToolTip.attach(b_ws_dup, TOOLTIP_TEXT.get("ftir_ws_duplicate", ""))
            ToolTip.attach(b_ws_col, TOOLTIP_TEXT.get("ftir_ws_graph_color", ""))
            ToolTip.attach(b_ws_del, TOOLTIP_TEXT.get("ftir_ws_delete", ""))
        except Exception:
            pass

        ttk.Label(left, text="FTIR datasets (current workspace)").grid(row=1, column=0, sticky="w", pady=(0, 6))

        tree = ttk.Treeview(left, columns=("active", "name", "n"), show="headings", selectmode="browse", height=9)
        tree.heading("active", text="")
        tree.heading("name", text="Name")
        tree.heading("n", text="Points")
        tree.column("active", width=28, stretch=False, anchor="center")
        tree.column("name", width=240, stretch=True)
        tree.column("n", width=70, stretch=False, anchor="e")
        tree.grid(row=2, column=0, sticky="nsew")
        self._tree = tree

        try:
            ToolTip.attach(tree, TOOLTIP_TEXT.get("ftir_tree", ""))
        except Exception:
            pass

        try:
            tree.bind("<<TreeviewSelect>>", lambda e: self._on_tree_select_set_active(), add=True)
            tree.bind("<Double-1>", lambda e: self._rename_selected(), add=True)
            tree.bind("<F2>", lambda e: self._rename_selected(), add=True)
            tree.bind("<Button-3>", self._on_tree_right_click, add=True)
        except Exception:
            pass

        btns = ttk.Frame(left)
        btns.grid(row=3, column=0, sticky="ew", pady=(8, 10))
        self._btn_load = ttk.Button(btns, text="Load FTIR…", command=self.load_ftir_dialog)
        self._btn_load.pack(side=tk.LEFT)
        self._btn_remove = ttk.Button(btns, text="Remove", command=self.remove_selected)
        self._btn_remove.pack(side=tk.LEFT, padx=(8, 0))
        self._btn_clear = ttk.Button(btns, text="Clear All", command=self.clear_all)
        self._btn_clear.pack(side=tk.LEFT, padx=(8, 0))

        try:
            if self._btn_load is not None:
                ToolTip.attach(self._btn_load, TOOLTIP_TEXT.get("ftir_load", ""))
            if self._btn_remove is not None:
                ToolTip.attach(self._btn_remove, TOOLTIP_TEXT.get("ftir_remove", ""))
            if self._btn_clear is not None:
                ToolTip.attach(self._btn_clear, TOOLTIP_TEXT.get("ftir_clear", ""))
        except Exception:
            pass

        # Overlays panel (persistent overlay groups + selection list)
        ov_outer = ttk.Frame(left)
        ov_outer.grid(row=4, column=0, sticky="nsew")
        ov_outer.columnconfigure(0, weight=1)
        ov_outer.rowconfigure(0, weight=2)
        ov_outer.rowconfigure(1, weight=2)

        groups_blk = ttk.LabelFrame(ov_outer, text="Overlays", padding=(8, 6, 8, 8))
        groups_blk.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        groups_blk.columnconfigure(0, weight=1)
        groups_blk.rowconfigure(1, weight=1)
        groups_blk.rowconfigure(5, weight=1)

        ttk.Label(groups_blk, text="Overlay Groups").grid(row=0, column=0, sticky="w")
        gtree = ttk.Treeview(groups_blk, columns=("name", "count", "ws"), show="headings", selectmode="browse", height=5)
        gtree.heading("name", text="Name")
        gtree.heading("count", text="#")
        gtree.heading("ws", text="WS")
        gtree.column("name", width=230, stretch=True)
        gtree.column("count", width=36, stretch=False, anchor="e")
        gtree.column("ws", width=60, stretch=False, anchor="center")
        gtree.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        self._overlay_groups_tree = gtree
        try:
            ToolTip.attach(gtree, TOOLTIP_TEXT.get("ftir_overlay_groups", ""))
        except Exception:
            pass
        try:
            gtree.bind("<<TreeviewSelect>>", lambda e: self._rebuild_overlay_group_members_list(), add=True)
            gtree.bind("<Double-1>", lambda e: self._activate_selected_overlay_group(), add=True)
        except Exception:
            pass

        gbtns = ttk.Frame(groups_blk)
        gbtns.grid(row=2, column=0, sticky="ew")
        b_ov_new = ttk.Button(gbtns, text="New Overlay from Selection", command=self._new_overlay_group_from_selection)
        b_ov_act = ttk.Button(gbtns, text="Activate Overlay", command=self._activate_selected_overlay_group)
        b_ov_ren = ttk.Button(gbtns, text="Rename…", command=self._rename_selected_overlay_group)
        b_ov_dup = ttk.Button(gbtns, text="Duplicate", command=self._duplicate_selected_overlay_group)
        b_ov_del = ttk.Button(gbtns, text="Delete", command=self._delete_selected_overlay_group)
        b_ov_clr = ttk.Button(gbtns, text="Clear Active Overlay", command=self._clear_active_overlay_group)
        b_ov_new.pack(side=tk.LEFT)
        b_ov_act.pack(side=tk.LEFT, padx=(8, 0))
        b_ov_ren.pack(side=tk.LEFT, padx=(8, 0))
        b_ov_dup.pack(side=tk.LEFT, padx=(8, 0))
        b_ov_del.pack(side=tk.LEFT, padx=(8, 0))
        b_ov_clr.pack(side=tk.LEFT, padx=(8, 0))

        try:
            ToolTip.attach(b_ov_new, TOOLTIP_TEXT.get("ftir_overlay_new", ""))
            ToolTip.attach(b_ov_act, TOOLTIP_TEXT.get("ftir_overlay_activate", ""))
            ToolTip.attach(b_ov_ren, TOOLTIP_TEXT.get("ftir_overlay_rename", ""))
            ToolTip.attach(b_ov_dup, TOOLTIP_TEXT.get("ftir_overlay_duplicate", ""))
            ToolTip.attach(b_ov_del, TOOLTIP_TEXT.get("ftir_overlay_delete", ""))
            ToolTip.attach(b_ov_clr, TOOLTIP_TEXT.get("ftir_overlay_clear_active", ""))
        except Exception:
            pass

        colors_row = ttk.Frame(groups_blk)
        colors_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(colors_row, text="Overlay colors").pack(side=tk.LEFT)
        ov_colors = ttk.Combobox(
            colors_row,
            textvariable=self._overlay_color_scheme_var,
            values=self._overlay_scheme_options(),
            state="readonly",
            width=20,
        )
        ov_colors.pack(side=tk.LEFT, padx=(8, 0))
        ov_colors.bind("<<ComboboxSelected>>", lambda _e: self._apply_overlay_color_scheme())
        ov_pick = ttk.Button(colors_row, text="Pick hue…", command=self._pick_overlay_single_hue_color)
        ov_pick.pack(side=tk.LEFT, padx=(8, 0))

        try:
            ToolTip.attach(ov_colors, TOOLTIP_TEXT.get("ftir_overlay_colors", ""))
            ToolTip.attach(ov_pick, TOOLTIP_TEXT.get("ftir_overlay_pick_hue", ""))
        except Exception:
            pass

        offset_row = ttk.Frame(groups_blk)
        offset_row.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(offset_row, text="Overlay offset").pack(side=tk.LEFT)
        ov_mode = ttk.Combobox(
            offset_row,
            textvariable=self._overlay_offset_mode_var,
            values=["Normal", "Offset Y", "Offset X"],
            state="readonly",
            width=14,
        )
        ov_mode.pack(side=tk.LEFT, padx=(8, 0))
        ov_mode.bind("<<ComboboxSelected>>", lambda _e: self._on_ftir_overlay_offset_changed())
        ttk.Label(offset_row, text="Value").pack(side=tk.LEFT, padx=(8, 0))
        ov_val = ttk.Entry(offset_row, textvariable=self._overlay_offset_var, width=10)
        ov_val.pack(side=tk.LEFT, padx=(4, 0))
        ov_val.bind("<KeyRelease>", lambda _e: self._on_ftir_overlay_offset_changed())

        ttk.Label(groups_blk, text="Members (selected group)").grid(row=5, column=0, sticky="w", pady=(10, 0))
        mtree = ttk.Treeview(groups_blk, columns=("member",), show="headings", selectmode="browse", height=5)
        mtree.heading("member", text="Workspace :: Dataset")
        mtree.column("member", width=320, stretch=True)
        mtree.grid(row=6, column=0, sticky="nsew", pady=(4, 8))
        self._overlay_members_tree = mtree

        try:
            ToolTip.attach(mtree, TOOLTIP_TEXT.get("ftir_overlay_members", ""))
        except Exception:
            pass

        mbtns = ttk.Frame(groups_blk)
        mbtns.grid(row=7, column=0, sticky="ew")
        b_set_active_member = ttk.Button(
            mbtns,
            text="Set Active = selected member",
            command=self._set_active_overlay_member_from_selected,
        )
        b_set_active_member.pack(side=tk.LEFT)
        try:
            ToolTip.attach(b_set_active_member, TOOLTIP_TEXT.get("ftir_overlay_set_active_member", ""))
        except Exception:
            pass

        select_blk = ttk.LabelFrame(ov_outer, text="Selection (for new overlay group)", padding=(8, 6, 8, 8))
        select_blk.grid(row=1, column=0, sticky="nsew")
        select_blk.columnconfigure(0, weight=1)
        select_blk.rowconfigure(2, weight=1)

        ttk.Label(select_blk, text="Filter").grid(row=0, column=0, sticky="w")
        ov_filter = ttk.Entry(select_blk, textvariable=self._overlay_filter_var)
        ov_filter.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        try:
            self._overlay_filter_var.trace_add("write", lambda *a: self._rebuild_overlay_selection_list())
        except Exception:
            pass

        sel_tree = ttk.Treeview(select_blk, columns=("item",), show="headings", selectmode="extended", height=7)
        sel_tree.heading("item", text="Workspace :: Dataset")
        sel_tree.column("item", width=320, stretch=True)
        sel_tree.grid(row=2, column=0, sticky="nsew")
        self._overlay_selection_tree = sel_tree

        try:
            ToolTip.attach(ov_filter, TOOLTIP_TEXT.get("ftir_overlay_filter", ""))
            ToolTip.attach(sel_tree, TOOLTIP_TEXT.get("ftir_overlay_selection", ""))
        except Exception:
            pass

        outer.add(left_outer, weight=1)

        # Right: plot + controls
        right = ttk.Frame(outer, padding=(8, 10, 10, 10))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        top = ttk.Frame(right)
        top.grid(row=0, column=0, sticky="ew")
        b_save_plot = ttk.Button(top, text="Save FTIR Plot…", command=self.save_plot_dialog)
        b_save_plot.pack(side=tk.LEFT)
        self._btn_peaks = ttk.Button(top, text="Peaks…", command=self._open_peaks_dialog)
        self._btn_peaks.pack(side=tk.LEFT, padx=(8, 0))
        self._btn_export_peaks = ttk.Button(top, text="Export Peaks…", command=self._export_peaks_dialog)
        self._btn_export_peaks.pack(side=tk.LEFT, padx=(8, 0))
        self._btn_add_bond_label = ttk.Button(top, text="Add Bond Label…", command=self._open_add_bond_label_dialog)
        self._btn_add_bond_label.pack(side=tk.LEFT, padx=(8, 0))
        cb_reverse = ttk.Checkbutton(top, text="Reverse x-axis (common FTIR)", variable=self._reverse_x_var, command=self._on_toggle_reverse)
        cb_reverse.pack(side=tk.LEFT, padx=(14, 0))

        cb_show_peaks_all = ttk.Checkbutton(
            top,
            text="Show peaks for all overlayed spectra",
            variable=self._show_peaks_all_overlay_var,
            command=self._schedule_redraw,
        )
        cb_show_peaks_all.pack(side=tk.LEFT, padx=(10, 0))

        try:
            ToolTip.attach(b_save_plot, TOOLTIP_TEXT.get("ftir_save_plot", ""))
            if self._btn_peaks is not None:
                ToolTip.attach(self._btn_peaks, TOOLTIP_TEXT.get("ftir_peaks", ""))
            if self._btn_export_peaks is not None:
                ToolTip.attach(self._btn_export_peaks, TOOLTIP_TEXT.get("ftir_export_peaks", ""))
            ToolTip.attach(cb_reverse, TOOLTIP_TEXT.get("ftir_reverse_x", ""))
            ToolTip.attach(cb_show_peaks_all, TOOLTIP_TEXT.get("ftir_show_peaks_all", ""))
        except Exception:
            pass

        status = ttk.Label(right, textvariable=self._status_var)
        status.grid(row=1, column=0, sticky="ew", pady=(8, 6))

        plot = ttk.Frame(right)
        plot.grid(row=2, column=0, sticky="nsew")
        plot.columnconfigure(0, weight=1)
        plot.rowconfigure(0, weight=1)
        plot.rowconfigure(1, weight=0)
        plot.rowconfigure(2, weight=0)

        fig = Figure(figsize=(10.5, 8.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title("FTIR")
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Absorbance")

        # Plotting is driven by cached line artists keyed by (workspace_id, dataset_id).
        line = None

        canvas = FigureCanvasTkAgg(fig, master=plot)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        try:
            toolbar = NavigationToolbar2Tk(canvas, plot, pack_toolbar=False)
            toolbar.update()
            toolbar.grid(row=1, column=0, sticky="ew")
        except Exception:
            toolbar = None

        coord_var = tk.StringVar(value="")
        coord_lbl = ttk.Label(plot, textvariable=coord_var, anchor="w")
        coord_lbl.grid(row=2, column=0, sticky="ew", pady=(2, 0))

        self._fig = fig
        self._ax = ax
        self._canvas = canvas
        self._toolbar = toolbar
        self._line = line
        self._coord_var = coord_var
        self._coord_label = coord_lbl

        try:
            self._mpl_nav = MatplotlibNavigator(
                canvas=canvas,
                ax=ax,
                status_label=coord_var,
            )
            self._mpl_nav.attach()
        except Exception:
            self._mpl_nav = None

        try:
            canvas.mpl_connect("draw_event", self._on_mpl_draw_event)
        except Exception:
            pass

        try:
            canvas.mpl_connect("pick_event", self._on_peak_pick_event)
        except Exception:
            pass

        try:
            canvas.mpl_connect("button_press_event", self._on_peak_drag_press)
            canvas.mpl_connect("motion_notify_event", self._on_peak_drag_motion)
            canvas.mpl_connect("button_release_event", self._on_peak_drag_release)
        except Exception:
            pass

        try:
            canvas.mpl_connect("key_press_event", self._on_ftir_keypress)
        except Exception:
            pass

        outer.add(right, weight=4)

        self._refresh_workspace_selector()
        self._rebuild_overlay_group_list()
        self._rebuild_overlay_group_members_list()
        self._rebuild_overlay_selection_list()
        self.refresh_from_workspace(select_active=True)

    # --- FTIR workspaces (in-memory only) ---

    def _init_ftir_workspaces_from_app_workspace(self) -> None:
        # Build a default FTIR workspace from the app Workspace model.
        if self.workspaces:
            return
        ws_id = uuid.uuid4().hex
        ftws = FTIRWorkspace(
            id=str(ws_id),
            name="Workspace 1",
            datasets=list(getattr(self.workspace, "ftir_datasets", []) or []),
            active_dataset_id=(None if not getattr(self.workspace, "active_ftir_id", None) else str(self.workspace.active_ftir_id)),
            line_color=self._default_color_for_workspace_id(str(ws_id)),
        )
        self.workspaces[str(ws_id)] = ftws
        self.active_workspace_id = str(ws_id)
        try:
            self._workspace_select_var.set(str(ftws.name))
        except Exception:
            pass

    def _active_workspace(self) -> FTIRWorkspace:
        if not self.workspaces:
            self._init_ftir_workspaces_from_app_workspace()
        ws = self.workspaces.get(str(self.active_workspace_id))
        if ws is None:
            # Fallback to an arbitrary workspace.
            first = next(iter(self.workspaces.values()))
            self.active_workspace_id = str(first.id)
            return first
        return ws

    def _default_color_for_workspace_id(self, ws_id: str) -> str:
        """Stable default color for a workspace.

        Returns a Tk-compatible color string ("#RRGGBB")."""
        s = str(ws_id or "")
        try:
            idx = int(s[:8], 16) % 10 if len(s) >= 8 else (abs(hash(s)) % 10)
        except Exception:
            idx = 0
        try:
            cycle = matplotlib.rcParams.get("axes.prop_cycle")
            if cycle is not None:
                colors = cycle.by_key().get("color")  # type: ignore[attr-defined]
                if colors:
                    # Prefer blue + black as the first two defaults (instead of blue + orange).
                    try:
                        colors = list(colors)
                        if len(colors) >= 2:
                            colors[1] = "#000000"
                    except Exception:
                        pass
                    base = str(colors[int(idx) % len(colors)])
                    return str(mcolors.to_hex(mcolors.to_rgb(base)))
        except Exception:
            pass
        # Conservative fallback (Matplotlib default blue)
        return "#1f77b4"

    def _workspace_line_color(self, ws_id: str) -> str:
        ws_obj = None
        try:
            ws_obj = (self.workspaces or {}).get(str(ws_id))
        except Exception:
            ws_obj = None
        c = None
        try:
            c = (None if ws_obj is None else getattr(ws_obj, "line_color", None))
        except Exception:
            c = None
        c = (str(c).strip() if c is not None else "")
        return c if c else self._default_color_for_workspace_id(str(ws_id))

    def _overlay_scheme_options(self) -> List[str]:
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

    def _apply_overlay_color_scheme(self) -> None:
        try:
            self._schedule_redraw()
        except Exception:
            pass

    def _pick_overlay_single_hue_color(self) -> None:
        current = str(self._overlay_single_hue_color or "#1f77b4")
        try:
            picked = colorchooser.askcolor(color=(current or None), title="Pick overlay hue", parent=self.app)[1]
        except Exception:
            picked = None
        if not picked:
            return
        self._overlay_single_hue_color = str(picked)
        try:
            self._overlay_color_scheme_var.set("Single hue…")
        except Exception:
            pass
        self._apply_overlay_color_scheme()

    def _overlay_colors_for_scheme(self, scheme: str, n: int) -> List[str]:
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
                h, l, s = colorsys.rgb_to_hls(*base_rgb)
            except Exception:
                h, l, s = (0.58, 0.45, 0.65)
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

    def _overlay_color_map_for_group(self, g: OverlayGroup) -> Dict[Tuple[str, str], str]:
        scheme = str(self._overlay_color_scheme_var.get() or "").strip()
        if scheme in ("", "Manual (workspace)"):
            return {}
        members = [(str(a), str(b)) for (a, b) in (getattr(g, "members", []) or [])]
        colors = self._overlay_colors_for_scheme(scheme, len(members))
        return {k: str(c) for k, c in zip(members, colors)}

    def _on_ftir_overlay_offset_changed(self) -> None:
        try:
            self._schedule_redraw()
        except Exception:
            pass

    def _edit_active_workspace_graph_color(self) -> None:
        ws = self._active_workspace()
        current = None
        try:
            current = (None if getattr(ws, "line_color", None) is None else str(ws.line_color))
        except Exception:
            current = None
        if not current:
            current = self._workspace_line_color(str(getattr(ws, "id", "")))
        # Tk's color chooser doesn't accept Matplotlib cycle aliases like "C1".
        # Normalize to a Tk-compatible "#RRGGBB" when possible.
        tk_current = None
        try:
            if current:
                tk_current = str(mcolors.to_hex(mcolors.to_rgb(str(current))))
        except Exception:
            tk_current = None
        c = colorchooser.askcolor(color=(tk_current or None), title=f"Pick workspace line color: {ws.name}", parent=self.app)[1]
        if not c:
            return
        try:
            ws.line_color = str(c)
        except Exception:
            pass
        self._schedule_redraw()

    def _find_workspace_id_by_name(self, name: str) -> Optional[str]:
        n = str(name or "").strip()
        for ws_id, ws in (self.workspaces or {}).items():
            if str(getattr(ws, "name", "")) == n:
                return str(ws_id)
        return None

    def _sync_active_workspace_to_app_workspace(self) -> None:
        # Keep legacy single-workspace persistence working:
        # mirror the currently active FTIR workspace into `self.workspace.ftir_datasets`.
        try:
            ws = self._active_workspace()
        except Exception:
            return
        try:
            self.workspace.ftir_datasets = list(getattr(ws, "datasets", []) or [])
            self.workspace.active_ftir_id = (None if not ws.active_dataset_id else str(ws.active_dataset_id))
        except Exception:
            pass

    def _all_dataset_keys(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for ws_id, ws in (self.workspaces or {}).items():
            for d in (getattr(ws, "datasets", []) or []):
                try:
                    out.append((str(ws_id), str(getattr(d, "id", ""))))
                except Exception:
                    continue
        return out

    def _get_dataset_by_key(self, key: Tuple[str, str]) -> Optional[FTIRDataset]:
        try:
            ws_id, ds_id = str(key[0]), str(key[1])
        except Exception:
            return None
        ws = self.workspaces.get(ws_id)
        if ws is None:
            return None
        for d in (getattr(ws, "datasets", []) or []):
            if str(getattr(d, "id", "")) == ds_id:
                return d
        return None

    def _unique_workspace_name(self, base: str, *, exclude_id: Optional[str] = None) -> str:
        b = str(base or "").strip() or "Workspace"
        used = set()
        for ws_id, ws in (self.workspaces or {}).items():
            if exclude_id is not None and str(ws_id) == str(exclude_id):
                continue
            used.add(str(getattr(ws, "name", "")).strip())
        if b not in used:
            return b
        i = 2
        while True:
            cand = f"{b} {i}"
            if cand not in used:
                return cand
            i += 1

    def _refresh_workspace_selector(self) -> None:
        names = [str(getattr(ws, "name", "")) for ws in (self.workspaces or {}).values()]
        try:
            if self._ws_combo is not None:
                self._ws_combo.configure(values=names)
        except Exception:
            pass
        try:
            ws = self._active_workspace()
            self._workspace_select_var.set(str(getattr(ws, "name", "")))
        except Exception:
            pass

    def _set_active_workspace_by_id(self, ws_id: str) -> None:
        if not ws_id or str(ws_id) not in (self.workspaces or {}):
            return
        # Keep method for existing callers; treat as an explicit workspace switch.
        ds_id: Optional[str] = None
        try:
            ws = self.workspaces.get(str(ws_id))
            ds_id = (None if ws is None else (None if not getattr(ws, "active_dataset_id", None) else str(ws.active_dataset_id)))
            if (not ds_id) and ws is not None and (getattr(ws, "datasets", None) or []):
                ds_id = str(ws.datasets[0].id)
        except Exception:
            ds_id = None
        self._set_active_dataset(str(ws_id), ds_id, reason="workspace_switch")

    def _on_workspace_selected(self) -> None:
        name = str(self._workspace_select_var.get() or "").strip()
        ws_id = self._find_workspace_id_by_name(name)
        if ws_id is None:
            return
        self._set_active_workspace_by_id(str(ws_id))

    def _new_workspace(self) -> None:
        base = f"Workspace {len(self.workspaces) + 1}" if self.workspaces else "Workspace 1"
        name = self._unique_workspace_name(base)
        ws_id = uuid.uuid4().hex
        self.workspaces[str(ws_id)] = FTIRWorkspace(id=str(ws_id), name=str(name), line_color=self._default_color_for_workspace_id(str(ws_id)))
        self._set_active_workspace_by_id(str(ws_id))

    def _rename_workspace(self) -> None:
        ws = self._active_workspace()
        new_name = simpledialog.askstring("Rename Workspace", "Workspace name:", initialvalue=str(getattr(ws, "name", "")), parent=self.app)
        if new_name is None:
            return
        new_name = str(new_name).strip()
        if not new_name:
            return
        ws.name = self._unique_workspace_name(new_name, exclude_id=str(ws.id))
        self._refresh_workspace_selector()
        self._rebuild_overlay_group_list()
        self._rebuild_overlay_group_members_list()
        self._rebuild_overlay_selection_list()

    def _duplicate_workspace(self) -> None:
        src = self._active_workspace()
        ws_id = uuid.uuid4().hex
        name = self._unique_workspace_name(f"{str(getattr(src, 'name', 'Workspace'))} (copy)")

        id_map: Dict[str, str] = {}
        datasets: List[FTIRDataset] = []
        for d in (getattr(src, "datasets", []) or []):
            new_id = uuid.uuid4().hex
            id_map[str(getattr(d, "id", ""))] = str(new_id)
            datasets.append(
                FTIRDataset(
                    id=str(new_id),
                    name=str(getattr(d, "name", "")),
                    path=getattr(d, "path", None),
                    x_full=np.asarray(getattr(d, "x_full", []), dtype=float),
                    y_full=np.asarray(getattr(d, "y_full", []), dtype=float),
                    x_disp=np.asarray(getattr(d, "x_disp", []), dtype=float),
                    y_disp=np.asarray(getattr(d, "y_disp", []), dtype=float),
                    y_mode=str(getattr(d, "y_mode", "absorbance")),
                    x_units=getattr(d, "x_units", None),
                    y_units=getattr(d, "y_units", None),
                    loaded_at_utc=getattr(d, "loaded_at_utc", None),
                    peak_settings=dict(getattr(d, "peak_settings", {}) or {}),
                    peaks=list(getattr(d, "peaks", []) or []),
                    peak_label_overrides=dict(getattr(d, "peak_label_overrides", {}) or {}),
                    peak_suppressed=set(getattr(d, "peak_suppressed", set()) or set()),
                    peak_label_positions=dict(getattr(d, "peak_label_positions", {}) or {}),
                )
            )

        active_old = getattr(src, "active_dataset_id", None)
        active_new = (id_map.get(str(active_old)) if active_old else None)
        src_color = None
        try:
            src_color = (None if getattr(src, "line_color", None) is None else str(src.line_color))
        except Exception:
            src_color = None
        self.workspaces[str(ws_id)] = FTIRWorkspace(
            id=str(ws_id),
            name=str(name),
            datasets=datasets,
            active_dataset_id=active_new,
            line_color=(src_color or self._default_color_for_workspace_id(str(ws_id))),
        )
        self._set_active_workspace_by_id(str(ws_id))

    def _delete_workspace(self) -> None:
        if not self.workspaces:
            return
        ws = self._active_workspace()
        if len(self.workspaces) <= 1:
            # Last workspace: clear it instead of deleting.
            self.clear_all()
            return

        ok = messagebox.askyesno("Delete Workspace", f"Delete workspace '{ws.name}'?", parent=self.app)
        if not ok:
            return

        del_id = str(ws.id)
        try:
            self.workspaces.pop(del_id, None)
        except Exception:
            pass
        # Remove deleted workspace from overlay groups.
        try:
            self._remove_workspace_id_from_overlay_groups(str(del_id))
        except Exception:
            pass

        # Drop cached line artists for the deleted workspace.
        try:
            for k, ln in list((self._line_artists or {}).items()):
                if str(k[0]) != del_id:
                    continue
                try:
                    ln.remove()
                except Exception:
                    pass
                try:
                    self._line_artists.pop(k, None)
                except Exception:
                    pass
        except Exception:
            pass

        # Choose a new active workspace.
        new_active = next(iter(self.workspaces.keys()))
        self.active_workspace_id = str(new_active)
        self._refresh_workspace_selector()
        self._sync_active_workspace_to_app_workspace()
        self.refresh_from_workspace(select_active=True)
        self._rebuild_overlay_group_list()
        self._rebuild_overlay_group_members_list()
        self._rebuild_overlay_selection_list()
        self._schedule_redraw()

    # --- Overlays (multiple saved overlay groups) ---

    def _parse_overlay_iid(self, iid: str) -> Optional[Tuple[str, str]]:
        s = str(iid or "")
        if "::" not in s:
            return None
        a, b = s.split("::", 1)
        a = a.strip()
        b = b.strip()
        if not a or not b:
            return None
        return (a, b)

    def _overlay_group_counter_next(self) -> int:
        best = 0
        for g in (self._overlay_groups or {}).values():
            try:
                name = str(getattr(g, "name", ""))
                if name.lower().startswith("overlay "):
                    rest = name[8:].strip()
                    n = int(rest.split()[0])
                    best = max(best, n)
            except Exception:
                continue
        return int(best) + 1

    def _overlay_group_ws_indicator(self, members: Sequence[FTIRDatasetKey]) -> str:
        ws_ids = set(str(k[0]) for k in (members or []) if isinstance(k, tuple) and len(k) == 2)
        if not ws_ids:
            return "—"
        return "1" if len(ws_ids) == 1 else "mix"

    def _get_selected_overlay_group_id(self) -> Optional[str]:
        tree = self._overlay_groups_tree
        if tree is None:
            return None
        try:
            sel = list(tree.selection() or [])
        except Exception:
            sel = []
        return (str(sel[0]) if sel else None)

    def _get_active_overlay_group(self) -> Optional[OverlayGroup]:
        gid = str(self._active_overlay_group_id or "").strip()
        if not gid:
            return None
        return (self._overlay_groups or {}).get(gid)

    def _effective_active_key(self) -> Optional[FTIRDatasetKey]:
        g = self._get_active_overlay_group()
        if g is not None:
            members = list(getattr(g, "members", []) or [])
            cand = (g.active_member if g.active_member is not None else (members[0] if members else None))
            if cand is not None and self._get_dataset_by_key((str(cand[0]), str(cand[1]))) is not None:
                return (str(cand[0]), str(cand[1]))
            for k in members:
                if self._get_dataset_by_key((str(k[0]), str(k[1]))) is not None:
                    return (str(k[0]), str(k[1]))
            return None

        # Normal mode: active dataset in active workspace.
        try:
            ws = self._active_workspace()
            aid = getattr(ws, "active_dataset_id", None)
            if aid:
                return (str(self.active_workspace_id), str(aid))
        except Exception:
            pass
        return None

    def _effective_active_dataset(self) -> Optional[FTIRDataset]:
        k = self._effective_active_key()
        if k is None:
            return None
        return self._get_dataset_by_key((str(k[0]), str(k[1])))

    def _overlay_selection_ordered_selected_keys(self) -> List[FTIRDatasetKey]:
        tree = self._overlay_selection_tree
        if tree is None:
            return []
        try:
            selected = set(str(x) for x in (tree.selection() or []))
        except Exception:
            selected = set()
        if not selected:
            return []

        out: List[FTIRDatasetKey] = []
        seen: set[FTIRDatasetKey] = set()
        try:
            for iid in list(tree.get_children("")):
                if str(iid) not in selected:
                    continue
                key = self._parse_overlay_iid(str(iid))
                if key is None:
                    continue
                k = (str(key[0]), str(key[1]))
                if k in seen:
                    continue
                out.append(k)
                seen.add(k)
        except Exception:
            pass
        return out

    def _default_overlay_group_name(self, members: Sequence[FTIRDatasetKey]) -> str:
        n = self._overlay_group_counter_next()
        parts: List[str] = []
        for key in (members or [])[:2]:
            d = self._get_dataset_by_key((str(key[0]), str(key[1])))
            if d is None:
                continue
            parts.append(str(getattr(d, "name", "dataset")))
        suffix = " + ".join(parts)
        if len(list(members or [])) > 2:
            suffix = (suffix + " + …") if suffix else "…"
        if suffix:
            return f"Overlay {n} ({suffix})"
        return f"Overlay {n}"

    def _new_overlay_group_from_selection(self) -> None:
        members = self._overlay_selection_ordered_selected_keys()
        if len(members) < 2:
            messagebox.showinfo("Overlay", "Select at least 2 datasets to create an overlay group.", parent=self.app)
            return

        gid = uuid.uuid4().hex
        g = OverlayGroup(
            group_id=str(gid),
            name=self._default_overlay_group_name(members),
            members=list(members),
            active_member=(members[0] if members else None),
            per_member_style={k: StyleState(linewidth=1.2) for k in members},
            created_at=float(time.time()),
        )
        self._overlay_groups[str(gid)] = g

        # Default: auto-activate new overlay group.
        self._active_overlay_group_id = str(gid)

        self._rebuild_overlay_group_list(select_group_id=str(gid))
        self._rebuild_overlay_group_members_list()
        self._schedule_redraw()

    def _activate_selected_overlay_group(self) -> None:
        gid = self._get_selected_overlay_group_id()
        if not gid:
            return
        if gid not in (self._overlay_groups or {}):
            return
        self._active_overlay_group_id = str(gid)

        g = self._overlay_groups.get(str(gid))
        if g is not None:
            if g.active_member is None and (getattr(g, "members", None) or []):
                g.active_member = (str(g.members[0][0]), str(g.members[0][1]))

        self._rebuild_overlay_group_list(select_group_id=str(gid))
        self._rebuild_overlay_group_members_list()
        self._schedule_redraw()

    def _clear_active_overlay_group(self) -> None:
        self._active_overlay_group_id = None
        self._rebuild_overlay_group_list(select_group_id=self._get_selected_overlay_group_id())
        self._schedule_redraw()

    def _rename_selected_overlay_group(self) -> None:
        gid = self._get_selected_overlay_group_id()
        if not gid:
            return
        g = (self._overlay_groups or {}).get(str(gid))
        if g is None:
            return
        new_name = simpledialog.askstring("Rename Overlay", "Overlay name:", initialvalue=str(getattr(g, "name", "")), parent=self.app)
        if new_name is None:
            return
        new_name = str(new_name).strip()
        if not new_name:
            return
        g.name = str(new_name)
        self._rebuild_overlay_group_list(select_group_id=str(gid))

    def _duplicate_selected_overlay_group(self) -> None:
        gid = self._get_selected_overlay_group_id()
        if not gid:
            return
        src = (self._overlay_groups or {}).get(str(gid))
        if src is None:
            return

        new_id = uuid.uuid4().hex
        new_name = f"{str(getattr(src, 'name', 'Overlay'))} (copy)"
        members = list(getattr(src, "members", []) or [])
        active_member = None
        try:
            if getattr(src, "active_member", None) is not None:
                active_member = (str(src.active_member[0]), str(src.active_member[1]))
        except Exception:
            active_member = None

        g = OverlayGroup(
            group_id=str(new_id),
            name=str(new_name),
            members=members,
            active_member=active_member,
            per_member_style=dict(getattr(src, "per_member_style", {}) or {}),
            created_at=float(time.time()),
        )
        self._overlay_groups[str(new_id)] = g
        self._rebuild_overlay_group_list(select_group_id=str(new_id))
        self._rebuild_overlay_group_members_list()

    def _delete_selected_overlay_group(self) -> None:
        gid = self._get_selected_overlay_group_id()
        if not gid:
            return
        g = (self._overlay_groups or {}).get(str(gid))
        if g is None:
            return
        ok = messagebox.askyesno("Delete Overlay", f"Delete overlay group '{getattr(g, 'name', '')}'?", parent=self.app)
        if not ok:
            return
        try:
            self._overlay_groups.pop(str(gid), None)
        except Exception:
            pass
        if str(self._active_overlay_group_id or "") == str(gid):
            self._active_overlay_group_id = None
        self._rebuild_overlay_group_list()
        self._rebuild_overlay_group_members_list()
        self._schedule_redraw()

    def _set_active_overlay_member_from_selected(self) -> None:
        g = self._get_active_overlay_group()
        if g is None:
            return
        mt = self._overlay_members_tree
        if mt is None:
            return
        try:
            sel = list(mt.selection() or [])
        except Exception:
            sel = []
        if not sel:
            return
        key = self._parse_overlay_iid(str(sel[0]))
        if key is None:
            return
        k = (str(key[0]), str(key[1]))
        g.active_member = k
        self._schedule_redraw()

    def _rebuild_overlay_group_list(self, *, select_group_id: Optional[str] = None) -> None:
        tree = self._overlay_groups_tree
        if tree is None:
            return
        try:
            for iid in list(tree.get_children("")):
                tree.delete(iid)
        except Exception:
            pass

        rows: List[Tuple[str, str, int, str]] = []
        for gid, g in (self._overlay_groups or {}).items():
            try:
                members = list(getattr(g, "members", []) or [])
                rows.append((str(gid), str(getattr(g, "name", "Overlay")), int(len(members)), self._overlay_group_ws_indicator(members)))
            except Exception:
                continue
        rows.sort(key=lambda r: r[1].lower())
        for gid, name, n, ws_ind in rows:
            try:
                tree.insert("", "end", iid=str(gid), values=(str(name), str(n), str(ws_ind)))
            except Exception:
                continue

        want = select_group_id
        if want is None:
            want = self._get_selected_overlay_group_id()
        if want is None and rows:
            want = str(rows[0][0])
        if want is not None:
            try:
                if tree.exists(str(want)):
                    tree.selection_set(str(want))
                    tree.see(str(want))
            except Exception:
                pass

    def _rebuild_overlay_group_members_list(self) -> None:
        mt = self._overlay_members_tree
        if mt is None:
            return
        try:
            for iid in list(mt.get_children("")):
                mt.delete(iid)
        except Exception:
            pass

        gid = self._get_selected_overlay_group_id()
        if not gid:
            return
        g = (self._overlay_groups or {}).get(str(gid))
        if g is None:
            return

        for key in (getattr(g, "members", None) or []):
            try:
                ws = self.workspaces.get(str(key[0]))
                ws_name = (str(getattr(ws, "name", "")) if ws is not None else str(key[0]))
                d = self._get_dataset_by_key((str(key[0]), str(key[1])))
                ds_name = (str(getattr(d, "name", "dataset")) if d is not None else str(key[1]))
                label = f"{ws_name} :: {ds_name}"
                iid = f"{key[0]}::{key[1]}"
                mt.insert("", "end", iid=str(iid), values=(str(label),))
            except Exception:
                continue

    def _rebuild_overlay_selection_list(self) -> None:
        tree = self._overlay_selection_tree
        if tree is None:
            return

        q = str(self._overlay_filter_var.get() or "").strip().lower()
        rows: List[Tuple[str, str]] = []  # (iid, label)
        for ws_id, ws in (self.workspaces or {}).items():
            ws_name = str(getattr(ws, "name", ""))
            for d in (getattr(ws, "datasets", []) or []):
                ds_id = str(getattr(d, "id", ""))
                if not ds_id:
                    continue
                ds_name = str(getattr(d, "name", ""))
                ds_path = str(getattr(d, "path", "") or "")
                label = f"{ws_name} :: {ds_name}"
                if q:
                    hay = f"{ws_name} {ds_name} {ds_path}".lower()
                    if q not in hay:
                        continue
                iid = f"{ws_id}::{ds_id}"
                rows.append((iid, label))

        rows.sort(key=lambda r: r[1].lower())
        try:
            for iid in list(tree.get_children("")):
                tree.delete(iid)
        except Exception:
            pass
        for iid, label in rows:
            try:
                tree.insert("", "end", iid=str(iid), values=(str(label),))
            except Exception:
                continue

    def _remove_dataset_key_from_overlay_groups(self, key: FTIRDatasetKey) -> None:
        k = (str(key[0]), str(key[1]))
        for g in (self._overlay_groups or {}).values():
            try:
                g.members = [m for m in (getattr(g, "members", []) or []) if (str(m[0]), str(m[1])) != k]
            except Exception:
                pass
            try:
                if g.active_member is not None and (str(g.active_member[0]), str(g.active_member[1])) == k:
                    g.active_member = None
            except Exception:
                pass
            try:
                g.per_member_style.pop(k, None)
            except Exception:
                pass

    def _remove_workspace_id_from_overlay_groups(self, ws_id: str) -> None:
        wid = str(ws_id or "").strip()
        if not wid:
            return
        for g in (self._overlay_groups or {}).values():
            try:
                g.members = [m for m in (getattr(g, "members", []) or []) if str(m[0]) != wid]
            except Exception:
                pass
            try:
                if g.active_member is not None and str(g.active_member[0]) == wid:
                    g.active_member = None
            except Exception:
                pass

    def refresh_from_workspace(self, *, select_active: bool) -> None:
        tree = self._tree
        if tree is None:
            return

        ws = self._active_workspace()

        self._ignore_tree_select = True
        try:
            for iid in list(tree.get_children("")):
                tree.delete(iid)
        except Exception:
            pass

        active_id = ws.active_dataset_id
        for d in list(getattr(ws, "datasets", []) or []):
            try:
                mark = "●" if (active_id is not None and str(d.id) == str(active_id)) else ""
                n_pts = int(np.asarray(getattr(d, "x_full", [])).size)
                tree.insert("", "end", iid=str(d.id), values=(mark, str(d.name), str(n_pts)))
            except Exception:
                continue

        if select_active and active_id is not None:
            try:
                if tree.exists(str(active_id)):
                    tree.selection_set(str(active_id))
                    tree.see(str(active_id))
            except Exception:
                pass
        # Important: TreeviewSelect can fire asynchronously after selection_set().
        # Keep ignore enabled until the event queue drains.
        def _release_ignore_and_redraw() -> None:
            self._ignore_tree_select = False
            self._schedule_redraw()

        try:
            self.app.after(0, _release_ignore_and_redraw)
        except Exception:
            _release_ignore_and_redraw()

    def _import_active_workspace_from_app_workspace(self) -> None:
        """One-way import: app Workspace -> current active FTIRWorkspace.

        Used only for backward-compatible workspace loading paths.
        """
        try:
            ws = self._active_workspace()
            ws.datasets = list(getattr(self.workspace, "ftir_datasets", []) or [])
            ws.active_dataset_id = (None if not getattr(self.workspace, "active_ftir_id", None) else str(self.workspace.active_ftir_id))
        except Exception:
            pass

    def apply_restored_ftir_state(self, payload: Optional[Dict[str, Any]]) -> None:
        """Apply restored multi-workspace + overlay-groups state after FTIR datasets have loaded."""
        # Ensure we have the loaded FTIRDataset objects available.
        self._import_active_workspace_from_app_workspace()

        dataset_by_id: Dict[str, FTIRDataset] = {}
        for d in (getattr(self.workspace, "ftir_datasets", []) or []):
            try:
                did = str(getattr(d, "id", ""))
                if did:
                    dataset_by_id[did] = d
            except Exception:
                continue

        if not isinstance(payload, dict):
            payload = {}

        # Restore FTIR workspaces
        restored_workspaces = payload.get("ftir_workspaces")
        if isinstance(restored_workspaces, list) and restored_workspaces:
            new_workspaces: Dict[str, FTIRWorkspace] = {}
            for row in restored_workspaces:
                if not isinstance(row, dict):
                    continue
                ws_id = str(row.get("id") or "").strip()
                if not ws_id:
                    continue
                name = str(row.get("name") or "Workspace")
                ds_ids = [str(x) for x in (row.get("dataset_ids") or []) if str(x)]
                datasets: List[FTIRDataset] = [dataset_by_id[x] for x in ds_ids if x in dataset_by_id]
                active_ds = (None if not row.get("active_dataset_id") else str(row.get("active_dataset_id")))
                line_color = (None if not row.get("line_color") else str(row.get("line_color")))
                if line_color:
                    try:
                        line_color = str(mcolors.to_hex(mcolors.to_rgb(str(line_color))))
                    except Exception:
                        line_color = str(line_color)
                if active_ds and active_ds not in dataset_by_id:
                    active_ds = None
                if (not active_ds) and datasets:
                    active_ds = str(getattr(datasets[0], "id", ""))
                if not line_color:
                    line_color = self._default_color_for_workspace_id(str(ws_id))

                wso = FTIRWorkspace(id=str(ws_id), name=str(name), datasets=datasets, active_dataset_id=active_ds, line_color=line_color)

                # Restore bond annotations
                try:
                    wso.bond_annotations = []
                    rows = row.get("bond_annotations")
                    if isinstance(rows, list):
                        for r in rows:
                            if not isinstance(r, dict):
                                continue
                            try:
                                xy = r.get("xytext")
                                if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                                    xy = (float(r.get("x_cm1")), float(r.get("y_value")))
                                wso.bond_annotations.append(
                                    FTIRBondAnnotation(
                                        dataset_id=str(r.get("dataset_id") or ""),
                                        text=str(r.get("text") or ""),
                                        x_cm1=float(r.get("x_cm1")),
                                        y_value=float(r.get("y_value")),
                                        xytext=(float(xy[0]), float(xy[1])),
                                        show_vline=bool(r.get("show_vline", False)),
                                        line_color=str(r.get("line_color") or "#444444"),
                                        text_color=str(r.get("text_color") or "#111111"),
                                        fontsize=int(r.get("fontsize") or 9),
                                        rotation=int(r.get("rotation") or 90),
                                        preset_id=(None if not r.get("preset_id") else str(r.get("preset_id"))),
                                    )
                                )
                            except Exception:
                                continue
                except Exception:
                    pass

                new_workspaces[ws_id] = wso

            if new_workspaces:
                self.workspaces = new_workspaces

        # Ensure we always have at least one workspace
        if not (self.workspaces or {}):
            ws_id = uuid.uuid4().hex
            self.workspaces = {
                str(ws_id): FTIRWorkspace(
                    id=str(ws_id),
                    name="Workspace 1",
                    datasets=list(dataset_by_id.values()),
                    line_color=self._default_color_for_workspace_id(str(ws_id)),
                )
            }

        # Restore active workspace
        active_ws_id = payload.get("active_ftir_workspace_id")
        if isinstance(active_ws_id, str) and active_ws_id in (self.workspaces or {}):
            self.active_workspace_id = str(active_ws_id)
        else:
            self.active_workspace_id = str(next(iter((self.workspaces or {}).keys())))

        # Restore overlay groups
        self._overlay_groups = {}
        og_rows = payload.get("ftir_overlay_groups")
        if isinstance(og_rows, list):
            for row in og_rows:
                if not isinstance(row, dict):
                    continue
                gid = str(row.get("group_id") or "").strip()
                if not gid:
                    continue
                name = str(row.get("name") or "Overlay")
                members: List[FTIRDatasetKey] = []
                for it in (row.get("members") or []):
                    try:
                        a, b = it
                        k = (str(a), str(b))
                        if self._get_dataset_by_key(k) is not None:
                            members.append(k)
                    except Exception:
                        continue
                active_member = None
                try:
                    am = row.get("active_member")
                    if isinstance(am, (list, tuple)) and len(am) == 2:
                        cand = (str(am[0]), str(am[1]))
                        if self._get_dataset_by_key(cand) is not None:
                            active_member = cand
                except Exception:
                    active_member = None

                per_style: Dict[FTIRDatasetKey, StyleState] = {}
                try:
                    ps = row.get("per_member_style") or {}
                    if isinstance(ps, dict):
                        for kk, vv in ps.items():
                            if not isinstance(kk, str) or "::" not in kk:
                                continue
                            a, b = kk.split("::", 1)
                            k = (str(a), str(b))
                            if self._get_dataset_by_key(k) is None:
                                continue
                            lw = 1.2
                            if isinstance(vv, dict) and vv.get("linewidth") is not None:
                                lw = float(vv.get("linewidth"))
                            per_style[k] = StyleState(linewidth=float(lw))
                except Exception:
                    per_style = {}

                self._overlay_groups[gid] = OverlayGroup(
                    group_id=str(gid),
                    name=str(name),
                    members=members,
                    active_member=active_member,
                    per_member_style=per_style,
                    created_at=float(row.get("created_at") or time.time()),
                )

        active_gid = payload.get("active_ftir_overlay_group_id")
        if isinstance(active_gid, str) and active_gid in (self._overlay_groups or {}):
            self._active_overlay_group_id = str(active_gid)
        else:
            self._active_overlay_group_id = None

        self._refresh_workspace_selector()
        self._sync_active_workspace_to_app_workspace()
        self.refresh_from_workspace(select_active=True)
        self._rebuild_overlay_group_list()
        self._rebuild_overlay_group_members_list()
        self._rebuild_overlay_selection_list()
        self._schedule_redraw()

    def reset_for_load(self) -> None:
        """Clear FTIR UI/session state (multi-workspaces + overlays) for FTIR-only restore."""
        # Close peaks dialog if open
        try:
            if getattr(self, "_peaks_dialog", None) is not None:
                try:
                    self._peaks_dialog.destroy()  # type: ignore[union-attr]
                except Exception:
                    pass
                self._peaks_dialog = None
        except Exception:
            pass

        # Clear plotted artists + overlay caches
        try:
            self._line_artists = {}
        except Exception:
            pass
        try:
            self._legend_artist = None
            self._legend_keys = tuple()
        except Exception:
            pass

        # Clear UI vars
        try:
            self._status_var.set("")
        except Exception:
            pass
        try:
            self._show_peaks_all_overlay_var.set(False)
        except Exception:
            pass
        try:
            self._overlay_filter_var.set("")
        except Exception:
            pass
        try:
            self._workspace_select_var.set("")
        except Exception:
            pass

        # Reset per-dataset prefs
        try:
            self._reverse_pref_by_id = {}
        except Exception:
            pass
        try:
            self._reverse_x_var.set(True)
        except Exception:
            pass

        # Clear in-memory FTIR workspaces + overlay groups
        self.workspaces = {}
        self.active_workspace_id = ""
        self._overlay_groups = {}
        self._active_overlay_group_id = None

        # Clear app Workspace model FTIR fields
        try:
            self.workspace.ftir_datasets.clear()
            self.workspace.active_ftir_id = None
        except Exception:
            pass

        # Recreate a default empty workspace
        try:
            self._init_ftir_workspaces_from_app_workspace()
        except Exception:
            pass

        # Refresh UI lists
        try:
            self._refresh_workspace_selector()
            self._rebuild_overlay_group_list()
            self._rebuild_overlay_group_members_list()
            self._rebuild_overlay_selection_list()
            self.refresh_from_workspace(select_active=True)
            self._schedule_redraw()
        except Exception:
            pass

    def _schedule_redraw(self) -> None:
        if self._redraw_after is not None:
            return

        def _do() -> None:
            self._redraw_after = None
            self._redraw()

        try:
            self._redraw_after = self.app.after(0, _do)
        except Exception:
            self._redraw_after = None
            _do()

    def load_ftir_dialog(self) -> None:
        if self._ftir_busy:
            try:
                self._status_var.set("FTIR load already in progress…")
            except Exception:
                pass
            return

        t0 = time.perf_counter()
        paths = filedialog.askopenfilenames(
            parent=self.app,
            title="Select FTIR CSV/TXT files",
            filetypes=[("FTIR files", "*.csv;*.txt"), ("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        self._ftir_timing("filedialog", time.perf_counter() - t0, extra=(f" n={len(paths)}" if paths else ""))
        if not paths:
            return

        # Let Tk breathe after the native dialog returns.
        try:
            self._status_var.set("Preparing FTIR load…")
        except Exception:
            pass
        # Keep UI responsive: do not Path.resolve()/exists() here (OneDrive can block).
        self.app.after(0, lambda: self._load_ftir_files_async([str(p) for p in paths]))

    def _ftir_timing(self, label: str, seconds: float, *, extra: str = "") -> None:
        if not FTIR_DEBUG:
            return
        msg = f"[FTIR] {label}: {seconds:.2f}s{extra}"
        try:
            print(msg)
        except Exception:
            pass
        try:
            self.app._set_status(msg)
        except Exception:
            pass
        try:
            self.app._log("INFO", msg)
        except Exception:
            pass

    def _request_mpl_draw_idle(self, reason: str = "") -> None:
        # UI thread only. Record when we requested a draw; log when it actually happens.
        if self._canvas is None:
            return
        self._mpl_draw_pending = True
        self._mpl_draw_req_t = time.perf_counter()
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

        # Watchdog: if draw never happens, we'll at least log the stall.
        try:
            if self._mpl_draw_watchdog_after is not None:
                self.app.after_cancel(self._mpl_draw_watchdog_after)
        except Exception:
            pass

        def _watchdog() -> None:
            self._mpl_draw_watchdog_after = None
            if not self._mpl_draw_pending:
                return
            dt = 0.0
            try:
                if self._mpl_draw_req_t is not None:
                    dt = float(time.perf_counter() - self._mpl_draw_req_t)
            except Exception:
                dt = 0.0
            self._ftir_timing("draw_pending", dt, extra=(f" reason={reason}" if reason else ""))

        try:
            self._mpl_draw_watchdog_after = self.app.after(1000, _watchdog)
        except Exception:
            self._mpl_draw_watchdog_after = None

    def _on_mpl_draw_event(self, evt=None) -> None:
        if not self._mpl_draw_pending:
            return
        self._mpl_draw_pending = False
        dt = 0.0
        try:
            if self._mpl_draw_req_t is not None:
                dt = float(time.perf_counter() - self._mpl_draw_req_t)
        except Exception:
            dt = 0.0
        self._ftir_timing("mpl_draw", dt)

    def _set_ftir_busy(self, busy: bool) -> None:
        self._ftir_busy = bool(busy)
        st = (tk.DISABLED if busy else tk.NORMAL)
        for b in (self._btn_load, self._btn_remove, self._btn_clear):
            try:
                if b is not None:
                    b.configure(state=st)
            except Exception:
                pass

    def _start_ftir_queue_poll(self) -> None:
        if self._ftir_poll_after is not None:
            return

        def _poll() -> None:
            self._ftir_poll_after = None
            self._poll_ftir_queue()
            if self._ftir_busy or self._peaks_busy:
                try:
                    self._ftir_poll_after = self.app.after(50, _poll)
                except Exception:
                    self._ftir_poll_after = None

        try:
            self._ftir_poll_after = self.app.after(0, _poll)
        except Exception:
            self._ftir_poll_after = None

    def _poll_ftir_queue(self) -> None:
        # UI thread only.
        while True:
            try:
                evt = self._ftir_event_q.get_nowait()
            except Exception:
                break
            if not evt:
                continue
            kind = str(evt[0])
            if kind == "progress":
                try:
                    i, n, name = int(evt[1]), int(evt[2]), str(evt[3])
                    self._set_loading_status(i, n, name)
                except Exception:
                    pass
            elif kind == "timing":
                try:
                    label = str(evt[1])
                    dt = float(evt[2])
                    extra = str(evt[3]) if len(evt) > 3 else ""
                    self._ftir_timing(label, dt, extra=extra)
                except Exception:
                    pass
            elif kind == "done":
                try:
                    results = evt[1]
                    activate_id = (str(evt[2]) if evt[2] else None)
                except Exception:
                    results = []
                    activate_id = None
                self._on_ftir_batch_loaded(results, activate_id=activate_id)
            elif kind == "peaks_done":
                try:
                    dataset_id = str(evt[1])
                    peak_settings = evt[2]
                    peaks = evt[3]
                except Exception:
                    dataset_id = ""
                    peak_settings = {}
                    peaks = []
                self._on_peaks_ready(dataset_id, peak_settings, peaks)
            elif kind == "peaks_done_key":
                try:
                    ws_id = str(evt[1])
                    ds_id = str(evt[2])
                    peak_settings = evt[3]
                    peaks = evt[4]
                except Exception:
                    ws_id = ""
                    ds_id = ""
                    peak_settings = {}
                    peaks = []
                if ws_id and ds_id:
                    self._on_peaks_ready_for_key((ws_id, ds_id), peak_settings, peaks)
            elif kind == "peaks_batch_done":
                try:
                    done = int(evt[1])
                    total = int(evt[2])
                except Exception:
                    done = 0
                    total = 0
                self._on_peaks_batch_done(done, total)
            else:
                # Unknown event
                continue

    def _load_ftir_files_async(self, paths: Sequence[str]) -> None:
        if self._ftir_busy:
            try:
                self._status_var.set("FTIR load already in progress…")
            except Exception:
                pass
            return

        # IMPORTANT: do not touch filesystem/path resolution on the Tk thread.
        selected = [str(p) for p in paths if str(p).strip()]
        if not selected:
            return

        def norm_path(s: str) -> str:
            try:
                return os.path.normcase(os.path.normpath(str(s)))
            except Exception:
                return str(s)

        # Snapshot existing paths/ids in the ACTIVE FTIR workspace (no resolve/exists).
        existing_by_norm: Dict[str, str] = {}
        try:
            ws = self._active_workspace()
            for d in (getattr(ws, "datasets", []) or []):
                try:
                    existing_by_norm[norm_path(str(getattr(d, "path", "")))] = str(getattr(d, "id", ""))
                except Exception:
                    continue
        except Exception:
            existing_by_norm = {}

        self._set_ftir_busy(True)
        self._start_ftir_queue_poll()

        try:
            self._status_var.set(f"Loading FTIR… ({len(selected)} file(s))")
        except Exception:
            pass

        t_start = time.perf_counter()

        def infer_from_meta(meta: Dict[str, Any]) -> Tuple[bool, str, Optional[str], Optional[str]]:
            x_units = None if not isinstance(meta, dict) else meta.get("XUNITS")
            y_units = None if not isinstance(meta, dict) else meta.get("YUNITS")
            y_mode = "absorbance"
            try:
                yu = (str(y_units).strip().lower() if y_units is not None else "")
                if "trans" in yu:
                    y_mode = "transmittance"
                elif "abs" in yu:
                    y_mode = "absorbance"
            except Exception:
                y_mode = "absorbance"

            reverse_recommended = False
            try:
                xu = (str(x_units).strip().lower() if x_units is not None else "")
                if ("1/cm" in xu) or ("cm-1" in xu) or ("cm^-1" in xu) or ("cm⁻¹" in xu):
                    reverse_recommended = True
            except Exception:
                reverse_recommended = False
            return bool(reverse_recommended), str(y_mode), (None if x_units is None else str(x_units)), (None if y_units is None else str(y_units))

        # Worker: no Tk calls; communicate via queue polled by UI thread.
        def worker(raw_paths: List[str], existing_norm: Dict[str, str]) -> None:
            t_plan0 = time.perf_counter()
            todo: List[str] = []
            activate_id: Optional[str] = None
            seen: set[str] = set()
            for rp in raw_paths:
                rp2 = str(rp).strip()
                if not rp2:
                    continue
                nrm = norm_path(rp2)
                if nrm in seen:
                    continue
                seen.add(nrm)
                if nrm in existing_norm:
                    activate_id = existing_norm.get(nrm) or activate_id
                    continue
                todo.append(rp2)
                # Cooperative yield: keep UI responsive even if many paths.
                if len(todo) and (len(todo) % 200 == 0):
                    time.sleep(0)
            try:
                self._ftir_event_q.put(("timing", "plan", time.perf_counter() - t_plan0, f" todo={len(todo)} existing={len(existing_norm)}"))
            except Exception:
                pass

            if not todo:
                try:
                    self._ftir_event_q.put(("done", [], activate_id))
                except Exception:
                    pass
                return

            results: List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool, str, Optional[str], Optional[str], Optional[str]]] = []
            n_total = int(len(todo))
            for i, pp in enumerate(todo, start=1):
                base = str(Path(pp).name)
                try:
                    t0p = time.perf_counter()
                    x, y, meta = _parse_ftir_xy_numpy(str(pp))
                    self._ftir_event_q.put(("timing", "parse", time.perf_counter() - t0p, f" file={base} n={int(x.size)}"))
                    if int(x.size) < 5:
                        raise ValueError("No usable numeric XY data found")

                    t0s = time.perf_counter()
                    try:
                        order = np.argsort(x)
                        x = x[order]
                        y = y[order]
                    except Exception:
                        pass
                    self._ftir_event_q.put(("timing", "sort", time.perf_counter() - t0s, f" file={base}"))

                    rr, y_mode, x_units, y_units = infer_from_meta(meta)

                    # Render decimation (plot only); keep full-res arrays stored.
                    x_plot = x
                    y_plot = y
                    try:
                        n = int(x.size)
                        max_n = 50_000
                        if n > max_n:
                            step = int(math.ceil(float(n) / float(max_n)))
                            x_plot = x[::step]
                            y_plot = y[::step]
                    except Exception:
                        x_plot = x
                        y_plot = y

                    results.append((str(pp), x, y, x_plot, y_plot, bool(rr), str(y_mode), None, x_units, y_units))
                except Exception as exc:
                    results.append((str(pp), None, None, None, None, True, "absorbance", str(exc), None, None))

                try:
                    self._ftir_event_q.put(("progress", i, n_total, base))
                except Exception:
                    pass

            try:
                self._ftir_event_q.put(("timing", "batch_total", time.perf_counter() - t_start, f" n={n_total}"))
            except Exception:
                pass
            try:
                self._ftir_event_q.put(("done", results, activate_id))
            except Exception:
                pass

        threading.Thread(target=worker, args=(selected, existing_by_norm), daemon=True).start()

    def _set_loading_status(self, i: int, n: int, name: str) -> None:
        try:
            self._status_var.set(f"Loading FTIR {i}/{n}: {name}")
        except Exception:
            pass
        try:
            self.app._set_status(f"Loading FTIR {i}/{n}: {name}")
        except Exception:
            pass

    def _on_ftir_batch_loaded(self, results: List[Tuple[Any, ...]], *, activate_id: Optional[str]) -> None:
        # UI thread only.
        try:
            if activate_id:
                ws = self._active_workspace()
                ws.active_dataset_id = str(activate_id)
        except Exception:
            pass

        ws = self._active_workspace()

        last_loaded_id: Optional[str] = None
        errs: List[str] = []
        for row in (results or []):
            try:
                (pp_s, wn, y, wn_plot, y_plot, reverse_recommended, y_mode, errtxt, x_units, y_units) = row
            except Exception:
                continue

            pp = Path(str(pp_s))
            if errtxt is not None or wn is None or y is None:
                errs.append(f"{pp.name}: {errtxt or 'Unknown error'}")
                continue

            ds_id = uuid.uuid4().hex
            loaded_at = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            ds = FTIRDataset(
                id=str(ds_id),
                # Do not resolve() on the UI thread (OneDrive paths can block).
                path=Path(pp).expanduser(),
                name=str(pp.name),
                x_full=np.asarray(wn, dtype=float),
                y_full=np.asarray(y, dtype=float),
                x_disp=(np.asarray(wn_plot, dtype=float) if wn_plot is not None else np.asarray(wn, dtype=float)),
                y_disp=(np.asarray(y_plot, dtype=float) if y_plot is not None else np.asarray(y, dtype=float)),
                y_mode=str(y_mode or "absorbance"),
                x_units=(None if x_units is None else str(x_units)),
                y_units=(None if y_units is None else str(y_units)),
                loaded_at_utc=str(loaded_at),
            )
            ws.datasets.append(ds)
            last_loaded_id = str(ds.id)

            try:
                self._reverse_pref_by_id[str(ds.id)] = bool(reverse_recommended)
            except Exception:
                pass

            try:
                self.app._log("INFO", f"Loaded FTIR: {pp.name} (n={int(np.asarray(wn).size)})")
            except Exception:
                pass

        if last_loaded_id is not None:
            try:
                self._set_active_dataset(str(self.active_workspace_id), str(last_loaded_id), reason="load_new")
            except Exception:
                ws.active_dataset_id = str(last_loaded_id)
                try:
                    self._reverse_x_var.set(bool(self._reverse_pref_by_id.get(str(last_loaded_id), True)))
                except Exception:
                    pass

        # Mirror to app Workspace for legacy save/load.
        self._sync_active_workspace_to_app_workspace()

        # If _set_active_dataset already refreshed, this will be a no-op refresh.
        self.refresh_from_workspace(select_active=True)

        if errs:
            try:
                self._status_var.set(f"FTIR: {len(errs)} file(s) failed (see Diagnostics)")
            except Exception:
                pass
            try:
                self.app._log("WARN", "FTIR load failures:\n" + "\n".join(errs[:50]))
            except Exception:
                pass

        try:
            self.app._set_status("FTIR load complete")
        except Exception:
            pass

        self._set_ftir_busy(False)

    def _read_table(self, path: Path) -> pd.DataFrame:
        p = Path(path).expanduser().resolve()
        # Robust + fast-ish heuristics:
        # - Avoid sep=None sniffing (can be extremely slow and can starve the UI via the GIL).
        # - Prefer the C engine with common separators; fall back to whitespace.
        common_kwargs: Dict[str, Any] = {
            "comment": "#",
            "encoding_errors": "ignore",
            "on_bad_lines": "skip",
            "low_memory": False,
        }

        # JASCO-style exports often have a metadata header followed by an `XYDATA` marker.
        # Parsing the whole file as a table can be slow and can confuse column inference,
        # so prefer reading only the numeric block when we detect this pattern.
        try:
            xy_idx: Optional[int] = None
            meta: Dict[str, str] = {}
            with p.open("r", errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if i > 5000:
                        break
                    s = line.strip("\r\n")
                    if s.strip().upper() == "XYDATA":
                        xy_idx = int(i)
                        break
                    try:
                        if "," in s:
                            k, v = s.split(",", 1)
                            k = str(k).strip().upper()
                            v = str(v).strip()
                            if k:
                                meta[k] = v
                    except Exception:
                        pass

            if xy_idx is not None:
                # Guess delimiter from a few rows after XYDATA.
                delim: Optional[str] = ","
                try:
                    samples: List[str] = []
                    with p.open("r", errors="ignore") as fh2:
                        for i, line in enumerate(fh2):
                            if i <= xy_idx:
                                continue
                            s = line.strip()
                            if not s:
                                continue
                            samples.append(s)
                            if len(samples) >= 5:
                                break

                    counts = {",": 0, ";": 0, "\t": 0}
                    for s in samples:
                        for d in counts:
                            counts[d] += int(s.count(d))
                    best = max(counts, key=lambda k: counts[k])
                    if int(counts[best]) == 0:
                        delim = None
                    else:
                        delim = str(best)
                except Exception:
                    delim = ","

                if delim is None:
                    df = pd.read_csv(
                        p,
                        sep=r"\s+",
                        engine="python",
                        skiprows=int(xy_idx) + 1,
                        header=None,
                        names=["x", "y"],
                        **common_kwargs,
                    )
                else:
                    df = pd.read_csv(
                        p,
                        sep=delim,
                        engine="c",
                        skiprows=int(xy_idx) + 1,
                        header=None,
                        names=["x", "y"],
                        skipinitialspace=True,
                        **common_kwargs,
                    )
                if int(getattr(df, "shape", (0, 0))[1]) >= 2:
                    try:
                        df.attrs["meta"] = dict(meta)
                    except Exception:
                        pass
                    return df
        except Exception:
            pass

        seps = [",", "\t", ";", "|"]
        for sep in seps:
            try:
                df = pd.read_csv(p, sep=sep, engine="c", **common_kwargs)
                if int(getattr(df, "shape", (0, 0))[1]) >= 2:
                    return df
            except Exception:
                continue

        # Whitespace-delimited (common for exported spectra)
        try:
            df = pd.read_csv(p, delim_whitespace=True, engine="c", **common_kwargs)
            if int(getattr(df, "shape", (0, 0))[1]) >= 2:
                return df
        except Exception:
            pass

        # Last resort: regex whitespace (python engine)
        return pd.read_csv(p, sep=r"\s+", engine="python", **common_kwargs)

    def _pick_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, bool, str, Optional[str], Optional[str]]:
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns")

        meta: Dict[str, Any] = {}
        try:
            meta = dict(getattr(df, "attrs", {}).get("meta") or {})
        except Exception:
            meta = {}
        x_units = None if not isinstance(meta, dict) else meta.get("XUNITS")
        y_units = None if not isinstance(meta, dict) else meta.get("YUNITS")

        cols = [str(c) for c in df.columns]
        col_lower = [c.strip().lower() for c in cols]

        def is_x_name(n: str) -> bool:
            return ("wavenumber" in n) or ("cm-1" in n) or ("cm^-1" in n) or ("cm⁻¹" in n) or (n in ("wn", "x"))

        x_candidates = [cols[i] for i, n in enumerate(col_lower) if is_x_name(n)]
        numeric_cols: List[str] = []
        for c in cols:
            try:
                v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                if int(np.sum(np.isfinite(v))) >= 5:
                    numeric_cols.append(c)
            except Exception:
                continue

        if not numeric_cols:
            raise ValueError("No usable numeric columns found")

        # Choose x column: preferred name if available, else numeric column with widest range.
        xcol: Optional[str] = None
        for c in x_candidates:
            if c in numeric_cols:
                xcol = c
                break
        if xcol is None:
            best_range = None
            for c in numeric_cols:
                v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size < 5:
                    continue
                r = float(np.nanmax(v) - np.nanmin(v))
                if best_range is None or r > best_range:
                    best_range = r
                    xcol = c

        if xcol is None:
            xcol = numeric_cols[0]

        # Choose y as the numeric column that yields the most finite pairs with x.
        x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
        best_y: Optional[Tuple[str, int]] = None
        for c in numeric_cols:
            if c == xcol:
                continue
            y = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            n = int(np.sum(np.isfinite(x) & np.isfinite(y)))
            if best_y is None or n > best_y[1]:
                best_y = (c, n)
        if best_y is None or best_y[1] < 5:
            raise ValueError("No usable numeric (x,y) column pair found")
        ycol = best_y[0]

        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 5:
            raise ValueError("Not enough numeric samples after filtering")

        # Preserve a hint of original direction (for default reverse-x)
        try:
            reverse_recommended = bool(x[0] > x[-1])
        except Exception:
            reverse_recommended = False

        order = np.argsort(x)
        x_sorted = np.asarray(x[order], dtype=float)
        y_sorted = np.asarray(y[order], dtype=float)

        # Infer y_mode from metadata when available
        y_mode = "absorbance"
        try:
            yu = (str(y_units).strip().lower() if y_units is not None else "")
            if "trans" in yu:
                y_mode = "transmittance"
            elif "abs" in yu:
                y_mode = "absorbance"
        except Exception:
            y_mode = "absorbance"

        # Recommend reverse-x for wavenumber-like axes (common FTIR convention)
        try:
            xu = (str(x_units).strip().lower() if x_units is not None else "")
            wavenumber_like = ("1/cm" in xu) or ("cm-1" in xu) or ("cm^-1" in xu) or ("cm⁻¹" in xu)
            if wavenumber_like:
                reverse_recommended = True
        except Exception:
            pass

        return x_sorted, y_sorted, bool(reverse_recommended), str(y_mode), (None if x_units is None else str(x_units)), (None if y_units is None else str(y_units))

    def _load_ftir_file(self, path: Path) -> None:
        # Legacy sync loader (kept for any internal callers); prefer _load_ftir_files_async for UI.
        p = Path(path).expanduser().resolve()
        if not p.exists():
            messagebox.showerror("FTIR", f"File not found:\n{p}", parent=self.app)
            return

        ws = self._active_workspace()

        # If already loaded, just activate it.
        for d in (getattr(ws, "datasets", []) or []):
            try:
                if getattr(d, "path", None) == p:
                    ws.active_dataset_id = str(d.id)
                    self._sync_active_workspace_to_app_workspace()
                    self.refresh_from_workspace(select_active=True)
                    return
            except Exception:
                continue

        try:
            df = self._read_table(p)
            wn, y, reverse_recommended, y_mode, x_units, y_units = self._pick_xy(df)
        except Exception as exc:
            messagebox.showerror("FTIR", f"Failed to load FTIR file:\n{p}\n\n{exc}", parent=self.app)
            return

        ds_id = uuid.uuid4().hex
        loaded_at = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        x_full = np.asarray(wn, dtype=float)
        y_full = np.asarray(y, dtype=float)
        x_disp = x_full
        y_disp = y_full
        try:
            n = int(x_full.size)
            max_n = 50_000
            if n > max_n:
                step = int(math.ceil(float(n) / float(max_n)))
                x_disp = x_full[::step]
                y_disp = y_full[::step]
        except Exception:
            x_disp = x_full
            y_disp = y_full

        ds = FTIRDataset(
            id=str(ds_id),
            path=p,
            name=str(p.name),
            x_full=x_full,
            y_full=y_full,
            x_disp=np.asarray(x_disp, dtype=float),
            y_disp=np.asarray(y_disp, dtype=float),
            y_mode=str(y_mode),
            x_units=(None if x_units is None else str(x_units)),
            y_units=(None if y_units is None else str(y_units)),
            loaded_at_utc=str(loaded_at),
        )
        ws.datasets.append(ds)
        ws.active_dataset_id = str(ds.id)
        self._sync_active_workspace_to_app_workspace()

        # Default: reverse x-axis if this file was likely descending (common FTIR)
        try:
            self._reverse_pref_by_id[str(ds.id)] = bool(reverse_recommended)
            self._reverse_x_var.set(bool(reverse_recommended))
        except Exception:
            pass

        try:
            self.app._log("INFO", f"Loaded FTIR: {p.name} (n={int(np.asarray(wn).size)})")
        except Exception:
            pass

        try:
            self.app._set_status(f"Loaded FTIR: {p.name}")
        except Exception:
            pass

    def _redraw(self) -> None:
        if self._ax is None or self._canvas is None:
            return
        t0 = time.perf_counter()
        ax = self._ax
        active_ws = self._active_workspace()

        g = self._get_active_overlay_group()
        overlay_on = bool(g is not None and (getattr(g, "members", None) or []))
        overlay_color_map: Dict[Tuple[str, str], str] = {}
        if overlay_on and g is not None:
            try:
                overlay_color_map = self._overlay_color_map_for_group(g)
            except Exception:
                overlay_color_map = {}

        active_key = self._effective_active_key()
        active_d = self._effective_active_dataset()
        try:
            if active_key is not None and str(active_key[0]) in (self.workspaces or {}):
                active_ws = self.workspaces[str(active_key[0])]
        except Exception:
            pass

        display_keys: set[Tuple[str, str]] = set()
        if overlay_on and g is not None:
            display_keys = set((str(a), str(b)) for (a, b) in (getattr(g, "members", []) or []))
            if active_key is not None:
                display_keys.add((str(active_key[0]), str(active_key[1])))
        else:
            if active_key is not None:
                display_keys.add((str(active_key[0]), str(active_key[1])))

        if not display_keys:
            try:
                self._status_var.set("(no FTIR dataset loaded)")
            except Exception:
                pass
            for ln in (self._line_artists or {}).values():
                try:
                    ln.set_visible(False)
                except Exception:
                    pass
            try:
                self._clear_peak_artists()
            except Exception:
                pass
            try:
                if self._legend_artist is not None:
                    self._legend_artist.remove()
                self._legend_artist = None
                self._legend_keys = tuple()
            except Exception:
                pass
            self._request_mpl_draw_idle("empty")
            return

        # Hide lines not in display_keys; keep artists for reuse.
        for key, ln in list((self._line_artists or {}).items()):
            if key in display_keys:
                continue
            try:
                ln.set_visible(False)
            except Exception:
                pass

        x_min_all: Optional[float] = None
        x_max_all: Optional[float] = None
        y_min_all: Optional[float] = None
        y_max_all: Optional[float] = None

        # Draw/update visible lines.
        visible_keys: List[Tuple[str, str]] = []
        overlay_order = sorted(display_keys)
        overlay_idx = {k: i for i, k in enumerate(overlay_order)}
        try:
            offset_mode = str(self._overlay_offset_mode_var.get() or "Normal")
        except Exception:
            offset_mode = "Normal"
        try:
            offset_val = float(self._overlay_offset_var.get() or 0.0)
        except Exception:
            offset_val = 0.0
        for key in sorted(display_keys):
            d = self._get_dataset_by_key(key)
            if d is None:
                continue

            x_plot, y_plot = self._get_plot_arrays(d)
            try:
                if int(np.asarray(x_plot).size) < 2 or int(np.asarray(y_plot).size) < 2:
                    continue
                mask = np.isfinite(x_plot) & np.isfinite(y_plot)
                if int(np.sum(mask)) < 2:
                    continue
                x_plot = np.asarray(x_plot, dtype=float)[mask]
                y_plot = np.asarray(y_plot, dtype=float)[mask]
            except Exception:
                continue

            if overlay_on and offset_mode != "Normal" and offset_val != 0.0:
                idx = overlay_idx.get(key, 0)
                try:
                    if offset_mode == "Offset Y":
                        y_plot = np.asarray(y_plot, dtype=float) + (float(idx) * float(offset_val))
                    elif offset_mode == "Offset X":
                        x_plot = np.asarray(x_plot, dtype=float) + (float(idx) * float(offset_val))
                except Exception:
                    pass

            ln = self._line_artists.get(key)
            if ln is None:
                try:
                    (ln,) = ax.plot([], [], lw=1.2)
                except Exception:
                    continue
                self._line_artists[key] = ln

            try:
                ln.set_data(x_plot, y_plot)
                ln.set_visible(True)
            except Exception:
                continue

            # Per-workspace color (stable within a workspace).
            try:
                if overlay_on and overlay_color_map and key in overlay_color_map:
                    ln.set_color(str(overlay_color_map.get(key)))
                else:
                    ln.set_color(self._workspace_line_color(str(key[0])))
            except Exception:
                pass

            # Legend label
            try:
                ws_obj = self.workspaces.get(str(key[0]))
                ws_name = (str(getattr(ws_obj, "name", "")) if ws_obj is not None else str(key[0]))
                ln.set_label(f"{ws_name}:{str(getattr(d, 'name', 'dataset'))}")
            except Exception:
                pass

            # Highlight active dataset
            is_active = (active_key is not None and key == active_key)
            try:
                if is_active:
                    ln.set_linewidth(2.6)
                    ln.set_alpha(1.0)
                    ln.set_zorder(4)
                else:
                    lw = 1.2
                    try:
                        if overlay_on and g is not None:
                            st = (getattr(g, "per_member_style", {}) or {}).get((str(key[0]), str(key[1])))
                            if st is not None and getattr(st, "linewidth", None) is not None:
                                lw = float(st.linewidth)
                    except Exception:
                        lw = 1.2
                    ln.set_linewidth(float(lw))
                    ln.set_alpha(0.75 if overlay_on else 1.0)
                    ln.set_zorder(2)
            except Exception:
                pass

            # Global limits
            try:
                x0 = float(np.nanmin(x_plot))
                x1 = float(np.nanmax(x_plot))
                y0 = float(np.nanmin(y_plot))
                y1 = float(np.nanmax(y_plot))
                if math.isfinite(x0) and math.isfinite(x1):
                    x_min_all = x0 if x_min_all is None else min(x_min_all, x0)
                    x_max_all = x1 if x_max_all is None else max(x_max_all, x1)
                if math.isfinite(y0) and math.isfinite(y1):
                    y_min_all = y0 if y_min_all is None else min(y_min_all, y0)
                    y_max_all = y1 if y_max_all is None else max(y_max_all, y1)
            except Exception:
                pass

            visible_keys.append(key)

        if not visible_keys:
            try:
                self._status_var.set("(no plottable FTIR data)")
            except Exception:
                pass
            self._request_mpl_draw_idle("no_data")
            return

        # Axis labels follow the active dataset (best-effort).
        if active_d is not None:
            ylabel = "Absorbance"
            try:
                if str(getattr(active_d, "y_mode", "absorbance") or "").strip().lower() == "transmittance":
                    ylabel = "Transmittance"
            except Exception:
                pass
            try:
                xu = str(getattr(active_d, "x_units", "") or "").strip()
                ax.set_xlabel(f"Wavenumber ({xu})" if xu else "Wavenumber")
            except Exception:
                ax.set_xlabel("Wavenumber")
            try:
                yu = str(getattr(active_d, "y_units", "") or "").strip()
                if yu and (ylabel.lower() not in yu.lower()):
                    ax.set_ylabel(f"{ylabel} ({yu})")
                else:
                    ax.set_ylabel(ylabel)
            except Exception:
                ax.set_ylabel(ylabel)

        # X limits (reverse-x applies to all displayed lines)
        try:
            if x_min_all is not None and x_max_all is not None:
                rev = bool(self._reverse_x_var.get())
                if rev:
                    ax.set_xlim(float(x_max_all), float(x_min_all))
                else:
                    ax.set_xlim(float(x_min_all), float(x_max_all))
        except Exception:
            pass

        # Y limits
        try:
            if y_min_all is not None and y_max_all is not None:
                y0 = float(y_min_all)
                y1 = float(y_max_all)
                if y1 == y0:
                    pad = 1.0 if y1 == 0.0 else abs(y1) * 0.05
                    ax.set_ylim(y0 - pad, y1 + pad)
                else:
                    pad = (y1 - y0) * 0.03
                    ax.set_ylim(y0 - pad, y1 + pad)
        except Exception:
            pass

        # Legend (overlay mode only)
        try:
            want_legend = bool(overlay_on and len(visible_keys) > 1)
            new_keys = tuple(sorted(visible_keys))
            if not want_legend:
                if self._legend_artist is not None:
                    try:
                        self._legend_artist.remove()
                    except Exception:
                        pass
                self._legend_artist = None
                self._legend_keys = tuple()
            else:
                if new_keys != tuple(self._legend_keys or tuple()):
                    if self._legend_artist is not None:
                        try:
                            self._legend_artist.remove()
                        except Exception:
                            pass
                    handles: List[Any] = []
                    labels: List[str] = []
                    for k in new_keys:
                        ln = self._line_artists.get(k)
                        if ln is None:
                            continue
                        try:
                            handles.append(ln)
                            labels.append(str(getattr(ln, "get_label")()))
                        except Exception:
                            continue
                    self._legend_artist = ax.legend(handles=handles, labels=labels, loc="best", fontsize=8)
                    self._legend_keys = new_keys
        except Exception:
            pass

        # Status line (active dataset + overlay count)
        try:
            if active_d is None:
                self._status_var.set("(no active FTIR dataset)")
            else:
                x_plot, y_plot = self._get_plot_arrays(active_d)
                mask = np.isfinite(x_plot) & np.isfinite(y_plot)
                x_plot = np.asarray(x_plot, dtype=float)[mask]
                y_plot = np.asarray(y_plot, dtype=float)[mask]
                x0 = float(np.nanmin(x_plot))
                x1 = float(np.nanmax(x_plot))
                y0 = float(np.nanmin(y_plot))
                y1 = float(np.nanmax(y_plot))
                n = int(np.asarray(getattr(active_d, "x_full", x_plot)).size)
                xu = str(getattr(active_d, "x_units", "") or "").strip()
                yu = str(getattr(active_d, "y_units", "") or "").strip()
                ym = str(getattr(active_d, "y_mode", "") or "").strip().lower() or "absorbance"
                wn_label = "wn" if ("cm" in xu.lower() or "1/" in xu.lower()) else "x"
                x_part = f"{wn_label}={x0:g}..{x1:g}" + (f" {xu}" if xu else "")
                y_part = f"y={y0:g}..{y1:g}" + (f" {yu}" if yu else "")
                ov_part = ""
                if overlay_on and g is not None:
                    ov_part = f" | overlay='{getattr(g, 'name', '')}' ({len(getattr(g, 'members', []) or [])})"
                self._status_var.set(f"{active_ws.name}:{active_d.name}  |  n={n}  |  {x_part}  |  {y_part}  |  {ym}{ov_part}")
        except Exception:
            pass

        # Peaks: active dataset only by default; optional marker-only peaks for all overlayed.
        try:
            if active_d is not None:
                show_all = bool(self._show_peaks_all_overlay_var.get()) and bool(overlay_on)

                if show_all:
                    # Stable set of displayed peaks per dataset while in overlay-all mode.
                    ids_active = []
                    try:
                        if active_key is not None:
                            ids_active = self._get_overlay_display_peak_ids((str(active_key[0]), str(active_key[1])), active_d, default_max=0)
                    except Exception:
                        ids_active = []
                    self._render_peaks_on_axes(
                        ax,
                        active_d,
                        dataset_key=active_key,
                        peak_color=self._peak_color_for_key(active_key),
                        pickable=True,
                        fontweight="bold",
                        include_peak_ids=ids_active,
                        include_summary=True,
                        overlay_order=visible_keys if overlay_on else None,
                    )
                    self._render_overlay_peak_markers(ax, visible_keys, active_key)
                else:
                    self._render_peaks_on_axes(
                        ax,
                        active_d,
                        dataset_key=active_key,
                        peak_color=self._peak_color_for_key(active_key),
                        pickable=True,
                        fontweight="bold",
                        overlay_order=visible_keys if overlay_on else None,
                    )
            else:
                self._clear_peak_artists()
        except Exception:
            pass

        # Bond labels: always rebuilt from persisted annotations.
        try:
            self._render_bond_annotations(ax, visible_keys=visible_keys, active_key=active_key)
        except Exception:
            pass

        if FTIR_DEBUG:
            try:
                print(
                    "[FTIR] redraw peaks",
                    "overlay=", bool(overlay_on),
                    "active=", active_key,
                    "sel=", (len(getattr(g, "members", []) or []) if g is not None else 0),
                    "cache=", len(getattr(self, "_overlay_peak_display_ids_by_key", {}) or {}),
                    "artists=", len(getattr(self, "_peak_texts", []) or []),
                )
                for k in sorted(display_keys):
                    try:
                        kk = (str(k[0]), str(k[1]))
                        d0 = self._get_dataset_by_key(kk)
                        n_state = len(list(getattr(d0, "peaks", []) or [])) if d0 is not None else 0
                        n_cache = len(list((self._overlay_peak_display_ids_by_key or {}).get(kk) or []))
                        n_art = len(list((self._peak_texts_by_key or {}).get(kk) or []))
                        print("  ", kk, "peaks=", n_state, "cache_ids=", n_cache, "ann=", n_art)
                    except Exception:
                        continue
            except Exception:
                pass

        self._request_mpl_draw_idle("plot")
        self._ftir_timing("plot", time.perf_counter() - t0)

        # No list refresh here: avoid redraw<->refresh recursion.

    def _active_dataset(self) -> Optional[FTIRDataset]:
        try:
            g = self._get_active_overlay_group()
            if g is not None and (getattr(g, "members", None) or []):
                return self._effective_active_dataset()
        except Exception:
            pass
        ws = self._active_workspace()
        aid = getattr(ws, "active_dataset_id", None)
        if not aid:
            return None
        for d in (getattr(ws, "datasets", []) or []):
            if str(getattr(d, "id", "")) == str(aid):
                return d
        return None

    def _get_plot_arrays(self, d: FTIRDataset) -> Tuple[np.ndarray, np.ndarray]:
        # Prefer precomputed display arrays.
        try:
            x2 = np.asarray(getattr(d, "x_disp", []), dtype=float)
            y2 = np.asarray(getattr(d, "y_disp", []), dtype=float)
            if int(x2.size) >= 2 and int(y2.size) >= 2:
                return x2, y2
        except Exception:
            pass

        # Lazy decimation (UI thread): cheap slicing only.
        try:
            x = np.asarray(getattr(d, "x_full", []), dtype=float)
            y = np.asarray(getattr(d, "y_full", []), dtype=float)
        except Exception:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        try:
            n = int(x.size)
            max_n = 50_000
            if n > max_n:
                step = int(math.ceil(float(n) / float(max_n)))
                x2 = x[::step]
                y2 = y[::step]
            else:
                x2 = x
                y2 = y
        except Exception:
            x2 = x
            y2 = y

        try:
            d.x_disp = np.asarray(x2, dtype=float)
            d.y_disp = np.asarray(y2, dtype=float)
        except Exception:
            pass
        return np.asarray(x2, dtype=float), np.asarray(y2, dtype=float)

    def _overlay_offset_for_key(
        self,
        key: Optional[Tuple[str, str]],
        *,
        order: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Tuple[float, float]:
        try:
            mode = str(self._overlay_offset_mode_var.get() or "Normal")
        except Exception:
            mode = "Normal"
        try:
            offset_val = float(self._overlay_offset_var.get() or 0.0)
        except Exception:
            offset_val = 0.0
        if mode == "Normal" or offset_val == 0.0 or key is None:
            return 0.0, 0.0

        if order is None:
            g = self._get_active_overlay_group()
            if g is not None:
                order = [(str(a), str(b)) for (a, b) in (getattr(g, "members", []) or [])]
        order = list(order or [])
        try:
            idx = order.index((str(key[0]), str(key[1])))
        except Exception:
            idx = 0

        if mode == "Offset Y":
            return 0.0, float(idx) * float(offset_val)
        if mode == "Offset X":
            return float(idx) * float(offset_val), 0.0
        return 0.0, 0.0

    def _selected_dataset_id(self) -> Optional[str]:
        tree = self._tree
        if tree is None:
            return None
        try:
            sel = tree.selection()
            return str(sel[0]) if sel else None
        except Exception:
            return None

    def _set_active_by_id(self, dataset_id: str) -> None:
        # Legacy entrypoint; treat as same-workspace selection.
        self._set_active_dataset(str(self.active_workspace_id), str(dataset_id), reason="same_workspace_select")

    def _set_active_dataset(
        self,
        ws_id: str,
        dataset_id: Optional[str],
        *,
        reason: Literal["same_workspace_select", "workspace_switch", "overlay_action", "load_new"],
    ) -> None:
        """Set active dataset.

        Overlay groups persist until explicitly deleted; switching datasets/workspaces does not clear overlays.
        """
        ws_id = str(ws_id or "")
        ds_id = (None if not dataset_id else str(dataset_id))
        if not ws_id or ws_id not in (self.workspaces or {}):
            return

        prev_ws_id = str(self.active_workspace_id or "")
        prev_ds_id: Optional[str] = None
        try:
            prev_ws = self.workspaces.get(prev_ws_id) if prev_ws_id else None
            prev_ds_id = (None if prev_ws is None else (None if not getattr(prev_ws, "active_dataset_id", None) else str(prev_ws.active_dataset_id)))
        except Exception:
            prev_ds_id = None

        # Choose default dataset id if not provided.
        try:
            ws_obj = self.workspaces.get(ws_id)
            if ws_obj is not None:
                if not ds_id:
                    ds_id = (None if not getattr(ws_obj, "active_dataset_id", None) else str(ws_obj.active_dataset_id))
                if (not ds_id) and (getattr(ws_obj, "datasets", None) or []):
                    ds_id = str(ws_obj.datasets[0].id)
        except Exception:
            pass

        # Apply active workspace + dataset
        self.active_workspace_id = str(ws_id)
        try:
            ws_obj = self.workspaces.get(str(ws_id))
            if ws_obj is not None:
                ws_obj.active_dataset_id = (None if not ds_id else str(ds_id))
        except Exception:
            pass

        self._refresh_workspace_selector()

        try:
            if ds_id and str(ds_id) in self._reverse_pref_by_id:
                self._reverse_x_var.set(bool(self._reverse_pref_by_id[str(ds_id)]))
        except Exception:
            pass

        # Mirror into app workspace for legacy save/load.
        self._sync_active_workspace_to_app_workspace()
        self.refresh_from_workspace(select_active=True)
        self._rebuild_overlay_group_list(select_group_id=self._get_selected_overlay_group_id())
        self._rebuild_overlay_group_members_list()
        self._rebuild_overlay_selection_list()
        self._schedule_redraw()

    def _on_tree_select_set_active(self) -> None:
        if bool(getattr(self, "_ignore_tree_select", False)):
            return
        sid = self._selected_dataset_id()
        if sid:
            # If selection change doesn't change active, do nothing (breaks event loops).
            try:
                ws = self._active_workspace()
                if str(getattr(ws, "active_dataset_id", "") or "") == str(sid):
                    return
            except Exception:
                pass
            self._set_active_dataset(str(self.active_workspace_id), str(sid), reason="same_workspace_select")

    def _on_toggle_reverse(self) -> None:
        aid = None
        try:
            aid = self._active_workspace().active_dataset_id
        except Exception:
            aid = None
        if aid:
            try:
                self._reverse_pref_by_id[str(aid)] = bool(self._reverse_x_var.get())
            except Exception:
                pass
        self._schedule_redraw()

    def remove_selected(self) -> None:
        ws = self._active_workspace()
        sid = self._selected_dataset_id() or getattr(ws, "active_dataset_id", None)
        if not sid:
            return
        sid = str(sid)
        # Remove
        before = list(getattr(ws, "datasets", []) or [])
        ws.datasets = [d for d in (getattr(ws, "datasets", []) or []) if str(getattr(d, "id", "")) != str(sid)]
        if len(before) == len(getattr(ws, "datasets", []) or []):
            return

        # Remove from overlay groups if present.
        try:
            self._remove_dataset_key_from_overlay_groups((str(self.active_workspace_id), str(sid)))
        except Exception:
            pass
        try:
            self._reverse_pref_by_id.pop(str(sid), None)
        except Exception:
            pass

        if str(getattr(ws, "active_dataset_id", "") or "") == str(sid):
            ws.active_dataset_id = (str(ws.datasets[0].id) if ws.datasets else None)

            try:
                # Ensure no overlay group still points at the removed id.
                self._remove_dataset_key_from_overlay_groups((str(self.active_workspace_id), str(sid)))
            except Exception:
                pass

        self._sync_active_workspace_to_app_workspace()
        self.refresh_from_workspace(select_active=True)
        self._rebuild_overlay_group_list(select_group_id=self._get_selected_overlay_group_id())
        self._rebuild_overlay_group_members_list()
        self._rebuild_overlay_selection_list()
        self._schedule_redraw()

    def clear_all(self) -> None:
        ws = self._active_workspace()
        ids = [str(getattr(d, "id", "")) for d in (getattr(ws, "datasets", []) or [])]
        try:
            ws.datasets.clear()
        except Exception:
            ws.datasets = []
        ws.active_dataset_id = None
        for sid in ids:
            try:
                self._reverse_pref_by_id.pop(str(sid), None)
            except Exception:
                pass

        try:
            self._remove_workspace_id_from_overlay_groups(str(self.active_workspace_id))
        except Exception:
            pass

        self._sync_active_workspace_to_app_workspace()
        self.refresh_from_workspace(select_active=False)
        self._rebuild_overlay_group_list(select_group_id=self._get_selected_overlay_group_id())
        self._rebuild_overlay_group_members_list()
        self._rebuild_overlay_selection_list()
        self._schedule_redraw()

    def _ensure_tree_menu(self) -> None:
        if self._tree_menu is not None:
            return
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Set Active", command=lambda: (self._set_active_by_id(self._selected_dataset_id() or "")))
        m.add_command(label="Rename…", command=self._rename_selected)
        m.add_separator()
        m.add_command(label="Remove", command=self.remove_selected)
        self._tree_menu = m

    def _on_tree_right_click(self, evt) -> None:
        tree = self._tree
        if tree is None:
            return
        try:
            iid = tree.identify_row(evt.y)
            if iid:
                tree.selection_set(iid)
        except Exception:
            pass
        self._ensure_tree_menu()
        try:
            if self._tree_menu is not None:
                self._tree_menu.tk_popup(int(evt.x_root), int(evt.y_root))
        finally:
            try:
                if self._tree_menu is not None:
                    self._tree_menu.grab_release()
            except Exception:
                pass

    def _rename_selected(self) -> None:
        ws = self._active_workspace()
        sid = self._selected_dataset_id() or getattr(ws, "active_dataset_id", None)
        if not sid:
            return
        d: Optional[FTIRDataset] = None
        for it in (getattr(ws, "datasets", []) or []):
            if str(getattr(it, "id", "")) == str(sid):
                d = it
                break
        if d is None:
            return
        new_name = simpledialog.askstring("Rename FTIR", "Display name:", initialvalue=str(d.name), parent=self.app)
        if new_name is None:
            return
        new_name = str(new_name).strip()
        if not new_name:
            try:
                new_name = str((d.path.name if getattr(d, "path", None) is not None else d.name))
            except Exception:
                new_name = str(d.name)
        d.name = new_name
        self._sync_active_workspace_to_app_workspace()
        self.refresh_from_workspace(select_active=True)

    def save_plot_dialog(self) -> None:
        try:
            snapshot, stem = self._snapshot_export_state()
        except Exception as exc:
            messagebox.showerror("FTIR Export", f"Failed to prepare export editor:\n\n{exc}", parent=self.app)
            return

        try:
            FTIRExportEditor(self.app, snapshot=snapshot, default_stem=stem)
        except Exception as exc:
            messagebox.showerror("FTIR Export", f"Failed to open export editor:\n\n{exc}", parent=self.app)

    def _snapshot_export_state(self) -> Tuple[Dict[str, Any], str]:
        """Create an export-only snapshot of what is currently displayed.

        This must NOT mutate the live FTIR view or datasets.
        """
        ws = self._active_workspace()
        if ws is None:
            raise RuntimeError("No active FTIR workspace")

        overlay_group = self._get_active_overlay_group()
        overlay_on = overlay_group is not None

        displayed_keys: List[Tuple[str, str]] = []
        active_key: Optional[Tuple[str, str]] = None

        if overlay_group is not None and (overlay_group.members or []):
            displayed_keys = [tuple(k) for k in list(overlay_group.members) if isinstance(k, (list, tuple)) and len(k) == 2]
            try:
                if overlay_group.active_member is not None:
                    active_key = tuple(overlay_group.active_member)
            except Exception:
                active_key = None
        else:
            ds = self._active_dataset()
            if ds is None:
                raise RuntimeError("No active FTIR dataset")
            active_key = (str(ws.id), str(ds.id))
            displayed_keys = [active_key]

        if not displayed_keys:
            raise RuntimeError("Nothing to export")
        if active_key is None:
            active_key = displayed_keys[0]

        show_peaks_all_overlay = False
        try:
            show_peaks_all_overlay = bool(self._show_peaks_all_overlay_var.get())
        except Exception:
            show_peaks_all_overlay = False

        displayed: List[Dict[str, Any]] = []
        for (ws_id, ds_id) in displayed_keys:
            w = (self.workspaces or {}).get(str(ws_id))
            if w is None:
                continue
            d = None
            for it in (getattr(w, "datasets", []) or []):
                if str(getattr(it, "id", "")) == str(ds_id):
                    d = it
                    break
            if d is None:
                continue

            try:
                x_disp = np.asarray(getattr(d, "x_disp", []), dtype=float)
                y_disp = np.asarray(getattr(d, "y_disp", []), dtype=float)
            except Exception:
                x_disp = np.asarray([], dtype=float)
                y_disp = np.asarray([], dtype=float)

            name = f"{w.name} • {d.name}" if getattr(w, "name", None) else str(d.name)

            line_color = None
            line_width = None
            try:
                ln = self._line_artists.get((str(ws_id), str(ds_id)))
                if ln is not None:
                    line_color = ln.get_color()
                    line_width = ln.get_linewidth()
            except Exception:
                pass

            peak_color = None
            try:
                peak_color = self._peak_color_for_key((str(ws_id), str(ds_id)))
            except Exception:
                peak_color = None

            # Try to reflect what's currently shown in the main FTIR view.
            is_active = bool(active_key is not None and (str(ws_id), str(ds_id)) == (str(active_key[0]), str(active_key[1])))
            show_peaks = bool(is_active or (overlay_on and show_peaks_all_overlay))
            # In the export editor we default labels the same as peaks when "show all" is enabled.
            show_labels = bool(is_active or (overlay_on and show_peaks_all_overlay))

            peak_settings = dict(getattr(d, "peak_settings", {}) or {})
            peaks_list = [dict(p) for p in (getattr(d, "peaks", []) or []) if isinstance(p, dict)]
            suppressed = set(getattr(d, "peak_suppressed", set()) or set())
            peak_label_positions_raw = dict(getattr(d, "peak_label_positions", {}) or {})
            peak_label_overrides = dict(getattr(d, "peak_label_overrides", {}) or {})

            peak_by_id: Dict[str, Dict[str, Any]] = {str(p.get("id") or ""): p for p in peaks_list if isinstance(p, dict)}
            try:
                dx, dy = self._overlay_offset_for_key((str(ws_id), str(ds_id)))
            except Exception:
                dx, dy = 0.0, 0.0

            peak_label_positions: Dict[str, Tuple[float, float]] = {}
            for pid, xy in (peak_label_positions_raw or {}).items():
                if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                    continue
                try:
                    x0 = float(xy[0])
                    y0 = float(xy[1])
                except Exception:
                    continue
                p = peak_by_id.get(str(pid))
                if p is None:
                    peak_label_positions[str(pid)] = (float(x0), float(y0))
                    continue
                try:
                    wn = float(p.get("wn"))
                    y_peak = float(p.get("y_display", p.get("y", 0.0)))
                except Exception:
                    peak_label_positions[str(pid)] = (float(x0), float(y0))
                    continue
                # Choose the position that is closer to the true peak tip (base coords).
                d0 = (float(x0) - wn) ** 2 + (float(y0) - y_peak) ** 2
                d1 = (float(x0) - float(dx) - wn) ** 2 + (float(y0) - float(dy) - y_peak) ** 2
                if d1 < d0:
                    peak_label_positions[str(pid)] = (float(x0) - float(dx), float(y0) - float(dy))
                else:
                    peak_label_positions[str(pid)] = (float(x0), float(y0))

            displayed.append(
                {
                    "key": (str(ws_id), str(ds_id)),
                    "name": str(name),
                    "x_disp": x_disp,
                    "y_disp": y_disp,
                    "line_color": line_color,
                    "line_width": line_width,
                    "peak_color": peak_color,
                    "peak_label_color": peak_color,
                    "visible": True,
                    "show_peaks": bool(show_peaks),
                    "show_labels": bool(show_labels),
                    "peak_settings": peak_settings,
                    "peaks": peaks_list,
                    "peak_suppressed": list(suppressed),
                    "peak_label_positions": peak_label_positions,
                    "peak_label_overrides": peak_label_overrides,
                }
            )

        if not displayed:
            raise RuntimeError("Nothing to export")

        reverse_x = False
        try:
            reverse_x = bool(self._reverse_x_var.get())
        except Exception:
            reverse_x = False

        snap: Dict[str, Any] = {
            "overlay_on": bool(overlay_on),
            "active_key": tuple(active_key),
            "displayed": displayed,
            "show_peaks_all_overlay": bool(show_peaks_all_overlay),
            "overlay_offset_mode": str(self._overlay_offset_mode_var.get() or "Normal"),
            "overlay_offset": float(self._overlay_offset_var.get() or 0.0),
            "reverse_x": bool(reverse_x),
            "title": "FTIR" if not overlay_on else "FTIR Overlay",
            "xlabel": "Wavenumber",
            "ylabel": "Absorbance",
            "grid_on": False,
            "axes_bg": "#ffffff",
            "peak_marker_size": 4,
            "legend_on": True,
            "legend_fontsize": 8,
            "bond_annotations": [],
        }

        # Bond annotations from the active FTIR workspace (export editor is a snapshot; edits are editor-local)
        try:
            bond_rows: List[Dict[str, Any]] = []
            for a in (getattr(ws, "bond_annotations", None) or []):
                try:
                    xy = getattr(a, "xytext", None)
                    if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                        xy = (float(getattr(a, "x_cm1")), float(getattr(a, "y_value")))
                    bond_rows.append(
                        {
                            "dataset_id": str(getattr(a, "dataset_id", "") or ""),
                            "text": str(getattr(a, "text", "") or ""),
                            "x_cm1": float(getattr(a, "x_cm1")),
                            "y_value": float(getattr(a, "y_value")),
                            "xytext": [float(xy[0]), float(xy[1])],
                            "show_vline": bool(getattr(a, "show_vline", False)),
                            "line_color": str(getattr(a, "line_color", "#444444") or "#444444"),
                            "text_color": str(getattr(a, "text_color", "#111111") or "#111111"),
                            "fontsize": int(getattr(a, "fontsize", 9) or 9),
                            "rotation": int(getattr(a, "rotation", 90) or 90),
                            "preset_id": (None if not getattr(a, "preset_id", None) else str(getattr(a, "preset_id"))),
                        }
                    )
                except Exception:
                    continue
            snap["bond_annotations"] = bond_rows
        except Exception:
            pass

        if overlay_on:
            stem = "ftir_overlay"
        else:
            try:
                stem = str(displayed[0].get("name") or "ftir").replace(" • ", "_")
                stem = "".join([c if c.isalnum() or c in ("_", "-", ".") else "_" for c in stem])
                stem = stem.strip("_ ") or "ftir"
            except Exception:
                stem = "ftir"

        return snap, stem

    def cycle_active(self, delta: int) -> None:
        ws = self._active_workspace()
        ds = list(getattr(ws, "datasets", []) or [])
        if not ds:
            return
        ids = [str(d.id) for d in ds]
        cur = getattr(ws, "active_dataset_id", None)
        if cur not in ids:
            self._set_active_by_id(ids[0])
            return
        i = ids.index(str(cur))
        j = (i + int(delta)) % len(ids)
        self._set_active_by_id(ids[j])

    # --- Peak picking UI ---

    def _open_peaks_dialog(self) -> None:
        if self._peaks_dialog is not None and bool(self._peaks_dialog.winfo_exists()):
            try:
                self._sync_peaks_vars_from_active_dataset()
                self._peaks_dialog.lift()
                self._peaks_dialog.focus_force()
            except Exception:
                pass
            return

        w = tk.Toplevel(self.app)
        w.title("FTIR Peaks")
        try:
            w.transient(self.app)
        except Exception:
            pass
        self._peaks_dialog = w

        def _on_close() -> None:
            try:
                self._peaks_dialog = None
            except Exception:
                pass
            try:
                w.destroy()
            except Exception:
                pass

        try:
            w.protocol("WM_DELETE_WINDOW", _on_close)
        except Exception:
            pass

        body = ttk.Frame(w, padding=12)
        body.grid(row=0, column=0, sticky="nsew")
        w.columnconfigure(0, weight=1)
        w.rowconfigure(0, weight=1)

        ttk.Label(body, text="Applies to the active FTIR dataset.").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(body, text="Use 'Apply to All' to compute peaks for every dataset in every FTIR workspace.").grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="e",
        )

        enabled = ttk.Checkbutton(body, text="Enable peak labels/markers", variable=self._peaks_enabled_var)
        enabled.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 10))
        try:
            ToolTip.attach(enabled, TOOLTIP_TEXT.get("ftir_peaks_enable", ""))
        except Exception:
            pass

        ttk.Label(body, text="Min prominence").grid(row=2, column=0, sticky="w")
        ent_prom = ttk.Entry(body, textvariable=self._peaks_min_prom_var, width=12)
        ent_prom.grid(row=2, column=1, sticky="w")
        try:
            ToolTip.attach(ent_prom, TOOLTIP_TEXT.get("ftir_peaks_min_prom", ""))
        except Exception:
            pass

        ttk.Label(body, text="Min distance (cm⁻¹)").grid(row=2, column=2, sticky="w", padx=(16, 0))
        ent_dist = ttk.Entry(body, textvariable=self._peaks_min_dist_var, width=12)
        ent_dist.grid(row=2, column=3, sticky="w")
        try:
            ToolTip.attach(ent_dist, TOOLTIP_TEXT.get("ftir_peaks_min_dist", ""))
        except Exception:
            pass

        ttk.Label(body, text="Label format").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ent_fmt = ttk.Entry(body, textvariable=self._peaks_label_fmt_var, width=26)
        ent_fmt.grid(row=3, column=1, columnspan=3, sticky="w", pady=(6, 0))
        try:
            ToolTip.attach(ent_fmt, TOOLTIP_TEXT.get("ftir_peaks_label_fmt", ""))
        except Exception:
            pass

        btns = ttk.Frame(body)
        btns.grid(row=4, column=0, columnspan=4, sticky="e", pady=(14, 0))
        b_unhide = ttk.Button(btns, text="Unhide All", command=self._unhide_all_peaks, style="Secondary.TButton")
        b_unhide.pack(side=tk.LEFT)
        b_apply = ttk.Button(btns, text="Apply", command=self._apply_peaks_dialog, style="Primary.TButton")
        b_apply.pack(side=tk.LEFT)
        b_apply_all = ttk.Button(btns, text="Apply to All", command=self._apply_peaks_dialog_all, style="Primary.TButton")
        b_apply_all.pack(side=tk.LEFT, padx=(8, 0))
        b_close = ttk.Button(btns, text="Close", command=_on_close, style="Secondary.TButton")
        b_close.pack(side=tk.LEFT, padx=(8, 0))

        try:
            ToolTip.attach(b_unhide, TOOLTIP_TEXT.get("ftir_peaks_unhide", ""))
            ToolTip.attach(b_apply, TOOLTIP_TEXT.get("ftir_peaks_apply", ""))
            ToolTip.attach(b_apply_all, TOOLTIP_TEXT.get("ftir_peaks_apply_all", ""))
            ToolTip.attach(b_close, TOOLTIP_TEXT.get("ftir_peaks_close", ""))
        except Exception:
            pass

        self._sync_peaks_vars_from_active_dataset()

    def _unhide_all_peaks(self) -> None:
        d = self._active_dataset()
        if d is None:
            return
        try:
            d.peak_suppressed.clear()
        except Exception:
            pass
        self._schedule_redraw()

    def _ensure_export_peaks_menu(self) -> None:
        if self._export_peaks_menu is not None:
            return
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Active dataset → CSV…", command=self._export_active_peaks_csv)
        m.add_command(label="All datasets → Excel workbook…", command=self._export_all_peaks_excel)
        self._export_peaks_menu = m

    def _export_peaks_dialog(self) -> None:
        # Minimal export dialog with a single option toggle.
        try:
            if self._export_peaks_dialog_win is not None and bool(self._export_peaks_dialog_win.winfo_exists()):
                try:
                    self._export_peaks_dialog_win.lift()
                    self._export_peaks_dialog_win.focus_force()
                except Exception:
                    pass
                return
        except Exception:
            pass

        win = tk.Toplevel(self.app)
        self._export_peaks_dialog_win = win
        win.title("Export Peaks")
        try:
            win.transient(self.app)
        except Exception:
            pass

        root = ttk.Frame(win, padding=(12, 10, 12, 12))
        root.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        ttk.Label(root, text="FTIR Peaks Export").grid(row=0, column=0, sticky="w")

        chk = ttk.Checkbutton(
            root,
            text="Include functional-group candidates (library v2)",
            variable=self._export_include_candidates_var,
        )
        chk.grid(row=1, column=0, sticky="w", pady=(8, 8))

        btns = ttk.Frame(root)
        btns.grid(row=2, column=0, sticky="ew")
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        def _close() -> None:
            try:
                if self._export_peaks_dialog_win is not None:
                    self._export_peaks_dialog_win.destroy()
            except Exception:
                pass
            self._export_peaks_dialog_win = None

        def _do_active_csv() -> None:
            include = bool(self._export_include_candidates_var.get())
            _close()
            self._export_active_peaks_csv(include_candidates=include)

        def _do_all_excel() -> None:
            include = bool(self._export_include_candidates_var.get())
            _close()
            self._export_all_peaks_excel(include_candidates=include)

        ttk.Button(btns, text="Active dataset → CSV…", command=_do_active_csv, style="Secondary.TButton").grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(btns, text="All datasets → Excel workbook…", command=_do_all_excel, style="Secondary.TButton").grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )

        ttk.Button(root, text="Close", command=_close).grid(row=3, column=0, sticky="e", pady=(10, 0))

        try:
            win.protocol("WM_DELETE_WINDOW", _close)
        except Exception:
            pass

        try:
            win.grab_set()
        except Exception:
            pass

        try:
            win.focus_force()
        except Exception:
            pass

    # --- Bond labels ---

    def _on_ftir_keypress(self, evt) -> None:
        key = str(getattr(evt, "key", "") or "")
        if not key:
            return
        if key.lower() == "escape":
            if bool(getattr(self, "_bond_placement_active", False)):
                self._bond_cancel_placement(reason="esc")
            return
        if key.lower() == "b":
            try:
                self._open_add_bond_label_dialog()
            except Exception:
                pass

    def _bond_common_presets(self) -> List[Dict[str, Any]]:
        # Small built-in fallback list.
        return [
            {"id": "common:ester_co", "label": "Ester C=O", "range_cm1": (1760, 1735)},
            {"id": "common:carboxylic_acid_co", "label": "Carboxylic acid C=O", "range_cm1": (1725, 1700)},
            {"id": "common:amide_i", "label": "Amide I", "range_cm1": (1690, 1630)},
            {"id": "common:amide_ii", "label": "Amide II", "range_cm1": (1560, 1510)},
            {"id": "common:oh_broad", "label": "O-H (broad)", "range_cm1": (3600, 2500)},
            {"id": "common:nh_stretch", "label": "N-H stretch", "range_cm1": (3500, 3300)},
            {"id": "common:co_stretch", "label": "C-O stretch", "range_cm1": (1300, 1000)},
            {"id": "common:cn", "label": "C=N", "range_cm1": (1690, 1620)},
            {"id": "common:ch", "label": "C-H", "range_cm1": (3100, 2850)},
        ]

    def _bond_presets(self) -> List[Dict[str, Any]]:
        presets: List[Dict[str, Any]] = []
        # Add library v2 entries if present
        try:
            from lab_gui.ftir_library import FTIR_LIBRARY_V2

            for e in (FTIR_LIBRARY_V2 or []):
                if not isinstance(e, dict):
                    continue
                pid = str(e.get("id") or "").strip()
                label = str(e.get("label") or "").strip()
                r = e.get("range_cm1")
                if not pid or not label or not isinstance(r, (list, tuple)) or len(r) != 2:
                    continue
                try:
                    r0, r1 = float(r[0]), float(r[1])
                except Exception:
                    continue
                presets.append({"id": pid, "label": label, "range_cm1": (r0, r1)})
        except Exception:
            presets = []

        # Always include common presets
        presets = self._bond_common_presets() + presets

        # De-dup by id
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for p in presets:
            try:
                pid = str(p.get("id") or "")
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                out.append(p)
            except Exception:
                continue
        return out

    def _open_add_bond_label_dialog(self) -> None:
        try:
            if self._bond_dialog is not None and bool(self._bond_dialog.winfo_exists()):
                self._bond_dialog.lift()
                self._bond_dialog.focus_force()
                return
        except Exception:
            pass

        if self._ax is None or self._canvas is None:
            return

        win = tk.Toplevel(self.app)
        self._bond_dialog = win
        win.title("Add Bond Label")
        try:
            win.transient(self.app)
        except Exception:
            pass

        presets = self._bond_presets()

        preset_var = tk.StringVar(value="")
        custom_text_var = tk.StringVar(value="")
        mode_var = tk.StringVar(value="click")
        range_filter_var = tk.BooleanVar(value=False)
        vline_var = tk.BooleanVar(value=False)
        text_color_var = tk.StringVar(value="#111111")
        line_color_var = tk.StringVar(value="#444444")
        fontsize_var = tk.IntVar(value=9)
        rotation_var = tk.IntVar(value=0)
        attach_var = tk.StringVar(value="Active only")
        dataset_choice_var = tk.StringVar(value="")

        # Visible dataset choices (for "Choose dataset")
        dataset_choices: List[Tuple[str, str]] = []  # (display, dataset_id)
        try:
            g = self._get_active_overlay_group()
            overlay_on = bool(g is not None and (getattr(g, "members", None) or []))
            keys: set[Tuple[str, str]] = set()
            ak = self._effective_active_key()
            if overlay_on and g is not None:
                keys |= set((str(a), str(b)) for (a, b) in (getattr(g, "members", []) or []))
            if ak is not None:
                keys.add((str(ak[0]), str(ak[1])))
            for k in sorted(keys):
                d = self._get_dataset_by_key(k)
                if d is None:
                    continue
                did = str(getattr(d, "id", "") or "")
                if not did:
                    continue
                try:
                    ws_obj = (self.workspaces or {}).get(str(k[0]))
                    ws_name = str(getattr(ws_obj, "name", k[0]))
                except Exception:
                    ws_name = str(k[0])
                dataset_choices.append((f"{ws_name}:{str(getattr(d, 'name', 'dataset'))}", did))
        except Exception:
            dataset_choices = []

        if dataset_choices:
            dataset_choice_var.set(dataset_choices[0][0])

        def _xview() -> Tuple[float, float]:
            try:
                lo, hi = self._ax.get_xlim()
                return (float(min(lo, hi)), float(max(lo, hi)))
            except Exception:
                return (-float("inf"), float("inf"))

        def _fmt_preset(p: Dict[str, Any]) -> str:
            r = p.get("range_cm1")
            try:
                return f"{str(p.get('label'))} ({float(r[0]):.0f}-{float(r[1]):.0f} cm⁻¹)"
            except Exception:
                return str(p.get("label") or "")

        preset_display_to_id: Dict[str, str] = {}
        preset_display_to_range: Dict[str, Tuple[float, float]] = {}

        def _refresh_presets() -> None:
            xmin, xmax = _xview()
            values: List[str] = []
            preset_display_to_id.clear()
            preset_display_to_range.clear()
            for p in presets:
                try:
                    r0, r1 = p.get("range_cm1")
                    r0 = float(r0)
                    r1 = float(r1)
                    pmin, pmax = (min(r0, r1), max(r0, r1))
                except Exception:
                    pmin, pmax = (-float("inf"), float("inf"))

                if bool(range_filter_var.get()):
                    # overlap with current view
                    if pmax < xmin or pmin > xmax:
                        continue

                disp = _fmt_preset(p)
                values.append(disp)
                preset_display_to_id[disp] = str(p.get("id") or "")
                preset_display_to_range[disp] = (float(pmin), float(pmax))

            try:
                combo["values"] = values
            except Exception:
                pass
            if values and (preset_var.get() not in values):
                preset_var.set(values[0])

        root = ttk.Frame(win, padding=(12, 10, 12, 12))
        root.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        ttk.Label(root, text="Bond preset").grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(root, textvariable=preset_var, state="readonly")
        combo.grid(row=1, column=0, sticky="ew")
        root.columnconfigure(0, weight=1)

        ttk.Checkbutton(root, text="Range filter (only presets in view)", variable=range_filter_var, command=_refresh_presets).grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )

        ttk.Label(root, text="Custom text (optional)").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(root, textvariable=custom_text_var).grid(row=4, column=0, sticky="ew")

        mode_box = ttk.Labelframe(root, text="Default position mode", padding=(8, 6))
        mode_box.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        ttk.Radiobutton(mode_box, text="Click to place", value="click", variable=mode_var).pack(anchor="w")
        ttk.Radiobutton(mode_box, text="Auto place at nearest peak in range", value="auto", variable=mode_var).pack(anchor="w")

        ttk.Checkbutton(root, text="Show vertical guide line", variable=vline_var).grid(row=6, column=0, sticky="w", pady=(8, 0))

        def _pick(var: tk.StringVar, title: str) -> None:
            try:
                c = colorchooser.askcolor(color=(var.get() or None), title=title, parent=win)[1]
                if c:
                    var.set(str(c))
            except Exception:
                return

        colors = ttk.Frame(root)
        colors.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(colors, text="Label color").pack(side=tk.LEFT)
        ttk.Entry(colors, textvariable=text_color_var, width=12).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(colors, text="Pick…", command=lambda: _pick(text_color_var, "Pick label color")).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(colors, text="Line color").pack(side=tk.LEFT, padx=(14, 0))
        ttk.Entry(colors, textvariable=line_color_var, width=12).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(colors, text="Pick…", command=lambda: _pick(line_color_var, "Pick line color")).pack(side=tk.LEFT, padx=(6, 0))

        fr = ttk.Frame(root)
        fr.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(fr, text="Font size").pack(side=tk.LEFT)
        ttk.Spinbox(fr, from_=6, to=30, textvariable=fontsize_var, width=6).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(fr, text="Rotation").pack(side=tk.LEFT, padx=(14, 0))
        ttk.Spinbox(fr, from_=-180, to=180, textvariable=rotation_var, width=6).pack(side=tk.LEFT, padx=(8, 0))

        attach = ttk.Labelframe(root, text="Attach to overlay dataset", padding=(8, 6))
        attach.grid(row=9, column=0, sticky="ew", pady=(10, 0))
        ttk.Radiobutton(attach, text="Active only", value="Active only", variable=attach_var).pack(anchor="w")
        ttk.Radiobutton(attach, text="All overlayed", value="All overlayed", variable=attach_var).pack(anchor="w")
        ttk.Radiobutton(attach, text="Choose dataset", value="Choose dataset", variable=attach_var).pack(anchor="w")

        choose_row = ttk.Frame(attach)
        choose_row.pack(fill="x", pady=(6, 0))
        ttk.Label(choose_row, text="Dataset").pack(side=tk.LEFT)
        ds_combo = ttk.Combobox(choose_row, textvariable=dataset_choice_var, state="readonly")
        ds_combo.pack(side=tk.LEFT, padx=(8, 0), fill="x", expand=True)
        ds_combo["values"] = [x[0] for x in dataset_choices]

        def _resolve_target_dataset_id() -> str:
            mode = str(attach_var.get() or "")
            if mode == "All overlayed":
                return "__ALL_OVERLAY__"
            if mode == "Choose dataset":
                disp = str(dataset_choice_var.get() or "")
                for name, did in dataset_choices:
                    if name == disp:
                        return str(did)
                # fallback
                if dataset_choices:
                    return str(dataset_choices[0][1])
            # Active only
            ad = self._effective_active_dataset()
            return str(getattr(ad, "id", "")) if ad is not None else ""

        def _build_opts() -> Dict[str, Any]:
            disp = str(preset_var.get() or "")
            preset_id = preset_display_to_id.get(disp)
            r = preset_display_to_range.get(disp)
            preset_text = ""
            try:
                # Strip the range suffix from display text
                preset_text = str(disp).split("(", 1)[0].strip()
            except Exception:
                preset_text = disp
            text = str(custom_text_var.get() or "").strip() or str(preset_text)

            return {
                "preset_id": (None if not preset_id else str(preset_id)),
                "preset_range": (None if not r else (float(r[0]), float(r[1]))),
                "text": str(text),
                "show_vline": bool(vline_var.get()),
                "text_color": str(text_color_var.get() or "#111111"),
                "line_color": str(line_color_var.get() or "#444444"),
                "fontsize": int(fontsize_var.get()),
                "rotation": int(rotation_var.get()),
                "target_dataset_id": _resolve_target_dataset_id(),
            }

        def _close() -> None:
            try:
                if self._bond_dialog is not None:
                    self._bond_dialog.destroy()
            except Exception:
                pass
            self._bond_dialog = None

        def _place() -> None:
            opts = _build_opts()
            if not str(opts.get("text") or "").strip():
                return
            if str(mode_var.get() or "") == "auto":
                _close()
                self._bond_autoplace(opts)
                return
            _close()
            self._bond_begin_placement(opts)

        btns = ttk.Frame(root)
        btns.grid(row=10, column=0, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Place label", command=_place, style="Primary.TButton").pack(side=tk.LEFT)
        ttk.Button(btns, text="Cancel", command=_close, style="Secondary.TButton").pack(side=tk.LEFT, padx=(8, 0))

        _refresh_presets()

        try:
            win.protocol("WM_DELETE_WINDOW", _close)
        except Exception:
            pass
        try:
            win.grab_set()
        except Exception:
            pass
        try:
            win.focus_force()
        except Exception:
            pass

    def _bond_cancel_placement(self, *, reason: str = "") -> None:
        if not bool(getattr(self, "_bond_placement_active", False)):
            return
        self._bond_placement_active = False
        self._bond_place_opts = {}
        try:
            if self._canvas is not None and self._bond_place_cid_click is not None:
                self._canvas.mpl_disconnect(self._bond_place_cid_click)
        except Exception:
            pass
        try:
            if self._canvas is not None and self._bond_place_cid_key is not None:
                self._canvas.mpl_disconnect(self._bond_place_cid_key)
        except Exception:
            pass
        self._bond_place_cid_click = None
        self._bond_place_cid_key = None
        try:
            self._status_var.set("" if not reason else f"Bond label placement cancelled ({reason}).")
        except Exception:
            pass

    def _bond_begin_placement(self, opts: Dict[str, Any]) -> None:
        if self._canvas is None:
            return
        # Cancel any existing placement
        try:
            self._bond_cancel_placement(reason="restart")
        except Exception:
            pass

        self._bond_place_opts = dict(opts or {})
        self._bond_placement_active = True
        try:
            self._status_var.set("Click on the FTIR plot to place the bond label. Esc to cancel.")
        except Exception:
            pass

        try:
            self._bond_place_cid_click = self._canvas.mpl_connect("button_press_event", self._on_bond_place_click)
            self._bond_place_cid_key = self._canvas.mpl_connect("key_press_event", self._on_bond_place_key)
        except Exception:
            self._bond_place_cid_click = None
            self._bond_place_cid_key = None

    def _on_bond_place_key(self, evt) -> None:
        key = str(getattr(evt, "key", "") or "")
        if key.lower() == "escape":
            self._bond_cancel_placement(reason="esc")

    def _on_bond_place_click(self, evt) -> None:
        if not bool(getattr(self, "_bond_placement_active", False)):
            return
        try:
            if evt is None or int(getattr(evt, "button", 0) or 0) != 1:
                return
            if getattr(evt, "inaxes", None) is None:
                return
            if evt.xdata is None:
                return
        except Exception:
            return

        x = float(evt.xdata)
        y = None
        try:
            ad = self._effective_active_dataset()
            if ad is not None:
                xp, yp = self._get_plot_arrays(ad)
                xp = np.asarray(xp, dtype=float)
                yp = np.asarray(yp, dtype=float)
                mask = np.isfinite(xp) & np.isfinite(yp)
                xp = xp[mask]
                yp = yp[mask]
                if xp.size >= 2:
                    try:
                        order = np.argsort(xp)
                        xp = xp[order]
                        yp = yp[order]
                    except Exception:
                        pass
                    y = float(np.interp(float(x), xp, yp))
        except Exception:
            y = None

        if y is None:
            try:
                y = float(evt.ydata) if evt.ydata is not None else 0.0
            except Exception:
                y = 0.0

        # Offset a little above
        try:
            y0, y1 = self._ax.get_ylim() if self._ax is not None else (y, y)
            off = (float(y1) - float(y0)) * 0.02
        except Exception:
            off = 0.0

        yy = float(y) + float(off)
        opts = dict(getattr(self, "_bond_place_opts", {}) or {})
        self._bond_add_annotation(opts, x_cm1=float(x), y_value=float(y), xytext=(float(x), float(yy)))
        self._bond_cancel_placement(reason="done")
        self._schedule_redraw()

    def _bond_add_annotation(self, opts: Dict[str, Any], *, x_cm1: float, y_value: float, xytext: Tuple[float, float]) -> None:
        ws = self._active_workspace()
        try:
            target = str(opts.get("target_dataset_id") or "").strip()
            if not target:
                # fallback to active dataset
                ad = self._effective_active_dataset()
                target = str(getattr(ad, "id", "")) if ad is not None else ""
            ann = FTIRBondAnnotation(
                dataset_id=str(target),
                text=str(opts.get("text") or ""),
                x_cm1=float(x_cm1),
                y_value=float(y_value),
                xytext=(float(xytext[0]), float(xytext[1])),
                show_vline=bool(opts.get("show_vline", False)),
                line_color=str(opts.get("line_color") or "#444444"),
                text_color=str(opts.get("text_color") or "#111111"),
                fontsize=int(opts.get("fontsize") or 9),
                rotation=int(opts.get("rotation") or 0),
                preset_id=(None if not opts.get("preset_id") else str(opts.get("preset_id"))),
            )
            ws.bond_annotations.append(ann)
        except Exception:
            return

    def _bond_autoplace(self, opts: Dict[str, Any]) -> None:
        d = self._effective_active_dataset()
        if d is None:
            return

        # Decide range
        r = opts.get("preset_range")
        if isinstance(r, (list, tuple)) and len(r) == 2:
            try:
                r0, r1 = float(r[0]), float(r[1])
                lo, hi = (min(r0, r1), max(r0, r1))
            except Exception:
                lo, hi = None, None
        else:
            lo, hi = None, None

        x_peak = None
        y_peak = None

        # Prefer picked peaks in range
        try:
            best = None
            best_prom = -float("inf")
            for p in (getattr(d, "peaks", None) or []):
                if not isinstance(p, dict):
                    continue
                try:
                    wn = float(p.get("wn"))
                except Exception:
                    continue
                if lo is not None and hi is not None and not (lo <= wn <= hi):
                    continue
                try:
                    prom = float(p.get("prominence", 0.0) or 0.0)
                except Exception:
                    prom = 0.0
                if prom > best_prom:
                    best_prom = prom
                    best = p
            if isinstance(best, dict):
                x_peak = float(best.get("wn"))
                y_peak = float(best.get("y_display", best.get("y", 0.0)))
        except Exception:
            pass

        # Otherwise, local extremum within range
        if x_peak is None or y_peak is None:
            try:
                x = np.asarray(getattr(d, "x_full", None) or getattr(d, "x_disp", None) or [], dtype=float)
                y = np.asarray(getattr(d, "y_full", None) or getattr(d, "y_disp", None) or [], dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]
                if x.size >= 2:
                    try:
                        order = np.argsort(x)
                        x = x[order]
                        y = y[order]
                    except Exception:
                        pass
                    if lo is not None and hi is not None:
                        inr = (x >= float(lo)) & (x <= float(hi))
                        if int(np.sum(inr)) >= 2:
                            xr = x[inr]
                            yr = y[inr]
                        else:
                            xr = x
                            yr = y
                    else:
                        xr = x
                        yr = y

                    ym = str(getattr(d, "y_mode", "absorbance") or "absorbance").lower()
                    idx = int(np.nanargmax(yr)) if "trans" not in ym else int(np.nanargmin(yr))
                    x_peak = float(xr[idx])
                    y_peak = float(yr[idx])
            except Exception:
                pass

        # Final fallback: midpoint
        if x_peak is None or y_peak is None:
            try:
                if lo is not None and hi is not None:
                    x_peak = float(lo + (hi - lo) * 0.5)
                else:
                    xmin, xmax = self._ax.get_xlim() if self._ax is not None else (0.0, 1.0)
                    x_peak = float(min(xmin, xmax) + (max(xmin, xmax) - min(xmin, xmax)) * 0.5)
            except Exception:
                x_peak = 0.0
            try:
                xp, yp = self._get_plot_arrays(d)
                xp = np.asarray(xp, dtype=float)
                yp = np.asarray(yp, dtype=float)
                mask = np.isfinite(xp) & np.isfinite(yp)
                xp = xp[mask]
                yp = yp[mask]
                if xp.size >= 2:
                    order = np.argsort(xp)
                    xp = xp[order]
                    yp = yp[order]
                    y_peak = float(np.interp(float(x_peak), xp, yp))
                else:
                    y_peak = 0.0
            except Exception:
                y_peak = 0.0

        try:
            y0, y1 = self._ax.get_ylim() if self._ax is not None else (y_peak, y_peak)
            off = (float(y1) - float(y0)) * 0.02
        except Exception:
            off = 0.0

        self._bond_add_annotation(opts, x_cm1=float(x_peak), y_value=float(y_peak), xytext=(float(x_peak), float(y_peak) + float(off)))
        self._schedule_redraw()

    def _build_peaks_export_rows(self, d: FTIRDataset) -> List[Dict[str, Any]]:
        peaks = list(getattr(d, "peaks", None) or [])
        if not peaks:
            return []

        settings = dict(getattr(d, "peak_settings", None) or {})
        overrides = dict(getattr(d, "peak_label_overrides", None) or {})
        label_fmt = str(settings.get("label_fmt") or "{wn:.1f}")

        try:
            settings_json = json.dumps(settings, ensure_ascii=False, sort_keys=True)
        except Exception:
            settings_json = str(settings)

        out: List[Dict[str, Any]] = []
        for p in peaks:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue
            try:
                wn0 = float(p.get("wn"))
            except Exception:
                continue
            try:
                y0 = float(p.get("y_display", p.get("y", 0.0)))
            except Exception:
                y0 = float("nan")
            try:
                prom0 = float(p.get("prominence", 0.0))
            except Exception:
                prom0 = float("nan")

            # width_cm1 if available (we don't always compute it)
            width_cm1 = None
            try:
                if p.get("width_cm1") is not None:
                    width_cm1 = float(p.get("width_cm1"))
                elif p.get("width") is not None:
                    width_cm1 = float(p.get("width"))
            except Exception:
                width_cm1 = None

            # Final displayed label (override if present, else computed from format)
            label_text = str(overrides.get(pid, "") or "").strip()
            if not label_text:
                try:
                    if FTIRPeak is not None:
                        label_text = format_peak_label(FTIRPeak(wn=wn0, y=y0, prominence=prom0), fmt=label_fmt)
                    else:
                        label_text = f"{wn0:.1f}"
                except Exception:
                    label_text = f"{wn0:.1f}"

            out.append(
                {
                    "file_name": Path(str(getattr(d, "path", "") or "")).name,
                    "file_path": str(getattr(d, "path", "") or ""),
                    "peak_wn_cm1": float(wn0),
                    "peak_y": (None if (y0 is None or (isinstance(y0, float) and not math.isfinite(y0))) else float(y0)),
                    "prominence": (None if (prom0 is None or (isinstance(prom0, float) and not math.isfinite(prom0))) else float(prom0)),
                    "width_cm1": (None if width_cm1 is None else float(width_cm1)),
                    "label_text": str(label_text),
                    "preprocessing_settings": str(settings_json),
                }
            )

        return out

    def _build_peaks_export_tables(
        self,
        d: FTIRDataset,
        *,
        include_candidates: bool,
        top_n: int = 3,
        min_score: float = 35.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build export rows for a single FTIR dataset.

        Returns:
            (main_rows, expanded_candidate_rows)
        """

        peaks = list(getattr(d, "peaks", None) or [])
        if not peaks:
            return [], []

        settings = dict(getattr(d, "peak_settings", None) or {})
        overrides = dict(getattr(d, "peak_label_overrides", None) or {})
        label_fmt = str(settings.get("label_fmt") or "{wn:.1f}")

        try:
            settings_json = json.dumps(settings, ensure_ascii=False, sort_keys=True)
        except Exception:
            settings_json = str(settings)

        file_name = Path(str(getattr(d, "path", "") or "")).name
        file_path = str(getattr(d, "path", "") or "")
        dataset_name = str(getattr(d, "name", "dataset") or "dataset")

        # Gather peak metrics
        base_rows: List[Dict[str, Any]] = []
        peaks_for_assign: List[Dict[str, Any]] = []

        for p in peaks:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue
            try:
                wn0 = float(p.get("wn"))
            except Exception:
                continue
            try:
                y0 = float(p.get("y_display", p.get("y", 0.0)))
            except Exception:
                y0 = float("nan")
            try:
                prom0 = float(p.get("prominence", 0.0))
            except Exception:
                prom0 = float("nan")

            # width_cm1 if available (we don't always compute it)
            width_cm1 = None
            try:
                if p.get("width_cm1") is not None:
                    width_cm1 = float(p.get("width_cm1"))
                elif p.get("width") is not None:
                    width_cm1 = float(p.get("width"))
            except Exception:
                width_cm1 = None

            # Final displayed label (override if present, else computed from format)
            label_text = str(overrides.get(pid, "") or "").strip()
            if not label_text:
                try:
                    if FTIRPeak is not None:
                        label_text = format_peak_label(FTIRPeak(wn=wn0, y=y0, prominence=prom0), fmt=label_fmt)
                    else:
                        label_text = f"{wn0:.1f}"
                except Exception:
                    label_text = f"{wn0:.1f}"

            height = None
            try:
                if isinstance(y0, float) and math.isfinite(y0):
                    height = float(y0)
            except Exception:
                height = None

            prominence = None
            try:
                if isinstance(prom0, float) and math.isfinite(prom0):
                    prominence = float(prom0)
            except Exception:
                prominence = None

            width = None
            try:
                if width_cm1 is not None and math.isfinite(float(width_cm1)):
                    width = float(width_cm1)
            except Exception:
                width = None

            sharpness = None
            try:
                if prominence is not None and width is not None and width > 0:
                    sharpness = float(prominence) / float(width)
            except Exception:
                sharpness = None

            base_rows.append(
                {
                    "_peak_id": pid,
                    "Peak(cm^-1)": float(wn0),
                    "Height/Intensity": height,
                    "Width/FWHM": width,
                    "Prominence": prominence,
                    "Label": str(label_text),
                    "file_name": file_name,
                    "file_path": file_path,
                    "preprocessing_settings": str(settings_json),
                }
            )
            peaks_for_assign.append(
                {
                    "wn": float(wn0),
                    "height": height,
                    "width": width,
                    "prominence": prominence,
                    "sharpness": sharpness,
                }
            )

        if not include_candidates:
            # Backward-compatible: return the old export rows.
            return self._build_peaks_export_rows(d), []

        # Compute candidates (pure code; UI imports live here only)
        try:
            from lab_gui.ftir_assignment import assign_ftir_peaks
            from lab_gui.ftir_library import FTIR_LIBRARY_V2, FTIR_LIBRARY_VERSION
        except Exception as exc:
            raise RuntimeError(f"Failed to import FTIR assignment modules: {exc}")

        lib_by_id: Dict[str, Dict[str, Any]] = {}
        try:
            for e in FTIR_LIBRARY_V2:
                if isinstance(e, dict) and e.get("id"):
                    lib_by_id[str(e.get("id"))] = e
        except Exception:
            lib_by_id = {}

        assignments = assign_ftir_peaks(
            peaks_for_assign,
            FTIR_LIBRARY_V2,
            spectrum_context={"dataset": dataset_name, "library_version": FTIR_LIBRARY_VERSION},
            top_n=int(top_n or 3),
            min_score=float(min_score or 0.0),
        )

        expanded: List[Dict[str, Any]] = []
        main: List[Dict[str, Any]] = []

        for idx, row in enumerate(base_rows):
            cand_list = []
            try:
                cand_list = list((assignments[idx] or {}).get("candidates") or [])
            except Exception:
                cand_list = []

            out_row = dict(row)
            # Remove internal helper key
            out_row.pop("_peak_id", None)

            # Add top-N candidate columns
            for k in range(1, 4):
                c = cand_list[k - 1] if (k - 1) < len(cand_list) else None
                label = None
                score = None
                reasons = ""
                lib_id = None
                if isinstance(c, dict):
                    label = str(c.get("label") or "") or None
                    try:
                        score = float(c.get("score"))
                    except Exception:
                        score = None
                    try:
                        rs = list(c.get("reasons") or [])
                        reasons = "; ".join(str(x) for x in rs if str(x).strip())
                    except Exception:
                        reasons = ""
                    lib_id = str(c.get("id") or "") or None

                # Excel (especially when opening CSV files) may misinterpret UTF-8 en-dash/em-dash.
                # Normalize to ASCII hyphen for export stability.
                if label is not None:
                    label = label.replace("\u2013", "-").replace("\u2014", "-")
                if reasons:
                    reasons = reasons.replace("\u2013", "-").replace("\u2014", "-")

                out_row[f"Candidate {k} Label"] = label
                out_row[f"Candidate {k} Score"] = score
                out_row[f"Candidate {k} Reasons"] = reasons

                if include_candidates and lib_id:
                    try:
                        entry = lib_by_id.get(lib_id) or {}
                        r = entry.get("range_cm1") or (None, None)
                        rng = ""
                        try:
                            rng = f"{float(r[0]):.0f}-{float(r[1]):.0f}"
                        except Exception:
                            rng = ""
                        expanded.append(
                            {
                                "Dataset": dataset_name,
                                "Peak(cm^-1)": float(out_row.get("Peak(cm^-1)") or 0.0),
                                "CandidateLabel": label,
                                "Score": score,
                                "Reasons": reasons,
                                "LibraryID": lib_id,
                                "LibraryRange": rng,
                                "Notes": str(entry.get("notes") or ""),
                            }
                        )
                    except Exception:
                        pass

            main.append(out_row)

        return main, expanded

    def _export_active_peaks_csv(self, *, include_candidates: Optional[bool] = None) -> None:
        d = self._active_dataset()
        if d is None:
            messagebox.showinfo("Export Peaks", "Load an FTIR dataset first.", parent=self.app)
            return
        if not (getattr(d, "peaks", None) or []):
            messagebox.showinfo("Export Peaks", "Run Peaks… first.", parent=self.app)
            return

        include = bool(self._export_include_candidates_var.get()) if include_candidates is None else bool(include_candidates)
        try:
            if include:
                rows, _ = self._build_peaks_export_tables(d, include_candidates=True)
            else:
                rows = self._build_peaks_export_rows(d)
        except Exception as exc:
            messagebox.showerror("Export Peaks", f"Failed to prepare peaks for export:\n\n{exc}", parent=self.app)
            return
        if not rows:
            messagebox.showinfo("Export Peaks", "Run Peaks… first.", parent=self.app)
            return

        default_name = f"{str(getattr(d, 'name', 'ftir')).strip() or 'ftir'}_peaks.csv"
        path = filedialog.asksaveasfilename(
            parent=self.app,
            title="Export FTIR peaks (active dataset)",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            df = pd.DataFrame(rows)
            # Excel on Windows detects UTF-8 more reliably with a BOM.
            df.to_csv(path, index=False, encoding="utf-8-sig")
            try:
                self.app._log("INFO", f"Exported FTIR peaks CSV: {Path(path).name}")
            except Exception:
                pass
        except Exception as exc:
            messagebox.showerror("Export Peaks", f"Failed to export CSV:\n\n{exc}", parent=self.app)

    def _export_all_peaks_excel(self, *, include_candidates: Optional[bool] = None) -> None:
        ds = list(getattr(self.workspace, "ftir_datasets", []) or [])
        if not ds:
            messagebox.showinfo("Export Peaks", "No FTIR datasets loaded.", parent=self.app)
            return

        any_rows = False
        for d in ds:
            if getattr(d, "peaks", None):
                any_rows = True
                break
        if not any_rows:
            messagebox.showinfo("Export Peaks", "Run Peaks… first.", parent=self.app)
            return

        path = filedialog.asksaveasfilename(
            parent=self.app,
            title="Export FTIR peaks (all datasets)",
            defaultextension=".xlsx",
            initialfile="ftir_peaks.xlsx",
            filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            import openpyxl  # noqa: F401
        except Exception:
            messagebox.showerror(
                "Export Peaks",
                "Excel export requires openpyxl.\n\nInstall it with: pip install openpyxl",
                parent=self.app,
            )
            return

        def _sheet_name(base: str, used: set[str]) -> str:
            s = str(base or "dataset").strip() or "dataset"
            # Excel sheet name constraints
            bad = set('[]:*?/\\')
            s = "".join(("_" if ch in bad else ch) for ch in s)
            s = s[:31]
            if not s:
                s = "dataset"
            name = s
            i = 2
            while name in used:
                suffix = f"_{i}"
                name = (s[: max(0, 31 - len(suffix))] + suffix)
                i += 1
            used.add(name)
            return name

        try:
            used: set[str] = set()
            include = bool(self._export_include_candidates_var.get()) if include_candidates is None else bool(include_candidates)
            expanded_all: List[Dict[str, Any]] = []
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for d in ds:
                    if not (getattr(d, "peaks", None) or []):
                        continue
                    if include:
                        rows, expanded = self._build_peaks_export_tables(d, include_candidates=True)
                        if expanded:
                            expanded_all.extend(expanded)
                    else:
                        rows = self._build_peaks_export_rows(d)
                    if not rows:
                        continue
                    df = pd.DataFrame(rows)
                    sheet = _sheet_name(str(getattr(d, "name", "dataset")), used)
                    df.to_excel(writer, sheet_name=sheet, index=False)

                if include and expanded_all:
                    # One row per (peak, candidate), across all datasets.
                    dfc = pd.DataFrame(expanded_all)
                    dfc.to_excel(writer, sheet_name=_sheet_name("Candidates", used), index=False)

            try:
                self.app._log("INFO", f"Exported FTIR peaks workbook: {Path(path).name}")
            except Exception:
                pass
        except Exception as exc:
            messagebox.showerror("Export Peaks", f"Failed to export Excel workbook:\n\n{exc}", parent=self.app)

    def _sync_peaks_vars_from_active_dataset(self) -> None:
        d = self._active_dataset()
        if d is None:
            return
        s = dict(getattr(d, "peak_settings", None) or {})
        try:
            if "enabled" in s:
                self._peaks_enabled_var.set(bool(s.get("enabled")))
            if "min_prominence" in s:
                self._peaks_min_prom_var.set(float(s.get("min_prominence") or 0.0))
            if "min_distance_cm1" in s:
                self._peaks_min_dist_var.set(float(s.get("min_distance_cm1") or 0.0))
            # Top-N removed from UI: always disabled.
            self._peaks_topn_var.set(0)
            if "label_fmt" in s:
                self._peaks_label_fmt_var.set(str(s.get("label_fmt") or "{wn:.1f}"))
        except Exception:
            pass

    def _apply_peaks_dialog(self) -> None:
        d = self._active_dataset()
        if d is None:
            messagebox.showinfo("FTIR Peaks", "Load an FTIR dataset first.", parent=self.app)
            return

        settings: Dict[str, Any] = {
            "enabled": bool(self._peaks_enabled_var.get()),
            # Preprocessing UI removed; keep preprocessing off.
            "smoothing": "none",
            "smoothing_window": 0,
            "poly_order": 0,
            "baseline": "none",
            "normalize": "none",
            "min_prominence": float(self._peaks_min_prom_var.get() or 0.0),
            "min_distance_cm1": float(self._peaks_min_dist_var.get() or 0.0),
            "top_n": 0,
            "label_fmt": str(self._peaks_label_fmt_var.get() or "{wn:.1f}"),
        }

        try:
            d.peak_settings = dict(settings)
        except Exception:
            pass

        if not settings.get("enabled"):
            self._schedule_redraw()
            return

        self._start_peaks_worker(
            str(getattr(d, "id", "")),
            np.asarray(getattr(d, "x_full", []), dtype=float),
            np.asarray(getattr(d, "y_full", []), dtype=float),
            str(getattr(d, "y_mode", "absorbance")),
            settings,
        )

    def _apply_peaks_dialog_all(self) -> None:
        # Apply current peaks settings to ALL datasets in ALL FTIR workspaces.
        keys = list(self._all_dataset_keys() or [])
        if not keys:
            messagebox.showinfo("FTIR Peaks", "No FTIR datasets loaded.", parent=self.app)
            return

        if self._peaks_busy:
            messagebox.showinfo("FTIR Peaks", "Peak picking is already running…", parent=self.app)
            return

        settings: Dict[str, Any] = {
            "enabled": bool(self._peaks_enabled_var.get()),
            # Preprocessing UI removed; keep preprocessing off.
            "smoothing": "none",
            "smoothing_window": 0,
            "poly_order": 0,
            "baseline": "none",
            "normalize": "none",
            "min_prominence": float(self._peaks_min_prom_var.get() or 0.0),
            "min_distance_cm1": float(self._peaks_min_dist_var.get() or 0.0),
            "top_n": 0,
            "label_fmt": str(self._peaks_label_fmt_var.get() or "{wn:.1f}"),
        }

        try:
            ok = messagebox.askyesno(
                "FTIR Peaks",
                f"Compute peaks for ALL FTIR datasets across all workspaces?\n\nDatasets: {len(keys)}\n\nThis may take some time.",
                parent=self.app,
            )
        except Exception:
            ok = True
        if not ok:
            return

        # Store settings on all datasets (best effort)
        for k in keys:
            d0 = self._get_dataset_by_key((str(k[0]), str(k[1])))
            if d0 is None:
                continue
            try:
                d0.peak_settings = dict(settings)
            except Exception:
                pass

        if not settings.get("enabled"):
            self._schedule_redraw()
            return

        self._start_peaks_batch_worker(keys, settings)

    def _start_peaks_worker(self, dataset_id: str, wn: np.ndarray, y: np.ndarray, y_mode: str, settings: Dict[str, Any]) -> None:
        if not dataset_id:
            return

        # Single-dataset mode.
        try:
            self._peaks_batch_active = False
            self._peaks_batch_redraw_pending = False
        except Exception:
            pass

        self._peaks_busy = True

        # Disable the Peaks button while working (best effort).
        try:
            if self._btn_peaks is not None:
                self._btn_peaks.configure(state=tk.DISABLED)
        except Exception:
            pass

        def worker() -> None:
            t0 = time.perf_counter()
            try:
                x = np.asarray(wn, dtype=float)
                yy = np.asarray(y, dtype=float)
                mask = np.isfinite(x) & np.isfinite(yy)
                x = x[mask]
                yy = yy[mask]
                if int(x.size) < 5:
                    raise ValueError("Not enough data")
                order = np.argsort(x)
                x = x[order]
                yy = yy[order]

                mode = (str(y_mode or "absorbance").strip().lower() or "absorbance")
                smoothing = str(settings.get("smoothing") or "none").strip().lower()
                w = int(settings.get("smoothing_window") or 0)
                po = int(settings.get("poly_order") or 0)
                if smoothing == "none":
                    w = 0
                # Note: preprocess_spectrum chooses SavGol if SciPy is present; otherwise moving average.
                x_p, y_p = preprocess_spectrum(
                    x,
                    yy,
                    mode=mode,
                    smoothing_window=int(w),
                    poly_order=int(po),
                    baseline=str(settings.get("baseline") or "none"),
                    normalize=str(settings.get("normalize") or "none"),
                )
                peaks = pick_peaks(
                    x_p,
                    y_p,
                    mode=mode,
                    min_prominence=float(settings.get("min_prominence") or 0.0),
                    min_distance_cm1=float(settings.get("min_distance_cm1") or 0.0),
                    top_n=0,
                )

                used: Dict[str, int] = {}
                out: List[Dict[str, Any]] = []
                for p in peaks:
                    wn0 = float(getattr(p, "wn", 0.0))
                    prom0 = float(getattr(p, "prominence", 0.0))
                    try:
                        y_disp = float(np.interp(wn0, x, yy))
                    except Exception:
                        y_disp = float(getattr(p, "y", 0.0))

                    base = f"{wn0:.2f}"
                    k = int(used.get(base, 0))
                    used[base] = k + 1
                    pid = base if k == 0 else f"{base}_{k}"

                    out.append(
                        {
                            "id": str(pid),
                            "wn": float(wn0),
                            "y_display": float(y_disp),
                            "prominence": float(prom0),
                        }
                    )

                try:
                    self._ftir_event_q.put(("peaks_done", str(dataset_id), dict(settings), out))
                except Exception:
                    pass
            except Exception as exc:
                try:
                    self._ftir_event_q.put(("timing", "peaks_error", 0.0, str(exc)))
                except Exception:
                    pass
                try:
                    self._ftir_event_q.put(("peaks_done", str(dataset_id), dict(settings), []))
                except Exception:
                    pass
            finally:
                try:
                    self._ftir_event_q.put(("timing", "peaks_compute", time.perf_counter() - t0, ""))
                except Exception:
                    pass

        # Ensure queue poll is running while we compute.
        self._start_ftir_queue_poll()
        threading.Thread(target=worker, daemon=True).start()

    def _start_peaks_batch_worker(self, keys: Sequence[Tuple[str, str]], settings: Dict[str, Any]) -> None:
        # Compute peaks for many datasets, sequentially, on one worker thread.
        keys2: List[Tuple[str, str]] = [
            (str(k[0]), str(k[1])) for k in (keys or []) if isinstance(k, (list, tuple)) and len(k) == 2
        ]
        if not keys2:
            return

        self._peaks_busy = True
        try:
            self._peaks_batch_active = True
            self._peaks_batch_redraw_pending = True
        except Exception:
            pass

        # Disable the Peaks button while working (best effort).
        try:
            if self._btn_peaks is not None:
                self._btn_peaks.configure(state=tk.DISABLED)
        except Exception:
            pass

        def worker() -> None:
            t0 = time.perf_counter()
            done = 0
            try:
                for (ws_id, ds_id) in keys2:
                    d = self._get_dataset_by_key((ws_id, ds_id))
                    if d is None:
                        continue

                    try:
                        wn = np.asarray(getattr(d, "x_full", []), dtype=float)
                        y = np.asarray(getattr(d, "y_full", []), dtype=float)
                        y_mode = str(getattr(d, "y_mode", "absorbance"))
                    except Exception:
                        continue

                    try:
                        x = np.asarray(wn, dtype=float)
                        yy = np.asarray(y, dtype=float)
                        mask = np.isfinite(x) & np.isfinite(yy)
                        x = x[mask]
                        yy = yy[mask]
                        if int(x.size) < 5:
                            raise ValueError("Not enough data")
                        order = np.argsort(x)
                        x = x[order]
                        yy = yy[order]

                        mode = (str(y_mode or "absorbance").strip().lower() or "absorbance")
                        smoothing = str(settings.get("smoothing") or "none").strip().lower()
                        w = int(settings.get("smoothing_window") or 0)
                        po = int(settings.get("poly_order") or 0)
                        if smoothing == "none":
                            w = 0
                        x_p, y_p = preprocess_spectrum(
                            x,
                            yy,
                            mode=mode,
                            smoothing_window=int(w),
                            poly_order=int(po),
                            baseline=str(settings.get("baseline") or "none"),
                            normalize=str(settings.get("normalize") or "none"),
                        )
                        peaks = pick_peaks(
                            x_p,
                            y_p,
                            mode=mode,
                            min_prominence=float(settings.get("min_prominence") or 0.0),
                            min_distance_cm1=float(settings.get("min_distance_cm1") or 0.0),
                            top_n=0,
                        )

                        used: Dict[str, int] = {}
                        out: List[Dict[str, Any]] = []
                        for p in peaks:
                            wn0 = float(getattr(p, "wn", 0.0))
                            prom0 = float(getattr(p, "prominence", 0.0))
                            try:
                                y_disp = float(np.interp(wn0, x, yy))
                            except Exception:
                                y_disp = float(getattr(p, "y", 0.0))

                            base = f"{wn0:.2f}"
                            k = int(used.get(base, 0))
                            used[base] = k + 1
                            pid = base if k == 0 else f"{base}_{k}"

                            out.append(
                                {
                                    "id": str(pid),
                                    "wn": float(wn0),
                                    "y_display": float(y_disp),
                                    "prominence": float(prom0),
                                }
                            )

                        try:
                            self._ftir_event_q.put(("peaks_done_key", str(ws_id), str(ds_id), dict(settings), out))
                        except Exception:
                            pass
                    except Exception:
                        # Still notify with empty list
                        try:
                            self._ftir_event_q.put(("peaks_done_key", str(ws_id), str(ds_id), dict(settings), []))
                        except Exception:
                            pass

                    done += 1

                try:
                    self._ftir_event_q.put(("peaks_batch_done", int(done), int(len(keys2))))
                except Exception:
                    pass
            finally:
                try:
                    self._ftir_event_q.put(("timing", "peaks_batch_compute", time.perf_counter() - t0, f"done={done}/{len(keys2)}"))
                except Exception:
                    pass

        self._start_ftir_queue_poll()
        threading.Thread(target=worker, daemon=True).start()

    def _on_peaks_ready_for_key(self, key: Tuple[str, str], peak_settings: Any, peaks: Any) -> None:
        # UI thread only. Applies peaks to the specified dataset (across any workspace).
        k = (str(key[0]), str(key[1]))
        d = self._get_dataset_by_key(k)
        if d is None:
            return

        try:
            if isinstance(peak_settings, dict):
                d.peak_settings = dict(peak_settings)
        except Exception:
            pass

        new_peaks: List[Dict[str, Any]] = []
        if isinstance(peaks, list):
            for row in peaks:
                if not isinstance(row, dict):
                    continue
                pid = str(row.get("id") or "").strip()
                if not pid:
                    continue
                try:
                    new_peaks.append(
                        {
                            "id": pid,
                            "wn": float(row.get("wn")),
                            "y_display": float(row.get("y_display", row.get("y", 0.0))),
                            "prominence": float(row.get("prominence", 0.0)),
                        }
                    )
                except Exception:
                    continue

        try:
            d.peaks = list(new_peaks)
        except Exception:
            pass

        # Peaks recomputed: invalidate overlay display cache for this dataset.
        try:
            self._overlay_peak_display_ids_by_key.pop(k, None)
        except Exception:
            pass

        # Keep overrides/suppression only for still-existing peak ids.
        try:
            valid = set(str(p.get("id")) for p in d.peaks if isinstance(p, dict) and p.get("id"))
            d.peak_label_overrides = {k: v for k, v in (getattr(d, "peak_label_overrides", {}) or {}).items() if k in valid}
            d.peak_suppressed = set(x for x in (getattr(d, "peak_suppressed", set()) or set()) if x in valid)
            d.peak_label_positions = {
                k: v
                for k, v in (getattr(d, "peak_label_positions", {}) or {}).items()
                if (k in valid and isinstance(v, tuple) and len(v) == 2)
            }
        except Exception:
            pass

    def _on_peaks_batch_done(self, done: int, total: int) -> None:
        # UI thread only.
        try:
            self._peaks_batch_active = False
        except Exception:
            pass
        self._peaks_busy = False
        try:
            if self._btn_peaks is not None:
                self._btn_peaks.configure(state=tk.NORMAL)
        except Exception:
            pass
        try:
            self.app._log("INFO", f"FTIR peaks applied to all: {int(done)}/{int(total)}")
        except Exception:
            pass

        self._schedule_redraw()

    def _on_peaks_ready(self, dataset_id: str, peak_settings: Any, peaks: Any) -> None:
        # UI thread only.
        self._peaks_busy = False
        try:
            if self._btn_peaks is not None:
                self._btn_peaks.configure(state=tk.NORMAL)
        except Exception:
            pass

        d: Optional[FTIRDataset] = None
        for it in (getattr(self._active_workspace(), "datasets", []) or []):
            if str(getattr(it, "id", "")) == str(dataset_id):
                d = it
                break
        if d is None:
            return

        try:
            if isinstance(peak_settings, dict):
                d.peak_settings = dict(peak_settings)
        except Exception:
            pass

        new_peaks: List[Dict[str, Any]] = []
        if isinstance(peaks, list):
            for row in peaks:
                if not isinstance(row, dict):
                    continue
                pid = str(row.get("id") or "").strip()
                if not pid:
                    continue
                try:
                    new_peaks.append(
                        {
                            "id": pid,
                            "wn": float(row.get("wn")),
                            "y_display": float(row.get("y_display", row.get("y", 0.0))),
                            "prominence": float(row.get("prominence", 0.0)),
                        }
                    )
                except Exception:
                    continue

        try:
            d.peaks = list(new_peaks)
        except Exception:
            pass

        # Peaks recomputed: invalidate overlay display cache for this dataset.
        try:
            k = (str(self.active_workspace_id), str(getattr(d, "id", "")))
            self._overlay_peak_display_ids_by_key.pop(k, None)
        except Exception:
            pass

        # Keep overrides/suppression only for still-existing peak ids.
        try:
            valid = set(str(p.get("id")) for p in d.peaks if isinstance(p, dict) and p.get("id"))
            d.peak_label_overrides = {k: v for k, v in (getattr(d, "peak_label_overrides", {}) or {}).items() if k in valid}
            d.peak_suppressed = set(x for x in (getattr(d, "peak_suppressed", set()) or set()) if x in valid)
            d.peak_label_positions = {
                k: v
                for k, v in (getattr(d, "peak_label_positions", {}) or {}).items()
                if (k in valid and isinstance(v, tuple) and len(v) == 2)
            }
        except Exception:
            pass

        self._schedule_redraw()

    def _clear_peak_artists(self) -> None:
        for t in list(self._peak_texts):
            try:
                t.remove()
            except Exception:
                pass
        for m in list(self._peak_markers):
            try:
                m.remove()
            except Exception:
                pass
        try:
            if self._peak_summary_text is not None:
                self._peak_summary_text.remove()
        except Exception:
            pass
        self._peak_texts = []
        self._peak_texts_by_key = {}
        self._peak_markers = []
        self._peak_summary_text = None
        self._peak_artist_to_info = {}

    def _clear_bond_artists(self) -> None:
        for t in list(getattr(self, "_bond_texts", []) or []):
            try:
                t.remove()
            except Exception:
                pass
        for ln in list(getattr(self, "_bond_vlines", []) or []):
            try:
                ln.remove()
            except Exception:
                pass
        self._bond_texts = []
        self._bond_vlines = []
        self._bond_artist_to_info = {}

    def _render_bond_annotations(
        self,
        ax: Any,
        *,
        visible_keys: Sequence[Tuple[str, str]],
        active_key: Optional[Tuple[str, str]],
    ) -> None:
        """Rebuild bond-label artists from persisted FTIRBondAnnotation objects."""
        try:
            self._clear_bond_artists()
        except Exception:
            pass

        ws = self._active_workspace()
        anns = list(getattr(ws, "bond_annotations", None) or [])
        if not anns:
            return

        # Which dataset ids are currently present on the plot?
        present_ids: set[str] = set()
        for k in (visible_keys or []):
            d0 = self._get_dataset_by_key((str(k[0]), str(k[1])))
            if d0 is None:
                continue
            try:
                did = str(getattr(d0, "id", ""))
                if did:
                    present_ids.add(did)
            except Exception:
                continue

        active_ds_id = None
        try:
            if active_key is not None:
                d = self._get_dataset_by_key((str(active_key[0]), str(active_key[1])))
                if d is not None:
                    active_ds_id = str(getattr(d, "id", ""))
        except Exception:
            active_ds_id = None

        ALL_OVERLAY = "__ALL_OVERLAY__"

        # Rebuild artists
        for idx, ann in enumerate(anns):
            if not isinstance(ann, FTIRBondAnnotation):
                continue

            target = str(getattr(ann, "dataset_id", "") or "")
            if not target:
                continue

            show = False
            if target == ALL_OVERLAY:
                show = True
            elif active_ds_id is not None and target == str(active_ds_id):
                show = True
            elif target in present_ids:
                show = True

            if not show:
                continue

            try:
                x = float(getattr(ann, "x_cm1"))
                xy = getattr(ann, "xytext", None)
                if isinstance(xy, tuple) and len(xy) == 2:
                    tx, ty = float(xy[0]), float(xy[1])
                else:
                    tx, ty = float(x), float(getattr(ann, "y_value"))

                txt = ax.text(
                    float(tx),
                    float(ty),
                    str(getattr(ann, "text", "") or ""),
                    color=str(getattr(ann, "text_color", "#111111") or "#111111"),
                    fontsize=int(getattr(ann, "fontsize", 9) or 9),
                    rotation=int(getattr(ann, "rotation", 90) or 90),
                    va="bottom",
                    ha="center",
                    clip_on=True,
                    zorder=10,
                )
                try:
                    txt.set_picker(True)
                except Exception:
                    pass

                self._bond_texts.append(txt)
                try:
                    self._bond_artist_to_info[txt] = (str(ws.id), str(target), int(idx))
                except Exception:
                    pass

                if bool(getattr(ann, "show_vline", False)):
                    try:
                        ln = ax.axvline(
                            float(x),
                            color=str(getattr(ann, "line_color", "#444444") or "#444444"),
                            linestyle="--",
                            linewidth=1.0,
                            zorder=6,
                        )
                        try:
                            ln.set_picker(True)
                        except Exception:
                            pass
                        self._bond_vlines.append(ln)
                        try:
                            self._bond_artist_to_info[ln] = (str(ws.id), str(target), int(idx))
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                continue

    def _get_peak_color(self) -> str:
        if self._peak_color:
            return str(self._peak_color)
        try:
            cycle = matplotlib.rcParams.get("axes.prop_cycle")
            if cycle is not None:
                colors = cycle.by_key().get("color")  # type: ignore[attr-defined]
                if colors:
                    self._peak_color = str(colors[0])
                    return str(self._peak_color)
        except Exception:
            pass
        self._peak_color = "C0"
        return "C0"

    def _tint_color_close(self, base: str) -> str:
        """Return a color close to base, but not identical (slightly tinted for readability)."""
        try:
            r, g, b = mcolors.to_rgb(base)
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            amt = 0.22
            if lum >= 0.78:
                # very light -> tint toward black
                rr, gg, bb = (r * (1.0 - amt), g * (1.0 - amt), b * (1.0 - amt))
            else:
                # otherwise -> tint toward white
                rr, gg, bb = (r + (1.0 - r) * amt, g + (1.0 - g) * amt, b + (1.0 - b) * amt)
            return str(mcolors.to_hex((rr, gg, bb)))
        except Exception:
            return str(base)

    def _peak_color_for_key(self, key: Optional[Tuple[str, str]]) -> str:
        try:
            if key is not None:
                ln = self._line_artists.get((str(key[0]), str(key[1])))
                if ln is not None:
                    return self._tint_color_close(str(ln.get_color()))
        except Exception:
            pass
        return self._tint_color_close(self._get_peak_color())

    def _prune_overlay_peak_display_cache(self) -> None:
        keep: set[Tuple[str, str]] = set()
        try:
            for g in (self._overlay_groups or {}).values():
                for a, b in (getattr(g, "members", None) or []):
                    keep.add((str(a), str(b)))
        except Exception:
            pass
        try:
            k = self._effective_active_key()
            if k is not None:
                keep.add((str(k[0]), str(k[1])))
        except Exception:
            pass
        self._overlay_peak_display_ids_by_key = {
            k: v for k, v in (self._overlay_peak_display_ids_by_key or {}).items() if (str(k[0]), str(k[1])) in keep
        }

    def _get_overlay_display_peak_ids(self, key: Tuple[str, str], d: FTIRDataset, *, default_max: int = 0) -> List[str]:
        k = (str(key[0]), str(key[1]))
        if k in (self._overlay_peak_display_ids_by_key or {}):
            return list(self._overlay_peak_display_ids_by_key.get(k) or [])

        peaks = list(getattr(d, "peaks", None) or [])
        suppressed = set(getattr(d, "peak_suppressed", None) or set())
        rows: List[Dict[str, Any]] = [
            p
            for p in peaks
            if isinstance(p, dict)
            and str(p.get("id") or "")
            and (str(p.get("id")) not in suppressed)
        ]

        def _sort_key(p: Dict[str, Any]) -> Tuple[float, float, str]:
            try:
                prom = float(p.get("prominence", 0.0) or 0.0)
            except Exception:
                prom = 0.0
            try:
                wn = float(p.get("wn", 0.0) or 0.0)
            except Exception:
                wn = 0.0
            pid = str(p.get("id") or "")
            # sort by prominence desc, then wn asc for stability
            return (-prom, wn, pid)

        try:
            rows.sort(key=_sort_key)
        except Exception:
            pass

        max_n = int(default_max or 0)
        if max_n > 0:
            rows = rows[:max_n]

        ids = [str(p.get("id")) for p in rows if isinstance(p, dict) and p.get("id")]
        self._overlay_peak_display_ids_by_key[k] = list(ids)
        return list(ids)

    def _render_peaks_on_axes(
        self,
        ax: Any,
        d: FTIRDataset,
        *,
        dataset_key: Optional[Tuple[str, str]] = None,
        peak_color: Optional[str] = None,
        pickable: bool = True,
        clear_first: bool = True,
        max_peaks: int = 0,
        include_summary: bool = True,
        fontweight: Optional[str] = None,
        include_peak_ids: Optional[Sequence[str]] = None,
        overlay_order: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        if clear_first:
            self._clear_peak_artists()
        peak_color = str(peak_color or self._get_peak_color())
        s = dict(getattr(d, "peak_settings", None) or {})
        if not bool(s.get("enabled", False)):
            return

        peaks = list(getattr(d, "peaks", None) or [])
        suppressed = set(getattr(d, "peak_suppressed", None) or set())
        overrides = dict(getattr(d, "peak_label_overrides", None) or {})
        positions = dict(getattr(d, "peak_label_positions", None) or {})
        fmt = str(s.get("label_fmt") or "{wn:.1f}")

        shown = [p for p in peaks if isinstance(p, dict) and str(p.get("id") or "") and (str(p.get("id")) not in suppressed)]

        if include_peak_ids is not None:
            want = [str(x) for x in (include_peak_ids or []) if str(x).strip()]
            by_id = {str(p.get("id")): p for p in shown if isinstance(p, dict) and p.get("id")}
            shown = [by_id[pid] for pid in want if pid in by_id]
        else:
            # Optional clutter control for overlay labels.
            if int(max_peaks or 0) > 0 and len(shown) > int(max_peaks or 0):
                def _prom(p: Dict[str, Any]) -> float:
                    try:
                        return float(p.get("prominence", 0.0) or 0.0)
                    except Exception:
                        return 0.0

                try:
                    shown.sort(key=_prom, reverse=True)
                except Exception:
                    pass
                shown = shown[: int(max_peaks or 0)]
        dx, dy = self._overlay_offset_for_key(dataset_key, order=overlay_order)
        xs: List[float] = []
        ys: List[float] = []
        for p in shown:
            try:
                xs.append(float(p.get("wn")) + float(dx))
                ys.append(float(p.get("y_display", p.get("y", 0.0))) + float(dy))
            except Exception:
                continue

        # Markers
        if xs and ys:
            try:
                (mline,) = ax.plot(
                    xs,
                    ys,
                    linestyle="none",
                    marker="o",
                    markersize=4,
                    color=peak_color,
                    markerfacecolor=peak_color,
                    markeredgecolor=peak_color,
                )
                self._peak_markers.append(mline)
            except Exception:
                pass

        # Labels
        for p in shown:
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue
            try:
                peak_wn = float(p.get("wn")) + float(dx)
                peak_y = float(p.get("y_display", p.get("y", 0.0))) + float(dy)
                prom0 = float(p.get("prominence", 0.0))
            except Exception:
                continue

            pos_x = float(peak_wn)
            pos_y = float(peak_y)

            # Use stored manual position if present
            try:
                if pid in positions:
                    pos = positions.get(pid)
                    if isinstance(pos, tuple) and len(pos) == 2:
                        pos_x = float(pos[0]) + float(dx)
                        pos_y = float(pos[1]) + float(dy)
            except Exception:
                pass

            label = str(overrides.get(pid, "") or "").strip()
            if not label:
                try:
                    if FTIRPeak is not None:
                        label = format_peak_label(FTIRPeak(wn=peak_wn - float(dx), y=peak_y - float(dy), prominence=prom0), fmt=fmt)
                    else:
                        label = f"{(peak_wn - float(dx)):.1f}"
                except Exception:
                    label = f"{(peak_wn - float(dx)):.1f}"

            try:
                txt = ax.annotate(
                    str(label),
                    xy=(float(peak_wn), float(peak_y)),
                    xytext=(float(pos_x), float(pos_y)),
                    textcoords="data",
                    xycoords="data",
                    va="bottom",
                    ha="left",
                    fontsize=8,
                    color=peak_color,
                    clip_on=True,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": peak_color,
                        "lw": 0.8,
                        "shrinkA": 0.0,
                        "shrinkB": 0.0,
                    },
                )
                try:
                    if fontweight:
                        txt.set_fontweight(str(fontweight))
                except Exception:
                    pass
                try:
                    txt.set_picker(bool(pickable))
                except Exception:
                    pass
                self._peak_texts.append(txt)
                if dataset_key is not None:
                    try:
                        k0 = (str(dataset_key[0]), str(dataset_key[1]))
                        self._peak_texts_by_key.setdefault(k0, []).append(txt)
                    except Exception:
                        pass
                if pickable and dataset_key is not None:
                    try:
                        self._peak_artist_to_info[txt] = (str(dataset_key[0]), str(dataset_key[1]), str(pid))
                    except Exception:
                        pass
            except Exception:
                continue

        # Summary annotation
        if not bool(include_summary):
            return
        try:
            n_total = len([p for p in peaks if isinstance(p, dict)])
            n_show = len(shown)
            summary = f"Peaks: {n_show}/{n_total}"
            if s.get("min_prominence") is not None:
                summary += f" | prom≥{float(s.get('min_prominence') or 0.0):g}"
            if s.get("min_distance_cm1") is not None:
                summary += f" | dist≥{float(s.get('min_distance_cm1') or 0.0):g}"
            self._peak_summary_text = ax.text(0.01, 0.99, summary, transform=ax.transAxes, va="top", ha="left", fontsize=9)
        except Exception:
            pass

    def _render_overlay_peak_markers(
        self,
        ax: Any,
        dataset_keys: Sequence[Tuple[str, str]],
        active_key: Optional[Tuple[str, str]],
    ) -> None:
        """Render non-interactive peak markers+labels for non-active overlayed datasets."""
        max_peaks_per_spectrum = 0
        for key in dataset_keys:
            try:
                if active_key is not None and tuple(key) == tuple(active_key):
                    continue
            except Exception:
                pass
            d = self._get_dataset_by_key(key)
            if d is None:
                continue
            ids = []
            try:
                ids = self._get_overlay_display_peak_ids((str(key[0]), str(key[1])), d, default_max=max_peaks_per_spectrum)
            except Exception:
                ids = []
            self._render_peaks_on_axes(
                ax,
                d,
                dataset_key=(str(key[0]), str(key[1])),
                peak_color=self._peak_color_for_key(key),
                pickable=True,
                clear_first=False,
                include_peak_ids=ids,
                include_summary=False,
                fontweight=None,
                overlay_order=dataset_keys,
            )

    def _on_peak_drag_press(self, evt) -> None:
        # During bond placement mode, ignore drag logic.
        try:
            if bool(getattr(self, "_bond_placement_active", False)):
                return
        except Exception:
            pass

        # Left-click on a label to start drag.
        try:
            if evt is None or int(getattr(evt, "button", 0) or 0) != 1:
                return
            if getattr(evt, "inaxes", None) is None:
                return
            if evt.xdata is None or evt.ydata is None:
                return
        except Exception:
            return

        # Bond labels first
        try:
            for t in list(getattr(self, "_bond_texts", []) or []):
                try:
                    contains, _ = t.contains(evt)
                    if not contains:
                        continue
                    info = (getattr(self, "_bond_artist_to_info", {}) or {}).get(t)
                    if not info:
                        continue
                    ws_id, _target, idx = str(info[0]), str(info[1]), int(info[2])
                    x0, y0 = t.get_position()
                    self._drag_bond_key = (ws_id, "bond")
                    self._drag_bond_idx = int(idx)
                    self._drag_bond_artist = t
                    self._drag_bond_dx = float(x0) - float(evt.xdata)
                    self._drag_bond_dy = float(y0) - float(evt.ydata)
                    return
                except Exception:
                    continue
        except Exception:
            pass

        hit_artist = None
        hit_info: Optional[Tuple[str, str, str]] = None
        for t in list(self._peak_texts):
            try:
                contains, _ = t.contains(evt)
                if contains:
                    hit_artist = t
                    hit_info = self._peak_artist_to_info.get(t)
                    break
            except Exception:
                continue

        if not hit_artist or not hit_info:
            return

        try:
            x0, y0 = hit_artist.get_position()
            self._drag_peak_key = (str(hit_info[0]), str(hit_info[1]))
            self._drag_peak_id = str(hit_info[2])
            self._drag_peak_artist = hit_artist
            self._drag_dx = float(x0) - float(evt.xdata)
            self._drag_dy = float(y0) - float(evt.ydata)
        except Exception:
            self._drag_peak_id = None
            self._drag_peak_key = None
            self._drag_peak_artist = None
            self._drag_dx = 0.0
            self._drag_dy = 0.0

    def _on_peak_drag_motion(self, evt) -> None:
        # Bond drag
        if getattr(self, "_drag_bond_idx", None) is not None and getattr(self, "_drag_bond_artist", None) is not None:
            try:
                if evt is None or getattr(evt, "inaxes", None) is None:
                    return
                if evt.xdata is None or evt.ydata is None:
                    return
            except Exception:
                return

            try:
                new_x = float(evt.xdata) + float(getattr(self, "_drag_bond_dx", 0.0) or 0.0)
                new_y = float(evt.ydata) + float(getattr(self, "_drag_bond_dy", 0.0) or 0.0)
                self._drag_bond_artist.set_position((new_x, new_y))
            except Exception:
                return

            # Persist
            try:
                ws = None
                try:
                    ws_id = (self._drag_bond_key[0] if self._drag_bond_key is not None else None)
                    if ws_id and isinstance(getattr(self, "workspaces", None), dict):
                        ws = (self.workspaces or {}).get(str(ws_id))
                except Exception:
                    ws = None
                if ws is None:
                    ws = self._active_workspace()
                idx = int(self._drag_bond_idx)
                anns = list(getattr(ws, "bond_annotations", None) or [])
                if 0 <= idx < len(anns) and isinstance(anns[idx], FTIRBondAnnotation):
                    ann = anns[idx]
                    ann.x_cm1 = float(new_x)
                    ann.y_value = float(new_y)
                    ann.xytext = (float(new_x), float(new_y))
                    # Move any vline artists for the same annotation
                    info_map = getattr(self, "_bond_artist_to_info", {}) or {}
                    for ln in list(getattr(self, "_bond_vlines", []) or []):
                        try:
                            inf = info_map.get(ln)
                            if not inf:
                                continue
                            if str(inf[0]) != str(getattr(ws, "id", "")):
                                continue
                            if int(inf[2]) != int(idx):
                                continue
                            try:
                                ln.set_xdata([float(new_x), float(new_x)])
                            except Exception:
                                pass
                        except Exception:
                            continue
            except Exception:
                pass

            try:
                if self._canvas is not None:
                    self._canvas.draw_idle()
            except Exception:
                pass
            return

        if not self._drag_peak_id or self._drag_peak_artist is None or self._drag_peak_key is None:
            return
        try:
            if evt is None or getattr(evt, "inaxes", None) is None:
                return
            if evt.xdata is None or evt.ydata is None:
                return
        except Exception:
            return

        try:
            new_x = float(evt.xdata) + float(self._drag_dx)
            new_y = float(evt.ydata) + float(self._drag_dy)
            self._drag_peak_artist.set_position((new_x, new_y))
        except Exception:
            return

        # Persist on the dataset that owns this label
        try:
            d = self._get_dataset_by_key((str(self._drag_peak_key[0]), str(self._drag_peak_key[1])))
            if d is not None:
                try:
                    dx, dy = self._overlay_offset_for_key((str(self._drag_peak_key[0]), str(self._drag_peak_key[1])))
                except Exception:
                    dx, dy = 0.0, 0.0
                # Store positions in base (un-offset) coordinates so overlay offsets don't double-shift.
                d.peak_label_positions[str(self._drag_peak_id)] = (float(new_x) - float(dx), float(new_y) - float(dy))
        except Exception:
            pass

        try:
            if self._canvas is not None:
                self._canvas.draw_idle()
        except Exception:
            pass

    def _on_peak_drag_release(self, evt) -> None:
        self._drag_bond_key = None
        self._drag_bond_idx = None
        self._drag_bond_artist = None
        self._drag_bond_dx = 0.0
        self._drag_bond_dy = 0.0

        self._drag_peak_id = None
        self._drag_peak_key = None
        self._drag_peak_artist = None
        self._drag_dx = 0.0
        self._drag_dy = 0.0

    def _ensure_peak_menu(self) -> None:
        if self._peak_menu is not None:
            return
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Edit label…", command=self._peak_menu_edit_label)
        m.add_command(label="Hide peak", command=self._peak_menu_hide)
        m.add_command(label="Show peak", command=self._peak_menu_show)
        m.add_separator()
        m.add_command(label="Delete peak", command=self._peak_menu_delete)
        self._peak_menu = m

    def _ensure_bond_menu(self) -> None:
        if getattr(self, "_bond_menu", None) is not None:
            return
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Edit label…", command=self._bond_menu_edit_label)
        m.add_command(label="Toggle guide line", command=self._bond_menu_toggle_vline)
        m.add_separator()
        m.add_command(label="Delete label", command=self._bond_menu_delete)
        self._bond_menu = m

    def _get_workspace_by_id(self, ws_id: str) -> Optional[FTIRWorkspace]:
        try:
            if isinstance(getattr(self, "workspaces", None), dict):
                return (self.workspaces or {}).get(str(ws_id))
        except Exception:
            return None
        return None

    def _open_label_editor(
        self,
        *,
        kind: str,
        ws_id: Optional[str] = None,
        ds_id: Optional[str] = None,
        peak_id: Optional[str] = None,
        ann_index: Optional[int] = None,
    ) -> None:
        """Unified label editor for FTIR peak labels and bond labels."""

        kind = str(kind or "").strip().lower()
        if kind not in {"peak", "bond"}:
            return

        d = None
        if kind == "peak":
            if not (ws_id and ds_id and peak_id):
                return
            d = self._get_dataset_by_key((str(ws_id), str(ds_id)))
            if d is None:
                return

        ws = None
        ann = None
        if kind == "bond":
            if not ws_id or ann_index is None:
                return
            ws = self._get_workspace_by_id(str(ws_id)) or self._active_workspace()
            anns = list(getattr(ws, "bond_annotations", None) or [])
            if int(ann_index) < 0 or int(ann_index) >= len(anns):
                return
            if not isinstance(anns[int(ann_index)], FTIRBondAnnotation):
                return
            ann = anns[int(ann_index)]

        win = tk.Toplevel(self.app)
        win.title("Edit Label")
        try:
            win.transient(self.app)
        except Exception:
            pass

        text_var = tk.StringVar(value="")
        show_vline_var = tk.BooleanVar(value=False)
        text_color_var = tk.StringVar(value="#111111")
        line_color_var = tk.StringVar(value="#444444")
        fontsize_var = tk.IntVar(value=9)
        rotation_var = tk.IntVar(value=90)

        if kind == "peak" and d is not None:
            cur = str((getattr(d, "peak_label_overrides", {}) or {}).get(str(peak_id), ""))
            if not cur:
                # best-effort default
                try:
                    peak = next((p for p in (d.peaks or []) if isinstance(p, dict) and str(p.get("id")) == str(peak_id)), None)
                    s = dict(getattr(d, "peak_settings", None) or {})
                    fmt = str(s.get("label_fmt") or "{wn:.1f}")
                    if isinstance(peak, dict):
                        wn0 = float(peak.get("wn") or 0.0)
                        y0 = float(peak.get("y_display", peak.get("y", 0.0)) or 0.0)
                        prom0 = float(peak.get("prominence", 0.0) or 0.0)
                        if FTIRPeak is not None:
                            cur = format_peak_label(FTIRPeak(wn=wn0, y=y0, prominence=prom0), fmt=fmt)
                        else:
                            cur = f"{wn0:.1f}"
                except Exception:
                    cur = ""
            text_var.set(str(cur))

        if kind == "bond" and ann is not None:
            text_var.set(str(getattr(ann, "text", "") or ""))
            show_vline_var.set(bool(getattr(ann, "show_vline", False)))
            text_color_var.set(str(getattr(ann, "text_color", "#111111") or "#111111"))
            line_color_var.set(str(getattr(ann, "line_color", "#444444") or "#444444"))
            fontsize_var.set(int(getattr(ann, "fontsize", 9) or 9))
            rotation_var.set(int(getattr(ann, "rotation", 90) or 90))

        outer = ttk.Frame(win, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        ttk.Label(outer, text="Text").grid(row=0, column=0, sticky="w")
        e_text = ttk.Entry(outer, textvariable=text_var)
        e_text.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        row = 1
        if kind == "bond":
            ttk.Checkbutton(outer, text="Show vertical guide line", variable=show_vline_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
            row += 1

            def _pick(var: tk.StringVar, title: str) -> None:
                try:
                    c = colorchooser.askcolor(color=(var.get() or None), title=title, parent=win)[1]
                    if c:
                        var.set(str(c))
                except Exception:
                    return

            cc = ttk.Frame(outer)
            cc.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            ttk.Label(cc, text="Label color").pack(side=tk.LEFT)
            ttk.Entry(cc, textvariable=text_color_var, width=12).pack(side=tk.LEFT, padx=(8, 0))
            ttk.Button(cc, text="Pick…", command=lambda: _pick(text_color_var, "Pick label color")).pack(side=tk.LEFT, padx=(6, 0))
            ttk.Label(cc, text="Line color").pack(side=tk.LEFT, padx=(14, 0))
            ttk.Entry(cc, textvariable=line_color_var, width=12).pack(side=tk.LEFT, padx=(8, 0))
            ttk.Button(cc, text="Pick…", command=lambda: _pick(line_color_var, "Pick line color")).pack(side=tk.LEFT, padx=(6, 0))
            row += 1

            fsr = ttk.Frame(outer)
            fsr.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            ttk.Label(fsr, text="Font size").pack(side=tk.LEFT)
            ttk.Spinbox(fsr, from_=6, to=30, textvariable=fontsize_var, width=6).pack(side=tk.LEFT, padx=(8, 0))
            ttk.Label(fsr, text="Rotation").pack(side=tk.LEFT, padx=(14, 0))
            ttk.Spinbox(fsr, from_=-180, to=180, textvariable=rotation_var, width=6).pack(side=tk.LEFT, padx=(8, 0))
            row += 1

        btns = ttk.Frame(outer)
        btns.grid(row=row, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def _save() -> None:
            if kind == "peak" and d is not None:
                new_label = str(text_var.get() or "")
                if not new_label.strip():
                    try:
                        d.peak_label_overrides.pop(str(peak_id), None)
                    except Exception:
                        pass
                else:
                    try:
                        d.peak_label_overrides[str(peak_id)] = str(new_label)
                    except Exception:
                        pass
                self._schedule_redraw()
                try:
                    win.destroy()
                except Exception:
                    pass
                return

            if kind == "bond" and ann is not None and ws is not None:
                try:
                    ann.text = str(text_var.get() or "")
                    ann.show_vline = bool(show_vline_var.get())
                    ann.text_color = str(text_color_var.get() or "#111111")
                    ann.line_color = str(line_color_var.get() or "#444444")
                    ann.fontsize = int(fontsize_var.get())
                    ann.rotation = int(rotation_var.get())
                except Exception:
                    pass
                self._schedule_redraw()
                try:
                    win.destroy()
                except Exception:
                    pass
                return

        def _delete_bond() -> None:
            if kind != "bond" or ws is None or ann_index is None:
                return
            try:
                idx = int(ann_index)
                if 0 <= idx < len(ws.bond_annotations):
                    ws.bond_annotations.pop(idx)
            except Exception:
                pass
            self._schedule_redraw()
            try:
                win.destroy()
            except Exception:
                pass

        ttk.Button(btns, text="Save", command=_save, style="Primary.TButton").pack(side=tk.LEFT)
        if kind == "bond":
            ttk.Button(btns, text="Delete", command=_delete_bond, style="Secondary.TButton").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="Cancel", command=lambda: win.destroy(), style="Secondary.TButton").pack(side=tk.LEFT, padx=(8, 0))

        try:
            e_text.focus_set()
        except Exception:
            pass

    def _bond_menu_edit_label(self) -> None:
        info = getattr(self, "_bond_menu_last_info", None)
        if not info:
            return
        ws_id, _target, idx = str(info[0]), str(info[1]), int(info[2])
        self._open_label_editor(kind="bond", ws_id=ws_id, ann_index=int(idx))

    def _bond_menu_toggle_vline(self) -> None:
        info = getattr(self, "_bond_menu_last_info", None)
        if not info:
            return
        ws_id, _target, idx = str(info[0]), str(info[1]), int(info[2])
        ws = self._get_workspace_by_id(str(ws_id)) or self._active_workspace()
        try:
            ann = ws.bond_annotations[int(idx)]
            if isinstance(ann, FTIRBondAnnotation):
                ann.show_vline = (not bool(getattr(ann, "show_vline", False)))
        except Exception:
            return
        self._schedule_redraw()

    def _bond_menu_delete(self) -> None:
        info = getattr(self, "_bond_menu_last_info", None)
        if not info:
            return
        ws_id, _target, idx = str(info[0]), str(info[1]), int(info[2])
        ws = self._get_workspace_by_id(str(ws_id)) or self._active_workspace()
        try:
            if 0 <= int(idx) < len(ws.bond_annotations):
                ws.bond_annotations.pop(int(idx))
        except Exception:
            return
        self._schedule_redraw()

    def _on_peak_pick_event(self, evt) -> None:
        try:
            artist = evt.artist
            # Bond labels first
            binfo = (getattr(self, "_bond_artist_to_info", {}) or {}).get(artist)
            info = (self._peak_artist_to_info.get(artist) if not binfo else None)
        except Exception:
            binfo = None
            info = None

        me = getattr(evt, "mouseevent", None)
        if me is None:
            return

        if binfo:
            try:
                self._bond_menu_last_info = (str(binfo[0]), str(binfo[1]), int(binfo[2]))
            except Exception:
                self._bond_menu_last_info = None

            # Double-click: edit
            try:
                if bool(getattr(me, "dblclick", False)):
                    self._bond_menu_edit_label()
                    return
            except Exception:
                pass

            # Right-click: context menu
            try:
                if int(getattr(me, "button", 0) or 0) == 3:
                    self._ensure_bond_menu()
                    x_root = None
                    y_root = None
                    ge = getattr(me, "guiEvent", None)
                    try:
                        x_root = int(getattr(ge, "x_root"))
                        y_root = int(getattr(ge, "y_root"))
                    except Exception:
                        x_root = int(self.app.winfo_pointerx())
                        y_root = int(self.app.winfo_pointery())

                    try:
                        if self._bond_menu is not None:
                            self._bond_menu.tk_popup(x_root, y_root)
                    finally:
                        try:
                            if self._bond_menu is not None:
                                self._bond_menu.grab_release()
                        except Exception:
                            pass
                    return
            except Exception:
                pass

            return

        if not info:
            return

        self._peak_menu_last_info = (str(info[0]), str(info[1]), str(info[2]))

        # Double-click: edit label
        try:
            if bool(getattr(me, "dblclick", False)):
                self._peak_menu_edit_label()
                return
        except Exception:
            pass

        # Right-click: context menu
        try:
            if int(getattr(me, "button", 0) or 0) == 3:
                self._ensure_peak_menu()
                x_root = None
                y_root = None
                ge = getattr(me, "guiEvent", None)
                try:
                    x_root = int(getattr(ge, "x_root"))
                    y_root = int(getattr(ge, "y_root"))
                except Exception:
                    x_root = int(self.app.winfo_pointerx())
                    y_root = int(self.app.winfo_pointery())

                try:
                    if self._peak_menu is not None:
                        self._peak_menu.tk_popup(x_root, y_root)
                finally:
                    try:
                        if self._peak_menu is not None:
                            self._peak_menu.grab_release()
                    except Exception:
                        pass
        except Exception:
            pass

    def _peak_menu_edit_label(self) -> None:
        info = self._peak_menu_last_info
        if not info:
            return
        ws_id, ds_id, pid = str(info[0]), str(info[1]), str(info[2])
        self._open_label_editor(kind="peak", ws_id=ws_id, ds_id=ds_id, peak_id=pid)

    def _peak_menu_hide(self) -> None:
        info = self._peak_menu_last_info
        if not info:
            return
        ws_id, ds_id, pid = str(info[0]), str(info[1]), str(info[2])
        d = self._get_dataset_by_key((ws_id, ds_id))
        if d is None:
            return
        try:
            d.peak_suppressed.add(pid)
        except Exception:
            pass
        try:
            self._overlay_peak_display_ids_by_key.pop((ws_id, ds_id), None)
        except Exception:
            pass
        self._schedule_redraw()

    def _peak_menu_show(self) -> None:
        info = self._peak_menu_last_info
        if not info:
            return
        ws_id, ds_id, pid = str(info[0]), str(info[1]), str(info[2])
        d = self._get_dataset_by_key((ws_id, ds_id))
        if d is None:
            return
        try:
            d.peak_suppressed.discard(pid)
        except Exception:
            pass
        try:
            self._overlay_peak_display_ids_by_key.pop((ws_id, ds_id), None)
        except Exception:
            pass
        self._schedule_redraw()

    def _peak_menu_delete(self) -> None:
        info = self._peak_menu_last_info
        if not info:
            return
        ws_id, ds_id, pid = str(info[0]), str(info[1]), str(info[2])
        d = self._get_dataset_by_key((ws_id, ds_id))
        if d is None:
            return
        try:
            d.peaks = [p for p in (d.peaks or []) if not (isinstance(p, dict) and str(p.get("id")) == pid)]
        except Exception:
            pass
        try:
            d.peak_label_overrides.pop(pid, None)
        except Exception:
            pass
        try:
            d.peak_suppressed.discard(pid)
        except Exception:
            pass
        try:
            self._overlay_peak_display_ids_by_key.pop((ws_id, ds_id), None)
        except Exception:
            pass
        self._schedule_redraw()


class App(tb.Window):
    @staticmethod
    def _normalize_theme_name(value: Any) -> str:
        try:
            s = str(value or "").strip().lower()
        except Exception:
            s = ""
        if s in ("dark", "darkly"):
            return "darkly"
        if s in ("light", "flatly"):
            return "flatly"
        return "flatly"

    def __init__(self) -> None:
        settings = load_settings()
        theme_name = App._normalize_theme_name(settings.get("theme"))
        super().__init__(themename=theme_name)
        self._ui_settings = settings
        self._theme_var = tk.StringVar(value=theme_name)
        self.title(APP_NAME)
        # Wider than tall (landscape), but adapt to the user's screen.
        try:
            sw = int(self.winfo_screenwidth())
            sh = int(self.winfo_screenheight())
            w = max(1200, min(2400, int(sw * 0.96)))
            h = max(650, min(1100, int(sh * 0.78)))
            self.geometry(f"{w}x{h}")
        except Exception:
            self.geometry("1800x750")

        # Surface Tk callback exceptions to users (common when running without a console).
        def _report_callback_exception(exc, val, tb) -> None:
            msg = "".join(traceback.format_exception(exc, val, tb))

            # Also persist to a crash log so we can debug issues that only show up in UI flows.
            try:
                log_dir = self._get_session_root_dir()
                log_path = Path(log_dir) / "crash.log"
                with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                    f.write("\n" + ("=" * 72) + "\n")
                    f.write(datetime.datetime.now().isoformat() + "\n")
                    f.write(msg)
            except Exception:
                pass
            try:
                self._log("ERROR", "Unhandled UI exception", exc=msg)
            except Exception:
                pass
            try:
                messagebox.showerror("Error", msg, parent=self)
            except Exception:
                try:
                    print(msg)
                except Exception:
                    pass

        try:
            self.report_callback_exception = _report_callback_exception  # type: ignore[assignment]
        except Exception:
            pass

        self.mzml_path: Optional[Path] = None
        self._reader: Optional[mzml.MzML] = None
        self._index: Optional[MzMLTICIndex] = None

        # Multi-module workspace (LCMS + FTIR)
        self.workspace = Workspace()

        # Workspace/session root directory (used for microscopy outputs). Defaults to a per-run folder,
        # but if a workspace JSON is saved/loaded, we use its directory as the session root.
        self._last_workspace_json_path: Optional[Path] = None
        self.lcms_workspace_path: Optional[Path] = None
        self._last_lcms_workspace_dir: Optional[Path] = None
        self._recent_lcms_workspaces: List[Dict[str, Any]] = []
        self._recent_lcms_menu: Optional[tk.Menu] = None
        self._session_root_dir: Path = self._default_session_root_dir()

        # Module views (tabs)
        self._module_notebook: Optional[ttk.Notebook] = None
        self._lcms_view: Optional[LCMSView] = None
        self._ftir_view: Optional[FTIRView] = None
        self._microscopy_view: Optional[MicroscopyView] = None
        self._plate_reader_view: Optional[PlateReaderView] = None
        self._data_studio_view: Optional[DataStudioView] = None

        # Multi-mzML workspace: keep only ONE open reader (active session)
        self._sessions: Dict[str, MzMLSession] = {}
        self._session_order: List[str] = []
        self._active_session_id: Optional[str] = None
        self._active_reader: Optional[mzml.MzML] = None
        self._session_load_counter: int = 0

        # LCMS overlay mode (multi-file comparison)
        self._overlay_session: Optional[OverlaySession] = None
        self._overlay_prev_active_session_id: Optional[str] = None
        self._overlay_active_dataset_id: Optional[str] = None
        self._overlay_selected_ms_rt: Optional[float] = None
        self._overlay_mode_var = tk.StringVar(value="Stacked")
        self._overlay_scheme_var = tk.StringVar(value="Auto (Tableau)")
        self._overlay_single_hue_color: str = "#1f77b4"
        self._overlay_show_uv_var = tk.BooleanVar(value=True)
        self._overlay_stack_spectra_var = tk.BooleanVar(value=False)
        self._overlay_show_labels_all_var = tk.BooleanVar(value=False)
        self._overlay_multi_drag_var = tk.BooleanVar(value=False)
        self._overlay_persist_var = tk.BooleanVar(value=False)
        self._overlay_tic_cache: Dict[Tuple[str, str], Tuple[List[SpectrumMeta], np.ndarray, np.ndarray]] = {}
        self._overlay_readers: Dict[str, mzml.MzML] = {}
        self._overlay_rt_tolerance_min: float = 0.25
        self._overlay_status_by_sid: Dict[str, str] = {}
        self._overlay_legend_tree: Optional[ttk.Treeview] = None
        self._overlay_legend_frame: Optional[ttk.LabelFrame] = None

        # Event-style listeners (used by non-modal windows like EIC)
        self._ms_position_listeners: List[Callable[[Optional[float]], None]] = []
        self._active_session_listeners: List[Callable[[], None]] = []

        # EIC cache: (mzml_path, polarity_filter, rounded_target_mz, tol_value, unit) -> (rts, intensities)
        self._sim_cache: Dict[Tuple[str, str, float, float, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._sim_cache_lock = threading.Lock()
        self._sim_last_params: Optional[Tuple[float, float, str, bool]] = None

        # Multi-UV workspace
        self._uv_sessions: Dict[str, UVSession] = {}
        self._uv_order: List[str] = []
        self._active_uv_id: Optional[str] = None
        self._uv_load_counter: int = 0

        # UV labeling from MS peaks (persistent across RT picks)
        self.uv_label_from_ms_var = tk.BooleanVar(value=False)
        self.uv_label_from_ms_top_n_var = tk.IntVar(value=3)  # 2 or 3
        self.uv_label_min_conf_var = tk.DoubleVar(value=0.0)

        # Panel visibility
        self.show_tic_var = tk.BooleanVar(value=True)
        self.show_spectrum_var = tk.BooleanVar(value=True)
        self.show_uv_var = tk.BooleanVar(value=True)

        # Workspace panel widgets
        self._ws_tree: Optional[ttk.Treeview] = None
        self._ws_menu: Optional[tk.Menu] = None
        self._ws_ignore_select: bool = False

        self._uv_ws_tree: Optional[ttk.Treeview] = None
        self._uv_ws_menu: Optional[tk.Menu] = None
        self._uv_ws_link_btn: Optional[ttk.Button] = None

        # Sidebar: progressive disclosure
        self._advanced_expanded_var = tk.BooleanVar(value=False)
        self._adv_show_polymer_var = tk.BooleanVar(value=False)
        self._adv_show_confidence_var = tk.BooleanVar(value=False)
        self._adv_show_alignment_diag_var = tk.BooleanVar(value=False)
        self._advanced_body: Optional[ttk.Frame] = None
        self._advanced_toggle_btn: Optional[ttk.Button] = None
        self._sidebar_notebook: Optional[ttk.Notebook] = None

        # Keyboard-first helpers
        self._rt_jump_entry: Optional[ttk.Entry] = None
        self._diagnostics_win: Optional[tk.Toplevel] = None
        self._diagnostics_vars: Dict[str, tk.StringVar] = {}
        self._nonmodal_dialog_stack: List[tk.Toplevel] = []

        # Status/Diagnostics panel (always visible)
        self._ctx_mzml_var = tk.StringVar(value="(no mzML)")
        self._ctx_uv_var = tk.StringVar(value="(no UV linked)")
        self._ctx_spectrum_var = tk.StringVar(value="")
        self._ctx_rt_var = tk.StringVar(value="")
        self._ctx_pol_var = tk.StringVar(value="")
        self._ctx_warn_var = tk.StringVar(value="")
        self._show_diag_panel_var = tk.BooleanVar(value=False)

        self._diag_text: Optional[tk.Text] = None
        self._diag_filter_warnings_only_var = tk.BooleanVar(value=False)
        self._diag_log_records: List[Tuple[str, str, str]] = []  # (ts, level, msg)
        self._plot_paned: Optional[Any] = None
        self._diag_frame: Optional[tk.Widget] = None

        # Brand logo (keep reference to avoid GC)
        self._logo_photo: Optional[Any] = None

        # Background watermark logo (best-effort)
        self._bg_canvas: Optional[tk.Canvas] = None
        self._bg_logo_photo: Optional[Any] = None
        self._bg_logo_pil: Optional[Any] = None
        self._bg_redraw_after: Optional[str] = None
        self._bg_alpha: float = 0.10
        self._bg_target_fill: float = 0.92
        self._bg_height_cap: float = 0.90
        self._bg_margin_px: int = 18
        self._uv_annotations: List[Any] = []
        self._uv_ann_key_by_objid: Dict[int, Tuple[float, int]] = {}

        self.rt_unit_var = tk.StringVar(value="minutes")
        self.polarity_var = tk.StringVar(value="all")  # all|positive|negative

        # Custom titles/labels (editable in Graph Settings)
        self.tic_title_var = tk.StringVar(value="TIC (MS1)")
        self.tic_xlabel_var = tk.StringVar(value="Retention time (min)")
        self.tic_ylabel_var = tk.StringVar(value="Intensity")

        self.spec_title_var = tk.StringVar(value="Spectrum (MS1)")
        self.spec_xlabel_var = tk.StringVar(value="m/z")
        self.spec_ylabel_var = tk.StringVar(value="Intensity")

        self.uv_title_var = tk.StringVar(value="UV chromatogram")
        self.uv_xlabel_var = tk.StringVar(value="Retention time (min)")
        self.uv_ylabel_var = tk.StringVar(value="Signal")

        # Plot appearance settings
        self.title_fontsize_var = tk.IntVar(value=12)
        self.label_fontsize_var = tk.IntVar(value=10)
        self.tick_fontsize_var = tk.IntVar(value=9)
        self._matplotlib_bg_var = tk.StringVar(value="#f5f5f5")

        # Axis limits (blank = auto)
        self.tic_xlim_min_var = tk.StringVar(value="")
        self.tic_xlim_max_var = tk.StringVar(value="")
        self.tic_ylim_min_var = tk.StringVar(value="")
        self.tic_ylim_max_var = tk.StringVar(value="")

        self.spec_xlim_min_var = tk.StringVar(value="")
        self.spec_xlim_max_var = tk.StringVar(value="")
        self.spec_ylim_min_var = tk.StringVar(value="")
        self.spec_ylim_max_var = tk.StringVar(value="")

        self.uv_xlim_min_var = tk.StringVar(value="")
        self.uv_xlim_max_var = tk.StringVar(value="")
        self.uv_ylim_min_var = tk.StringVar(value="")
        self.uv_ylim_max_var = tk.StringVar(value="")

        # Spectrum annotation settings
        self.annotate_peaks_var = tk.BooleanVar(value=False)
        self.annotate_top_n_var = tk.IntVar(value=10)
        self.annotate_min_rel_var = tk.DoubleVar(value=0.05)  # fraction of max intensity
        self.drag_annotations_var = tk.BooleanVar(value=True)

        # Custom spectrum labels (stored per-spectrum/RT selection)
        self._custom_labels_by_spectrum: Dict[str, List[CustomLabel]] = {}

        # Polymer / reaction matching settings
        self.poly_enabled_var = tk.BooleanVar(value=False)
        self.poly_monomers_text_var = tk.StringVar(value="")
        self.poly_bond_delta_var = tk.DoubleVar(value=-18.010565)  # dehydration default
        self.poly_extra_delta_var = tk.DoubleVar(value=0.0)  # applied once per chain
        self.poly_adduct_mass_var = tk.DoubleVar(value=1.007276)  # +H default
        self.poly_decarb_enabled_var = tk.BooleanVar(value=False)
        self.poly_oxid_enabled_var = tk.BooleanVar(value=False)
        self.poly_cluster_enabled_var = tk.BooleanVar(value=False)
        self.poly_cluster_adduct_mass_var = tk.DoubleVar(value=-1.007276)  # -H default (2M-H)
        self.poly_adduct_na_var = tk.BooleanVar(value=False)
        self.poly_adduct_k_var = tk.BooleanVar(value=False)
        self.poly_adduct_cl_var = tk.BooleanVar(value=False)
        self.poly_adduct_formate_var = tk.BooleanVar(value=False)
        self.poly_adduct_acetate_var = tk.BooleanVar(value=False)
        self.poly_charges_var = tk.StringVar(value="1")
        self.poly_max_dp_var = tk.IntVar(value=12)
        self.poly_tol_value_var = tk.DoubleVar(value=0.02)
        self.poly_tol_unit_var = tk.StringVar(value="Da")  # Da|ppm
        self.poly_min_rel_int_var = tk.DoubleVar(value=0.01)  # fraction of max intensity

        self._busy_dialog: Optional[tk.Toplevel] = None
        self._busy_bar: Optional[ttk.Progressbar] = None

        self._fig: Optional[Figure] = None
        self._ax_tic = None
        self._ax_spec = None
        self._ax_uv = None
        self._canvas: Optional[FigureCanvasTkAgg] = None
        self._toolbar: Optional[NavigationToolbar2Tk] = None
        self._tic_line = None
        self._tic_marker = None
        self._uv_rt_marker = None
        self._mpl_cid: Optional[int] = None

        self._filtered_meta: List[SpectrumMeta] = []
        self._filtered_rts: Optional[np.ndarray] = None
        self._filtered_tics: Optional[np.ndarray] = None

        self._current_spectrum_mz: Optional[np.ndarray] = None
        self._current_spectrum_int: Optional[np.ndarray] = None
        self._current_spectrum_meta: Optional[SpectrumMeta] = None
        self._current_scan_index: Optional[int] = None

        self._spectrum_annotations: List[Any] = []
        self._active_annotation: Optional[Any] = None
        self._active_annotation_ax = None

        # Per-annotation metadata for edit/delete
        # key: id(ann) -> tuple describing what to edit/delete
        # - ('custom', spectrum_id, custom_index)
        # - ('auto', spectrum_id, mz_key)
        # - ('poly', spectrum_id, kind, mz_key)
        self._spec_ann_key_by_objid: Dict[int, Tuple[Any, ...]] = {}

        # Persistent overrides/suppression for auto/poly labels per spectrum
        # value None => suppressed; otherwise override text
        self._spec_label_overrides: Dict[str, Dict[Tuple[str, float], Optional[str]]] = {}

        # Navigation controls
        self._rt_jump_var = tk.StringVar(value="")
        self._last_polarity_filter: str = "all"

        # Find m/z navigation (session-only)
        self._mz_find_mz_var = tk.StringVar(value="")
        self._mz_find_tol_var = tk.StringVar(value="10")
        self._mz_find_unit_var = tk.StringVar(value="ppm")  # ppm|Da
        self._mz_find_mode_var = tk.StringVar(value="Nearest")  # Nearest|Forward|Backward
        self._mz_find_min_int_var = tk.StringVar(value="0")

        self._mz_find_cache: Dict[Tuple[str, str, float, float, str, float], List[int]] = {}
        self._mz_find_peak_cache: Dict[Tuple[str, str, float, float, str, float], Dict[int, Tuple[float, float, float]]] = {}

        self._last_mz_find_params: Optional[Tuple[float, float, str, str, float]] = None
        self._last_mz_find_last_hit_idx: Optional[int] = None

        self._mz_find_history: List[Tuple[float, float, str, str, float]] = []
        self._mz_find_history_var = tk.StringVar(value="")
        self._mz_find_history_map: Dict[str, Tuple[float, float, str, str, float]] = {}
        self._mz_find_history_combo: Optional[ttk.Combobox] = None

        self._mz_find_dialog: Optional[tk.Toplevel] = None

        self._alignment_diag_win: Optional[tk.Toplevel] = None
        self._instructions_win: Optional[tk.Toplevel] = None

        # Simple concurrency guard for EIC runs (best-effort cancellation by token)
        self._sim_run_token: int = 0

        self._mz_find_highlight_artist: Optional[Any] = None
        self._mz_find_highlight_target_mz: Optional[float] = None
        self._mz_find_highlight_tol_da: Optional[float] = None

        self._selected_rt_min: Optional[float] = None

        # TIC RT region selection (LCMS)
        self.tic_region_select_var = tk.BooleanVar(value=False)
        self._tic_region_dragging: bool = False
        self._tic_region_start_rt: Optional[float] = None
        self._tic_region_end_rt: Optional[float] = None
        self._tic_region_active_rt: Optional[Tuple[float, float]] = None
        self._tic_region_span_artist: Optional[Any] = None
        self._tic_region_clear_btn: Optional[ttk.Button] = None
        self._region_force_poly_match: bool = False

        # LCMS-only workspace restore context
        self._lcms_workspace_restore_ctx: Optional[Dict[str, Any]] = None

        # Alignment between UV RT axis and MS RT axis (minutes).
        # If UV is earlier than MS by +0.125 min, then: MS_RT ≈ UV_RT + offset.
        self._uv_ms_rt_offset_min: float = 0.125
        self.uv_ms_rt_offset_var = tk.StringVar(value=f"{self._uv_ms_rt_offset_min:.3f}")

        # Optional auto-alignment anchors (piecewise linear mapping).
        self.uv_ms_align_enabled_var = tk.BooleanVar(value=False)
        self._uv_ms_align_uv_rts: Optional[np.ndarray] = None
        self._uv_ms_align_ms_rts: Optional[np.ndarray] = None

        self._build_ui()

        # Ensure status reflects active tab.
        try:
            self._update_status_by_tab()
        except Exception:
            pass

    def destroy(self) -> None:
        try:
            # Persist per-session recall fields.
            self._save_active_session_state()
        except Exception:
            pass
        try:
            if self._active_reader is not None:
                self._active_reader.close()
            elif self._reader is not None:
                self._reader.close()
        except Exception:
            pass
        self._active_reader = None
        self._reader = None
        super().destroy()

    # --- session root (microscopy outputs) ---

    def _default_session_root_dir(self) -> Path:
        """Default live-analysis root.

        We do not create a new per-run session folder under AppData. If the user doesn't save/load
        a workspace JSON, outputs default under the current working directory.
        """
        try:
            root = Path.cwd()
        except Exception:
            root = Path(os.environ.get("USERPROFILE") or os.environ.get("APPDATA") or ".")
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return root

    def _set_session_root_dir_from_workspace_path(self, workspace_json_path: Path) -> None:
        """Set session root to the directory containing the workspace JSON."""
        try:
            p = Path(workspace_json_path).expanduser().resolve()
        except Exception:
            p = Path(workspace_json_path)
        self._last_workspace_json_path = p
        try:
            self._session_root_dir = p.parent
        except Exception:
            self._session_root_dir = self._default_session_root_dir()

    def _get_session_root_dir(self) -> Path:
        """Return the current session root directory (created if missing)."""
        root = getattr(self, "_session_root_dir", None)
        if root is None:
            root = self._default_session_root_dir()
            self._session_root_dir = root
        try:
            Path(root).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return Path(root)

    def _update_status_by_tab(self) -> None:
        nb = getattr(self, "_module_notebook", None)
        if nb is None:
            self._update_status_current()
            return

        try:
            tab_id = nb.select()
            tab_text = str(nb.tab(tab_id, "text") or "")
        except Exception:
            tab_text = ""

        if tab_text.strip().lower() == "microscopy":
            mv = getattr(self, "_microscopy_view", None)
            if mv is not None:
                try:
                    self._set_status(str(mv.status_text()))
                    return
                except Exception:
                    pass

        if tab_text.strip().lower() == "plate reader":
            pv = getattr(self, "_plate_reader_view", None)
            if pv is not None:
                try:
                    self._set_status(str(pv.status_text()))
                    return
                except Exception:
                    pass

        if tab_text.strip().lower() == "data studio":
            dv = getattr(self, "_data_studio_view", None)
            if dv is not None:
                try:
                    self._set_status(str(dv.status_text()))
                    return
                except Exception:
                    pass

        # Default: LCMS-style status string (existing behavior)
        self._update_status_current()

    def _build_ui(self) -> None:
        self._apply_theme()

        self._load_brand_logo()

        # Watermark background (behind all other widgets)
        try:
            bg = tk.Canvas(self, highlightthickness=0, bd=0)
            bg.place(x=0, y=0, relwidth=1, relheight=1)
            bg.lower()
            self._bg_canvas = bg
            self.bind("<Configure>", self._on_root_configure)
        except Exception:
            self._bg_canvas = None

        self._build_menu()

        # Root layout: content / status (header removed)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        content = ttk.Frame(self, padding=(10, 10, 10, 10))
        content.grid(row=0, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        nb = ttk.Notebook(content)
        nb.grid(row=0, column=0, sticky="nsew")
        self._module_notebook = nb

        lcms_tab = ttk.Frame(nb)
        ftir_tab = ttk.Frame(nb)
        microscopy_tab = ttk.Frame(nb)
        plate_reader_tab = ttk.Frame(nb)
        data_studio_tab = ttk.Frame(nb)
        nb.add(lcms_tab, text="LCMS")
        nb.add(ftir_tab, text="FTIR")
        nb.add(microscopy_tab, text="Microscopy")
        nb.add(plate_reader_tab, text="Plate Reader")
        nb.add(data_studio_tab, text="Data Studio")

        try:
            nb.bind("<<NotebookTabChanged>>", lambda _e: self._update_status_by_tab(), add=True)
        except Exception:
            pass

        # Build module UIs
        self._lcms_view = LCMSView(lcms_tab, self, self.workspace)
        self._lcms_view.pack(fill=tk.BOTH, expand=True)

        self._ftir_view = FTIRView(ftir_tab, self, self.workspace)
        self._ftir_view.pack(fill=tk.BOTH, expand=True)

        self._microscopy_view = MicroscopyView(microscopy_tab, self, self.workspace)
        self._microscopy_view.pack(fill=tk.BOTH, expand=True)

        self._plate_reader_view = PlateReaderView(plate_reader_tab, self, self.workspace)
        self._plate_reader_view.pack(fill=tk.BOTH, expand=True)

        self._data_studio_view = DataStudioView(data_studio_tab, self, self.workspace)
        self._data_studio_view.pack(fill=tk.BOTH, expand=True)

        status = ttk.Label(
            self,
            text="Open an mzML file to begin",
            relief="sunken",
            anchor="w",
            padding=(8, 4),
        )
        status.grid(row=2, column=0, sticky="ew")
        self._status = status

        # Shortcuts remain global, but route to active tab where appropriate.
        self._bind_shortcuts()

    def _build_diagnostics_panel(self, parent: tk.Widget) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ctx = ttk.LabelFrame(parent, text="Current Context", padding=8)
        ctx.grid(row=0, column=0, sticky="ew")
        ctx.columnconfigure(1, weight=1)

        ttk.Label(ctx, text="mzML:").grid(row=0, column=0, sticky="w")
        ttk.Label(ctx, textvariable=self._ctx_mzml_var).grid(row=0, column=1, sticky="w")
        ttk.Label(ctx, text="UV:").grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(ctx, textvariable=self._ctx_uv_var).grid(row=1, column=1, sticky="w", pady=(2, 0))
        ttk.Label(ctx, text="Spectrum:").grid(row=2, column=0, sticky="w", pady=(2, 0))
        ttk.Label(ctx, textvariable=self._ctx_spectrum_var).grid(row=2, column=1, sticky="w", pady=(2, 0))
        ttk.Label(ctx, text="RT:").grid(row=3, column=0, sticky="w", pady=(2, 0))
        ttk.Label(ctx, textvariable=self._ctx_rt_var).grid(row=3, column=1, sticky="w", pady=(2, 0))
        ttk.Label(ctx, text="Polarity:").grid(row=4, column=0, sticky="w", pady=(2, 0))
        ttk.Label(ctx, textvariable=self._ctx_pol_var).grid(row=4, column=1, sticky="w", pady=(2, 0))

        warn_lbl = ttk.Label(ctx, textvariable=self._ctx_warn_var, wraplength=520, justify="left")
        warn_lbl.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        logf = ttk.LabelFrame(parent, text="Log", padding=8)
        logf.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(1, weight=1)

        top = ttk.Frame(logf)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        cb = ttk.Checkbutton(
            top,
            text="Warnings only",
            variable=self._diag_filter_warnings_only_var,
            command=self._diag_refresh_text,
        )
        cb.grid(row=0, column=0, sticky="w")

        btns = ttk.Frame(top)
        btns.grid(row=0, column=1, sticky="e")
        ttk.Button(btns, text="Save…", command=self._diag_save).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="Copy", command=self._diag_copy).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(btns, text="Clear", command=self._diag_clear).grid(row=0, column=2)

        text_frame = ttk.Frame(logf)
        text_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        ysb = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        ysb.grid(row=0, column=1, sticky="ns")
        txt = tk.Text(text_frame, height=7, wrap="word", yscrollcommand=ysb.set)
        txt.grid(row=0, column=0, sticky="nsew")
        ysb.configure(command=txt.yview)
        try:
            txt.configure(state="disabled")
        except Exception:
            pass

        self._diag_text = txt

    def _apply_diag_panel_visibility(self) -> None:
        paned = getattr(self, "_plot_paned", None)
        frame = getattr(self, "_diag_frame", None)
        if paned is None or frame is None:
            return
        try:
            panes = set(paned.panes())
        except Exception:
            panes = set()
        want = bool(self._show_diag_panel_var.get())
        try:
            if want:
                if str(frame) not in panes:
                    paned.add(frame, weight=1)
            else:
                if str(frame) in panes:
                    paned.forget(frame)
        except Exception:
            pass

    def _log(self, level: str, message: str, exc: Optional[Union[BaseException, str]] = None) -> None:
        lvl = (level or "INFO").upper()
        if exc is not None:
            if isinstance(exc, str):
                msg = f"{message}\n{exc}"
            else:
                msg = f"{message}\n{exc!r}"
        else:
            msg = str(message)
        self._diag_append(lvl, msg)

    def _warn(self, message: str) -> None:
        try:
            self._ctx_warn_var.set(f"Warning: {message}")
        except Exception:
            pass
        self._diag_append("WARN", str(message))

    def _diag_append(self, level: str, message: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        lvl = (level or "INFO").upper()
        msg = str(message)
        self._diag_log_records.append((ts, lvl, msg))
        self._diag_refresh_text(append_last=True)

    def _diag_refresh_text(self, append_last: bool = False) -> None:
        txt = self._diag_text
        if txt is None:
            return
        warnings_only = bool(self._diag_filter_warnings_only_var.get())

        def include(rec: Tuple[str, str, str]) -> bool:
            if not warnings_only:
                return True
            return rec[1] in ("WARN", "WARNING", "ERROR")

        try:
            txt.configure(state="normal")
        except Exception:
            return

        if append_last and len(self._diag_log_records) >= 1:
            rec = self._diag_log_records[-1]
            if include(rec):
                txt.insert("end", f"[{rec[0]}] [{rec[1]}] {rec[2]}\n")
                txt.see("end")
            try:
                txt.configure(state="disabled")
            except Exception:
                pass
            return

        try:
            txt.delete("1.0", "end")
            for rec in self._diag_log_records:
                if include(rec):
                    txt.insert("end", f"[{rec[0]}] [{rec[1]}] {rec[2]}\n")
            txt.see("end")
        finally:
            try:
                txt.configure(state="disabled")
            except Exception:
                pass

    def _diag_clear(self) -> None:
        self._diag_log_records.clear()
        self._diag_refresh_text(append_last=False)

    def _diag_copy(self) -> None:
        txt = self._diag_text
        if txt is None:
            return
        try:
            data = txt.get("1.0", "end-1c")
        except Exception:
            return
        if not data.strip():
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(data)
        except Exception:
            pass

    def _diag_save(self) -> None:
        txt = self._diag_text
        if txt is None:
            return
        try:
            data = txt.get("1.0", "end-1c")
        except Exception:
            return
        if not data.strip():
            return
        try:
            path = filedialog.asksaveasfilename(
                parent=self,
                title="Save diagnostics log",
                defaultextension=".txt",
                filetypes=[("Text", "*.txt"), ("All files", "*.*")],
            )
        except Exception:
            path = ""
        if not path:
            return
        try:
            Path(path).write_text(data, encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Save log", f"Failed to save log:\n\n{exc}", parent=self)

    def _update_current_context_panel(self) -> None:
        try:
            mzml = self._active_session_display_name() if getattr(self, "_active_session_id", None) else (self.mzml_path.name if self.mzml_path else "(no mzML)")
        except Exception:
            mzml = "(no mzML)"
        uv_sess = self._active_uv_session()
        uv = uv_sess.path.name if uv_sess is not None else "(no UV linked)"
        pol = (self.polarity_var.get() or "all").strip()

        try:
            self._ctx_mzml_var.set(str(mzml))
            self._ctx_uv_var.set(str(uv))
            self._ctx_pol_var.set(str(pol))
        except Exception:
            pass

        if self._current_spectrum_meta is None:
            try:
                self._ctx_spectrum_var.set("")
                self._ctx_rt_var.set("")
            except Exception:
                pass
        else:
            meta = self._current_spectrum_meta
            sid = getattr(meta, "spectrum_id", None)
            try:
                if sid:
                    self._ctx_spectrum_var.set(str(sid))
                else:
                    self._ctx_spectrum_var.set("(spectrum loaded)")
                self._ctx_rt_var.set(f"{float(meta.rt_min):.4f} min")
            except Exception:
                pass

        # A small warning summary line (populated by `_warn`)
        try:
            if not (self._ctx_warn_var.get() or "").strip():
                self._ctx_warn_var.set("")
        except Exception:
            pass

    def _load_brand_logo(self) -> None:
        """Best-effort load of assets/lab_logo.png; never raises."""
        self._logo_photo = None
        self._bg_logo_pil = None
        logo_path = _resolve_logo_path()
        if logo_path is None:
            return
        try:
            img = tk.PhotoImage(file=str(logo_path))
        except Exception:
            return

        # Keep it a reasonable size for the header (subsample only supports integers)
        try:
            h = int(img.height())
            if h > 64:
                factor = max(2, int(np.ceil(h / 64.0)))
                img = img.subsample(factor, factor)
        except Exception:
            pass

        self._logo_photo = img
        try:
            self.iconphoto(True, img)
        except Exception:
            pass

        # Also load a Pillow RGBA image for watermark resizing/alpha.
        try:
            from PIL import Image

            self._bg_logo_pil = Image.open(str(logo_path)).convert("RGBA")
        except Exception:
            self._bg_logo_pil = None

        # Initial watermark draw (if canvas exists)
        self._schedule_bg_redraw()

    def _on_root_configure(self, _evt) -> None:
        # Debounced redraw on window resize.
        self._schedule_bg_redraw()

    def _schedule_bg_redraw(self) -> None:
        if self._bg_canvas is None:
            return
        try:
            if self._bg_redraw_after is not None:
                self.after_cancel(self._bg_redraw_after)
        except Exception:
            pass
        try:
            self._bg_redraw_after = self.after(120, self._redraw_bg_watermark)
        except Exception:
            self._bg_redraw_after = None

    def _redraw_bg_watermark(self) -> None:
        if self._bg_canvas is None:
            return

        c = self._bg_canvas
        try:
            c.delete("all")
        except Exception:
            return

        # Use Pillow for alpha/resizing; if not available, skip silently.
        pil_img = getattr(self, "_bg_logo_pil", None)
        if pil_img is None:
            return

        try:
            from PIL import Image, ImageTk
        except Exception:
            return

        try:
            w = max(1, int(c.winfo_width()))
            h = max(1, int(c.winfo_height()))
        except Exception:
            return

        try:
            iw, ih = pil_img.size
            if iw <= 0 or ih <= 0:
                return

            # Target size: mostly fill width, but keep inside window.
            target_w = max(1, int(self._bg_target_fill * float(w)))
            scale = float(target_w) / float(iw)
            target_h = int(float(ih) * scale)
            if target_h > int(self._bg_height_cap * float(h)):
                scale = (self._bg_height_cap * float(h)) / float(ih)
                target_w = int(float(iw) * scale)
                target_h = int(float(ih) * scale)

            target_w = max(1, int(target_w))
            target_h = max(1, int(target_h))

            im = pil_img.resize((target_w, target_h), resample=Image.LANCZOS)

            # Apply global alpha multiplier.
            alpha_mul = max(0.02, min(0.35, float(self._bg_alpha)))
            r, g, b, a = im.split()
            a = a.point(lambda p: int(max(0, min(255, round(p * alpha_mul)))))
            im.putalpha(a)

            photo = ImageTk.PhotoImage(im)
            self._bg_logo_photo = photo  # keep reference

            x = int(self._bg_margin_px)
            y = int(self._bg_margin_px)
            c.create_image(x, y, anchor="nw", image=photo)
        except Exception:
            # Never let watermark break the UI.
            self._bg_logo_photo = None
            return

    def _draw_mpl_watermark(self) -> None:
        """Draw a translucent logo inside the Matplotlib figure (top-left)."""
        if self._fig is None:
            return
        pil_img = getattr(self, "_bg_logo_pil", None)
        if pil_img is None:
            return
        try:
            from PIL import Image
        except Exception:
            return

        # Remove previous watermark artists
        try:
            prev = getattr(self, "_mpl_watermark_artists", [])
            for a in prev:
                try:
                    a.remove()
                except Exception:
                    pass
        except Exception:
            pass
        self._mpl_watermark_artists = []

        try:
            fw, fh = self._fig.get_size_inches()
            dpi = float(getattr(self._fig, "dpi", 100) or 100)
            px_w = max(1, int(fw * dpi))
            px_h = max(1, int(fh * dpi))
        except Exception:
            return

        try:
            iw, ih = pil_img.size
            if iw <= 0 or ih <= 0:
                return

            # Similar sizing logic as Tk watermark, but in figure pixels
            target_w = max(1, int(self._bg_target_fill * float(px_w)))
            scale = float(target_w) / float(iw)
            target_h = int(float(ih) * scale)
            if target_h > int(self._bg_height_cap * float(px_h)):
                scale = (self._bg_height_cap * float(px_h)) / float(ih)
                target_w = int(float(iw) * scale)
                target_h = int(float(ih) * scale)

            target_w = max(1, int(target_w))
            target_h = max(1, int(target_h))

            im = pil_img.resize((target_w, target_h), resample=Image.LANCZOS)
            alpha_mul = max(0.02, min(0.35, float(self._bg_alpha)))
            r, g, b, a = im.split()
            a = a.point(lambda p: int(max(0, min(255, round(p * alpha_mul)))))
            im.putalpha(a)

            # Add as figure image in normalized coords
            arr = np.asarray(im)
            x0 = float(self._bg_margin_px) / float(px_w)
            y0 = 1.0 - (float(self._bg_margin_px) / float(px_h))
            w0 = float(target_w) / float(px_w)
            h0 = float(target_h) / float(px_h)

            axw = self._fig.add_axes([x0, y0 - h0, w0, h0], zorder=-10)
            axw.axis("off")
            artist = axw.imshow(arr)
            self._mpl_watermark_artists = [artist, axw]
        except Exception:
            self._mpl_watermark_artists = []
            return

    def _apply_theme(self) -> None:
        self._apply_brand_theme()

    def _apply_brand_theme(self) -> None:
        """Best-effort ttk styling that layers on top of ttkbootstrap themes."""
        style = getattr(self, "style", None) or ttk.Style(self)
        colors = getattr(style, "colors", None)

        bg = getattr(colors, "bg", BG_LIGHT)
        fg = getattr(colors, "fg", TEXT_DARK)
        primary = getattr(colors, "primary", PRIMARY_TEAL)

        try:
            self.configure(bg=bg)
        except Exception:
            pass

        # Base widget styles (padding + minimal colors)
        try:
            style.configure("TFrame", background=bg)
            style.configure("TLabel", background=bg, foreground=fg)
            style.configure("TButton", padding=(10, 7))
            style.configure("TCheckbutton", background=bg, foreground=fg)
            style.configure("TRadiobutton", background=bg, foreground=fg)
        except Exception:
            pass

        # Labelframe + label
        try:
            style.configure("TLabelframe", background=bg)
            style.configure("TLabelframe.Label", background=bg, foreground=fg)
        except Exception:
            pass

        # Treeview row height + heading font
        try:
            style.configure("Treeview", rowheight=26)
        except Exception:
            pass

        try:
            import tkinter.font as tkfont

            heading_font = tkfont.nametofont("TkDefaultFont").copy()
            heading_font.configure(weight="bold")
            style.configure("Treeview.Heading", font=heading_font)
        except Exception:
            pass

        # Minimal padding only; keep theme colors
        try:
            style.configure("TButton", padding=(10, 7))
            style.configure("TMenubutton", padding=(10, 7))
            style.configure("Header.TLabel", background=bg, foreground=fg)
        except Exception:
            pass

        # Header font (best-effort)
        try:
            import tkinter.font as tkfont

            header_font = tkfont.nametofont("TkDefaultFont").copy()
            header_font.configure(size=max(12, int(header_font.cget("size")) + 4), weight="bold")
            style.configure("Header.TLabel", font=header_font)
        except Exception:
            pass

    def _set_theme(self, theme_name: str) -> None:
        theme_name = App._normalize_theme_name(theme_name)
        try:
            self.style.theme_use(theme_name)
        except Exception:
            return

        try:
            self._theme_var.set(theme_name)
        except Exception:
            pass

        try:
            self._ui_settings["theme"] = theme_name
            save_settings(self._ui_settings)
        except Exception:
            pass

        try:
            self._apply_theme()
        except Exception:
            pass


    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open…\tCtrl+O", command=self._dispatch_open)
        file_menu.add_command(label="Open multiple…\tCtrl+Shift+O", command=self._dispatch_open_many)
        file_menu.add_command(label="Add UV CSV…\tCtrl+U", command=self._dispatch_add_uv_single)
        file_menu.add_command(label="Add multiple UV CSV…\tCtrl+Shift+U", command=self._dispatch_add_uv_many)
        file_menu.add_separator()
        file_menu.add_command(label="Load FTIR…", command=lambda: (self._ftir_view.load_ftir_dialog() if self._ftir_view else None))
        file_menu.add_command(label="Save FTIR Plot…", command=lambda: (self._ftir_view.save_plot_dialog() if self._ftir_view else None))
        file_menu.add_command(label="Save FTIR Workspace…", command=self._save_ftir_workspace)
        file_menu.add_command(label="Load FTIR Workspace…", command=self._load_ftir_workspace)
        file_menu.add_command(label="Save Microscopy Workspace…", command=self._save_microscopy_workspace)
        file_menu.add_command(label="Load Microscopy Workspace…", command=self._load_microscopy_workspace)
        file_menu.add_separator()
        file_menu.add_command(label="Save LCMS Workspace…\tCtrl+Shift+S", command=self._save_workspace)
        file_menu.add_command(label="Load LCMS Workspace…\tCtrl+Shift+L", command=self._load_workspace)
        file_menu.add_command(label="Reveal LCMS Workspace in Explorer", command=self._reveal_lcms_workspace_in_explorer)
        recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent LCMS Workspaces", menu=recent_menu)
        self._recent_lcms_menu = recent_menu
        self._refresh_recent_lcms_menu()
        file_menu.add_separator()
        file_menu.add_command(label="Close mzML (remove active)\tCtrl+W", command=lambda: self._dispatch_lcms_only(self._close_active_session))
        file_menu.add_separator()
        file_menu.add_command(label="Export primary…\tCtrl+E", command=self._dispatch_export_primary)
        file_menu.add_command(label="Export All Labels (Excel)…", command=self._export_all_labels_xlsx)
        file_menu.add_command(label="Save TIC Plot…", command=self._save_tic_plot)
        file_menu.add_command(label="Save Spectrum Plot…", command=self._save_spectrum_plot)
        file_menu.add_command(label="Save UV Plot…", command=self._save_uv_plot)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Edit Graph…\tCtrl+G", command=self._open_graph_settings)
        tools_menu.add_command(label="Annotate Peaks…\tCtrl+P", command=self._open_annotation_settings)
        tools_menu.add_command(label="Custom Labels…\tCtrl+L", command=self._open_custom_labels)
        tools_menu.add_command(label="Polymer Match…\tCtrl+M", command=self._open_polymer_match)
        tools_menu.add_separator()
        tools_menu.add_command(label="Reset View\tCtrl+0", command=self._reset_view_all)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        theme_menu = tk.Menu(view_menu, tearoff=0)
        theme_menu.add_radiobutton(
            label="Dark",
            value="darkly",
            variable=self._theme_var,
            command=lambda: self._set_theme("darkly"),
        )
        theme_menu.add_radiobutton(
            label="Light",
            value="flatly",
            variable=self._theme_var,
            command=lambda: self._set_theme("flatly"),
        )
        view_menu.add_cascade(label="Theme", menu=theme_menu)
        view_menu.add_separator()
        view_menu.add_checkbutton(
            label="Show Diagnostics Panel",
            variable=self._show_diag_panel_var,
            command=self._apply_diag_panel_visibility,
        )

        def _pick_matplotlib_bg() -> None:
            try:
                initial = (self._matplotlib_bg_var.get() or "").strip() or "#f5f5f5"
            except Exception:
                initial = "#f5f5f5"
            try:
                _rgb, hex_color = colorchooser.askcolor(title="Matplotlib background", initialcolor=initial, parent=self)
            except Exception:
                hex_color = None
            if hex_color:
                try:
                    self._matplotlib_bg_var.set(str(hex_color))
                except Exception:
                    pass
                self._apply_matplotlib_bg_current()

        view_menu.add_command(label="Matplotlib Background…", command=_pick_matplotlib_bg)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Shortcuts", command=self._show_shortcuts)
        help_menu.add_command(label="Instructions / User Guide…\tF1", command=self._open_instructions_window)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _show_shortcuts(self) -> None:
        messagebox.showinfo(
            "Shortcuts",
            "Ctrl+O  Open (routes to active tab)\n"
            "Ctrl+Shift+O  Open multiple (LCMS)\n"
            "Ctrl+U  Add UV CSV (LCMS)\n"
            "Ctrl+Shift+U  Add multiple UV CSV (LCMS)\n"
            "Ctrl+Shift+S  Save LCMS workspace\n"
            "Ctrl+Shift+L  Load LCMS workspace\n"
            "Ctrl+W  Close mzML (remove active)\n"
            "Ctrl+E  Export primary (LCMS export / FTIR save plot)\n"
            "Ctrl+F  Find m/z\n"
            "Ctrl+G  Graph settings\n"
            "Ctrl+P  Peak annotation settings\n"
            "Ctrl+L  Custom labels\n"
            "Ctrl+M  Polymer match\n"
            "Ctrl+0  Reset view\n"
            "Ctrl+Alt+O  Overlay selected datasets\n"
            "Ctrl+Alt+C  Clear overlay\n"
            "Ctrl+Alt+Left/Right  Cycle overlay dataset\n"
            "F1  Instructions / User Guide\n"
            "Left/Right  Prev/Next spectrum\n"
            "Home/End  First/Last spectrum\n"
            "\nTip: Right-click (or double-click) a label to edit/delete.",
            parent=self,
        )

    def _show_about(self) -> None:
        logo_path = _resolve_logo_path()
        logo_note = str(logo_path) if logo_path is not None else "(not found)"
        messagebox.showinfo(
            "About",
            f"{APP_NAME}\n"
            f"Version {APP_VERSION}\n\n"
            "Lab theme enabled.\n"
            f"Logo: {logo_note}",
            parent=self,
        )

    def _active_module_name(self) -> str:
        nb = self._module_notebook
        if nb is None:
            return "LCMS"
        try:
            idx = int(nb.index("current"))
            txt = str(nb.tab(idx, "text") or "").strip()
            return txt or "LCMS"
        except Exception:
            return "LCMS"

    def _is_lcms_active(self) -> bool:
        return self._active_module_name().upper() == "LCMS"

    def _is_ftir_active(self) -> bool:
        return self._active_module_name().upper() == "FTIR"

    def _dispatch_open(self) -> None:
        if self._is_ftir_active() and self._ftir_view is not None:
            self._ftir_view.load_ftir_dialog()
            return
        self._open_mzml()

    def _dispatch_open_many(self) -> None:
        if self._is_ftir_active() and self._ftir_view is not None:
            # Minimal: FTIR supports one-at-a-time loading for now.
            self._ftir_view.load_ftir_dialog()
            return
        self._open_mzml_many()

    def _dispatch_add_uv_single(self) -> None:
        if not self._is_lcms_active():
            return
        self._open_uv_csv_single()

    def _dispatch_add_uv_many(self) -> None:
        if not self._is_lcms_active():
            return
        self._open_uv_csv_many()

    def _dispatch_export_primary(self) -> None:
        if self._is_ftir_active() and self._ftir_view is not None:
            self._ftir_view.save_plot_dialog()
            return
        self._export_spectrum_csv()

    def _dispatch_lcms_only(self, fn: Callable[[], None]) -> None:
        if not self._is_lcms_active():
            return
        fn()

    def _dispatch_cycle(self, delta: int) -> None:
        if self._is_ftir_active() and self._ftir_view is not None:
            self._ftir_view.cycle_active(int(delta))
            return
        self._cycle_active_session(int(delta))

    def _add_ftir_from_path_async(
        self,
        csv_path: Path,
        *,
        make_active: bool,
        restore_ctx: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
        name: Optional[str] = None,
        y_mode: Optional[str] = None,
        x_units: Optional[str] = None,
        y_units: Optional[str] = None,
        peak_settings: Optional[Dict[str, Any]] = None,
        peaks: Optional[List[Dict[str, Any]]] = None,
        peak_label_overrides: Optional[Dict[str, str]] = None,
        peak_suppressed: Optional[List[str]] = None,
        peak_label_positions: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = Path(csv_path).expanduser().resolve()
        if not p.exists():
            messagebox.showerror("FTIR", f"File not found:\n{p}", parent=self)
            ctx = (restore_ctx if isinstance(restore_ctx, dict) else getattr(self, "_workspace_restore_ctx", None))
            if isinstance(ctx, dict):
                try:
                    ctx["done_ftir"] = int(ctx.get("done_ftir", 0)) + 1
                except Exception:
                    ctx["done_ftir"] = 1
                if restore_ctx is not None:
                    self._maybe_finalize_ftir_only_restore(ctx)
                else:
                    self._maybe_finalize_workspace_restore()
            return

        try:
            for d in self.workspace.ftir_datasets:
                if getattr(d, "path", None) == p:
                    if make_active:
                        self.workspace.active_ftir_id = str(getattr(d, "id", ""))
                    try:
                        if self._ftir_view is not None:
                            self._ftir_view.refresh_from_workspace(select_active=True)
                    except Exception:
                        pass
                    ctx = (restore_ctx if isinstance(restore_ctx, dict) else getattr(self, "_workspace_restore_ctx", None))
                    if isinstance(ctx, dict):
                        try:
                            ctx["done_ftir"] = int(ctx.get("done_ftir", 0)) + 1
                        except Exception:
                            ctx["done_ftir"] = 1
                        if restore_ctx is not None:
                            self._maybe_finalize_ftir_only_restore(ctx)
                        else:
                            self._maybe_finalize_workspace_restore()
                    return
        except Exception:
            pass

        self._set_status(f"Loading FTIR: {p.name}")

        def infer_from_meta(meta: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
            x_units = None if not isinstance(meta, dict) else meta.get("XUNITS")
            y_units = None if not isinstance(meta, dict) else meta.get("YUNITS")
            y_mode_inferred = "absorbance"
            try:
                yu = (str(y_units).strip().lower() if y_units is not None else "")
                if "trans" in yu:
                    y_mode_inferred = "transmittance"
                elif "abs" in yu:
                    y_mode_inferred = "absorbance"
            except Exception:
                y_mode_inferred = "absorbance"
            return str(y_mode_inferred), (None if x_units is None else str(x_units)), (None if y_units is None else str(y_units))

        def worker() -> None:
            try:
                # Use the same XY-only parser as the FTIR workstation.
                meta: Dict[str, Any] = {}

                xs, ys, meta = _parse_ftir_xy_only(str(p))
                if len(xs) < 5:
                    raise ValueError("No usable numeric XY data found")

                x = np.asarray(xs, dtype=float)
                y = np.asarray(ys, dtype=float)
                try:
                    order = np.argsort(x)
                    x = x[order]
                    y = y[order]
                except Exception:
                    pass

                inferred_mode, inferred_xu, inferred_yu = infer_from_meta(meta)
                final_mode = (str(y_mode).strip() if y_mode else str(inferred_mode))
                final_xu = (str(x_units).strip() if x_units else inferred_xu)
                final_yu = (str(y_units).strip() if y_units else inferred_yu)

                self.after(
                    0,
                    lambda: self._on_ftir_ready(
                        p,
                        x,
                        y,
                        make_active,
                        restore_ctx,
                        dataset_id,
                        name,
                        final_mode,
                        final_xu,
                        final_yu,
                        peak_settings,
                        peaks,
                        peak_label_overrides,
                        peak_suppressed,
                        peak_label_positions,
                        None,
                    ),
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda: self._on_ftir_ready(
                        p,
                        None,
                        None,
                        make_active,
                        restore_ctx,
                        dataset_id,
                        name,
                        y_mode,
                        x_units,
                        y_units,
                        peak_settings,
                        peaks,
                        peak_label_overrides,
                        peak_suppressed,
                        peak_label_positions,
                        exc,
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_ftir_ready(
        self,
        path: Path,
        x: Optional[np.ndarray],
        y: Optional[np.ndarray],
        make_active: bool,
        restore_ctx: Optional[Dict[str, Any]],
        dataset_id: Optional[str],
        name: Optional[str],
        y_mode: Optional[str],
        x_units: Optional[str],
        y_units: Optional[str],
        peak_settings: Optional[Dict[str, Any]],
        peaks: Optional[List[Dict[str, Any]]],
        peak_label_overrides: Optional[Dict[str, str]],
        peak_suppressed: Optional[List[str]],
        peak_label_positions: Optional[Dict[str, Any]],
        err: Optional[Exception],
    ) -> None:
        if err is not None or x is None or y is None:
            try:
                messagebox.showerror("FTIR", f"Failed to load FTIR CSV:\n{path}\n\n{err}", parent=self)
            except Exception:
                pass
            try:
                self._log("ERROR", f"Failed to load FTIR: {path}", exc=err)
            except Exception:
                pass
        else:
            try:
                p = Path(path).expanduser().resolve()
                existing = None
                for d in self.workspace.ftir_datasets:
                    if getattr(d, "path", None) == p:
                        existing = d
                        break

                if existing is None:
                    ds_id = (str(dataset_id).strip() if dataset_id else "") or uuid.uuid4().hex
                    loaded_at = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                    x_full = np.asarray(x, dtype=float)
                    y_full = np.asarray(y, dtype=float)
                    x_disp = x_full
                    y_disp = y_full
                    try:
                        n = int(x_full.size)
                        max_n = 50_000
                        if n > max_n:
                            step = int(math.ceil(float(n) / float(max_n)))
                            x_disp = x_full[::step]
                            y_disp = y_full[::step]
                    except Exception:
                        x_disp = x_full
                        y_disp = y_full

                    existing = FTIRDataset(
                        id=str(ds_id),
                        path=p,
                        name=(str(name).strip() if name else str(p.name)),
                        x_full=x_full,
                        y_full=y_full,
                        x_disp=np.asarray(x_disp, dtype=float),
                        y_disp=np.asarray(y_disp, dtype=float),
                        y_mode=(str(y_mode).strip() if y_mode else "absorbance"),
                        x_units=(None if x_units is None else str(x_units)),
                        y_units=(None if y_units is None else str(y_units)),
                        loaded_at_utc=str(loaded_at),
                    )
                    self.workspace.ftir_datasets.append(existing)
                else:
                    try:
                        if dataset_id and str(dataset_id).strip():
                            existing.id = str(dataset_id).strip()
                        if name and str(name).strip():
                            existing.name = str(name).strip()
                        if y_mode and str(y_mode).strip():
                            existing.y_mode = str(y_mode).strip()
                        if x_units and str(x_units).strip():
                            existing.x_units = str(x_units).strip()
                        if y_units and str(y_units).strip():
                            existing.y_units = str(y_units).strip()
                    except Exception:
                        pass

                # Restore peak picking state (best effort)
                try:
                    if isinstance(peak_settings, dict):
                        existing.peak_settings = dict(peak_settings)
                    if isinstance(peaks, list):
                        existing.peaks = list(peaks)
                    if isinstance(peak_label_overrides, dict):
                        existing.peak_label_overrides = {str(k): str(v) for k, v in peak_label_overrides.items()}
                    if isinstance(peak_suppressed, list):
                        existing.peak_suppressed = set(str(x) for x in peak_suppressed if str(x).strip())
                    if isinstance(peak_label_positions, dict):
                        pos_out: Dict[str, Tuple[float, float]] = {}
                        for k, v in peak_label_positions.items():
                            try:
                                if isinstance(v, (list, tuple)) and len(v) == 2:
                                    pos_out[str(k)] = (float(v[0]), float(v[1]))
                            except Exception:
                                continue
                        existing.peak_label_positions = pos_out
                except Exception:
                    pass

                if make_active:
                    self.workspace.active_ftir_id = str(getattr(existing, "id", ""))
                try:
                    if self._ftir_view is not None:
                        self._ftir_view.refresh_from_workspace(select_active=True)
                except Exception:
                    pass
                try:
                    self._log("INFO", f"Loaded FTIR: {Path(path).name} (n={int(np.asarray(x).size)})")
                except Exception:
                    pass
            except Exception as exc:
                try:
                    self._log("ERROR", "Failed to attach FTIR dataset", exc=exc)
                except Exception:
                    pass

        ctx = getattr(self, "_workspace_restore_ctx", None)
        if restore_ctx is not None and isinstance(restore_ctx, dict):
            try:
                restore_ctx["done_ftir"] = int(restore_ctx.get("done_ftir", 0)) + 1
            except Exception:
                restore_ctx["done_ftir"] = 1
            self._maybe_finalize_ftir_only_restore(restore_ctx)
        elif isinstance(ctx, dict):
            try:
                ctx["done_ftir"] = int(ctx.get("done_ftir", 0)) + 1
            except Exception:
                ctx["done_ftir"] = 1
            self._maybe_finalize_workspace_restore()

    # --- FTIR-only workspace persistence ---

    def _ftir_state_to_dict(self) -> Dict[str, Any]:
        """Capture FTIR-only session state into a JSON-serializable dict."""
        fv = getattr(self, "_ftir_view", None)

        ftir_files_rows: List[Dict[str, Any]] = []
        ftir_workspaces_rows: List[Dict[str, Any]] = []
        active_ftir_workspace_id: Optional[str] = None
        ftir_overlay_groups_rows: List[Dict[str, Any]] = []
        active_ftir_overlay_group_id: Optional[str] = None

        if fv is not None and isinstance(getattr(fv, "workspaces", None), dict) and fv.workspaces:
            active_ftir_workspace_id = str(getattr(fv, "active_workspace_id", "") or "") or None

            seen_ds: set[str] = set()
            for ws_id, ws_obj in (fv.workspaces or {}).items():
                ds_ids: List[str] = []
                for d in (getattr(ws_obj, "datasets", []) or []):
                    did = str(getattr(d, "id", ""))
                    if not did:
                        continue
                    ds_ids.append(did)
                    if did in seen_ds:
                        continue
                    seen_ds.add(did)
                    ftir_files_rows.append(
                        {
                            "id": str(getattr(d, "id", "")),
                            "path": str(d.path),
                            "name": str(getattr(d, "name", d.path.name)),
                            "y_mode": str(getattr(d, "y_mode", "absorbance")),
                            "x_units": (None if getattr(d, "x_units", None) is None else str(getattr(d, "x_units", ""))),
                            "y_units": (None if getattr(d, "y_units", None) is None else str(getattr(d, "y_units", ""))),
                            "peak_settings": (getattr(d, "peak_settings", None) or {}),
                            "peaks": (getattr(d, "peaks", None) or []),
                            "peak_label_overrides": (getattr(d, "peak_label_overrides", None) or {}),
                            "peak_suppressed": sorted(list(getattr(d, "peak_suppressed", None) or set())),
                            "peak_label_positions": {
                                str(k): [float(v[0]), float(v[1])]
                                for k, v in (getattr(d, "peak_label_positions", None) or {}).items()
                                if isinstance(v, tuple) and len(v) == 2
                            },
                        }
                    )

                ftir_workspaces_rows.append(
                    {
                        "id": str(getattr(ws_obj, "id", ws_id)),
                        "name": str(getattr(ws_obj, "name", "Workspace")),
                        "dataset_ids": ds_ids,
                        "active_dataset_id": (None if not getattr(ws_obj, "active_dataset_id", None) else str(getattr(ws_obj, "active_dataset_id"))),
                        "line_color": (None if getattr(ws_obj, "line_color", None) is None else str(getattr(ws_obj, "line_color"))),
                        "bond_annotations": [],
                    }
                )

                # Bond annotations (workspace-level)
                try:
                    bond_rows: List[Dict[str, Any]] = []
                    for a in (getattr(ws_obj, "bond_annotations", None) or []):
                        try:
                            xy = getattr(a, "xytext", None)
                            if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                                xy = (float(getattr(a, "x_cm1")), float(getattr(a, "y_value")))
                            bond_rows.append(
                                {
                                    "dataset_id": str(getattr(a, "dataset_id", "") or ""),
                                    "text": str(getattr(a, "text", "") or ""),
                                    "x_cm1": float(getattr(a, "x_cm1")),
                                    "y_value": float(getattr(a, "y_value")),
                                    "xytext": [float(xy[0]), float(xy[1])],
                                    "show_vline": bool(getattr(a, "show_vline", False)),
                                    "line_color": str(getattr(a, "line_color", "#444444") or "#444444"),
                                    "text_color": str(getattr(a, "text_color", "#111111") or "#111111"),
                                    "fontsize": int(getattr(a, "fontsize", 9) or 9),
                                    "rotation": int(getattr(a, "rotation", 90) or 90),
                                    "preset_id": (None if not getattr(a, "preset_id", None) else str(getattr(a, "preset_id"))),
                                }
                            )
                        except Exception:
                            continue
                    ftir_workspaces_rows[-1]["bond_annotations"] = bond_rows
                except Exception:
                    pass

            try:
                active_ftir_overlay_group_id = (None if not getattr(fv, "_active_overlay_group_id", None) else str(getattr(fv, "_active_overlay_group_id")))
            except Exception:
                active_ftir_overlay_group_id = None

            for gid, g in (getattr(fv, "_overlay_groups", {}) or {}).items():
                try:
                    members = [(str(a), str(b)) for (a, b) in (getattr(g, "members", []) or [])]
                except Exception:
                    members = []
                try:
                    am = getattr(g, "active_member", None)
                    active_member = ([str(am[0]), str(am[1])] if (am is not None and isinstance(am, tuple) and len(am) == 2) else None)
                except Exception:
                    active_member = None

                per_style: Dict[str, Any] = {}
                try:
                    for k, st in (getattr(g, "per_member_style", {}) or {}).items():
                        kk = f"{str(k[0])}::{str(k[1])}"
                        per_style[kk] = {"linewidth": float(getattr(st, "linewidth", 1.2) or 1.2)}
                except Exception:
                    per_style = {}

                ftir_overlay_groups_rows.append(
                    {
                        "group_id": str(getattr(g, "group_id", gid)),
                        "name": str(getattr(g, "name", "Overlay")),
                        "created_at": float(getattr(g, "created_at", time.time()) or time.time()),
                        "members": [[a, b] for (a, b) in members],
                        "active_member": active_member,
                        "per_member_style": per_style,
                    }
                )

        # UI prefs (best-effort)
        reverse_pref_by_id: Dict[str, bool] = {}
        show_peaks_all_overlay = False
        try:
            if fv is not None:
                reverse_pref_by_id = {str(k): bool(v) for k, v in (getattr(fv, "_reverse_pref_by_id", {}) or {}).items()}
        except Exception:
            reverse_pref_by_id = {}
        try:
            if fv is not None:
                show_peaks_all_overlay = bool(getattr(fv, "_show_peaks_all_overlay_var").get())
        except Exception:
            show_peaks_all_overlay = False

        active_ftir_id: Optional[str] = None
        try:
            active_ftir_id = (None if getattr(self.workspace, "active_ftir_id", None) is None else str(self.workspace.active_ftir_id))
        except Exception:
            active_ftir_id = None

        return {
            "schema_version": int(WORKSPACE_SCHEMA_VERSION),
            "app": str(APP_NAME),
            "created_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "ftir": {
                "ftir_files": ftir_files_rows,
                "ftir_workspaces": ftir_workspaces_rows,
                "active_ftir_workspace_id": active_ftir_workspace_id,
                "ftir_overlay_groups": ftir_overlay_groups_rows,
                "active_ftir_overlay_group_id": active_ftir_overlay_group_id,
                "active_ftir_id": active_ftir_id,
                "ui": {
                    "reverse_pref_by_id": reverse_pref_by_id,
                    "show_peaks_all_overlay": bool(show_peaks_all_overlay),
                },
            },
        }

    def _clear_ftir_for_load(self) -> None:
        """Clear only FTIR state (datasets + FTIR view state)."""
        try:
            self.workspace.ftir_datasets.clear()
            self.workspace.active_ftir_id = None
        except Exception:
            pass
        try:
            if self._ftir_view is not None:
                self._ftir_view.reset_for_load()
        except Exception:
            pass

    def _apply_ftir_workspace_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("FTIR workspace JSON must be an object")

        schema_version = int(state.get("schema_version") or 0)
        if schema_version != int(WORKSPACE_SCHEMA_VERSION):
            raise ValueError(f"Unsupported FTIR workspace schema_version={schema_version} (expected {WORKSPACE_SCHEMA_VERSION})")

        ft = state.get("ftir") or {}
        if not isinstance(ft, dict):
            ft = {}

        # Replace only FTIR state
        self._clear_ftir_for_load()

        pending_ftir: List[Dict[str, Any]] = []
        for item in (ft.get("ftir_files") or []):
            if not isinstance(item, dict):
                continue
            p = str(item.get("path") or "").strip()
            if not p:
                continue
            pending_ftir.append(
                {
                    "id": item.get("id"),
                    "path": str(Path(p).expanduser().resolve()),
                    "name": item.get("name"),
                    "y_mode": item.get("y_mode"),
                    "x_units": item.get("x_units"),
                    "y_units": item.get("y_units"),
                    "peak_settings": item.get("peak_settings"),
                    "peaks": item.get("peaks"),
                    "peak_label_overrides": item.get("peak_label_overrides"),
                    "peak_suppressed": item.get("peak_suppressed"),
                    "peak_label_positions": item.get("peak_label_positions"),
                }
            )

        ui = ft.get("ui") or {}
        if isinstance(ui, dict):
            try:
                if self._ftir_view is not None and isinstance(ui.get("reverse_pref_by_id"), dict):
                    self._ftir_view._reverse_pref_by_id = {str(k): bool(v) for k, v in ui.get("reverse_pref_by_id").items()}  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                if self._ftir_view is not None and ui.get("show_peaks_all_overlay") is not None:
                    self._ftir_view._show_peaks_all_overlay_var.set(bool(ui.get("show_peaks_all_overlay")))  # type: ignore[attr-defined]
            except Exception:
                pass

        ctx: Dict[str, Any] = {
            "pending_ftir": pending_ftir,
            "expected_ftir": 0,
            "done_ftir": 0,
            "missing_ftir": [],
            "active_ftir_id": (None if not ft.get("active_ftir_id") else str(ft.get("active_ftir_id"))),
            "ftir_restore_payload": {
                "ftir_workspaces": ft.get("ftir_workspaces"),
                "active_ftir_workspace_id": ft.get("active_ftir_workspace_id"),
                "ftir_overlay_groups": ft.get("ftir_overlay_groups"),
                "active_ftir_overlay_group_id": ft.get("active_ftir_overlay_group_id"),
            },
        }

        for item in pending_ftir:
            p_str = str((item or {}).get("path") or "").strip()
            if not p_str:
                continue
            p = Path(p_str)
            if not p.exists():
                ctx["missing_ftir"].append(p_str)
                continue
            ctx["expected_ftir"] += 1
            self._add_ftir_from_path_async(
                p,
                make_active=False,
                restore_ctx=ctx,
                dataset_id=(None if not isinstance(item, dict) else item.get("id")),
                name=(None if not isinstance(item, dict) else item.get("name")),
                y_mode=(None if not isinstance(item, dict) else item.get("y_mode")),
                x_units=(None if not isinstance(item, dict) else item.get("x_units")),
                y_units=(None if not isinstance(item, dict) else item.get("y_units")),
                peak_settings=(None if not isinstance(item, dict) else item.get("peak_settings")),
                peaks=(None if not isinstance(item, dict) else item.get("peaks")),
                peak_label_overrides=(None if not isinstance(item, dict) else item.get("peak_label_overrides")),
                peak_suppressed=(None if not isinstance(item, dict) else item.get("peak_suppressed")),
                peak_label_positions=(None if not isinstance(item, dict) else item.get("peak_label_positions")),
            )

        self._maybe_finalize_ftir_only_restore(ctx)

    def _maybe_finalize_ftir_only_restore(self, ctx: Dict[str, Any]) -> None:
        if not isinstance(ctx, dict):
            return
        if int(ctx.get("done_ftir", 0)) < int(ctx.get("expected_ftir", 0)):
            return

        # Restore active FTIR (by id)
        try:
            active_ftir_id = ctx.get("active_ftir_id")
            if isinstance(active_ftir_id, str) and active_ftir_id:
                if any(str(getattr(d, "id", "")) == str(active_ftir_id) for d in (self.workspace.ftir_datasets or [])):
                    self.workspace.active_ftir_id = str(active_ftir_id)
        except Exception:
            pass

        try:
            if self._ftir_view is not None:
                self._ftir_view.apply_restored_ftir_state(ctx.get("ftir_restore_payload"))
        except Exception:
            pass

        missing_ftir = list(ctx.get("missing_ftir") or [])
        if missing_ftir:
            msg = "Some FTIR files were missing and skipped:\n\n" + "\n".join(missing_ftir[:15]) + ("\n…" if len(missing_ftir) > 15 else "")
            try:
                messagebox.showwarning("FTIR workspace loaded", msg, parent=self)
            except Exception:
                pass

        try:
            self._set_status("FTIR workspace loaded")
        except Exception:
            pass

    def _save_ftir_workspace(self) -> None:
        try:
            state = self._ftir_state_to_dict()
        except Exception as exc:
            messagebox.showerror("Save FTIR Workspace", f"Failed to capture FTIR state:\n\n{exc}", parent=self)
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save FTIR Workspace",
            defaultextension=".json",
            filetypes=[("FTIR Workspace JSON", "*.json"), ("All files", "*.*")],
            initialfile="ftir_workspace.json",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Save FTIR Workspace", f"Failed to write file:\n\n{exc}", parent=self)
            return

        messagebox.showinfo("Save FTIR Workspace", f"Saved FTIR workspace:\n\n{path}", parent=self)

    def _load_ftir_workspace(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Load FTIR Workspace",
            filetypes=[("FTIR Workspace JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            messagebox.showerror("Load FTIR Workspace", f"Failed to read JSON:\n\n{exc}", parent=self)
            return

        try:
            self._apply_ftir_workspace_dict(state)
        except Exception as exc:
            messagebox.showerror("Load FTIR Workspace", f"Failed to load FTIR workspace:\n\n{exc}", parent=self)
            return

    def _bind_shortcuts(self) -> None:
        # File
        self.bind_all("<Control-o>", lambda e: (self._dispatch_open(), "break"))
        self.bind_all("<Control-Shift-O>", lambda e: (self._dispatch_open_many(), "break"))
        self.bind_all("<Control-Shift-o>", lambda e: (self._dispatch_open_many(), "break"))
        self.bind_all("<Control-u>", lambda e: (self._dispatch_add_uv_single(), "break"))
        self.bind_all("<Control-Shift-U>", lambda e: (self._dispatch_add_uv_many(), "break"))
        self.bind_all("<Control-Shift-u>", lambda e: (self._dispatch_add_uv_many(), "break"))
        self.bind_all("<Control-Shift-S>", lambda e: (self._save_workspace(), "break"))
        self.bind_all("<Control-Shift-s>", lambda e: (self._save_workspace(), "break"))
        self.bind_all("<Control-Shift-L>", lambda e: (self._load_workspace(), "break"))
        self.bind_all("<Control-Shift-l>", lambda e: (self._load_workspace(), "break"))
        self.bind_all("<Control-w>", lambda e: (self._dispatch_lcms_only(self._close_active_session), "break"))
        self.bind_all("<Control-e>", lambda e: (self._dispatch_export_primary(), "break"))

        # Cycle active mzML sessions
        self.bind_all("<Control-Tab>", lambda e: (self._dispatch_cycle(+1), "break"))
        self.bind_all("<Control-Shift-Tab>", lambda e: (self._dispatch_cycle(-1), "break"))
        # Some Tk builds report Shift+Tab as ISO_Left_Tab
        self.bind_all("<Control-ISO_Left_Tab>", lambda e: (self._dispatch_cycle(-1), "break"))

        # Tools
        self.bind_all("<Control-g>", lambda e: (self._dispatch_lcms_only(self._open_graph_settings), "break"))
        self.bind_all("<Control-p>", lambda e: (self._dispatch_lcms_only(self._open_annotation_settings), "break"))
        self.bind_all("<Control-l>", lambda e: (self._dispatch_lcms_only(self._open_custom_labels), "break"))
        self.bind_all("<Control-m>", lambda e: (self._dispatch_lcms_only(self._open_polymer_match), "break"))
        self.bind_all("<Control-0>", lambda e: (self._dispatch_lcms_only(self._reset_view_all), "break"))

        # Overlay shortcuts
        self.bind_all("<Control-Alt-o>", lambda e: (self._dispatch_lcms_only(self._start_overlay_selected), "break"))
        self.bind_all("<Control-Alt-c>", lambda e: (self._dispatch_lcms_only(self._clear_overlay), "break"))
        self.bind_all("<Control-Alt-Right>", lambda e: (self._dispatch_lcms_only(lambda: self._cycle_overlay_active_dataset(+1)), "break"))
        self.bind_all("<Control-Alt-Left>", lambda e: (self._dispatch_lcms_only(lambda: self._cycle_overlay_active_dataset(-1)), "break"))

        # Find m/z (optional)
        self.bind_all("<Control-f>", lambda e: (self._dispatch_lcms_only(self._open_find_mz_dialog), "break"))

        # Navigation
        self.bind_all("<Left>", lambda e: (self._dispatch_lcms_only(lambda: self._step_spectrum(-1)), "break"))
        self.bind_all("<Right>", lambda e: (self._dispatch_lcms_only(lambda: self._step_spectrum(+1)), "break"))
        self.bind_all("<Home>", lambda e: (self._dispatch_lcms_only(lambda: self._go_to_index(0)), "break"))
        self.bind_all("<End>", lambda e: (self._dispatch_lcms_only(self._go_last), "break"))

        # Keyboard-first navigation (single keys; do not steal from text fields)
        self.bind_all("<m>", self._on_key_find_mz, add=True)
        self.bind_all("<M>", self._on_key_find_mz, add=True)
        self.bind_all("<s>", self._on_key_sim, add=True)
        self.bind_all("<S>", self._on_key_sim, add=True)

    def _cycle_active_session(self, delta: int) -> None:
        if not self._session_order:
            return
        if self._active_session_id not in self._session_order:
            self._set_active_session(self._session_order[0])
            return
        try:
            i = int(self._session_order.index(self._active_session_id))
        except Exception:
            i = 0
        n = int(len(self._session_order))
        j = (int(i) + int(delta)) % n
        self._set_active_session(self._session_order[int(j)])

    def _cycle_overlay_active_dataset(self, delta: int) -> None:
        if not self._is_overlay_active():
            return
        ids = self._overlay_dataset_ids()
        if not ids:
            return
        cur = str(self._active_session_id or "")
        if cur not in ids:
            self._set_active_session(str(ids[0]))
            return
        try:
            i = int(ids.index(cur))
        except Exception:
            i = 0
        n = int(len(ids))
        j = (int(i) + int(delta)) % n
        self._set_active_session(str(ids[int(j)]))

    def _toggle_advanced(self) -> None:
        cur = bool(self._advanced_expanded_var.get())
        try:
            self._advanced_expanded_var.set(not cur)
        except Exception:
            pass
        self._apply_advanced_visibility()

    def _apply_advanced_visibility(self) -> None:
        expanded = bool(self._advanced_expanded_var.get())
        if self._advanced_toggle_btn is not None:
            try:
                self._advanced_toggle_btn.configure(text=("Hide ▲" if expanded else "Show ▼"))
            except Exception:
                pass
        if self._advanced_body is not None:
            try:
                if expanded:
                    self._advanced_body.grid()
                else:
                    self._advanced_body.grid_remove()
            except Exception:
                pass

        nb = self._sidebar_notebook
        tab_poly = getattr(self, "_tab_poly_frame", None)
        if nb is not None and tab_poly is not None:
            try:
                poly_wanted = bool(self._adv_show_polymer_var.get())
                tabs = set(nb.tabs())
                tab_id = str(tab_poly)
                if poly_wanted and tab_id not in tabs:
                    nb.add(tab_poly, text="Polymer")
                if (not poly_wanted) and tab_id in tabs:
                    nb.forget(tab_poly)
            except Exception:
                pass

        # Alignment diagnostics controls
        nav_diag = getattr(self, "_nav_diag_btn", None)
        if nav_diag is not None:
            try:
                if bool(self._adv_show_alignment_diag_var.get()):
                    nav_diag.grid()
                else:
                    nav_diag.grid_remove()
            except Exception:
                pass

    def _open_jump_to_mz_dialog(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Jump to m/z")
        dlg.resizable(False, False)
        dlg.transient(self)

        frm = ttk.Frame(dlg, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="m/z").grid(row=0, column=0, sticky="w")
        mz_ent = ttk.Entry(frm, textvariable=self._mz_find_mz_var, width=14)
        mz_ent.grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Label(frm, text="Tolerance").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tol_ent = ttk.Entry(frm, textvariable=self._mz_find_tol_var, width=14)
        tol_ent.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(8, 0))
        unit = ttk.Combobox(frm, textvariable=self._mz_find_unit_var, values=["ppm", "Da"], state="readonly", width=8)
        unit.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(8, 0))

        buttons = ttk.Frame(frm)
        buttons.grid(row=2, column=0, columnspan=3, sticky="e", pady=(14, 0))

        def do_jump() -> None:
            try:
                self._mz_find_mode_var.set("Nearest")
                self._mz_find_min_int_var.set("0")
            except Exception:
                pass
            self._find_mz_jump()

        jump_btn = ttk.Button(buttons, text="Jump", command=do_jump)
        jump_btn.grid(row=0, column=0, padx=(0, 8))
        close_btn = ttk.Button(buttons, text="Close", command=dlg.destroy)
        close_btn.grid(row=0, column=1)

        try:
            mz_ent.focus_set()
            mz_ent.selection_range(0, tk.END)
            mz_ent.bind("<Return>", lambda e: (do_jump(), "break"))
            tol_ent.bind("<Return>", lambda e: (do_jump(), "break"))
        except Exception:
            pass
        self.bind_all("<a>", self._on_key_auto_align, add=True)
        self.bind_all("<A>", self._on_key_auto_align, add=True)
        self.bind_all("<d>", self._on_key_diagnostics, add=True)
        self.bind_all("<D>", self._on_key_diagnostics, add=True)
        self.bind_all("<bracketleft>", self._on_key_step_coarse_back, add=True)
        self.bind_all("<bracketright>", self._on_key_step_coarse_fwd, add=True)
        self.bind_all("<Shift-Left>", self._on_key_step_medium_back, add=True)
        self.bind_all("<Shift-Right>", self._on_key_step_medium_fwd, add=True)
        self.bind_all("<slash>", self._on_key_focus_rt_jump, add=True)
        self.bind_all("<Escape>", self._on_key_escape_close, add=True)

        # Help
        self.bind_all("<F1>", lambda e: (self._open_instructions_window(), "break"))

    def _shortcut_in_text_input(self, widget: Any) -> bool:
        """True if the focused widget is a text-entry control where single keys should type."""
        if widget is None:
            return False
        try:
            cls = str(widget.winfo_class() or "")
        except Exception:
            return False
        return cls in {
            "Entry",
            "TEntry",
            "Text",
            "TCombobox",
            "Combobox",
            "Spinbox",
            "TSpinbox",
        }

    def _register_nonmodal_dialog(self, win: tk.Toplevel) -> None:
        """Track non-modal dialogs for Esc-to-close when focus is on the main window."""
        if win is None:
            return
        try:
            if win in self._nonmodal_dialog_stack:
                self._nonmodal_dialog_stack.remove(win)
            self._nonmodal_dialog_stack.append(win)
        except Exception:
            return

        def _on_focus_in(_evt=None) -> None:
            try:
                if win in self._nonmodal_dialog_stack:
                    self._nonmodal_dialog_stack.remove(win)
                self._nonmodal_dialog_stack.append(win)
            except Exception:
                pass

        def _on_destroy(_evt=None) -> None:
            try:
                if win in self._nonmodal_dialog_stack:
                    self._nonmodal_dialog_stack.remove(win)
            except Exception:
                pass

        try:
            win.bind("<FocusIn>", _on_focus_in, add=True)
            win.bind("<Destroy>", _on_destroy, add=True)
        except Exception:
            pass

    def _close_topmost_nonmodal_dialog(self) -> bool:
        """Return True if a dialog was closed."""
        try:
            w = self.focus_get()
        except Exception:
            w = None
        try:
            top = w.winfo_toplevel() if w is not None else None
        except Exception:
            top = None

        if isinstance(top, tk.Toplevel) and top is not self:
            try:
                top.destroy()
                return True
            except Exception:
                return False

        for win in list(reversed(self._nonmodal_dialog_stack)):
            try:
                if win is None or not bool(win.winfo_exists()):
                    continue
                if win is self:
                    continue
                win.destroy()
                return True
            except Exception:
                continue
        return False

    def _on_key_find_mz(self, event=None):
        if event is not None and self._shortcut_in_text_input(getattr(event, "widget", None)):
            return None
        self._open_find_mz_dialog()
        return "break"

    def _on_key_sim(self, event=None):
        if event is not None and self._shortcut_in_text_input(getattr(event, "widget", None)):
            return None
        self._open_sim_dialog()
        return "break"

    def _on_key_auto_align(self, event=None):
        if event is not None and self._shortcut_in_text_input(getattr(event, "widget", None)):
            return None
        self._auto_align_uv_ms()
        return "break"

    def _on_key_diagnostics(self, event=None):
        if event is not None and self._shortcut_in_text_input(getattr(event, "widget", None)):
            return None
        self._open_diagnostics_window()
        return "break"

    def _on_key_step_coarse_back(self, event=None):
        if event is not None and self._shortcut_in_text_input(getattr(event, "widget", None)):
            return None
        self._step_spectrum(-10)
        return "break"

    def _on_key_step_coarse_fwd(self, event=None):
        if event is not None and self._shortcut_in_text_input(getattr(event, "widget", None)):
            return None
        self._step_spectrum(+10)
        return "break"

    def _on_key_step_medium_back(self, event=None):
        if event is not None and self._shortcut_in_text_input(getattr(event, "widget", None)):
            return None
        self._step_spectrum(-5)
        return "break"

    def _on_key_step_medium_fwd(self, event=None):
        if event is not None and self._shortcut_in_text_input(getattr(event, "widget", None)):
            return None
        self._step_spectrum(+5)
        return "break"

    def _on_key_focus_rt_jump(self, event=None):
        if event is not None and self._shortcut_in_text_input(getattr(event, "widget", None)):
            return None
        ent = self._rt_jump_entry
        if ent is None:
            return "break"
        try:
            ent.focus_set()
            ent.selection_range(0, tk.END)
        except Exception:
            pass
        return "break"

    def _on_key_escape_close(self, _event=None):
        closed = self._close_topmost_nonmodal_dialog()
        return "break" if closed else None

    def _open_diagnostics_window(self) -> None:
        if self._diagnostics_win is not None:
            try:
                if bool(self._diagnostics_win.winfo_exists()):
                    self._refresh_diagnostics_window()
                    self._diagnostics_win.deiconify()
                    self._diagnostics_win.lift()
                    return
            except Exception:
                self._diagnostics_win = None

        dlg = tk.Toplevel(self)
        dlg.title("Diagnostics")
        dlg.resizable(False, False)
        dlg.transient(self)

        frm = ttk.Frame(dlg, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")

        self._diagnostics_vars = {}

        def mk_row(r: int, label: str, key: str) -> None:
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=(0, 6))
            var = tk.StringVar(value="")
            self._diagnostics_vars[key] = var
            ttk.Label(frm, textvariable=var).grid(row=r, column=1, sticky="w", padx=(12, 0), pady=(0, 6))

        mk_row(0, "Active mzML:", "mzml")
        mk_row(1, "Linked UV:", "uv")
        mk_row(2, "Polarity filter:", "pol")
        mk_row(3, "Current RT (min):", "rt")
        mk_row(4, "UV↔MS offset (min):", "offset")
        mk_row(5, "Auto-align:", "align")
        mk_row(6, "UV label min conf (%):", "uv_conf")

        btns = ttk.Frame(frm)
        btns.grid(row=7, column=0, columnspan=2, sticky="e", pady=(6, 0))
        ttk.Button(btns, text="Refresh", command=self._refresh_diagnostics_window).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="Close", command=dlg.destroy).grid(row=0, column=1)

        try:
            dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        except Exception:
            pass

        try:
            self._register_nonmodal_dialog(dlg)
        except Exception:
            pass

        self._diagnostics_win = dlg
        self._refresh_diagnostics_window()

    def _refresh_diagnostics_window(self) -> None:
        if not isinstance(self._diagnostics_vars, dict) or not self._diagnostics_vars:
            return

        mzml = self._active_session_display_name() if getattr(self, "_active_session_id", None) else (self.mzml_path.name if self.mzml_path else "(no mzML)")
        uv_sess = self._active_uv_session()
        uv = uv_sess.path.name if uv_sess is not None else "(no UV linked)"
        try:
            pol = (self.polarity_var.get() or "all").strip()
        except Exception:
            pol = "all"

        rt = "—"
        try:
            if self._current_spectrum_meta is not None:
                rt = f"{float(self._current_spectrum_meta.rt_min):.4f}"
        except Exception:
            rt = "—"
        try:
            off = f"{float(self._uv_ms_rt_offset_min):.3f}"
        except Exception:
            off = "—"
        try:
            align = "enabled" if (bool(self.uv_ms_align_enabled_var.get()) and self._uv_ms_align_uv_rts is not None) else "disabled"
        except Exception:
            align = "disabled"
        try:
            uv_conf = f"{float(self.uv_label_min_conf_var.get()):g}"
        except Exception:
            uv_conf = "0"

        try:
            self._diagnostics_vars["mzml"].set(str(mzml))
            self._diagnostics_vars["uv"].set(str(uv))
            self._diagnostics_vars["pol"].set(str(pol))
            self._diagnostics_vars["rt"].set(str(rt))
            self._diagnostics_vars["offset"].set(str(off))
            self._diagnostics_vars["align"].set(str(align))
            self._diagnostics_vars["uv_conf"].set(str(uv_conf))
        except Exception:
            pass

    # --- Workspace persistence ---

    def _workspace_state_to_dict(self) -> Dict[str, Any]:
        """Capture workspace + UI state into a JSON-serializable dict."""
        try:
            self._save_active_session_state()
        except Exception:
            pass

        def _var_get(var: Any, default: Any) -> Any:
            try:
                return var.get()
            except Exception:
                return default

        def _encode_custom_labels(custom: Dict[str, List[CustomLabel]]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for spec_id, items in (custom or {}).items():
                rows: List[Dict[str, Any]] = []
                for it in items or []:
                    try:
                        rows.append({"label": str(it.label), "mz": float(it.mz), "snap": bool(it.snap_to_nearest_peak)})
                    except Exception:
                        continue
                out[str(spec_id)] = rows
            return out

        def _encode_overrides(overrides: Dict[str, Dict[Tuple[str, float], Optional[str]]]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for spec_id, m in (overrides or {}).items():
                items: List[Dict[str, Any]] = []
                for (kind, mz_key), val in (m or {}).items():
                    try:
                        items.append({"kind": str(kind), "mz": float(mz_key), "value": (None if val is None else str(val))})
                    except Exception:
                        continue
                out[str(spec_id)] = items
            return out

        def _encode_uv_labels_by_uv_id(uv_labels_by_uv_id: Dict[str, Any]) -> Dict[str, Any]:
            # Convert UV-id keys to UV-path keys so restoration survives new UUIDs.
            out: Dict[str, Any] = {}
            for uv_id, labels_by_uvrt in (uv_labels_by_uv_id or {}).items():
                uv_sess = self._uv_sessions.get(str(uv_id))
                if uv_sess is None:
                    continue
                uv_path = str(uv_sess.path)
                uvrt_rows: List[Dict[str, Any]] = []
                if isinstance(labels_by_uvrt, dict):
                    for uv_rt, states in labels_by_uvrt.items():
                        try:
                            uv_rt_f = float(uv_rt)
                        except Exception:
                            continue
                        st_rows: List[Dict[str, Any]] = []
                        for st in (states or []):
                            try:
                                xy = getattr(st, "xytext", (0.0, 0.0))
                                st_rows.append(
                                    {
                                        "text": str(getattr(st, "text", "")),
                                        "xytext": [float(xy[0]), float(xy[1])],
                                        "confidence": float(getattr(st, "confidence", 0.0) or 0.0),
                                        "rt_delta_min": float(getattr(st, "rt_delta_min", 0.0) or 0.0),
                                        "uv_peak_score": float(getattr(st, "uv_peak_score", 0.0) or 0.0),
                                        "tic_peak_score": float(getattr(st, "tic_peak_score", 0.0) or 0.0),
                                    }
                                )
                            except Exception:
                                continue
                        uvrt_rows.append({"uv_rt": float(uv_rt_f), "labels": st_rows})
                out[uv_path] = uvrt_rows
            return out

        # UV registry
        uv_files: List[Dict[str, Any]] = []
        for uv_id in list(self._uv_order):
            sess = self._uv_sessions.get(str(uv_id))
            if sess is None:
                continue
            uv_files.append({"path": str(sess.path), "load_order": int(getattr(sess, "load_order", 0) or 0)})

        # mzML registry
        mzml_files: List[Dict[str, Any]] = []
        for sid in list(self._session_order):
            sess = self._sessions.get(str(sid))
            if sess is None:
                continue
            linked_uv_path: Optional[str] = None
            try:
                if sess.linked_uv_id and str(sess.linked_uv_id) in self._uv_sessions:
                    linked_uv_path = str(self._uv_sessions[str(sess.linked_uv_id)].path)
            except Exception:
                linked_uv_path = None

            mzml_files.append(
                {
                    "path": str(sess.path),
                    "display_name": str(sess.display_name),
                    "load_order": int(sess.load_order),
                    "overlay_color": str(getattr(sess, "overlay_color", "") or ""),
                    "last_selected_rt_min": (None if sess.last_selected_rt_min is None else float(sess.last_selected_rt_min)),
                    "last_scan_index": (None if sess.last_scan_index is None else int(sess.last_scan_index)),
                    "last_polarity_filter": (None if sess.last_polarity_filter is None else str(sess.last_polarity_filter)),
                    "custom_labels_by_spectrum": _encode_custom_labels(getattr(sess, "custom_labels_by_spectrum", {}) or {}),
                    "spec_label_overrides": _encode_overrides(getattr(sess, "spec_label_overrides", {}) or {}),
                    "linked_uv_path": linked_uv_path,
                    "uv_labels_by_uv_path": _encode_uv_labels_by_uv_id(getattr(sess, "uv_labels_by_uv_id", {}) or {}),
                }
            )

        # FTIR registry (prefer FTIRView multi-workspace state if available)
        ftir_files_rows: List[Dict[str, Any]] = []
        ftir_workspaces_rows: List[Dict[str, Any]] = []
        active_ftir_workspace_id: Optional[str] = None
        ftir_overlay_groups_rows: List[Dict[str, Any]] = []
        active_ftir_overlay_group_id: Optional[str] = None

        try:
            fv = getattr(self, "_ftir_view", None)
            if fv is not None and isinstance(getattr(fv, "workspaces", None), dict) and fv.workspaces:
                active_ftir_workspace_id = str(getattr(fv, "active_workspace_id", "") or "") or None

                # Flatten unique datasets across all workspaces
                seen_ds: set[str] = set()
                for ws_id, ws_obj in (fv.workspaces or {}).items():
                    ds_ids: List[str] = []
                    for d in (getattr(ws_obj, "datasets", []) or []):
                        did = str(getattr(d, "id", ""))
                        if not did:
                            continue
                        ds_ids.append(did)
                        if did in seen_ds:
                            continue
                        seen_ds.add(did)
                        ftir_files_rows.append(
                            {
                                "id": str(getattr(d, "id", "")),
                                "path": str(d.path),
                                "name": str(getattr(d, "name", d.path.name)),
                                "y_mode": str(getattr(d, "y_mode", "absorbance")),
                                "x_units": (None if getattr(d, "x_units", None) is None else str(getattr(d, "x_units", ""))),
                                "y_units": (None if getattr(d, "y_units", None) is None else str(getattr(d, "y_units", ""))),
                                "peak_settings": (getattr(d, "peak_settings", None) or {}),
                                "peaks": (getattr(d, "peaks", None) or []),
                                "peak_label_overrides": (getattr(d, "peak_label_overrides", None) or {}),
                                "peak_suppressed": sorted(list(getattr(d, "peak_suppressed", None) or set())),
                                "peak_label_positions": {
                                    str(k): [float(v[0]), float(v[1])]
                                    for k, v in (getattr(d, "peak_label_positions", None) or {}).items()
                                    if isinstance(v, tuple) and len(v) == 2
                                },
                            }
                        )

                    ftir_workspaces_rows.append(
                        {
                            "id": str(getattr(ws_obj, "id", ws_id)),
                            "name": str(getattr(ws_obj, "name", "Workspace")),
                            "dataset_ids": ds_ids,
                            "active_dataset_id": (None if not getattr(ws_obj, "active_dataset_id", None) else str(getattr(ws_obj, "active_dataset_id"))),
                            "line_color": (None if getattr(ws_obj, "line_color", None) is None else str(getattr(ws_obj, "line_color"))),
                        }
                    )

                # Overlay groups
                try:
                    active_ftir_overlay_group_id = (None if not getattr(fv, "_active_overlay_group_id", None) else str(getattr(fv, "_active_overlay_group_id")))
                except Exception:
                    active_ftir_overlay_group_id = None

                for gid, g in (getattr(fv, "_overlay_groups", {}) or {}).items():
                    try:
                        members = [(str(a), str(b)) for (a, b) in (getattr(g, "members", []) or [])]
                    except Exception:
                        members = []
                    try:
                        am = getattr(g, "active_member", None)
                        active_member = ([str(am[0]), str(am[1])] if (am is not None and isinstance(am, tuple) and len(am) == 2) else None)
                    except Exception:
                        active_member = None

                    per_style: Dict[str, Any] = {}
                    try:
                        for k, st in (getattr(g, "per_member_style", {}) or {}).items():
                            kk = f"{str(k[0])}::{str(k[1])}"
                            per_style[kk] = {"linewidth": float(getattr(st, "linewidth", 1.2) or 1.2)}
                    except Exception:
                        per_style = {}

                    ftir_overlay_groups_rows.append(
                        {
                            "group_id": str(getattr(g, "group_id", gid)),
                            "name": str(getattr(g, "name", "Overlay")),
                            "created_at": float(getattr(g, "created_at", time.time()) or time.time()),
                            "members": [[a, b] for (a, b) in members],
                            "active_member": active_member,
                            "per_member_style": per_style,
                        }
                    )
        except Exception:
            ftir_files_rows = []
            ftir_workspaces_rows = []
            active_ftir_workspace_id = None
            ftir_overlay_groups_rows = []
            active_ftir_overlay_group_id = None

        if not ftir_files_rows:
            # Legacy fallback
            ftir_files_rows = [
                {
                    "id": str(getattr(d, "id", "")),
                    "path": str(d.path),
                    "name": str(getattr(d, "name", d.path.name)),
                    "y_mode": str(getattr(d, "y_mode", "absorbance")),
                    "x_units": (None if getattr(d, "x_units", None) is None else str(getattr(d, "x_units", ""))),
                    "y_units": (None if getattr(d, "y_units", None) is None else str(getattr(d, "y_units", ""))),
                    "peak_settings": (getattr(d, "peak_settings", None) or {}),
                    "peaks": (getattr(d, "peaks", None) or []),
                    "peak_label_overrides": (getattr(d, "peak_label_overrides", None) or {}),
                    "peak_suppressed": sorted(list(getattr(d, "peak_suppressed", None) or set())),
                    "peak_label_positions": {
                        str(k): [float(v[0]), float(v[1])]
                        for k, v in (getattr(d, "peak_label_positions", None) or {}).items()
                        if isinstance(v, tuple) and len(v) == 2
                    },
                }
                for d in (self.workspace.ftir_datasets if getattr(self, "workspace", None) is not None else [])
            ]

        # Microscopy registry
        microscopy_workspaces_rows: List[Dict[str, Any]] = []
        active_microscopy_workspace_id: Optional[str] = None
        try:
            active_microscopy_workspace_id = (
                None
                if getattr(self.workspace, "active_microscopy_workspace_id", None) is None
                else str(self.workspace.active_microscopy_workspace_id)
            )
        except Exception:
            active_microscopy_workspace_id = None

        try:
            for ws_obj in (getattr(self.workspace, "microscopy_workspaces", None) or []):
                ds_rows: List[Dict[str, Any]] = []
                for d in (getattr(ws_obj, "datasets", None) or []):
                    ds_rows.append(
                        {
                            "id": str(getattr(d, "id", "")),
                            "display_name": str(getattr(d, "display_name", "")),
                            "file_path": str(getattr(d, "file_path", "")),
                            "workspace_id": str(getattr(d, "workspace_id", getattr(ws_obj, "id", ""))),
                            "created_at": str(getattr(d, "created_at", "")),
                            "notes": str(getattr(d, "notes", "")),
                            "output_dir": str(getattr(d, "output_dir", "")),
                            "last_macro_run": (None if getattr(d, "last_macro_run", None) is None else str(getattr(d, "last_macro_run"))),
                        }
                    )
                microscopy_workspaces_rows.append(
                    {
                        "id": str(getattr(ws_obj, "id", "")),
                        "name": str(getattr(ws_obj, "name", "Workspace")),
                        "datasets": ds_rows,
                    }
                )
        except Exception:
            microscopy_workspaces_rows = []
            active_microscopy_workspace_id = None

        state: Dict[str, Any] = {
            "schema_version": int(WORKSPACE_SCHEMA_VERSION),
            "app": str(APP_NAME),
            "created_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "window": {"geometry": str(self.geometry())},
            "workspace": {
                "active_mzml_path": (str(self._sessions[self._active_session_id].path) if (self._active_session_id and self._active_session_id in self._sessions) else None),
                "active_uv_path": (str(self._uv_sessions[self._active_uv_id].path) if (self._active_uv_id and self._active_uv_id in self._uv_sessions) else None),
                "mzml_files": mzml_files,
                "uv_files": uv_files,
                "ftir_files": ftir_files_rows,
                "ftir_workspaces": ftir_workspaces_rows,
                "active_ftir_workspace_id": active_ftir_workspace_id,
                "ftir_overlay_groups": ftir_overlay_groups_rows,
                "active_ftir_overlay_group_id": active_ftir_overlay_group_id,
                "microscopy_workspaces": microscopy_workspaces_rows,
                "active_microscopy_workspace_id": active_microscopy_workspace_id,
                "active_ftir_id": (None if getattr(self, "workspace", None) is None else (None if self.workspace.active_ftir_id is None else str(self.workspace.active_ftir_id))),
                "active_ftir_path": (
                    None
                    if (getattr(self, "workspace", None) is None or self.workspace.active_ftir_id is None)
                    else next((str(d.path) for d in self.workspace.ftir_datasets if str(getattr(d, "id", "")) == str(self.workspace.active_ftir_id)), None)
                ),
            },
            "ui": {
                "rt_unit": str(_var_get(self.rt_unit_var, "minutes")),
                "polarity": str(_var_get(self.polarity_var, "all")),
                "show_tic": bool(_var_get(self.show_tic_var, True)),
                "show_spectrum": bool(_var_get(self.show_spectrum_var, True)),
                "show_uv": bool(_var_get(self.show_uv_var, True)),
                "uv_label_from_ms": bool(_var_get(self.uv_label_from_ms_var, False)),
                "uv_label_from_ms_top_n": int(_var_get(self.uv_label_from_ms_top_n_var, 3)),
                "uv_label_min_conf": float(_var_get(self.uv_label_min_conf_var, 0.0)),
                "titles": {
                    "tic_title": str(_var_get(self.tic_title_var, "")),
                    "tic_xlabel": str(_var_get(self.tic_xlabel_var, "")),
                    "tic_ylabel": str(_var_get(self.tic_ylabel_var, "")),
                    "spec_title": str(_var_get(self.spec_title_var, "")),
                    "spec_xlabel": str(_var_get(self.spec_xlabel_var, "")),
                    "spec_ylabel": str(_var_get(self.spec_ylabel_var, "")),
                    "uv_title": str(_var_get(self.uv_title_var, "")),
                    "uv_xlabel": str(_var_get(self.uv_xlabel_var, "")),
                    "uv_ylabel": str(_var_get(self.uv_ylabel_var, "")),
                },
                "fonts": {
                    "title": int(_var_get(self.title_fontsize_var, 12)),
                    "label": int(_var_get(self.label_fontsize_var, 10)),
                    "tick": int(_var_get(self.tick_fontsize_var, 9)),
                },
                "limits": {
                    "tic": {
                        "x_min": str(_var_get(self.tic_xlim_min_var, "")),
                        "x_max": str(_var_get(self.tic_xlim_max_var, "")),
                        "y_min": str(_var_get(self.tic_ylim_min_var, "")),
                        "y_max": str(_var_get(self.tic_ylim_max_var, "")),
                    },
                    "spec": {
                        "x_min": str(_var_get(self.spec_xlim_min_var, "")),
                        "x_max": str(_var_get(self.spec_xlim_max_var, "")),
                        "y_min": str(_var_get(self.spec_ylim_min_var, "")),
                        "y_max": str(_var_get(self.spec_ylim_max_var, "")),
                    },
                    "uv": {
                        "x_min": str(_var_get(self.uv_xlim_min_var, "")),
                        "x_max": str(_var_get(self.uv_xlim_max_var, "")),
                        "y_min": str(_var_get(self.uv_ylim_min_var, "")),
                        "y_max": str(_var_get(self.uv_ylim_max_var, "")),
                    },
                },
                "annotations": {
                    "annotate_peaks": bool(_var_get(self.annotate_peaks_var, False)),
                    "annotate_top_n": int(_var_get(self.annotate_top_n_var, 10)),
                    "annotate_min_rel": float(_var_get(self.annotate_min_rel_var, 0.05)),
                    "drag_annotations": bool(_var_get(self.drag_annotations_var, True)),
                },
                "polymer": {
                    "enabled": bool(_var_get(self.poly_enabled_var, False)),
                    "monomers_text": str(_var_get(self.poly_monomers_text_var, "")),
                    "bond_delta": float(_var_get(self.poly_bond_delta_var, -18.010565)),
                    "extra_delta": float(_var_get(self.poly_extra_delta_var, 0.0)),
                    "adduct_mass": float(_var_get(self.poly_adduct_mass_var, 1.007276)),
                    "decarb": bool(_var_get(self.poly_decarb_enabled_var, False)),
                    "oxid": bool(_var_get(self.poly_oxid_enabled_var, False)),
                    "cluster": bool(_var_get(self.poly_cluster_enabled_var, False)),
                    "cluster_adduct_mass": float(_var_get(self.poly_cluster_adduct_mass_var, -1.007276)),
                    "adduct_na": bool(_var_get(self.poly_adduct_na_var, False)),
                    "adduct_k": bool(_var_get(self.poly_adduct_k_var, False)),
                    "adduct_cl": bool(_var_get(self.poly_adduct_cl_var, False)),
                    "adduct_formate": bool(_var_get(self.poly_adduct_formate_var, False)),
                    "adduct_acetate": bool(_var_get(self.poly_adduct_acetate_var, False)),
                    "charges": str(_var_get(self.poly_charges_var, "1")),
                    "max_dp": int(_var_get(self.poly_max_dp_var, 12)),
                    "tol_value": float(_var_get(self.poly_tol_value_var, 0.02)),
                    "tol_unit": str(_var_get(self.poly_tol_unit_var, "Da")),
                    "min_rel_int": float(_var_get(self.poly_min_rel_int_var, 0.01)),
                },
                "uv_ms_alignment": {
                    "offset_min": float(getattr(self, "_uv_ms_rt_offset_min", 0.0) or 0.0),
                    "align_enabled": bool(_var_get(self.uv_ms_align_enabled_var, False)),
                    "anchors_uv": ([] if getattr(self, "_uv_ms_align_uv_rts", None) is None else [float(x) for x in np.asarray(self._uv_ms_align_uv_rts, dtype=float).tolist()]),
                    "anchors_ms": ([] if getattr(self, "_uv_ms_align_ms_rts", None) is None else [float(x) for x in np.asarray(self._uv_ms_align_ms_rts, dtype=float).tolist()]),
                },
                "sim": {
                    "last_params": (None if self._sim_last_params is None else [float(self._sim_last_params[0]), float(self._sim_last_params[1]), str(self._sim_last_params[2]), bool(self._sim_last_params[3])]),
                },
            },
        }

        # Optional overlay persistence (ephemeral by default)
        try:
            if self._overlay_session is not None and bool(self._overlay_persist_var.get()):
                state["ui"]["overlay"] = {
                    "dataset_ids": list(self._overlay_session.dataset_ids),
                    "mode": str(self._overlay_mode_var.get()),
                    "colors": dict(self._overlay_session.colors),
                    "color_scheme": str(self._overlay_scheme_var.get() or "Auto (Tableau)"),
                    "single_hue_color": str(self._overlay_single_hue_color or "#1f77b4"),
                    "persist": True,
                    "show_uv": bool(self._overlay_show_uv_var.get()),
                    "stack_spectra": bool(self._overlay_stack_spectra_var.get()),
                    "show_labels_all": bool(self._overlay_show_labels_all_var.get()),
                    "multi_drag": bool(self._overlay_multi_drag_var.get()),
                    "active_dataset_id": str(self._overlay_active_dataset_id or "") or None,
                }
        except Exception:
            pass
        return state

    def _clear_workspace_for_load(self) -> None:
        """Clear mzML/UV workspaces without confirmation prompts."""
        try:
            self._save_active_session_state()
        except Exception:
            pass

        try:
            if self._active_reader is not None:
                self._active_reader.close()
            elif self._reader is not None:
                self._reader.close()
        except Exception:
            pass
        self._active_reader = None
        self._reader = None

        self._sessions.clear()
        self._session_order.clear()
        self._active_session_id = None
        self._session_load_counter = 0
        self.mzml_path = None
        self._index = None

        # Workspace model
        try:
            self.workspace.lcms_datasets.clear()
            self.workspace.active_lcms = None
            self.workspace.ftir_datasets.clear()
            self.workspace.active_ftir_id = None
            self.workspace.microscopy_workspaces.clear()
            self.workspace.active_microscopy_workspace_id = None
        except Exception:
            pass

        self._uv_sessions.clear()
        self._uv_order.clear()
        self._active_uv_id = None
        self._uv_load_counter = 0

        # Overlay state
        try:
            self._clear_overlay()
        except Exception:
            self._overlay_session = None
            self._overlay_selected_ms_rt = None

        self._custom_labels_by_spectrum = {}
        self._spec_label_overrides = {}

        self._filtered_meta = []
        self._filtered_rts = None
        self._filtered_tics = None
        self._current_scan_index = None
        self._current_spectrum_meta = None
        self._current_spectrum_mz = None
        self._current_spectrum_int = None
        self._selected_rt_min = None

        try:
            if self._ws_tree is not None:
                for iid in list(self._ws_tree.get_children("")):
                    try:
                        self._ws_tree.delete(iid)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if self._uv_ws_tree is not None:
                for iid in list(self._uv_ws_tree.get_children("")):
                    try:
                        self._uv_ws_tree.delete(iid)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            self._tic_line = None
            self._tic_marker = None
            self._uv_rt_marker = None
            self._rebuild_plot_axes()
            self._plot_uv()
            self._update_status_by_tab()
        except Exception:
            pass

        try:
            if self._microscopy_view is not None:
                self._microscopy_view.refresh_from_workspace(select_first=False)
        except Exception:
            pass

    def _clear_lcms_for_load(self) -> None:
        """Clear LCMS (mzML/UV) state without touching FTIR or microscopy."""
        try:
            self._save_active_session_state()
        except Exception:
            pass

        try:
            if self._active_reader is not None:
                self._active_reader.close()
            elif self._reader is not None:
                self._reader.close()
        except Exception:
            pass
        self._active_reader = None
        self._reader = None

        self._sessions.clear()
        self._session_order.clear()
        self._active_session_id = None
        self._session_load_counter = 0
        self.mzml_path = None
        self._index = None

        try:
            self.workspace.lcms_datasets.clear()
            self.workspace.active_lcms = None
        except Exception:
            pass

        self._uv_sessions.clear()
        self._uv_order.clear()
        self._active_uv_id = None
        self._uv_load_counter = 0

        try:
            self._clear_overlay()
        except Exception:
            self._overlay_session = None
            self._overlay_selected_ms_rt = None

        self._custom_labels_by_spectrum = {}
        self._spec_label_overrides = {}

        self._filtered_meta = []
        self._filtered_rts = None
        self._filtered_tics = None
        self._current_scan_index = None
        self._current_spectrum_meta = None
        self._current_spectrum_mz = None
        self._current_spectrum_int = None
        self._selected_rt_min = None

        try:
            if self._ws_tree is not None:
                for iid in list(self._ws_tree.get_children("")):
                    try:
                        self._ws_tree.delete(iid)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if self._uv_ws_tree is not None:
                for iid in list(self._uv_ws_tree.get_children("")):
                    try:
                        self._uv_ws_tree.delete(iid)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            self._tic_line = None
            self._tic_marker = None
            self._uv_rt_marker = None
            self._rebuild_plot_axes()
            self._plot_uv()
            self._update_status_by_tab()
        except Exception:
            pass

    def _apply_workspace_dict(self, state: Dict[str, Any]) -> None:
        """Apply a saved workspace dict to the current app (async loads)."""
        if not isinstance(state, dict):
            raise ValueError("Workspace JSON must be an object")

        schema_version = int(state.get("schema_version") or 0)
        if schema_version != int(WORKSPACE_SCHEMA_VERSION):
            raise ValueError(f"Unsupported workspace schema_version={schema_version} (expected {WORKSPACE_SCHEMA_VERSION})")

        ws = state.get("workspace") or {}
        ui = state.get("ui") or {}

        # Replace current workspace
        self._clear_workspace_for_load()

        # Restore Microscopy workspaces (best-effort; older sessions may not have these fields)
        def _safe_dirname(s: str) -> str:
            t = "".join(ch for ch in str(s) if ch.isalnum() or ch in ("-", "_", " ")).strip().replace(" ", "_")
            return t or "item"

        try:
            mic_rows = ws.get("microscopy_workspaces") or []
            mic_active = ws.get("active_microscopy_workspace_id")
        except Exception:
            mic_rows = []
            mic_active = None

        restored_mic_wss: List[MicroscopyWorkspace] = []
        try:
            session_root = self._get_session_root_dir()
        except Exception:
            session_root = Path.cwd()

        if isinstance(mic_rows, list):
            for w in mic_rows:
                if not isinstance(w, dict):
                    continue
                ws_id = str(w.get("id") or "") or str(uuid.uuid4())
                ws_name = str(w.get("name") or "Workspace")
                ws_obj = MicroscopyWorkspace(id=ws_id, name=ws_name, datasets=[])

                ds_rows = w.get("datasets") or []
                if isinstance(ds_rows, list):
                    for d in ds_rows:
                        if not isinstance(d, dict):
                            continue
                        did = str(d.get("id") or "") or str(uuid.uuid4())
                        file_path = str(d.get("file_path") or "").strip()
                        display_name = str(d.get("display_name") or (Path(file_path).stem if file_path else "dataset"))
                        created_at = str(d.get("created_at") or "")
                        notes = str(d.get("notes") or "")
                        last_macro_run = (None if d.get("last_macro_run") in (None, "") else str(d.get("last_macro_run")))

                        out_dir = str(d.get("output_dir") or "").strip()
                        if not out_dir:
                            out_dir = str(session_root / "microscopy" / _safe_dirname(ws_name) / _safe_dirname(Path(file_path).stem if file_path else display_name))

                        try:
                            Path(out_dir).mkdir(parents=True, exist_ok=True)
                        except Exception:
                            pass

                        ds_obj = MicroscopyDataset(
                            id=did,
                            display_name=display_name,
                            file_path=file_path,
                            workspace_id=ws_id,
                            created_at=created_at,
                            notes=notes,
                            output_dir=out_dir,
                            last_macro_run=last_macro_run,
                        )
                        ws_obj.datasets.append(ds_obj)

                restored_mic_wss.append(ws_obj)

        try:
            self.workspace.microscopy_workspaces = restored_mic_wss
            self.workspace.active_microscopy_workspace_id = (None if not mic_active else str(mic_active))
        except Exception:
            pass

        try:
            if self._microscopy_view is not None:
                self._microscopy_view.refresh_from_workspace(select_first=False)
        except Exception:
            pass

        # Restore UI vars that affect loading/indexing first
        try:
            rt_unit = str(ui.get("rt_unit") or "minutes")
            if rt_unit in ("minutes", "seconds"):
                self.rt_unit_var.set(rt_unit)
        except Exception:
            pass
        try:
            pol = str(ui.get("polarity") or "all")
            if pol in ("all", "positive", "negative"):
                self.polarity_var.set(pol)
        except Exception:
            pass

        # Other UI state (best-effort)
        def _set_var(var: Any, val: Any) -> None:
            try:
                var.set(val)
            except Exception:
                pass

        _set_var(self.show_tic_var, bool(ui.get("show_tic", True)))
        _set_var(self.show_spectrum_var, bool(ui.get("show_spectrum", True)))
        _set_var(self.show_uv_var, bool(ui.get("show_uv", True)))

        _set_var(self.uv_label_from_ms_var, bool(ui.get("uv_label_from_ms", False)))
        try:
            _set_var(self.uv_label_from_ms_top_n_var, int(ui.get("uv_label_from_ms_top_n", 3)))
        except Exception:
            pass
        try:
            _set_var(self.uv_label_min_conf_var, float(ui.get("uv_label_min_conf", 0.0)))
        except Exception:
            pass

        titles = ui.get("titles") or {}
        _set_var(self.tic_title_var, str(titles.get("tic_title", "")))
        _set_var(self.tic_xlabel_var, str(titles.get("tic_xlabel", "")))
        _set_var(self.tic_ylabel_var, str(titles.get("tic_ylabel", "")))
        _set_var(self.spec_title_var, str(titles.get("spec_title", "")))
        _set_var(self.spec_xlabel_var, str(titles.get("spec_xlabel", "")))
        _set_var(self.spec_ylabel_var, str(titles.get("spec_ylabel", "")))
        _set_var(self.uv_title_var, str(titles.get("uv_title", "")))
        _set_var(self.uv_xlabel_var, str(titles.get("uv_xlabel", "")))
        _set_var(self.uv_ylabel_var, str(titles.get("uv_ylabel", "")))

        fonts = ui.get("fonts") or {}
        try:
            _set_var(self.title_fontsize_var, int(fonts.get("title", self.title_fontsize_var.get())))
            _set_var(self.label_fontsize_var, int(fonts.get("label", self.label_fontsize_var.get())))
            _set_var(self.tick_fontsize_var, int(fonts.get("tick", self.tick_fontsize_var.get())))
        except Exception:
            pass

        limits = ui.get("limits") or {}
        tic_lim = limits.get("tic") or {}
        spec_lim = limits.get("spec") or {}
        uv_lim = limits.get("uv") or {}
        _set_var(self.tic_xlim_min_var, str(tic_lim.get("x_min", "")))
        _set_var(self.tic_xlim_max_var, str(tic_lim.get("x_max", "")))
        _set_var(self.tic_ylim_min_var, str(tic_lim.get("y_min", "")))
        _set_var(self.tic_ylim_max_var, str(tic_lim.get("y_max", "")))
        _set_var(self.spec_xlim_min_var, str(spec_lim.get("x_min", "")))
        _set_var(self.spec_xlim_max_var, str(spec_lim.get("x_max", "")))
        _set_var(self.spec_ylim_min_var, str(spec_lim.get("y_min", "")))
        _set_var(self.spec_ylim_max_var, str(spec_lim.get("y_max", "")))
        _set_var(self.uv_xlim_min_var, str(uv_lim.get("x_min", "")))
        _set_var(self.uv_xlim_max_var, str(uv_lim.get("x_max", "")))
        _set_var(self.uv_ylim_min_var, str(uv_lim.get("y_min", "")))
        _set_var(self.uv_ylim_max_var, str(uv_lim.get("y_max", "")))

        ann = ui.get("annotations") or {}
        _set_var(self.annotate_peaks_var, bool(ann.get("annotate_peaks", False)))
        try:
            _set_var(self.annotate_top_n_var, int(ann.get("annotate_top_n", 10)))
        except Exception:
            pass
        try:
            _set_var(self.annotate_min_rel_var, float(ann.get("annotate_min_rel", 0.05)))
        except Exception:
            pass
        _set_var(self.drag_annotations_var, bool(ann.get("drag_annotations", True)))

        poly = ui.get("polymer") or {}
        _set_var(self.poly_enabled_var, bool(poly.get("enabled", False)))
        _set_var(self.poly_monomers_text_var, str(poly.get("monomers_text", "")))
        try:
            _set_var(self.poly_bond_delta_var, float(poly.get("bond_delta", -18.010565)))
            _set_var(self.poly_extra_delta_var, float(poly.get("extra_delta", 0.0)))
            _set_var(self.poly_adduct_mass_var, float(poly.get("adduct_mass", 1.007276)))
            _set_var(self.poly_cluster_adduct_mass_var, float(poly.get("cluster_adduct_mass", -1.007276)))
            _set_var(self.poly_tol_value_var, float(poly.get("tol_value", 0.02)))
            _set_var(self.poly_min_rel_int_var, float(poly.get("min_rel_int", 0.01)))
        except Exception:
            pass
        _set_var(self.poly_decarb_enabled_var, bool(poly.get("decarb", False)))
        _set_var(self.poly_oxid_enabled_var, bool(poly.get("oxid", False)))
        _set_var(self.poly_cluster_enabled_var, bool(poly.get("cluster", False)))
        _set_var(self.poly_adduct_na_var, bool(poly.get("adduct_na", False)))
        _set_var(self.poly_adduct_k_var, bool(poly.get("adduct_k", False)))
        _set_var(self.poly_adduct_cl_var, bool(poly.get("adduct_cl", False)))
        _set_var(self.poly_adduct_formate_var, bool(poly.get("adduct_formate", False)))
        _set_var(self.poly_adduct_acetate_var, bool(poly.get("adduct_acetate", False)))
        _set_var(self.poly_charges_var, str(poly.get("charges", "1")))
        try:
            _set_var(self.poly_max_dp_var, int(poly.get("max_dp", 12)))
        except Exception:
            pass
        _set_var(self.poly_tol_unit_var, str(poly.get("tol_unit", "Da")))

        align = ui.get("uv_ms_alignment") or {}
        try:
            self._uv_ms_rt_offset_min = float(align.get("offset_min", getattr(self, "_uv_ms_rt_offset_min", 0.0)))
            self.uv_ms_rt_offset_var.set(f"{float(self._uv_ms_rt_offset_min):.3f}")
        except Exception:
            pass
        _set_var(self.uv_ms_align_enabled_var, bool(align.get("align_enabled", False)))
        try:
            uv_anc = [float(x) for x in (align.get("anchors_uv") or [])]
            ms_anc = [float(x) for x in (align.get("anchors_ms") or [])]
            if uv_anc and ms_anc and len(uv_anc) == len(ms_anc):
                self._uv_ms_align_uv_rts = np.asarray(uv_anc, dtype=float)
                self._uv_ms_align_ms_rts = np.asarray(ms_anc, dtype=float)
            else:
                self._uv_ms_align_uv_rts = None
                self._uv_ms_align_ms_rts = None
        except Exception:
            self._uv_ms_align_uv_rts = None
            self._uv_ms_align_ms_rts = None

        sim = ui.get("sim") or {}
        try:
            lp = sim.get("last_params")
            if isinstance(lp, list) and len(lp) == 4:
                self._sim_last_params = (float(lp[0]), float(lp[1]), str(lp[2]), bool(lp[3]))
        except Exception:
            pass

        # Prepare restore context for async callbacks
        pending_mzml: Dict[str, Any] = {}
        for item in (ws.get("mzml_files") or []):
            if not isinstance(item, dict):
                continue
            p = str(item.get("path") or "").strip()
            if p:
                pending_mzml[str(Path(p).expanduser().resolve())] = dict(item)

        pending_uv: Dict[str, Any] = {}
        for item in (ws.get("uv_files") or []):
            if not isinstance(item, dict):
                continue
            p = str(item.get("path") or "").strip()
            if p:
                pending_uv[str(Path(p).expanduser().resolve())] = dict(item)

        active_mzml_path = ws.get("active_mzml_path")
        active_mzml_resolved = (str(Path(active_mzml_path).expanduser().resolve()) if active_mzml_path else None)
        active_uv_path = ws.get("active_uv_path")
        active_uv_resolved = (str(Path(active_uv_path).expanduser().resolve()) if active_uv_path else None)

        pending_ftir: List[Dict[str, Any]] = []
        for item in (ws.get("ftir_files") or []):
            if isinstance(item, str):
                p = str(item).strip()
                if p:
                    pending_ftir.append({"path": str(Path(p).expanduser().resolve())})
                continue
            if not isinstance(item, dict):
                continue
            p = str(item.get("path") or "").strip()
            if p:
                pending_ftir.append(
                    {
                        "id": item.get("id"),
                        "path": str(Path(p).expanduser().resolve()),
                        "name": item.get("name"),
                        "y_mode": item.get("y_mode"),
                        "x_units": item.get("x_units"),
                        "y_units": item.get("y_units"),
                        "peak_settings": item.get("peak_settings"),
                        "peaks": item.get("peaks"),
                        "peak_label_overrides": item.get("peak_label_overrides"),
                        "peak_suppressed": item.get("peak_suppressed"),
                        "peak_label_positions": item.get("peak_label_positions"),
                    }
                )

        active_ftir_id = ws.get("active_ftir_id")
        active_ftir_id_s = (str(active_ftir_id).strip() if active_ftir_id else None)
        active_ftir_path = ws.get("active_ftir_path")
        active_ftir_resolved = (str(Path(active_ftir_path).expanduser().resolve()) if active_ftir_path else None)

        ctx: Dict[str, Any] = {
            "pending_mzml": pending_mzml,
            "pending_uv": pending_uv,
            "pending_ftir": pending_ftir,
            "expected_mzml": 0,
            "expected_uv": 0,
            "expected_ftir": 0,
            "done_mzml": 0,
            "done_uv": 0,
            "done_ftir": 0,
            "missing_mzml": [],
            "missing_uv": [],
            "missing_ftir": [],
            "active_mzml_path": active_mzml_resolved,
            "active_uv_path": active_uv_resolved,
            "active_ftir_id": active_ftir_id_s,
            "active_ftir_path": active_ftir_resolved,
            "overlay_state": (ui.get("overlay") if isinstance(ui, dict) else None),
            "ftir_restore_payload": {
                "ftir_workspaces": ws.get("ftir_workspaces"),
                "active_ftir_workspace_id": ws.get("active_ftir_workspace_id"),
                "ftir_overlay_groups": ws.get("ftir_overlay_groups"),
                "active_ftir_overlay_group_id": ws.get("active_ftir_overlay_group_id"),
            },
        }
        self._workspace_restore_ctx = ctx  # type: ignore[attr-defined]

        # Kick off loads (UV first so path->id mapping becomes available early)
        for p_str in pending_uv.keys():
            p = Path(p_str)
            if not p.exists():
                ctx["missing_uv"].append(p_str)
                continue
            ctx["expected_uv"] += 1
            self._add_uv_session_from_path_async(p)

        for p_str in pending_mzml.keys():
            p = Path(p_str)
            if not p.exists():
                ctx["missing_mzml"].append(p_str)
                continue
            ctx["expected_mzml"] += 1
            self._add_session_from_path_async(p, make_active=False)

        for item in pending_ftir:
            p_str = str((item or {}).get("path") or "").strip()
            if not p_str:
                continue
            p = Path(p_str)
            if not p.exists():
                ctx["missing_ftir"].append(p_str)
                continue
            ctx["expected_ftir"] += 1
            self._add_ftir_from_path_async(
                p,
                make_active=False,
                dataset_id=(None if not isinstance(item, dict) else item.get("id")),
                name=(None if not isinstance(item, dict) else item.get("name")),
                y_mode=(None if not isinstance(item, dict) else item.get("y_mode")),
                x_units=(None if not isinstance(item, dict) else item.get("x_units")),
                y_units=(None if not isinstance(item, dict) else item.get("y_units")),
                peak_settings=(None if not isinstance(item, dict) else item.get("peak_settings")),
                peaks=(None if not isinstance(item, dict) else item.get("peaks")),
                peak_label_overrides=(None if not isinstance(item, dict) else item.get("peak_label_overrides")),
                peak_suppressed=(None if not isinstance(item, dict) else item.get("peak_suppressed")),
                peak_label_positions=(None if not isinstance(item, dict) else item.get("peak_label_positions")),
            )

        self._maybe_finalize_workspace_restore()

    def _maybe_finalize_workspace_restore(self) -> None:
        ctx = getattr(self, "_workspace_restore_ctx", None)
        if not isinstance(ctx, dict):
            return
        if int(ctx.get("done_uv", 0)) < int(ctx.get("expected_uv", 0)):
            return
        if int(ctx.get("done_mzml", 0)) < int(ctx.get("expected_mzml", 0)):
            return
        if int(ctx.get("done_ftir", 0)) < int(ctx.get("expected_ftir", 0)):
            return

        pending_mzml: Dict[str, Any] = ctx.get("pending_mzml") or {}
        uv_path_to_id: Dict[str, str] = {}
        for uv_id, uv_sess in (self._uv_sessions or {}).items():
            try:
                uv_path_to_id[str(uv_sess.path)] = str(uv_id)
            except Exception:
                continue
        uv_load_order_by_path: Dict[str, int] = ctx.get("uv_load_order_by_path") or {}

        def _decode_custom_labels(payload: Any) -> Dict[str, List[CustomLabel]]:
            out: Dict[str, List[CustomLabel]] = {}
            if not isinstance(payload, dict):
                return out
            for spec_id, items in payload.items():
                rows: List[CustomLabel] = []
                if isinstance(items, list):
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        try:
                            rows.append(CustomLabel(label=str(it.get("label", "")), mz=float(it.get("mz")), snap_to_nearest_peak=bool(it.get("snap", True))))
                        except Exception:
                            continue
                out[str(spec_id)] = rows
            return out

        def _decode_overrides(payload: Any) -> Dict[str, Dict[Tuple[str, float], Optional[str]]]:
            out: Dict[str, Dict[Tuple[str, float], Optional[str]]] = {}
            if not isinstance(payload, dict):
                return out
            for spec_id, items in payload.items():
                m: Dict[Tuple[str, float], Optional[str]] = {}
                if isinstance(items, list):
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        try:
                            kind = str(it.get("kind", ""))
                            mz_key = float(it.get("mz"))
                            val = it.get("value")
                            m[(kind, mz_key)] = (None if val is None else str(val))
                        except Exception:
                            continue
                out[str(spec_id)] = m
            return out

        def _decode_uv_labels(payload: Any) -> Dict[float, List[UVLabelState]]:
            out: Dict[float, List[UVLabelState]] = {}
            if not isinstance(payload, list):
                return out
            for row in payload:
                if not isinstance(row, dict):
                    continue
                try:
                    uv_rt = float(row.get("uv_rt"))
                except Exception:
                    continue
                labels: List[UVLabelState] = []
                for st in (row.get("labels") or []):
                    if not isinstance(st, dict):
                        continue
                    try:
                        xy = st.get("xytext") or [0.0, 0.0]
                        labels.append(
                            UVLabelState(
                                text=str(st.get("text", "")),
                                xytext=(float(xy[0]), float(xy[1])),
                                confidence=float(st.get("confidence", 0.0) or 0.0),
                                rt_delta_min=float(st.get("rt_delta_min", 0.0) or 0.0),
                                uv_peak_score=float(st.get("uv_peak_score", 0.0) or 0.0),
                                tic_peak_score=float(st.get("tic_peak_score", 0.0) or 0.0),
                            )
                        )
                    except Exception:
                        continue
                out[float(uv_rt)] = labels
            return out

        # Restore mzML order as saved
        mzml_order_paths = list(ctx.get("mzml_order_paths") or [])
        if mzml_order_paths:
            path_to_sid: Dict[str, str] = {}
            for sid, sess in (self._sessions or {}).items():
                try:
                    path_to_sid[str(sess.path)] = str(sid)
                except Exception:
                    continue
            new_order: List[str] = []
            for p in mzml_order_paths:
                sid = path_to_sid.get(str(p))
                if sid and sid not in new_order:
                    new_order.append(str(sid))
            for sid in list(self._session_order):
                if sid not in new_order:
                    new_order.append(str(sid))
            self._session_order = list(new_order)
            max_load = 0
            for i, sid in enumerate(self._session_order, start=1):
                try:
                    order_val = int(i)
                    self._sessions[sid].load_order = order_val
                    if order_val > max_load:
                        max_load = order_val
                except Exception:
                    pass
            try:
                self._session_load_counter = max(int(self._session_load_counter or 0), int(max_load))
            except Exception:
                pass
            if self._ws_tree is not None:
                for sid in self._session_order:
                    try:
                        if self._ws_tree.exists(str(sid)):
                            self._ws_tree.move(str(sid), "", "end")
                    except Exception:
                        pass

        # Restore UV order as saved
        uv_order_paths = list(ctx.get("uv_order_paths") or [])
        if uv_order_paths:
            new_uv_order: List[str] = []
            for p in uv_order_paths:
                uv_id = uv_path_to_id.get(str(p))
                if uv_id and uv_id not in new_uv_order:
                    new_uv_order.append(str(uv_id))
            for uv_id in list(self._uv_order):
                if uv_id not in new_uv_order:
                    new_uv_order.append(str(uv_id))
            self._uv_order = list(new_uv_order)
            max_uv_load = 0
            for i, uv_id in enumerate(self._uv_order, start=1):
                try:
                    uv_path = str(self._uv_sessions[uv_id].path)
                    saved_order = uv_load_order_by_path.get(uv_path)
                    order_val = int(saved_order) if saved_order is not None else int(i)
                    self._uv_sessions[uv_id].load_order = order_val
                    if order_val > max_uv_load:
                        max_uv_load = order_val
                except Exception:
                    continue
            try:
                self._uv_load_counter = max(int(self._uv_load_counter or 0), int(max_uv_load))
            except Exception:
                pass
            if self._uv_ws_tree is not None:
                for uv_id in self._uv_order:
                    try:
                        if self._uv_ws_tree.exists(str(uv_id)):
                            self._uv_ws_tree.move(str(uv_id), "", "end")
                    except Exception:
                        pass

        # Apply per-session saved fields now that UV sessions are loaded
        for sid, sess in list(self._sessions.items()):
            path_key = str(sess.path)
            saved = pending_mzml.get(path_key)
            if not isinstance(saved, dict):
                continue
            try:
                sess.display_name = str(saved.get("display_name") or sess.display_name)
            except Exception:
                pass
            try:
                sess.overlay_color = str(saved.get("overlay_color") or getattr(sess, "overlay_color", ""))
            except Exception:
                pass
            try:
                sess.last_selected_rt_min = (None if saved.get("last_selected_rt_min") is None else float(saved.get("last_selected_rt_min")))
            except Exception:
                pass
            try:
                sess.last_scan_index = (None if saved.get("last_scan_index") is None else int(saved.get("last_scan_index")))
            except Exception:
                pass
            try:
                sess.last_polarity_filter = (None if saved.get("last_polarity_filter") is None else str(saved.get("last_polarity_filter")))
            except Exception:
                pass

            sess.custom_labels_by_spectrum = _decode_custom_labels(saved.get("custom_labels_by_spectrum"))
            sess.spec_label_overrides = _decode_overrides(saved.get("spec_label_overrides"))

            # Linked UV
            linked_uv_path = saved.get("linked_uv_path")
            if isinstance(linked_uv_path, str) and linked_uv_path in uv_path_to_id:
                sess.linked_uv_id = uv_path_to_id[linked_uv_path]
            else:
                sess.linked_uv_id = None

            # UV labels remap (UV path -> UV id)
            uv_labels_by_uv_path = saved.get("uv_labels_by_uv_path")
            sess.uv_labels_by_uv_id = {}
            if isinstance(uv_labels_by_uv_path, dict):
                for uv_path, payload in uv_labels_by_uv_path.items():
                    if not isinstance(uv_path, str):
                        continue
                    uv_id = uv_path_to_id.get(uv_path)
                    if not uv_id:
                        continue
                    sess.uv_labels_by_uv_id[str(uv_id)] = _decode_uv_labels(payload)

            try:
                self._refresh_ws_tree_row(str(sid))
            except Exception:
                pass

        # Restore overlay session if persisted
        overlay_state = ctx.get("overlay_state")
        if isinstance(overlay_state, dict):
            try:
                ids = [str(i) for i in (overlay_state.get("dataset_ids") or []) if str(i) in self._sessions]
                if len(ids) >= 2:
                    colors = {str(k): str(v) for k, v in (overlay_state.get("colors") or {}).items()}
                    for sid in ids:
                        sess = self._sessions.get(str(sid))
                        if sess is not None:
                            sess.overlay_selected = True
                            if str(sid) not in colors:
                                colors[str(sid)] = self._ensure_overlay_color(str(sid))
                            try:
                                self._refresh_ws_tree_row(str(sid))
                            except Exception:
                                pass
                    self._overlay_session = OverlaySession(
                        dataset_ids=list(ids),
                        mode=str(overlay_state.get("mode") or "Stacked"),
                        colors=dict(colors),
                        persist=bool(overlay_state.get("persist", False)),
                        show_uv=bool(overlay_state.get("show_uv", True)),
                        stack_spectra=bool(overlay_state.get("stack_spectra", False)),
                        show_labels_all=bool(overlay_state.get("show_labels_all", False)),
                        multi_drag=bool(overlay_state.get("multi_drag", False)),
                        active_dataset_id=(None if not overlay_state.get("active_dataset_id") else str(overlay_state.get("active_dataset_id"))),
                    )
                    try:
                        self._overlay_mode_var.set(str(overlay_state.get("mode") or "Stacked"))
                        self._overlay_show_uv_var.set(bool(overlay_state.get("show_uv", True)))
                        self._overlay_stack_spectra_var.set(bool(overlay_state.get("stack_spectra", False)))
                        self._overlay_show_labels_all_var.set(bool(overlay_state.get("show_labels_all", False)))
                        self._overlay_multi_drag_var.set(bool(overlay_state.get("multi_drag", False)))
                        self._overlay_persist_var.set(bool(overlay_state.get("persist", False)))
                        self._overlay_scheme_var.set(str(overlay_state.get("color_scheme") or "Auto (Tableau)"))
                        self._overlay_single_hue_color = str(overlay_state.get("single_hue_color") or "#1f77b4")
                    except Exception:
                        pass
                    self._refresh_overlay_view()
                    try:
                        self._apply_overlay_color_scheme(ids=list(ids))
                    except Exception:
                        pass
            except Exception:
                pass


        # Activate saved active session (by path)
        active_path = ctx.get("active_mzml_path")
        active_sid: Optional[str] = None
        if isinstance(active_path, str):
            for sid, sess in self._sessions.items():
                if str(sess.path) == active_path:
                    active_sid = str(sid)
                    break
        if active_sid is None and self._session_order:
            active_sid = str(self._session_order[0])

        if active_sid is not None:
            try:
                self._set_active_session(active_sid)
            except Exception:
                pass
        else:
            # No mzML session active (or none loaded): restore active UV selection if possible
            active_uv_path = ctx.get("active_uv_path")
            if isinstance(active_uv_path, str) and active_uv_path in uv_path_to_id:
                self._active_uv_id = uv_path_to_id[active_uv_path]
                try:
                    self._plot_uv()
                except Exception:
                    pass

        # Restore FTIR active dataset (by id, fallback to path)
        try:
            active_ftir_id = ctx.get("active_ftir_id")
            if isinstance(active_ftir_id, str) and active_ftir_id:
                if any(str(getattr(d, "id", "")) == str(active_ftir_id) for d in self.workspace.ftir_datasets):
                    self.workspace.active_ftir_id = str(active_ftir_id)
            if not self.workspace.active_ftir_id:
                active_ftir_path = ctx.get("active_ftir_path")
                if isinstance(active_ftir_path, str) and active_ftir_path:
                    for d in self.workspace.ftir_datasets:
                        if str(getattr(d, "path", "")) == str(active_ftir_path):
                            self.workspace.active_ftir_id = str(getattr(d, "id", ""))
                            break
            if self._ftir_view is not None:
                self._ftir_view.apply_restored_ftir_state(ctx.get("ftir_restore_payload"))
        except Exception:
            pass

        missing_mzml = list(ctx.get("missing_mzml") or [])
        missing_uv = list(ctx.get("missing_uv") or [])
        missing_ftir = list(ctx.get("missing_ftir") or [])
        if missing_mzml or missing_uv or missing_ftir:
            msg = "Some files were missing and skipped:\n\n"
            if missing_mzml:
                msg += "Missing mzML:\n" + "\n".join(missing_mzml[:15]) + ("\n…" if len(missing_mzml) > 15 else "") + "\n\n"
            if missing_uv:
                msg += "Missing UV CSV:\n" + "\n".join(missing_uv[:15]) + ("\n…" if len(missing_uv) > 15 else "")
                msg += "\n\n" if missing_ftir else ""
            if missing_ftir:
                msg += "Missing FTIR CSV:\n" + "\n".join(missing_ftir[:15]) + ("\n…" if len(missing_ftir) > 15 else "")
            try:
                messagebox.showwarning("Workspace loaded", msg, parent=self)
            except Exception:
                pass

        # Done
        try:
            delattr(self, "_workspace_restore_ctx")
        except Exception:
            self._workspace_restore_ctx = None  # type: ignore[attr-defined]

    def _maybe_finalize_lcms_workspace_restore(self) -> None:
        ctx = getattr(self, "_lcms_workspace_restore_ctx", None)
        if not isinstance(ctx, dict):
            return
        if int(ctx.get("done_uv", 0)) < int(ctx.get("expected_uv", 0)):
            return
        if int(ctx.get("done_mzml", 0)) < int(ctx.get("expected_mzml", 0)):
            return

        pending_mzml: Dict[str, Any] = ctx.get("pending_mzml") or {}
        uv_path_to_id: Dict[str, str] = {}
        for uv_id, uv_sess in (self._uv_sessions or {}).items():
            try:
                uv_path_to_id[str(uv_sess.path)] = str(uv_id)
            except Exception:
                continue

        def _decode_annotations(payload: Any) -> Tuple[Dict[str, List[CustomLabel]], Dict[str, Dict[Tuple[str, float], Optional[str]]]]:
            custom_out: Dict[str, List[CustomLabel]] = {}
            overrides_out: Dict[str, Dict[Tuple[str, float], Optional[str]]] = {}
            if not isinstance(payload, dict):
                return custom_out, overrides_out
            for spec_id, items in payload.items():
                if not isinstance(items, list):
                    continue
                custom_rows: List[CustomLabel] = []
                override_rows: Dict[Tuple[str, float], Optional[str]] = {}
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    try:
                        mz_val = float(it.get("mz"))
                    except Exception:
                        continue
                    kind = str(it.get("kind") or "")
                    text = it.get("text")
                    suppressed = bool(it.get("suppressed", False))
                    if kind == "custom":
                        try:
                            custom_rows.append(CustomLabel(label=str(text or ""), mz=mz_val, snap_to_nearest_peak=True))
                        except Exception:
                            continue
                    else:
                        override_rows[(kind, mz_val)] = (None if suppressed else ("" if text is None else str(text)))
                if custom_rows:
                    custom_out[str(spec_id)] = custom_rows
                if override_rows:
                    overrides_out[str(spec_id)] = override_rows
            return custom_out, overrides_out

        # Apply per-session saved fields now that UV sessions are loaded
        for sid, sess in list(self._sessions.items()):
            path_key = str(sess.path)
            saved = pending_mzml.get(path_key)
            if not isinstance(saved, dict):
                continue
            pol = saved.get("polarity")
            if isinstance(pol, str) and pol in ("all", "positive", "negative"):
                sess.last_polarity_filter = str(pol)
            try:
                if saved.get("last_scan_index") is not None:
                    sess.last_scan_index = int(saved.get("last_scan_index"))
            except Exception:
                pass
            try:
                if saved.get("last_selected_rt_min") is not None:
                    sess.last_selected_rt_min = float(saved.get("last_selected_rt_min"))
            except Exception:
                pass

            ann_payload = saved.get("annotations")
            custom_labels, overrides = _decode_annotations(ann_payload)
            if custom_labels:
                sess.custom_labels_by_spectrum = custom_labels
            if overrides:
                sess.spec_label_overrides = overrides

            # Restore linked UV per session (if saved)
            try:
                linked_uv_path = saved.get("linked_uv_path")
                if isinstance(linked_uv_path, str) and linked_uv_path:
                    uv_id = uv_path_to_id.get(str(linked_uv_path))
                    if uv_id:
                        sess.linked_uv_id = str(uv_id)
                        try:
                            for d in getattr(self.workspace, "lcms_datasets", []) or []:
                                if str(getattr(d, "session_id", "")) == str(sid):
                                    d.uv_csv_path = str(self._uv_sessions[str(uv_id)].path)
                                    break
                        except Exception:
                            pass
            except Exception:
                pass

        # Linked UV
        linked_uv = ctx.get("linked_uv")
        if isinstance(linked_uv, dict):
            mzml_path = linked_uv.get("mzml_path")
            uv_path = linked_uv.get("uv_csv_path")
            if isinstance(mzml_path, str) and isinstance(uv_path, str):
                uv_id = uv_path_to_id.get(str(uv_path))
                if uv_id:
                    for sid, sess in list(self._sessions.items()):
                        if str(sess.path) == str(mzml_path):
                            sess.linked_uv_id = str(uv_id)
                            break
        try:
            self._refresh_uv_tree_links()
        except Exception:
            pass

        # Activate saved active session (by path)
        active_path = ctx.get("active_mzml_path")
        active_sid: Optional[str] = None
        if isinstance(active_path, str):
            for sid, sess in self._sessions.items():
                if str(sess.path) == active_path:
                    active_sid = str(sid)
                    break
        if active_sid is None and self._session_order:
            active_sid = str(self._session_order[0])
        if active_sid is not None:
            try:
                self._set_active_session(active_sid)
            except Exception:
                pass

        missing_mzml = list(ctx.get("missing_mzml") or [])
        missing_uv = list(ctx.get("missing_uv") or [])
        if missing_mzml or missing_uv:
            msg = "Some files were missing and skipped:\n\n"
            if missing_mzml:
                msg += "Missing mzML:\n" + "\n".join(missing_mzml[:15]) + ("\n…" if len(missing_mzml) > 15 else "") + "\n\n"
            if missing_uv:
                msg += "Missing UV CSV:\n" + "\n".join(missing_uv[:15]) + ("\n…" if len(missing_uv) > 15 else "")
            try:
                messagebox.showwarning("LCMS Workspace loaded", msg, parent=self)
            except Exception:
                pass

        try:
            if self.lcms_workspace_path is not None:
                self._set_status(f"Loaded LCMS workspace: {self.lcms_workspace_path.name}")
            else:
                self._set_status("Loaded LCMS workspace")
            self._update_status_by_tab()
        except Exception:
            pass

        try:
            delattr(self, "_lcms_workspace_restore_ctx")
        except Exception:
            self._lcms_workspace_restore_ctx = None

    def _save_workspace(self) -> None:
        initial_dir = self._default_lcms_workspace_dir()
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save LCMS Workspace",
            defaultextension=".lcms_workspace.json",
            filetypes=[("LCMS Workspace", "*.lcms_workspace.json"), ("JSON", "*.json"), ("All files", "*.*")],
            initialdir=(str(initial_dir) if initial_dir else None),
            initialfile="lcms_workspace.lcms_workspace.json",
        )
        if not path:
            return

        save_path = self._ensure_lcms_workspace_extension(Path(path))
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            payload = build_lcms_workspace_dict(self)
        except Exception as exc:
            messagebox.showerror("Save LCMS Workspace", f"Failed to capture LCMS state:\n\n{exc}", parent=self)
            return

        try:
            json.dumps(payload, default=json_default)
        except Exception as exc:
            messagebox.showerror("Save LCMS Workspace", f"Failed to serialize LCMS workspace:\n\n{exc}", parent=self)
            return

        try:
            atomic_write_json(save_path, payload)
        except Exception as exc:
            messagebox.showerror("Save LCMS Workspace", f"Failed to write file:\n\n{exc}", parent=self)
            return

        try:
            if not save_path.exists() or save_path.stat().st_size <= 0:
                raise IOError("Saved file is empty or missing.")
        except Exception as exc:
            messagebox.showerror("Save LCMS Workspace", f"Failed to verify saved file:\n\n{exc}", parent=self)
            return

        self.lcms_workspace_path = save_path
        self._last_lcms_workspace_dir = save_path.parent
        try:
            size = int(save_path.stat().st_size)
        except Exception:
            size = 0
        self._set_status(f"LCMS workspace saved: {save_path} ({size} bytes)")
        self._add_recent_lcms_workspace(save_path)

    def _get_active_microscopy_workspace(self) -> Optional[MicroscopyWorkspace]:
        try:
            wss = getattr(self.workspace, "microscopy_workspaces", None) or []
        except Exception:
            wss = []
        if not isinstance(wss, list) or not wss:
            return None

        try:
            active_id = getattr(self.workspace, "active_microscopy_workspace_id", None)
        except Exception:
            active_id = None

        if active_id:
            for ws in wss:
                if str(getattr(ws, "id", "")) == str(active_id):
                    return ws
        return wss[0]

    def _encode_microscopy_workspace(self, ws_obj: MicroscopyWorkspace) -> Dict[str, Any]:
        ds_rows: List[Dict[str, Any]] = []
        for d in (getattr(ws_obj, "datasets", None) or []):
            ds_rows.append(
                {
                    "id": str(getattr(d, "id", "")),
                    "display_name": str(getattr(d, "display_name", "")),
                    "file_path": str(getattr(d, "file_path", "")),
                    "workspace_id": str(getattr(d, "workspace_id", getattr(ws_obj, "id", ""))),
                    "created_at": str(getattr(d, "created_at", "")),
                    "notes": str(getattr(d, "notes", "")),
                    "output_dir": str(getattr(d, "output_dir", "")),
                    "last_macro_run": (None if getattr(d, "last_macro_run", None) is None else str(getattr(d, "last_macro_run"))),
                }
            )
        return {"id": str(getattr(ws_obj, "id", "")), "name": str(getattr(ws_obj, "name", "Workspace")), "datasets": ds_rows}

    def _decode_microscopy_workspace(self, row: Dict[str, Any], *, fallback_root: Path) -> MicroscopyWorkspace:
        def _safe_dirname(s: str) -> str:
            t = "".join(ch for ch in str(s) if ch.isalnum() or ch in ("-", "_", " ")).strip().replace(" ", "_")
            return t or "item"

        ws_id = str(row.get("id") or "") or str(uuid.uuid4())
        ws_name = str(row.get("name") or "Workspace")
        ws_obj = MicroscopyWorkspace(id=ws_id, name=ws_name, datasets=[])

        ds_rows = row.get("datasets") or []
        if isinstance(ds_rows, list):
            for d in ds_rows:
                if not isinstance(d, dict):
                    continue
                did = str(d.get("id") or "") or str(uuid.uuid4())
                file_path = str(d.get("file_path") or "").strip()
                display_name = str(d.get("display_name") or (Path(file_path).stem if file_path else "dataset"))
                created_at = str(d.get("created_at") or "")
                notes = str(d.get("notes") or "")
                last_macro_run = (None if d.get("last_macro_run") in (None, "") else str(d.get("last_macro_run")))

                out_dir = str(d.get("output_dir") or "").strip()
                if not out_dir:
                    out_dir = str(fallback_root / "microscopy" / _safe_dirname(ws_name) / _safe_dirname(Path(file_path).stem if file_path else display_name))

                try:
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                ws_obj.datasets.append(
                    MicroscopyDataset(
                        id=did,
                        display_name=display_name,
                        file_path=file_path,
                        workspace_id=ws_id,
                        created_at=created_at,
                        notes=notes,
                        output_dir=out_dir,
                        last_macro_run=last_macro_run,
                    )
                )

        return ws_obj

    def _save_microscopy_workspace(self) -> None:
        ws_obj = self._get_active_microscopy_workspace()
        if ws_obj is None:
            messagebox.showinfo("Save Microscopy Workspace", "No microscopy workspace to save.")
            return

        default_name = "microscopy_workspace.microscopy.json"
        try:
            nm = str(getattr(ws_obj, "name", "workspace") or "workspace").strip()
            nm = "".join(ch for ch in nm if ch.isalnum() or ch in ("-", "_", " ")).strip().replace(" ", "_")
            if nm:
                default_name = f"{nm}.microscopy.json"
        except Exception:
            pass

        path = filedialog.asksaveasfilename(
            title="Save Microscopy Workspace",
            defaultextension=".microscopy.json",
            initialfile=default_name,
            filetypes=[("Microscopy workspace", "*.microscopy.json"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        payload: Dict[str, Any] = {
            "schema": "microscopy_workspace",
            "schema_version": 1,
            "app": str(APP_NAME),
            "saved_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "workspace": self._encode_microscopy_workspace(ws_obj),
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self._set_status(f"Saved Microscopy workspace: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Save Microscopy Workspace", f"Failed to save:\n\n{exc}")

    def _load_microscopy_workspace(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Microscopy Workspace",
            filetypes=[("Microscopy workspace", "*.microscopy.json"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            messagebox.showerror("Load Microscopy Workspace", f"Failed to read file:\n\n{exc}")
            return

        if not isinstance(payload, dict):
            messagebox.showerror("Load Microscopy Workspace", "Invalid file format (expected JSON object).")
            return

        if str(payload.get("schema") or "") != "microscopy_workspace":
            messagebox.showerror("Load Microscopy Workspace", "Invalid file (not a microscopy workspace snapshot).")
            return

        try:
            ver = int(payload.get("schema_version") or 0)
        except Exception:
            ver = 0
        if ver != 1:
            messagebox.showerror("Load Microscopy Workspace", f"Unsupported microscopy workspace schema_version={ver}.")
            return

        ws_row = payload.get("workspace")
        if not isinstance(ws_row, dict):
            messagebox.showerror("Load Microscopy Workspace", "Invalid file (missing workspace object).")
            return

        fallback_root = Path(path).resolve().parent
        try:
            ws_obj = self._decode_microscopy_workspace(ws_row, fallback_root=fallback_root)
        except Exception as exc:
            messagebox.showerror("Load Microscopy Workspace", f"Failed to parse workspace:\n\n{exc}")
            return

        try:
            wss = getattr(self.workspace, "microscopy_workspaces", None)
            if not isinstance(wss, list):
                self.workspace.microscopy_workspaces = []
                wss = self.workspace.microscopy_workspaces
        except Exception:
            self.workspace.microscopy_workspaces = []
            wss = self.workspace.microscopy_workspaces

        # Replace by id if already present, else append.
        replaced = False
        try:
            for i, existing in enumerate(list(wss)):
                if str(getattr(existing, "id", "")) == str(ws_obj.id):
                    wss[i] = ws_obj
                    replaced = True
                    break
        except Exception:
            pass
        if not replaced:
            try:
                wss.append(ws_obj)
            except Exception:
                pass

        try:
            self.workspace.active_microscopy_workspace_id = str(ws_obj.id)
        except Exception:
            pass

        try:
            if self._microscopy_view is not None:
                self._microscopy_view.refresh_from_workspace(select_first=True)
        except Exception:
            pass

        try:
            self._set_status(f"Loaded Microscopy workspace: {ws_obj.name}")
        except Exception:
            pass

        try:
            state = self._workspace_state_to_dict()
        except Exception as exc:
            messagebox.showerror("Save Workspace", f"Failed to capture workspace state:\n\n{exc}", parent=self)
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            messagebox.showerror("Save Workspace", f"Failed to write file:\n\n{exc}", parent=self)
            return

        messagebox.showinfo("Save Workspace", f"Saved workspace:\n\n{path}", parent=self)

    def _load_workspace(self) -> None:
        if self._sessions or self._uv_sessions:
            if not messagebox.askyesno(
                "Load LCMS Workspace",
                "Loading an LCMS workspace will replace the current LCMS (mzML/UV) workspace. Continue?",
                parent=self,
            ):
                return

        initial_dir = self._default_lcms_workspace_dir()
        path = filedialog.askopenfilename(
            parent=self,
            title="Load LCMS Workspace",
            filetypes=[("LCMS Workspace", "*.lcms_workspace.json"), ("JSON", "*.json"), ("All files", "*.*")],
            initialdir=(str(initial_dir) if initial_dir else None),
        )
        if not path:
            return

        self._load_lcms_workspace_from_path(Path(path))

    def _default_lcms_workspace_dir(self) -> Optional[Path]:
        if self._last_lcms_workspace_dir is not None:
            return self._last_lcms_workspace_dir
        try:
            docs = Path.home() / "Documents"
            if docs.exists():
                return docs
        except Exception:
            pass
        try:
            return Path.home()
        except Exception:
            return None

    def _ensure_lcms_workspace_extension(self, path: Path) -> Path:
        p = Path(path)
        low = str(p).lower()
        if low.endswith(".lcms_workspace.json"):
            return p
        if p.suffix.lower() == ".json":
            return p.with_suffix(".lcms_workspace.json")
        return p.with_name(p.name + ".lcms_workspace.json")

    def _add_recent_lcms_workspace(self, path: Path) -> None:
        try:
            p = Path(path).expanduser().resolve()
        except Exception:
            p = Path(path)
        entry = {"path": str(p), "name": p.name, "saved_at": datetime.datetime.now().isoformat(timespec="seconds")}
        try:
            self._recent_lcms_workspaces = [e for e in (self._recent_lcms_workspaces or []) if str(e.get("path")) != str(p)]
        except Exception:
            self._recent_lcms_workspaces = []
        self._recent_lcms_workspaces.insert(0, entry)
        self._recent_lcms_workspaces = self._recent_lcms_workspaces[:10]
        self._refresh_recent_lcms_menu()

    def _refresh_recent_lcms_menu(self) -> None:
        menu = self._recent_lcms_menu
        if menu is None:
            return
        try:
            menu.delete(0, "end")
        except Exception:
            return
        if not self._recent_lcms_workspaces:
            menu.add_command(label="(None)", state="disabled")
            return
        for entry in list(self._recent_lcms_workspaces):
            p = str(entry.get("path") or "")
            name = str(entry.get("name") or p)
            ts = str(entry.get("saved_at") or "")
            label = f"{name} ({ts})" if ts else name
            menu.add_command(label=label, command=lambda path=p: self._load_lcms_workspace_from_path(Path(path)))

    def _reveal_lcms_workspace_in_explorer(self) -> None:
        if self.lcms_workspace_path is None:
            self._set_status("No LCMS workspace path available.")
            return
        try:
            subprocess.Popen(["explorer", "/select,", str(self.lcms_workspace_path)])
        except Exception as exc:
            messagebox.showerror("Reveal LCMS Workspace", f"Failed to open Explorer:\n\n{exc}", parent=self)

    def _load_lcms_workspace_from_path(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            messagebox.showerror("Load LCMS Workspace", f"Failed to read file:\n\n{exc}", parent=self)
            return

        if not isinstance(data, dict):
            messagebox.showerror("Load LCMS Workspace", "Invalid file format (expected JSON object).", parent=self)
            return
        if str(data.get("schema") or "") != "LCMS_WORKSPACE":
            messagebox.showerror("Load LCMS Workspace", "Invalid file (not an LCMS workspace).", parent=self)
            return
        try:
            ver = int(data.get("version") or 0)
        except Exception:
            ver = 0
        if ver != 1:
            messagebox.showerror("Load LCMS Workspace", f"Unsupported LCMS workspace version={ver}.", parent=self)
            return

        try:
            self.lcms_workspace_path = Path(path).expanduser().resolve()
            self._last_lcms_workspace_dir = self.lcms_workspace_path.parent
        except Exception:
            pass

        # Apply LCMS UI settings (best-effort)
        try:
            tic_settings = data.get("tic_settings") or {}
            self.show_tic_var.set(bool(tic_settings.get("show_tic", True)))
            self.tic_xlim_min_var.set(str(tic_settings.get("x_min", "")))
            self.tic_xlim_max_var.set(str(tic_settings.get("x_max", "")))
            self.tic_ylim_min_var.set(str(tic_settings.get("y_min", "")))
            self.tic_ylim_max_var.set(str(tic_settings.get("y_max", "")))
            self.tic_title_var.set(str(tic_settings.get("title", "")))
        except Exception:
            pass

        try:
            poly = data.get("polymer_settings") or {}
            self.poly_enabled_var.set(bool(poly.get("enabled", False)))
            monomers = poly.get("monomers") or []
            if isinstance(monomers, list):
                self.poly_monomers_text_var.set("\n".join(str(m) for m in monomers))
            self.poly_max_dp_var.set(int(poly.get("max_dp", 12)))
            self.poly_tol_value_var.set(float(poly.get("tolerance", 0.02)))
            self.poly_tol_unit_var.set(str(poly.get("tolerance_unit", "Da")))
            mode = poly.get("positive_mode") or {}
            if isinstance(mode, dict):
                self.poly_bond_delta_var.set(float(mode.get("bond_delta", -18.010565)))
                self.poly_extra_delta_var.set(float(mode.get("extra_delta", 0.0)))
                self.poly_adduct_mass_var.set(float(mode.get("adduct_mass", 1.007276)))
                self.poly_cluster_adduct_mass_var.set(float(mode.get("cluster_adduct_mass", -1.007276)))
                self.poly_adduct_na_var.set(bool(mode.get("adduct_na", False)))
                self.poly_adduct_k_var.set(bool(mode.get("adduct_k", False)))
                self.poly_adduct_cl_var.set(bool(mode.get("adduct_cl", False)))
                self.poly_adduct_formate_var.set(bool(mode.get("adduct_formate", False)))
                self.poly_adduct_acetate_var.set(bool(mode.get("adduct_acetate", False)))
                self.poly_charges_var.set(str(mode.get("charges", "1")))
                self.poly_decarb_enabled_var.set(bool(mode.get("decarb", False)))
                self.poly_oxid_enabled_var.set(bool(mode.get("oxid", False)))
                self.poly_cluster_enabled_var.set(bool(mode.get("cluster", False)))
                self.poly_min_rel_int_var.set(float(mode.get("min_rel_int", 0.01)))
        except Exception:
            pass

        base_dir = Path(path).expanduser().resolve().parent

        def _resolve_path(p: str) -> Path:
            pp = Path(p).expanduser()
            return pp if pp.is_absolute() else (base_dir / pp)

        mzml_rows = data.get("mzml_files") or []
        if not isinstance(mzml_rows, list):
            mzml_rows = []

        active_idx = 0
        try:
            active_idx = int(data.get("active_mzml_index") or 0)
        except Exception:
            active_idx = 0

        linked_uv = data.get("linked_uv") if isinstance(data.get("linked_uv"), dict) else None
        uv_rows = data.get("uv_files") or []
        if not isinstance(uv_rows, list):
            uv_rows = []
        annotations_by_mzml = data.get("annotations_by_mzml") if isinstance(data.get("annotations_by_mzml"), dict) else {}
        annotations_active = data.get("annotations") if isinstance(data.get("annotations"), dict) else None

        self._clear_lcms_for_load()

        ctx: Dict[str, Any] = {
            "pending_mzml": {},
            "pending_uv": {},
            "expected_mzml": 0,
            "expected_uv": 0,
            "done_mzml": 0,
            "done_uv": 0,
            "missing_mzml": [],
            "missing_uv": [],
            "active_mzml_path": None,
            "linked_uv": None,
            "mzml_order_paths": [],
            "uv_order_paths": [],
            "uv_load_order_by_path": {},
        }

        current_scan_index = None
        try:
            if data.get("current_scan_index") is not None:
                current_scan_index = int(data.get("current_scan_index"))
        except Exception:
            current_scan_index = None

        rt_unit = None
        ordered_mzml_rows = list(mzml_rows)
        try:
            indexed_mzml_rows = list(enumerate(list(mzml_rows)))
            ordered_mzml_rows = [
                row
                for _i, row in sorted(
                    indexed_mzml_rows,
                    key=lambda item: (
                        0 if isinstance((item[1] or {}).get("load_order"), (int, float, str)) else 1,
                        int(float((item[1] or {}).get("load_order"))) if isinstance((item[1] or {}).get("load_order"), (int, float, str)) else 0,
                        int(item[0]),
                    ),
                )
            ]
        except Exception:
            ordered_mzml_rows = list(mzml_rows)

        resolved_mzml_paths: List[str] = []
        for row in ordered_mzml_rows:
            if not isinstance(row, dict):
                continue
            p = str(row.get("path") or "").strip()
            if not p:
                continue
            rp = _resolve_path(p)
            resolved_mzml_paths.append(str(rp))
            rt_unit = rt_unit or str(row.get("rt_unit") or "").strip()
            saved = dict(row)
            saved["path"] = str(rp)
            try:
                l_uv = saved.get("linked_uv_path")
                if l_uv:
                    saved["linked_uv_path"] = str(_resolve_path(str(l_uv)))
            except Exception:
                pass
            ann = annotations_by_mzml.get(str(p)) if isinstance(annotations_by_mzml, dict) else None
            if ann is None:
                ann = annotations_by_mzml.get(str(rp)) if isinstance(annotations_by_mzml, dict) else None
            if isinstance(ann, dict):
                saved["annotations"] = ann
            ctx["pending_mzml"][str(rp)] = saved
            ctx["mzml_order_paths"].append(str(rp))

        ordered_uv_rows = list(uv_rows)
        try:
            indexed_uv_rows = list(enumerate(list(uv_rows)))
            ordered_uv_rows = [
                row
                for _i, row in sorted(
                    indexed_uv_rows,
                    key=lambda item: (
                        0 if isinstance((item[1] or {}).get("load_order"), (int, float, str)) else 1,
                        int(float((item[1] or {}).get("load_order"))) if isinstance((item[1] or {}).get("load_order"), (int, float, str)) else 0,
                        int(item[0]),
                    ),
                )
            ]
        except Exception:
            ordered_uv_rows = list(uv_rows)

        resolved_uv_paths: List[str] = []
        for row in ordered_uv_rows:
            if isinstance(row, str):
                p = str(row).strip()
                saved = {"path": p}
            elif isinstance(row, dict):
                p = str(row.get("path") or "").strip()
                saved = dict(row)
            else:
                continue
            if not p:
                continue
            rp = _resolve_path(p)
            resolved = str(rp)
            if resolved not in ctx["pending_uv"]:
                saved["path"] = resolved
                ctx["pending_uv"][resolved] = saved
            resolved_uv_paths.append(resolved)
            try:
                if isinstance(saved.get("load_order"), (int, float, str)):
                    ctx["uv_load_order_by_path"][resolved] = int(float(saved.get("load_order")))
            except Exception:
                pass

        if rt_unit in ("minutes", "seconds"):
            try:
                self.rt_unit_var.set(str(rt_unit))
            except Exception:
                pass

        if 0 <= int(active_idx) < len(resolved_mzml_paths):
            ctx["active_mzml_path"] = resolved_mzml_paths[int(active_idx)]
            if current_scan_index is not None and ctx["active_mzml_path"] in ctx["pending_mzml"]:
                try:
                    ctx["pending_mzml"][ctx["active_mzml_path"]]["last_scan_index"] = int(current_scan_index)
                except Exception:
                    pass
            if annotations_active and not annotations_by_mzml:
                try:
                    ctx["pending_mzml"][ctx["active_mzml_path"]]["annotations"] = dict(annotations_active)
                except Exception:
                    pass

        if isinstance(linked_uv, dict):
            mzml_p = str(linked_uv.get("mzml_path") or "").strip()
            uv_p = str(linked_uv.get("uv_csv_path") or "").strip()
            if mzml_p and uv_p:
                uv_offset = float(linked_uv.get("uv_ms_offset", getattr(self, "_uv_ms_rt_offset_min", 0.0) or 0.0))
                ctx["linked_uv"] = {
                    "mzml_path": str(_resolve_path(mzml_p)),
                    "uv_csv_path": str(_resolve_path(uv_p)),
                    "uv_ms_offset": uv_offset,
                }
                resolved_uv = str(_resolve_path(uv_p))
                if resolved_uv not in ctx["pending_uv"]:
                    ctx["pending_uv"][resolved_uv] = {"path": resolved_uv}
                    resolved_uv_paths.append(resolved_uv)
                try:
                    self._uv_ms_rt_offset_min = float(uv_offset)
                    self.uv_ms_rt_offset_var.set(f"{float(self._uv_ms_rt_offset_min):.3f}")
                except Exception:
                    pass

        if resolved_uv_paths:
            ctx["uv_order_paths"] = list(resolved_uv_paths)

        self._lcms_workspace_restore_ctx = ctx

        for p_str in list(ctx["pending_uv"].keys()):
            p = Path(p_str)
            if not p.exists():
                ctx["missing_uv"].append(p_str)
                continue
            ctx["expected_uv"] += 1
            self._add_uv_session_from_path_async(p)

        for p_str in list(ctx["pending_mzml"].keys()):
            p = Path(p_str)
            if not p.exists():
                ctx["missing_mzml"].append(p_str)
                continue
            ctx["expected_mzml"] += 1
            self._add_session_from_path_async(p, make_active=False)

        self._maybe_finalize_lcms_workspace_restore()

    def _open_instructions_window(self) -> None:
        if self._instructions_win is not None:
            try:
                if bool(self._instructions_win.winfo_exists()):
                    self._instructions_win.deiconify()
                    self._instructions_win.lift()
                    try:
                        self._instructions_win.focus_force()
                    except Exception:
                        pass
                    return
            except Exception:
                self._instructions_win = None

        win = InstructionWindow(self)
        self._instructions_win = win

    def _set_status(self, text: str) -> None:
        try:
            self._status.configure(text=str(text))
        except Exception:
            pass

    def _update_status_current(self) -> None:
        mzml = self._active_session_display_name() if getattr(self, "_active_session_id", None) else (self.mzml_path.name if self.mzml_path else "(no mzML)")
        uv_sess = self._active_uv_session()
        uv = uv_sess.path.name if uv_sess is not None else "(no UV linked)"
        pol = (self.polarity_var.get() or "all").strip()
        off = float(self._uv_ms_rt_offset_min)
        align = "auto-align ON" if (bool(self.uv_ms_align_enabled_var.get()) and self._uv_ms_align_uv_rts is not None) else "auto-align off"
        if self._current_spectrum_meta is None or self._current_scan_index is None or not self._filtered_meta:
            self._set_status(
                f"mzML: {mzml} | UV: {uv} | Polarity filter: {pol} | UV↔MS offset={off:.3f} min ({align}) | Click TIC/UV to load a spectrum | Right-click label to edit"
            )
            self._update_now_viewing_header()
            self._update_current_context_panel()
            return

        meta = self._current_spectrum_meta
        i = int(self._current_scan_index) + 1
        n = int(len(self._filtered_meta))
        self._set_status(
            f"mzML: {mzml} | UV: {uv} | {i}/{n} | RT={meta.rt_min:.4f} min | {meta.polarity or 'unknown'} | UV↔MS offset={off:.3f} min ({align}) | Right-click label to edit | Arrows: prev/next"
        )
        self._update_now_viewing_header()
        self._update_current_context_panel()

    def _update_now_viewing_header(self) -> None:
        try:
            var = getattr(self, "_now_view_var", None)
        except Exception:
            var = None
        if var is None:
            return
        mzml = self._active_session_display_name() if getattr(self, "_active_session_id", None) else (self.mzml_path.name if self.mzml_path else "(no mzML)")
        uv_sess = self._active_uv_session()
        uv = uv_sess.path.name if uv_sess is not None else "(no UV linked)"
        pol = (self.polarity_var.get() or "all").strip()
        off = float(self._uv_ms_rt_offset_min)
        align = "auto-align ON" if (bool(self.uv_ms_align_enabled_var.get()) and self._uv_ms_align_uv_rts is not None) else "auto-align off"
        if self._current_spectrum_meta is None:
            var.set(f"Now viewing: mzML={mzml} • UV={uv} • polarity={pol} • offset={off:.3f} min • {align}")
            self._update_current_context_panel()
            return
        meta = self._current_spectrum_meta
        var.set(
            f"Now viewing: mzML={mzml} • UV={uv} • RT={float(meta.rt_min):.4f} min • pol={meta.polarity or 'unknown'} • offset={off:.3f} min • {align}"
        )
        self._update_current_context_panel()

    def _apply_quick_annotate_settings(self) -> None:
        try:
            top_n = int(self.annotate_top_n_var.get())
            if top_n < 0:
                raise ValueError
            min_rel = float(self.annotate_min_rel_var.get())
            if not (0.0 <= min_rel <= 1.0):
                raise ValueError
            uv_top_n = int(self.uv_label_from_ms_top_n_var.get())
            if uv_top_n not in (2, 3):
                raise ValueError
        except Exception:
            messagebox.showerror(
                "Invalid value",
                "Top N must be >= 0, min rel must be 0..1, and UV peaks must be 2 or 3.",
                parent=self,
            )
            return

        self._redraw_spectrum_only()
        self._maybe_store_uv_ms_labels_for_current_spectrum(anchor_rt_min=None)
        self._plot_uv()
        self._update_status_current()

    def _reset_view_all(self) -> None:
        # Clear axis limits and reset toolbar zoom.
        for v in [
            self.tic_xlim_min_var,
            self.tic_xlim_max_var,
            self.tic_ylim_min_var,
            self.tic_ylim_max_var,
            self.spec_xlim_min_var,
            self.spec_xlim_max_var,
            self.spec_ylim_min_var,
            self.spec_ylim_max_var,
            self.uv_xlim_min_var,
            self.uv_xlim_max_var,
            self.uv_ylim_min_var,
            self.uv_ylim_max_var,
        ]:
            try:
                v.set("")
            except Exception:
                pass
        try:
            if self._toolbar is not None:
                self._toolbar.home()
        except Exception:
            pass
        self._redraw_all()

    # --- Workspace / multi-mzML sessions ---
    def _active_session_display_name(self) -> str:
        sid = getattr(self, "_active_session_id", None)
        if sid and sid in self._sessions:
            return str(self._sessions[sid].display_name)
        return self.mzml_path.name if self.mzml_path else "(no mzML)"

    def _get_session_id_by_path(self, path: Path) -> Optional[str]:
        p = Path(path).expanduser().resolve()
        for sid, sess in self._sessions.items():
            try:
                if sess.path == p:
                    return sid
            except Exception:
                continue
        return None

    def _open_mzml_many(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select one or more mzML files",
            filetypes=[("mzML files", "*.mzML"), ("All files", "*.*")],
        )
        if not paths:
            return
        self._add_mzml_paths([Path(p) for p in paths], make_first_active=(self._active_session_id is None))

    def _add_mzml_paths(self, paths: Sequence[Path], *, make_first_active: bool) -> None:
        make_active_next = bool(make_first_active)
        for p in paths:
            mzml_path = Path(p).expanduser().resolve()
            if not mzml_path.exists():
                messagebox.showerror("File not found", f"mzML file not found:\n{mzml_path}", parent=self)
                continue

            existing = self._get_session_id_by_path(mzml_path)
            if existing is not None:
                try:
                    if self._ws_tree is not None:
                        self._ws_tree.selection_set(existing)
                        self._ws_tree.see(existing)
                except Exception:
                    pass
                if make_active_next:
                    self._set_active_session(existing)
                    make_active_next = False
                continue

            self._add_session_from_path_async(mzml_path, make_active=make_active_next)
            make_active_next = False

    def _add_session_from_path_async(self, mzml_path: Path, *, make_active: bool) -> None:
        session_id = uuid.uuid4().hex
        self._session_load_counter += 1
        load_order = int(self._session_load_counter)

        if self._ws_tree is not None:
            try:
                # columns: overlay, active, color, name, ms1, pol
                self._ws_tree.insert("", "end", iid=session_id, values=("", "", "", mzml_path.name, "Loading…", "…"))
            except Exception:
                pass

        self._set_status(f"Indexing: {mzml_path.name}")
        rt_unit = self.rt_unit_var.get()

        def worker() -> None:
            try:
                idx = MzMLTICIndex(mzml_path, rt_unit=rt_unit)
                idx.build()
                self.after(0, lambda: self._on_session_index_ready(session_id, mzml_path, idx, load_order, make_active, None))
            except Exception as exc:
                self.after(0, lambda: self._on_session_index_ready(session_id, mzml_path, None, load_order, make_active, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_session_index_ready(
        self,
        session_id: str,
        mzml_path: Path,
        idx: Optional[MzMLTICIndex],
        load_order: int,
        make_active: bool,
        err: Optional[Exception],
    ) -> None:
        if err is not None or idx is None:
            try:
                if self._ws_tree is not None and self._ws_tree.exists(session_id):
                    self._ws_tree.delete(session_id)
            except Exception:
                pass
            messagebox.showerror("Error", f"Failed to index mzML:\n{mzml_path}\n\n{err}", parent=self)
            try:
                self._log("ERROR", f"Failed to index mzML: {mzml_path}", exc=err)
            except Exception:
                pass
            self._update_status_current()
            ctx = getattr(self, "_workspace_restore_ctx", None)
            if isinstance(ctx, dict):
                try:
                    ctx["done_mzml"] = int(ctx.get("done_mzml", 0)) + 1
                except Exception:
                    ctx["done_mzml"] = 1
                self._maybe_finalize_workspace_restore()
            ctx_lcms = getattr(self, "_lcms_workspace_restore_ctx", None)
            if isinstance(ctx_lcms, dict):
                try:
                    ctx_lcms["done_mzml"] = int(ctx_lcms.get("done_mzml", 0)) + 1
                except Exception:
                    ctx_lcms["done_mzml"] = 1
                self._maybe_finalize_lcms_workspace_restore()
            return

        # Index stats/warnings
        st = getattr(idx, "stats", None)
        try:
            if isinstance(st, dict) and st:
                fatal = st.get("fatal_error")
                skipped = int(st.get("skipped_no_rt", 0)) + int(st.get("skipped_no_intensity", 0)) + int(st.get("skipped_error", 0))
                kept = int(st.get("ms1_kept", 0))
                if skipped > 0:
                    self._warn(
                        f"{mzml_path.name}: indexed {kept} MS1 scans; skipped {skipped} (no RT={int(st.get('skipped_no_rt', 0))}, no intensity={int(st.get('skipped_no_intensity', 0))}, errors={int(st.get('skipped_error', 0))})."
                    )
                if fatal:
                    self._warn(f"{mzml_path.name}: {fatal}")
                self._log("INFO", f"Indexed mzML: {mzml_path.name} (MS1 kept={kept})")
        except Exception:
            pass

        ms1 = list(idx.ms1)
        ms1.sort(key=lambda m: float(m.rt_min))
        ms1_count = len(ms1)
        rt_min = float(ms1[0].rt_min) if ms1 else None
        rt_max = float(ms1[-1].rt_min) if ms1 else None
        pols = {m.polarity for m in ms1 if m.polarity}
        pol_sum = "/".join(sorted(pols)) if pols else "unknown"

        sess = MzMLSession(
            session_id=session_id,
            path=Path(mzml_path).expanduser().resolve(),
            index=idx,
            load_order=int(load_order),
            display_name=mzml_path.stem,
            custom_labels_by_spectrum={},
            spec_label_overrides={},
            ms1_count=int(ms1_count),
            rt_min=rt_min,
            rt_max=rt_max,
            polarity_summary=str(pol_sum),
        )

        try:
            sess.overlay_color = self._next_overlay_color()
        except Exception:
            pass

        self._sessions[session_id] = sess
        self._session_order.append(session_id)

        # Workspace model (LCMS datasets)
        try:
            sid = str(session_id)
            if not any(str(d.session_id) == sid for d in self.workspace.lcms_datasets):
                self.workspace.lcms_datasets.append(LCMSDataset(session_id=sid, mzml_path=sess.path, uv_csv_path=None))
            if self.workspace.active_lcms is None:
                self.workspace.active_lcms = sid
        except Exception:
            pass

        rt_txt = "—"
        if rt_min is not None and rt_max is not None:
            rt_txt = f"{rt_min:.2f}..{rt_max:.2f} ({pol_sum})"

        if self._ws_tree is not None:
            try:
                active_mark = "●" if (self._active_session_id == session_id) else ""
                overlay_mark = "■" if bool(getattr(sess, "overlay_selected", False)) else "□"
                color_mark = "■" if str(getattr(sess, "overlay_color", "")).strip() else ""
                values = (
                    overlay_mark,
                    active_mark,
                    color_mark,
                    sess.display_name,
                    str(sess.ms1_count),
                    str(sess.polarity_summary or "unknown"),
                )
                if self._ws_tree.exists(session_id):
                    self._ws_tree.item(session_id, values=values)
                else:
                    self._ws_tree.insert("", "end", iid=session_id, values=values)
            except Exception:
                pass

        if self._active_session_id is None or bool(make_active):
            self._set_active_session(session_id)
        else:
            self._update_status_current()

        ctx = getattr(self, "_workspace_restore_ctx", None)
        if isinstance(ctx, dict):
            try:
                ctx["done_mzml"] = int(ctx.get("done_mzml", 0)) + 1
            except Exception:
                ctx["done_mzml"] = 1
            self._maybe_finalize_workspace_restore()
        ctx_lcms = getattr(self, "_lcms_workspace_restore_ctx", None)
        if isinstance(ctx_lcms, dict):
            try:
                ctx_lcms["done_mzml"] = int(ctx_lcms.get("done_mzml", 0)) + 1
            except Exception:
                ctx_lcms["done_mzml"] = 1
            self._maybe_finalize_lcms_workspace_restore()

    def _save_active_session_state(self) -> None:
        sid = self._active_session_id
        if sid is None or sid not in self._sessions:
            return
        sess = self._sessions[sid]
        sess.last_selected_rt_min = float(self._selected_rt_min) if self._selected_rt_min is not None else None
        sess.last_scan_index = int(self._current_scan_index) if self._current_scan_index is not None else None
        try:
            sess.last_polarity_filter = str(self.polarity_var.get())
        except Exception:
            sess.last_polarity_filter = None
        # Store active per-session label dicts
        sess.custom_labels_by_spectrum = self._custom_labels_by_spectrum
        sess.spec_label_overrides = self._spec_label_overrides

    def _set_active_session(self, session_id: str) -> None:
        if not session_id or session_id not in self._sessions:
            return
        if self._active_session_id == session_id:
            return

        self._save_active_session_state()

        try:
            if self._active_reader is not None:
                self._active_reader.close()
        except Exception:
            pass
        self._active_reader = None
        self._reader = None

        sess = self._sessions[session_id]
        self._active_session_id = session_id

        if self._is_overlay_active():
            self._overlay_active_dataset_id = str(session_id)
            try:
                self._overlay_session = OverlaySession(
                    dataset_ids=list(self._overlay_session.dataset_ids) if self._overlay_session else [str(session_id)],
                    mode=str(self._overlay_mode_var.get()),
                    colors=(dict(self._overlay_session.colors) if self._overlay_session else {}),
                    persist=bool(self._overlay_persist_var.get()),
                    show_uv=bool(self._overlay_show_uv_var.get()),
                    stack_spectra=bool(self._overlay_stack_spectra_var.get()),
                    show_labels_all=bool(self._overlay_show_labels_all_var.get()),
                    multi_drag=bool(self._overlay_multi_drag_var.get()),
                    active_dataset_id=str(session_id),
                )
            except Exception:
                pass

        # Workspace model (LCMS active selection)
        try:
            self.workspace.active_lcms = str(session_id)
        except Exception:
            pass

        # Swap legacy single-file fields to keep the rest of the app unchanged.
        self.mzml_path = sess.path
        self._index = sess.index

        # Swap per-session label state
        self._custom_labels_by_spectrum = sess.custom_labels_by_spectrum
        self._spec_label_overrides = sess.spec_label_overrides

        # Restore per-session polarity (optional)
        if sess.last_polarity_filter in ("all", "positive", "negative"):
            try:
                self.polarity_var.set(str(sess.last_polarity_filter))
            except Exception:
                pass

        # Open reader for active session (only one open at a time)
        try:
            self._active_reader = mzml.MzML(str(sess.path))
            self._reader = self._active_reader
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to open mzML:\n{sess.path}\n\n{exc}", parent=self)
            self._active_reader = None
            self._reader = None
            return

        # Restore selection
        self._current_scan_index = sess.last_scan_index
        self._selected_rt_min = sess.last_selected_rt_min
        self._current_spectrum_meta = None
        self._current_spectrum_mz = None
        self._current_spectrum_int = None

        try:
            if self._ws_tree is not None and self._ws_tree.exists(session_id):
                self._ws_ignore_select = True
                self._ws_tree.selection_set(session_id)
                self._ws_tree.see(session_id)
        except Exception:
            pass
        finally:
            self._ws_ignore_select = False

        self._refresh_ws_active_markers()

        # Switch linked UV to this session (if any)
        self._sync_active_uv_id()
        self._update_uv_ws_controls()

        self._refresh_tic()
        self._plot_uv()
        self._update_status_current()

        # Notify non-modal windows (e.g., EIC) to refresh for the new active mzML.
        self._notify_active_session_changed()

    def _selected_session_id_from_tree(self) -> Optional[str]:
        if self._ws_tree is None:
            return None
        try:
            sel = self._ws_tree.selection()
            return str(sel[0]) if sel else None
        except Exception:
            return None

    def _set_active_from_workspace(self) -> None:
        sid = self._selected_session_id_from_tree()
        if not sid:
            messagebox.showinfo("Workspace", "Select an mzML in the Workspace list.", parent=self)
            return
        self._set_active_session(sid)

    def _remove_selected_session(self) -> None:
        sid = self._selected_session_id_from_tree() or self._active_session_id
        if not sid:
            return
        self._remove_session(sid)

    def _close_active_session(self) -> None:
        if self._active_session_id is None:
            return
        self._remove_session(self._active_session_id)

    def _remove_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            return
        sess = self._sessions[session_id]
        if not messagebox.askyesno(
            "Remove mzML",
            f"Remove from workspace?\n\n{sess.display_name}\n{sess.path}",
            parent=self,
        ):
            return

        was_active = (self._active_session_id == session_id)
        if was_active:
            self._save_active_session_state()
            try:
                if self._active_reader is not None:
                    self._active_reader.close()
            except Exception:
                pass
            self._active_reader = None
            self._reader = None
            self._active_session_id = None
            self.mzml_path = None
            self._index = None

        self._sessions.pop(session_id, None)
        try:
            self._session_order.remove(session_id)
        except Exception:
            pass

        try:
            for k in list(self._overlay_tic_cache.keys()):
                if str(k[0]) == str(session_id):
                    self._overlay_tic_cache.pop(k, None)
        except Exception:
            pass

        # Update overlay session if needed
        if self._overlay_session is not None:
            try:
                ids = [sid for sid in self._overlay_session.dataset_ids if str(sid) != str(session_id)]
                if len(ids) < 2:
                    self._clear_overlay()
                else:
                    self._overlay_session = OverlaySession(
                        dataset_ids=list(ids),
                        mode=str(self._overlay_session.mode),
                        colors=dict(self._overlay_session.colors),
                        persist=bool(self._overlay_session.persist),
                        show_uv=bool(self._overlay_session.show_uv),
                        stack_spectra=bool(self._overlay_session.stack_spectra),
                        show_labels_all=bool(self._overlay_session.show_labels_all),
                        multi_drag=bool(self._overlay_session.multi_drag),
                        active_dataset_id=(None if self._overlay_session.active_dataset_id == str(session_id) else self._overlay_session.active_dataset_id),
                    )
                    self._refresh_overlay_view()
            except Exception:
                pass

        # Workspace model (LCMS datasets)
        try:
            self.workspace.lcms_datasets = [d for d in self.workspace.lcms_datasets if str(d.session_id) != str(session_id)]
            if self.workspace.active_lcms == str(session_id):
                self.workspace.active_lcms = str(self._active_session_id) if self._active_session_id is not None else None
        except Exception:
            pass

        if self._ws_tree is not None:
            try:
                if self._ws_tree.exists(session_id):
                    self._ws_tree.delete(session_id)
            except Exception:
                pass

        if was_active and self._session_order:
            self._set_active_session(self._session_order[0])
            return

        if not self._session_order:
            self._filtered_meta = []
            self._filtered_rts = None
            self._filtered_tics = None
            self._current_scan_index = None
            self._current_spectrum_meta = None
            self._current_spectrum_mz = None
            self._current_spectrum_int = None
            self._custom_labels_by_spectrum = {}
            self._spec_label_overrides = {}
            self._active_uv_id = None
            self._tic_line = None
            self._tic_marker = None
            self._rebuild_plot_axes()
            self._plot_uv()
            self._update_status_current()

    def _rename_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            return
        sess = self._sessions[session_id]
        new_name = simpledialog.askstring("Rename", "Display name:", initialvalue=str(sess.display_name), parent=self)
        if new_name is None:
            return
        new_name = str(new_name).strip()
        if not new_name:
            new_name = sess.path.stem
        sess.display_name = new_name

        if self._ws_tree is not None:
            try:
                vals = list(self._ws_tree.item(session_id, "values") or [])
                if vals:
                    if len(vals) >= 4:
                        vals[3] = sess.display_name
                    self._ws_tree.item(session_id, values=tuple(vals))
            except Exception:
                pass

        self._update_now_viewing_header()
        self._update_status_current()

    def _copy_session_path(self, session_id: str) -> None:
        if session_id not in self._sessions:
            return
        p = str(self._sessions[session_id].path)
        try:
            self.clipboard_clear()
            self.clipboard_append(p)
        except Exception:
            pass
        self._set_status(f"Copied path: {p}")

    def _ensure_ws_menu(self) -> None:
        if self._ws_menu is not None:
            return
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Set Active", command=self._set_active_from_workspace)
        m.add_command(label="Link UV…", command=self._link_uv_from_context_menu)
        m.add_command(label="Rename…", command=lambda: self._rename_session(self._selected_session_id_from_tree() or ""))
        m.add_command(label="Open containing folder", command=self._open_selected_mzml_folder)
        m.add_separator()
        m.add_command(label="Remove", command=self._remove_selected_session)
        m.add_command(label="Copy path", command=lambda: self._copy_session_path(self._selected_session_id_from_tree() or ""))
        self._ws_menu = m

    def _on_ws_right_click(self, evt) -> None:
        if self._ws_tree is None:
            return
        try:
            iid = self._ws_tree.identify_row(evt.y)
            if iid:
                self._ws_tree.selection_set(iid)
        except Exception:
            pass
        self._ensure_ws_menu()
        try:
            if self._ws_menu is not None:
                self._ws_menu.tk_popup(int(evt.x_root), int(evt.y_root))
        finally:
            try:
                if self._ws_menu is not None:
                    self._ws_menu.grab_release()
            except Exception:
                pass

    def _overlay_palette(self) -> List[str]:
        base = list(mcolors.TABLEAU_COLORS.values())
        extra = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        return base + [c for c in extra if c not in base]

    def _overlay_scheme_options(self) -> List[str]:
        return [
            "Auto (Tableau)",
            "Manual (per-dataset)",
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

    def _on_overlay_scheme_changed(self) -> None:
        try:
            self._apply_overlay_color_scheme()
        except Exception:
            pass
        try:
            if self._is_overlay_active():
                self._refresh_overlay_view()
        except Exception:
            pass

    def _pick_overlay_single_hue_color(self) -> None:
        current = str(self._overlay_single_hue_color or "#1f77b4")
        try:
            picked = colorchooser.askcolor(color=(current or None), title="Pick overlay hue", parent=self)[1]
        except Exception:
            picked = None
        if not picked:
            return
        self._overlay_single_hue_color = str(picked)
        try:
            self._overlay_scheme_var.set("Single hue…")
        except Exception:
            pass
        self._on_overlay_scheme_changed()

    def _overlay_colors_for_scheme(self, scheme: str, n: int) -> List[str]:
        if n <= 0:
            return []
        scheme = str(scheme or "").strip()
        if scheme in ("", "Auto (Tableau)"):
            palette = self._overlay_palette()
            return [str(palette[i % len(palette)]) for i in range(n)]
        if scheme == "Single hue…":
            try:
                base_rgb = mcolors.to_rgb(str(self._overlay_single_hue_color or "#1f77b4"))
            except Exception:
                base_rgb = (0.12, 0.47, 0.71)
            try:
                h, l, s = colorsys.rgb_to_hls(*base_rgb)
            except Exception:
                h, l, s = (0.58, 0.45, 0.65)
            lo = 0.22
            hi = 0.88
            vals = np.linspace(lo, hi, n)
            return [mcolors.to_hex(colorsys.hls_to_rgb(float(h), float(v), float(s))) for v in vals]
        # Scientific palettes from matplotlib/ColorBrewer
        try:
            cmap = cm.get_cmap(str(scheme).lower())
        except Exception:
            try:
                cmap = cm.get_cmap(str(scheme))
            except Exception:
                palette = self._overlay_palette()
                return [str(palette[i % len(palette)]) for i in range(n)]
        xs = np.linspace(0.06, 0.94, n)
        return [mcolors.to_hex(cmap(float(x))) for x in xs]

    def _apply_overlay_color_scheme(self, *, ids: Optional[List[str]] = None) -> None:
        scheme = str(self._overlay_scheme_var.get() or "").strip()
        if scheme == "Manual (per-dataset)":
            return
        if ids is None:
            ids = self._overlay_dataset_ids() if self._is_overlay_active() else self._overlay_selected_ids_from_flags()
        ids = [str(sid) for sid in (ids or []) if str(sid) in self._sessions]
        if not ids:
            return
        colors = self._overlay_colors_for_scheme(scheme, len(ids))
        for sid, col in zip(ids, colors):
            sess = self._sessions.get(str(sid))
            if sess is None:
                continue
            try:
                sess.overlay_color = str(col)
            except Exception:
                pass
            try:
                self._refresh_ws_tree_row(str(sid))
            except Exception:
                pass
        if self._overlay_session is not None:
            try:
                self._overlay_session = OverlaySession(
                    dataset_ids=list(self._overlay_session.dataset_ids),
                    mode=str(self._overlay_mode_var.get()),
                    colors={str(sid): str(self._sessions[str(sid)].overlay_color) for sid in ids if str(sid) in self._sessions},
                    persist=bool(self._overlay_persist_var.get()),
                    show_uv=bool(self._overlay_show_uv_var.get()),
                    stack_spectra=bool(self._overlay_stack_spectra_var.get()),
                    show_labels_all=bool(self._overlay_show_labels_all_var.get()),
                    multi_drag=bool(self._overlay_multi_drag_var.get()),
                    active_dataset_id=str(self._overlay_active_dataset_id or self._active_session_id or "") or None,
                )
            except Exception:
                pass

    def _next_overlay_color(self) -> str:
        used = {str(getattr(s, "overlay_color", "")) for s in self._sessions.values() if getattr(s, "overlay_color", "")}
        for c in self._overlay_palette():
            if c not in used:
                return str(c)
        return "#4e79a7"

    def _ensure_overlay_color(self, session_id: str) -> str:
        if session_id not in self._sessions:
            return "#4e79a7"
        sess = self._sessions[session_id]
        col = str(getattr(sess, "overlay_color", "")).strip()
        if not col:
            col = self._next_overlay_color()
            try:
                sess.overlay_color = str(col)
            except Exception:
                pass
        return str(col)

    def _refresh_ws_tree_row(self, session_id: str) -> None:
        if self._ws_tree is None or session_id not in self._sessions:
            return
        if not self._ws_tree.exists(session_id):
            return
        sess = self._sessions[session_id]
        overlay_mark = "■" if bool(getattr(sess, "overlay_selected", False)) else "□"
        active_mark = "●" if (self._active_session_id == session_id) else ""
        color_mark = "■" if str(getattr(sess, "overlay_color", "")).strip() else ""
        values = (
            overlay_mark,
            active_mark,
            color_mark,
            str(sess.display_name),
            str(sess.ms1_count),
            str(sess.polarity_summary or "unknown"),
        )
        try:
            self._ws_tree.item(session_id, values=values)
        except Exception:
            pass

    def _on_ws_left_click(self, evt) -> Optional[str]:
        if self._ws_tree is None:
            return None
        try:
            col = self._ws_tree.identify_column(evt.x)
            iid = self._ws_tree.identify_row(evt.y)
        except Exception:
            return None
        if not iid or iid not in self._sessions:
            return None

        # Column indices: #1 overlay, #2 active, #3 color, #4 name, #5 ms1, #6 pol
        if col == "#1":
            sess = self._sessions[str(iid)]
            try:
                sess.overlay_selected = not bool(getattr(sess, "overlay_selected", False))
            except Exception:
                pass
            try:
                self._apply_overlay_color_scheme(ids=self._overlay_selected_ids_from_flags())
            except Exception:
                self._ensure_overlay_color(str(iid))
            self._refresh_ws_tree_row(str(iid))
            if self._overlay_session is not None:
                ids = self._overlay_selected_ids_from_flags()
                if len(ids) >= 2:
                    try:
                        self._overlay_session = OverlaySession(
                            dataset_ids=list(ids),
                            mode=str(self._overlay_mode_var.get()),
                            colors=dict(self._overlay_session.colors),
                            persist=bool(self._overlay_persist_var.get()),
                            show_uv=bool(self._overlay_show_uv_var.get()),
                            stack_spectra=bool(self._overlay_stack_spectra_var.get()),
                            show_labels_all=bool(self._overlay_show_labels_all_var.get()),
                            multi_drag=bool(self._overlay_multi_drag_var.get()),
                            active_dataset_id=str(self._overlay_active_dataset_id or self._active_session_id or "") or None,
                        )
                    except Exception:
                        pass
                    self._refresh_overlay_view()
                else:
                    self._clear_overlay()
            return "break"

        if col == "#3":
            sess = self._sessions[str(iid)]
            current = str(getattr(sess, "overlay_color", "") or "")
            try:
                picked = colorchooser.askcolor(color=(current or None), title="Pick overlay color", parent=self)[1]
            except Exception:
                picked = None
            if picked:
                try:
                    sess.overlay_color = str(picked)
                except Exception:
                    pass
                self._refresh_ws_tree_row(str(iid))
                if self._is_overlay_active():
                    self._refresh_overlay_view()
            return "break"
        return None

    def _open_selected_mzml_folder(self) -> None:
        sid = self._selected_session_id_from_tree()
        if not sid or sid not in self._sessions:
            return
        try:
            p = Path(self._sessions[sid].path)
        except Exception:
            return
        try:
            # Windows: open folder in Explorer
            os.startfile(str(p.parent))  # type: ignore[attr-defined]
        except Exception:
            try:
                import subprocess

                subprocess.Popen(["explorer", str(p.parent)])
            except Exception:
                pass

    def _link_uv_from_context_menu(self) -> None:
        sid = self._selected_session_id_from_tree()
        if not sid:
            return
        # If a UV is selected, link it; otherwise prompt.
        uv_id = self._selected_uv_id_from_tree()
        if uv_id and uv_id in self._uv_sessions:
            self._link_uv_to_mzml(session_id=str(sid), uv_id=str(uv_id))
            return
        if not self._uv_sessions:
            messagebox.showinfo("Link UV", "Add a UV CSV first.", parent=self)
            return

        dlg = tk.Toplevel(self)
        dlg.title("Link UV")
        dlg.resizable(False, False)
        dlg.transient(self)

        frm = ttk.Frame(dlg, padding=12)
        frm.grid(row=0, column=0)
        ttk.Label(frm, text="Choose UV CSV:").grid(row=0, column=0, sticky="w")
        uv_items = [(uid, self._uv_sessions[uid].path.name) for uid in list(self._uv_order) if uid in self._uv_sessions]
        names = [name for _uid, name in uv_items] or [self._uv_sessions[uid].path.name for uid in self._uv_sessions.keys()]
        choice = tk.StringVar(value=(names[0] if names else ""))
        combo = ttk.Combobox(frm, textvariable=choice, values=names, state="readonly", width=42)
        combo.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, sticky="e", pady=(12, 0))

        def do_link() -> None:
            sel_name = (choice.get() or "").strip()
            picked = None
            for uid, nm in uv_items:
                if nm == sel_name:
                    picked = uid
                    break
            if picked is None:
                for uid, uv_sess in self._uv_sessions.items():
                    if uv_sess.path.name == sel_name:
                        picked = uid
                        break
            if picked is not None:
                self._link_uv_to_mzml(session_id=str(sid), uv_id=str(picked))
            try:
                dlg.destroy()
            except Exception:
                pass

        ttk.Button(btns, text="Link", command=do_link).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="Close", command=dlg.destroy).grid(row=0, column=1)

    # --- UV workspace (multi-UV sessions, linked per mzML session) ---
    def _selected_uv_id_from_tree(self) -> Optional[str]:
        if self._uv_ws_tree is None:
            return None
        try:
            sel = self._uv_ws_tree.selection()
            return str(sel[0]) if sel else None
        except Exception:
            return None

    def _get_uv_id_by_path(self, path: Path) -> Optional[str]:
        p = Path(path).expanduser().resolve()
        for uid, sess in self._uv_sessions.items():
            try:
                if sess.path == p:
                    return uid
            except Exception:
                continue
        return None

    def _uv_display_name(self, uv_id: Optional[str]) -> str:
        if not uv_id:
            return "—"
        sess = self._uv_sessions.get(str(uv_id))
        if sess is None:
            return "—"
        try:
            return sess.path.stem
        except Exception:
            return "—"

    def _update_ws_row_uv(self, session_id: str) -> None:
        # Backward-compatible hook: link display moved to the UV list.
        self._refresh_uv_tree_links()

    def _sync_active_uv_id(self) -> None:
        sid = self._active_session_id
        if not sid or sid not in self._sessions:
            self._active_uv_id = None
            return
        linked = self._sessions[sid].linked_uv_id
        if linked and str(linked) in self._uv_sessions:
            self._active_uv_id = str(linked)
        else:
            self._active_uv_id = None

    def _active_uv_session(self) -> Optional[UVSession]:
        self._sync_active_uv_id()
        if not self._active_uv_id:
            return None
        return self._uv_sessions.get(str(self._active_uv_id))

    def _active_uv_xy(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        uv = self._active_uv_session()
        if uv is None:
            return None, None
        return uv.rt_min, uv.signal

    def _active_uv_labels_by_uvrt(self, *, create: bool) -> Dict[float, List[UVLabelState]]:
        sid = self._active_session_id
        if not sid or sid not in self._sessions:
            return {}
        self._sync_active_uv_id()
        uv_id = self._active_uv_id
        if not uv_id:
            return {}
        mzsess = self._sessions[sid]
        if create:
            return mzsess.uv_labels_by_uv_id.setdefault(str(uv_id), {})
        return mzsess.uv_labels_by_uv_id.get(str(uv_id), {})

    def _update_uv_ws_controls(self) -> None:
        btn = self._uv_ws_link_btn
        if btn is None:
            return
        uv_id = self._selected_uv_id_from_tree()
        sid = self._selected_session_id_from_tree() or self._active_session_id
        ok = bool(uv_id) and (uv_id in self._uv_sessions) and bool(sid) and (sid in self._sessions)
        try:
            btn.configure(state=("normal" if ok else "disabled"))
        except Exception:
            pass

    def _on_ws_select(self, _evt=None) -> None:
        if bool(getattr(self, "_ws_ignore_select", False)):
            return
        sid = self._selected_session_id_from_tree()
        if not sid:
            return
        if sid not in self._sessions:
            return
        self._set_active_session(str(sid))

    def _refresh_ws_active_markers(self) -> None:
        if self._ws_tree is None:
            return
        try:
            for sid in list(self._sessions.keys()):
                if not self._ws_tree.exists(sid):
                    continue
                vals = list(self._ws_tree.item(sid, "values") or [])
                if len(vals) != 6:
                    continue
                vals[1] = "●" if (self._active_session_id == sid) else ""
                self._ws_tree.item(sid, values=tuple(vals))
        except Exception:
            pass

    def _uv_linked_to_summary(self, uv_id: str) -> str:
        linked_names: List[str] = []
        for _sid, mzsess in (self._sessions or {}).items():
            try:
                if str(mzsess.linked_uv_id) == str(uv_id):
                    linked_names.append(str(mzsess.display_name))
            except Exception:
                continue
        if not linked_names:
            return "—"
        if len(linked_names) == 1:
            return linked_names[0]
        return f"{len(linked_names)} mzML"

    def _refresh_uv_tree_links(self) -> None:
        if self._uv_ws_tree is None:
            return
        try:
            for uv_id in list(self._uv_sessions.keys()):
                if not self._uv_ws_tree.exists(str(uv_id)):
                    continue
                vals = list(self._uv_ws_tree.item(str(uv_id), "values") or [])
                if len(vals) != 4:
                    continue
                vals[0] = self._uv_linked_to_summary(str(uv_id))
                self._uv_ws_tree.item(str(uv_id), values=tuple(vals))
        except Exception:
            pass

    def _link_uv_to_mzml(self, *, session_id: str, uv_id: str) -> None:
        if not session_id or session_id not in self._sessions:
            return
        if not uv_id or uv_id not in self._uv_sessions:
            return

        mzsess = self._sessions[str(session_id)]
        prev = mzsess.linked_uv_id
        if prev is not None and str(prev) != str(uv_id):
            try:
                mzsess.uv_labels_by_uv_id.clear()
            except Exception:
                mzsess.uv_labels_by_uv_id = {}

        mzsess.linked_uv_id = str(uv_id)
        self._refresh_uv_tree_links()

        if self._active_session_id == str(session_id):
            self._sync_active_uv_id()
            self._plot_uv()
            self._update_status_current()

    def _link_selected_uv_to_selected_mzml(self) -> None:
        uv_id = self._selected_uv_id_from_tree()
        if not uv_id:
            messagebox.showinfo("Link UV", "Select a UV CSV in the UV list.", parent=self)
            return
        sid = self._selected_session_id_from_tree() or self._active_session_id
        if not sid:
            messagebox.showinfo("Link UV", "Select an mzML in the mzML list.", parent=self)
            return
        self._link_uv_to_mzml(session_id=str(sid), uv_id=str(uv_id))

    def _open_uv_csv_single(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a UV chromatogram CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self._add_uv_paths([Path(path)])

    def _open_uv_csv_many(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select one or more UV chromatogram CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not paths:
            return
        self._add_uv_paths([Path(p) for p in paths])

    def _add_uv_paths(self, paths: Sequence[Path]) -> None:
        for p in paths:
            csv_path = Path(p).expanduser().resolve()
            if not csv_path.exists():
                messagebox.showerror("File not found", f"CSV file not found:\n{csv_path}", parent=self)
                continue

            existing = self._get_uv_id_by_path(csv_path)
            if existing is not None:
                try:
                    if self._uv_ws_tree is not None:
                        self._uv_ws_tree.selection_set(existing)
                        self._uv_ws_tree.see(existing)
                except Exception:
                    pass
                continue

            self._add_uv_session_from_path_async(csv_path)

        self._update_uv_ws_controls()

    def _add_uv_session_from_path_async(self, csv_path: Path) -> None:
        uv_id = uuid.uuid4().hex
        self._uv_load_counter += 1
        load_order = int(self._uv_load_counter)

        if self._uv_ws_tree is not None:
            try:
                # columns: linked, name, rt, sig
                self._uv_ws_tree.insert("", "end", iid=uv_id, values=("—", csv_path.name, "Loading…", "…"))
            except Exception:
                pass

        self._set_status(f"Loading UV: {csv_path.name}")

        def prompt_settings(
            *,
            cols: List[str],
            preview: List[Tuple[Any, ...]],
            default_x: str,
            default_y: str,
            unit_guess: str,
            reason: str,
        ) -> Tuple[str, str, str]:
            result: Dict[str, str] = {"x": default_x, "y": default_y, "unit": unit_guess}
            done = threading.Event()

            def ui() -> None:
                dlg = tk.Toplevel(self)
                dlg.title("UV import settings")
                dlg.resizable(True, False)
                dlg.transient(self)

                frm = ttk.Frame(dlg, padding=12)
                frm.grid(row=0, column=0, sticky="nsew")
                frm.columnconfigure(1, weight=1)

                ttk.Label(frm, text="We couldn't confidently detect time/signal columns.").grid(row=0, column=0, columnspan=2, sticky="w")
                ttk.Label(frm, text=str(reason), wraplength=620, justify="left").grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 10))

                ttk.Label(frm, text="Time (x) column").grid(row=2, column=0, sticky="w")
                xvar = tk.StringVar(value=str(default_x))
                xcb = ttk.Combobox(frm, textvariable=xvar, values=cols, state="readonly")
                xcb.grid(row=2, column=1, sticky="ew", padx=(10, 0))

                ttk.Label(frm, text="Signal (y) column").grid(row=3, column=0, sticky="w", pady=(8, 0))
                yvar = tk.StringVar(value=str(default_y))
                ycb = ttk.Combobox(frm, textvariable=yvar, values=cols, state="readonly")
                ycb.grid(row=3, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))

                ttk.Label(frm, text="Time units").grid(row=4, column=0, sticky="w", pady=(8, 0))
                uvar = tk.StringVar(value=str(unit_guess))
                ucb = ttk.Combobox(frm, textvariable=uvar, values=["minutes", "seconds"], state="readonly", width=12)
                ucb.grid(row=4, column=1, sticky="w", padx=(10, 0), pady=(8, 0))

                prevf = ttk.LabelFrame(frm, text="Preview (first rows)", padding=8)
                prevf.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
                tv = ttk.Treeview(prevf, columns=[f"c{i}" for i in range(len(cols))], show="headings", height=6)
                for i, c in enumerate(cols):
                    tv.heading(f"c{i}", text=str(c))
                    tv.column(f"c{i}", width=110, stretch=True)
                for row in preview[:25]:
                    vals = list(row)
                    if len(vals) < len(cols):
                        vals += [""] * (len(cols) - len(vals))
                    tv.insert("", "end", values=tuple(vals[: len(cols)]))
                tv.grid(row=0, column=0, sticky="ew")

                btns = ttk.Frame(frm)
                btns.grid(row=6, column=0, columnspan=2, sticky="e", pady=(12, 0))

                def ok() -> None:
                    result["x"] = (xvar.get() or default_x)
                    result["y"] = (yvar.get() or default_y)
                    result["unit"] = (uvar.get() or unit_guess)
                    done.set()
                    try:
                        dlg.destroy()
                    except Exception:
                        pass

                def cancel() -> None:
                    done.set()
                    try:
                        dlg.destroy()
                    except Exception:
                        pass

                ttk.Button(btns, text="OK", command=ok).grid(row=0, column=0, padx=(0, 8))
                ttk.Button(btns, text="Cancel", command=cancel).grid(row=0, column=1)

                try:
                    dlg.protocol("WM_DELETE_WINDOW", cancel)
                except Exception:
                    pass
                try:
                    xcb.focus_set()
                except Exception:
                    pass

            self.after(0, ui)
            done.wait()
            return (str(result["x"]), str(result["y"]), str(result["unit"]))

        def worker() -> None:
            try:
                df = pd.read_csv(csv_path)
                info = infer_uv_columns(df)
                cols = list(info.get("cols") or [])
                xcol = str(info.get("xcol") or "")
                ycol = str(info.get("ycol") or "")
                unit_guess = str(info.get("unit_guess") or "minutes")

                preview_rows: List[Tuple[Any, ...]] = preview_dataframe_rows(df, n=10)

                if bool(info.get("low_conf")):
                    try:
                        self._warn(f"{csv_path.name}: low-confidence column detection; using inferred columns without prompt")
                    except Exception:
                        pass

                rt_min, signal, rt_range, import_warnings = parse_uv_arrays(df, xcol=str(xcol), ycol=str(ycol), unit_guess=str(unit_guess))
                sess = UVSession(
                    uv_id=str(uv_id),
                    path=Path(csv_path).expanduser().resolve(),
                    rt_min=rt_min,
                    signal=signal,
                    xcol=str(xcol),
                    ycol=str(ycol),
                    n_points=int(rt_min.size),
                    rt_min_range=(float(rt_range[0]), float(rt_range[1])),
                    load_order=int(load_order),
                    import_warnings=list(import_warnings),
                )
                self.after(0, lambda: self._on_uv_session_ready(sess, None))
            except Exception as exc:
                self.after(0, lambda: self._on_uv_session_ready(None, exc, uv_id=str(uv_id), csv_path=Path(csv_path)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_uv_session_ready(
        self,
        sess: Optional[UVSession],
        err: Optional[Exception],
        *,
        uv_id: Optional[str] = None,
        csv_path: Optional[Path] = None,
    ) -> None:
        if err is not None or sess is None:
            try:
                if self._uv_ws_tree is not None and uv_id and self._uv_ws_tree.exists(str(uv_id)):
                    self._uv_ws_tree.delete(str(uv_id))
            except Exception:
                pass
            messagebox.showerror("Error", f"Failed to load UV CSV:\n{csv_path}\n\n{err}", parent=self)
            try:
                self._log("ERROR", f"Failed to load UV CSV: {csv_path}", exc=err)
            except Exception:
                pass
            self._update_status_current()
            self._update_uv_ws_controls()
            ctx = getattr(self, "_workspace_restore_ctx", None)
            if isinstance(ctx, dict):
                try:
                    ctx["done_uv"] = int(ctx.get("done_uv", 0)) + 1
                except Exception:
                    ctx["done_uv"] = 1
                self._maybe_finalize_workspace_restore()
            ctx_lcms = getattr(self, "_lcms_workspace_restore_ctx", None)
            if isinstance(ctx_lcms, dict):
                try:
                    ctx_lcms["done_uv"] = int(ctx_lcms.get("done_uv", 0)) + 1
                except Exception:
                    ctx_lcms["done_uv"] = 1
                self._maybe_finalize_lcms_workspace_restore()
            return

        self._uv_sessions[sess.uv_id] = sess
        self._uv_order.append(sess.uv_id)

        try:
            self._log("INFO", f"Loaded UV CSV: {sess.path.name} (n={int(sess.n_points)})")
            for w in list(getattr(sess, "import_warnings", []) or []):
                self._warn(f"{sess.path.name}: {w}")
        except Exception:
            pass

        rt_txt = f"{sess.rt_min_range[0]:.2f}..{sess.rt_min_range[1]:.2f}"
        if self._uv_ws_tree is not None:
            try:
                if self._uv_ws_tree.exists(sess.uv_id):
                    self._uv_ws_tree.item(
                        sess.uv_id,
                        values=(self._uv_linked_to_summary(str(sess.uv_id)), sess.path.stem, rt_txt, str(sess.ycol)),
                    )
                else:
                    self._uv_ws_tree.insert(
                        "",
                        "end",
                        iid=sess.uv_id,
                        values=(self._uv_linked_to_summary(str(sess.uv_id)), sess.path.stem, rt_txt, str(sess.ycol)),
                    )
            except Exception:
                pass

        self._update_uv_ws_controls()
        self._refresh_uv_tree_links()
        self._update_status_current()

        ctx = getattr(self, "_workspace_restore_ctx", None)
        if isinstance(ctx, dict):
            try:
                ctx["done_uv"] = int(ctx.get("done_uv", 0)) + 1
            except Exception:
                ctx["done_uv"] = 1
            self._maybe_finalize_workspace_restore()
        ctx_lcms = getattr(self, "_lcms_workspace_restore_ctx", None)
        if isinstance(ctx_lcms, dict):
            try:
                ctx_lcms["done_uv"] = int(ctx_lcms.get("done_uv", 0)) + 1
            except Exception:
                ctx_lcms["done_uv"] = 1
            self._maybe_finalize_lcms_workspace_restore()

    def _link_uv_to_active_mzml(self, uv_id: str) -> None:
        sid = self._active_session_id
        if not sid:
            messagebox.showinfo("Link UV", "Load and activate an mzML session first.", parent=self)
            return
        self._link_uv_to_mzml(session_id=str(sid), uv_id=str(uv_id))

    def _link_selected_uv_to_active_mzml(self) -> None:
        uv_id = self._selected_uv_id_from_tree()
        if not uv_id:
            messagebox.showinfo("Link UV", "Select a UV CSV in the UV Workspace list.", parent=self)
            return
        if uv_id not in self._uv_sessions:
            return
        self._link_uv_to_active_mzml(str(uv_id))

    def _remove_selected_uv(self) -> None:
        uv_id = self._selected_uv_id_from_tree()
        if not uv_id:
            return
        self._remove_uv(str(uv_id))

    def _remove_uv(self, uv_id: str) -> None:
        if uv_id not in self._uv_sessions:
            return
        sess = self._uv_sessions[uv_id]
        if not messagebox.askyesno(
            "Remove UV",
            f"Remove from UV workspace?\n\n{sess.path.stem}\n{sess.path}",
            parent=self,
        ):
            return

        # Unlink from all mzML sessions
        for sid, mzsess in list(self._sessions.items()):
            if mzsess.linked_uv_id == uv_id:
                mzsess.linked_uv_id = None
                try:
                    mzsess.uv_labels_by_uv_id.pop(uv_id, None)
                except Exception:
                    pass
                self._refresh_uv_tree_links()

        self._uv_sessions.pop(uv_id, None)
        try:
            self._uv_order.remove(uv_id)
        except Exception:
            pass
        if self._uv_ws_tree is not None:
            try:
                if self._uv_ws_tree.exists(uv_id):
                    self._uv_ws_tree.delete(uv_id)
            except Exception:
                pass

        # If this affected the active mzML's link, sync active UV and redraw.
        self._sync_active_uv_id()
        self._plot_uv()
        self._update_status_current()
        self._update_uv_ws_controls()
        self._refresh_uv_tree_links()

    def _normalize_stem(self, s: str) -> str:
        s = (s or "").strip().lower()
        out = []
        prev_us = False
        for ch in s:
            if ch.isalnum():
                out.append(ch)
                prev_us = False
            else:
                if not prev_us:
                    out.append("_")
                    prev_us = True
        norm = "".join(out).strip("_")
        return norm

    def _auto_link_uv_by_name(self) -> None:
        if not self._sessions:
            messagebox.showinfo("Auto-link", "Load mzML sessions first.", parent=self)
            return
        if not self._uv_sessions:
            messagebox.showinfo("Auto-link", "Load UV CSV files first.", parent=self)
            return

        uv_by_norm: Dict[str, str] = {}
        for uv_id, uv_sess in self._uv_sessions.items():
            uv_by_norm[self._normalize_stem(uv_sess.path.stem)] = uv_id

        linked = 0
        not_matched: List[str] = []

        for sid, mzsess in self._sessions.items():
            if mzsess.linked_uv_id:
                continue
            mz_norm = self._normalize_stem(mzsess.path.stem)

            best: Optional[str] = uv_by_norm.get(mz_norm)
            if best is None:
                # contains
                for uv_norm, uv_id in uv_by_norm.items():
                    if uv_norm and (uv_norm in mz_norm or mz_norm in uv_norm):
                        best = uv_id
                        break
            if best is None:
                # prefix
                for uv_norm, uv_id in uv_by_norm.items():
                    if uv_norm and (mz_norm.startswith(uv_norm) or uv_norm.startswith(mz_norm)):
                        best = uv_id
                        break

            if best is None:
                not_matched.append(mzsess.display_name)
                continue

            mzsess.linked_uv_id = str(best)
            linked += 1

        # Sync active UV and redraw if the active session got linked
        self._sync_active_uv_id()
        self._plot_uv()
        self._update_status_current()
        self._refresh_uv_tree_links()

        msg = f"Linked {linked} mzML session(s)."
        if not_matched:
            msg += "\n\nNot matched:\n" + "\n".join(not_matched[:30])
            if len(not_matched) > 30:
                msg += f"\n… (+{len(not_matched) - 30} more)"
        messagebox.showinfo("Auto-link by name", msg, parent=self)

    def _copy_uv_path(self, uv_id: str) -> None:
        if uv_id not in self._uv_sessions:
            return
        p = str(self._uv_sessions[uv_id].path)
        try:
            self.clipboard_clear()
            self.clipboard_append(p)
        except Exception:
            pass
        self._set_status(f"Copied path: {p}")

    def _ensure_uv_ws_menu(self) -> None:
        if self._uv_ws_menu is not None:
            return
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Link to active mzML", command=self._link_selected_uv_to_active_mzml)
        m.add_command(label="Remove", command=self._remove_selected_uv)
        m.add_separator()
        m.add_command(label="Copy path", command=lambda: self._copy_uv_path(self._selected_uv_id_from_tree() or ""))
        self._uv_ws_menu = m

    def _on_uv_ws_right_click(self, evt) -> None:
        if self._uv_ws_tree is None:
            return
        try:
            iid = self._uv_ws_tree.identify_row(evt.y)
            if iid:
                self._uv_ws_tree.selection_set(iid)
        except Exception:
            pass
        self._ensure_uv_ws_menu()
        try:
            if self._uv_ws_menu is not None:
                self._uv_ws_menu.tk_popup(int(evt.x_root), int(evt.y_root))
        finally:
            try:
                if self._uv_ws_menu is not None:
                    self._uv_ws_menu.grab_release()
            except Exception:
                pass

    def _on_panels_changed(self) -> None:
        self._rebuild_plot_axes()
        # Re-plot without blowing away selection.
        self._redraw_all()

    def _rebuild_plot_axes(self) -> None:
        if self._fig is None or self._canvas is None:
            return

        show_tic = bool(self.show_tic_var.get())
        show_spec = bool(self.show_spectrum_var.get())
        show_uv = bool(self.show_uv_var.get())

        visible: List[str] = []
        if show_tic:
            visible.append("tic")
        if show_spec:
            visible.append("spectrum")
        if show_uv:
            visible.append("uv")

        self._fig.clear()
        self._ax_tic = None
        self._ax_spec = None
        self._ax_uv = None

        if not visible:
            ax = self._fig.add_subplot(1, 1, 1)
            ax.axis("off")
            ax.text(0.5, 0.5, "All panels hidden", ha="center", va="center", transform=ax.transAxes)
            try:
                self._canvas.draw_idle()
            except Exception:
                pass
            return

        n = len(visible)
        # Stacked layout: top-to-bottom, stretch to fill.
        try:
            self._fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.94, hspace=0.38)
        except Exception:
            pass

        for i, kind in enumerate(visible, start=1):
            ax = self._fig.add_subplot(n, 1, i)
            if kind == "tic":
                self._ax_tic = ax
                ax.set_title(self.tic_title_var.get())
                ax.set_xlabel(self.tic_xlabel_var.get())
                ax.set_ylabel(self.tic_ylabel_var.get())
            elif kind == "spectrum":
                self._ax_spec = ax
                ax.set_title("Spectrum at selected RT")
                ax.set_xlabel(self.spec_xlabel_var.get())
                ax.set_ylabel(self.spec_ylabel_var.get())
                if self._current_spectrum_meta is None:
                    ax.text(0.5, 0.5, "Click TIC/UV to load a spectrum", ha="center", va="center", transform=ax.transAxes)
            elif kind == "uv":
                self._ax_uv = ax
                ax.set_title(self.uv_title_var.get())
                ax.set_xlabel(self.uv_xlabel_var.get())
                ax.set_ylabel(self.uv_ylabel_var.get())
                if self._active_uv_session() is None:
                    ax.text(0.5, 0.5, "No UV linked (use UV Workspace → Link)", ha="center", va="center", transform=ax.transAxes)

        self._apply_plot_style()
        self._draw_mpl_watermark()
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _apply_uv_ms_offset(self) -> None:
        raw = (self.uv_ms_rt_offset_var.get() or "").strip()
        if not raw:
            raw = f"{self._uv_ms_rt_offset_min:.3f}"
        try:
            val = float(raw)
        except Exception:
            messagebox.showerror("Invalid", "UV↔MS offset must be a number (minutes).", parent=self)
            self.uv_ms_rt_offset_var.set(f"{self._uv_ms_rt_offset_min:.3f}")
            return

        # Keep a reasonable range to avoid accidental huge shifts.
        if abs(val) > 30.0:
            messagebox.showerror("Invalid", "UV↔MS offset seems too large. Use minutes (e.g., 0.125).", parent=self)
            self.uv_ms_rt_offset_var.set(f"{self._uv_ms_rt_offset_min:.3f}")
            return

        self._uv_ms_rt_offset_min = float(val)
        self.uv_ms_rt_offset_var.set(f"{self._uv_ms_rt_offset_min:.6f}".rstrip("0").rstrip("."))

        # Re-align UV marker based on current MS selection.
        if self._current_spectrum_meta is not None:
            self._selected_rt_min = float(self._map_ms_to_uv_rt(float(self._current_spectrum_meta.rt_min)))

        if self._active_uv_session() is not None:
            self._plot_uv()
        self._update_status_current()

    def _has_uv_ms_alignment(self) -> bool:
        return (
            self._uv_ms_align_uv_rts is not None
            and self._uv_ms_align_ms_rts is not None
            and self._uv_ms_align_uv_rts.size >= 3
            and self._uv_ms_align_ms_rts.size >= 3
        )

    def _map_uv_to_ms_rt(self, uv_rt_min: float) -> float:
        uv_rt = float(uv_rt_min)
        off = float(self._uv_ms_rt_offset_min)
        if bool(self.uv_ms_align_enabled_var.get()) and self._has_uv_ms_alignment():
            uv_rts = self._uv_ms_align_uv_rts
            ms_rts = self._uv_ms_align_ms_rts
            if uv_rts is not None and ms_rts is not None and uv_rts.size >= 3:
                if uv_rt < float(uv_rts[0]) or uv_rt > float(uv_rts[-1]):
                    return uv_rt + off
                return float(np.interp(uv_rt, uv_rts, ms_rts))
        return uv_rt + off

    def _map_ms_to_uv_rt(self, ms_rt_min: float) -> float:
        ms_rt = float(ms_rt_min)
        off = float(self._uv_ms_rt_offset_min)
        if bool(self.uv_ms_align_enabled_var.get()) and self._has_uv_ms_alignment():
            uv_rts = self._uv_ms_align_uv_rts
            ms_rts = self._uv_ms_align_ms_rts
            if uv_rts is not None and ms_rts is not None and ms_rts.size >= 3:
                if ms_rt < float(ms_rts[0]) or ms_rt > float(ms_rts[-1]):
                    return ms_rt - off
                return float(np.interp(ms_rt, ms_rts, uv_rts))
        return ms_rt - off

    def _on_uv_ms_align_enabled_changed(self) -> None:
        if bool(self.uv_ms_align_enabled_var.get()):
            if not self._has_uv_ms_alignment():
                messagebox.showinfo(
                    "Auto-align",
                    "Auto-align has not been computed yet.\n\nClick ‘Auto-align UV↔MS’ (Navigate tab) to compute it.",
                    parent=self,
                )
                self.uv_ms_align_enabled_var.set(False)
                return

        if self._current_spectrum_meta is not None:
            try:
                self._selected_rt_min = float(self._map_ms_to_uv_rt(float(self._current_spectrum_meta.rt_min)))
            except Exception:
                pass
        if self._active_uv_session() is not None:
            self._plot_uv()
        self._update_status_current()

    def _open_alignment_diagnostics(self) -> None:
        if not self._has_uv_ms_alignment():
            messagebox.showinfo(
                "Alignment Diagnostics",
                "No alignment available.\n\nRun ‘Auto-align UV↔MS’ first (Navigate tab).",
                parent=self,
            )
            return
        if self._alignment_diag_win is not None:
            try:
                if bool(self._alignment_diag_win.winfo_exists()):
                    self._alignment_diag_win.deiconify()
                    self._alignment_diag_win.lift()
                    try:
                        self._alignment_diag_win.focus_force()
                    except Exception:
                        pass
                    try:
                        if hasattr(self._alignment_diag_win, "refresh"):
                            self._alignment_diag_win.refresh()
                    except Exception:
                        pass
                    return
            except Exception:
                pass
        win = AlignmentDiagnostics(self)
        self._alignment_diag_win = win

    def _smooth_1d(self, y: np.ndarray, window: int) -> np.ndarray:
        arr = np.asarray(y, dtype=float)
        if arr.size < 5:
            return arr
        w = int(window)
        if w < 3:
            return arr
        if w % 2 == 0:
            w += 1
        w = max(3, min(w, 301))
        kernel = np.ones(w, dtype=float) / float(w)
        try:
            return np.convolve(arr, kernel, mode="same")
        except Exception:
            return arr

    def _pick_peaks_time_series(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        top_n: int = 25,
        min_height: float = 0.15,
        min_spacing_min: float = 0.20,
        smooth_target_min: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 5 or y.size < 5:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        x = x[mask]
        y = y[mask]
        if x.size < 5:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        if np.any(np.diff(x) < 0):
            order = np.argsort(x)
            x = x[order]
            y = y[order]

        dx = np.diff(x)
        dt = float(np.median(dx[np.isfinite(dx) & (dx > 0)])) if dx.size else 0.0
        w = 9
        if dt > 0:
            w = int(round(float(smooth_target_min) / float(dt)))
        y_s = self._smooth_1d(y, window=w)

        lo = float(np.percentile(y_s, 10))
        hi = float(np.percentile(y_s, 99))
        denom = hi - lo
        if denom <= 0:
            denom = float(np.nanmax(y_s) - np.nanmin(y_s))
        if not np.isfinite(denom) or denom <= 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        yn = (y_s - lo) / float(denom)
        yn = np.clip(yn, 0.0, 1.0)

        if yn.size < 3:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        mid = yn[1:-1]
        is_peak = (mid >= yn[:-2]) & (mid > yn[2:]) & (mid >= float(min_height))
        cand = np.where(is_peak)[0] + 1
        if cand.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        scores = yn[cand]
        order = np.argsort(scores)[::-1]
        chosen_idx: List[int] = []
        chosen_rt: List[float] = []
        for k in order.tolist():
            idx = int(cand[int(k)])
            rt = float(x[idx])
            ok = True
            for rt0 in chosen_rt:
                if abs(rt - rt0) < float(min_spacing_min):
                    ok = False
                    break
            if not ok:
                continue
            chosen_idx.append(idx)
            chosen_rt.append(rt)
            if len(chosen_idx) >= int(top_n):
                break

        if not chosen_idx:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        chosen_idx = sorted(chosen_idx, key=lambda i: float(x[int(i)]))
        peaks_x = np.asarray([float(x[int(i)]) for i in chosen_idx], dtype=float)
        peaks_h = np.asarray([float(yn[int(i)]) for i in chosen_idx], dtype=float)
        return peaks_x, peaks_h

    def _match_peaks_monotonic(
        self,
        uv_rts: np.ndarray,
        uv_h: np.ndarray,
        ms_rts: np.ndarray,
        ms_h: np.ndarray,
        *,
        offset_guess_min: float,
        window_min: float = 1.5,
        amp_weight: float = 0.2,
    ) -> List[Tuple[int, int]]:
        uv_rts = np.asarray(uv_rts, dtype=float)
        uv_h = np.asarray(uv_h, dtype=float)
        ms_rts = np.asarray(ms_rts, dtype=float)
        ms_h = np.asarray(ms_h, dtype=float)
        n = int(uv_rts.size)
        m = int(ms_rts.size)
        if n == 0 or m == 0:
            return []

        dp_score = np.full((n + 1, m + 1), 0.0, dtype=float)
        dp_count = np.full((n + 1, m + 1), 0, dtype=int)
        prev_i = np.full((n + 1, m + 1), -1, dtype=int)
        prev_j = np.full((n + 1, m + 1), -1, dtype=int)
        prev_move = np.full((n + 1, m + 1), 0, dtype=np.int8)  # 1 skip uv, 2 skip ms, 3 match

        def better(ns: float, nc: int, os: float, oc: int) -> bool:
            if ns > os + 1e-12:
                return True
            if abs(ns - os) <= 1e-12 and nc > oc:
                return True
            return False

        for i in range(n + 1):
            for j in range(m + 1):
                base_s = float(dp_score[i, j])
                base_c = int(dp_count[i, j])

                if i < n:
                    if better(base_s, base_c, float(dp_score[i + 1, j]), int(dp_count[i + 1, j])):
                        dp_score[i + 1, j] = base_s
                        dp_count[i + 1, j] = base_c
                        prev_i[i + 1, j] = i
                        prev_j[i + 1, j] = j
                        prev_move[i + 1, j] = 1

                if j < m:
                    if better(base_s, base_c, float(dp_score[i, j + 1]), int(dp_count[i, j + 1])):
                        dp_score[i, j + 1] = base_s
                        dp_count[i, j + 1] = base_c
                        prev_i[i, j + 1] = i
                        prev_j[i, j + 1] = j
                        prev_move[i, j + 1] = 2

                if i < n and j < m:
                    delta = float(ms_rts[j] - uv_rts[i])
                    err = abs(delta - float(offset_guess_min))
                    if err <= float(window_min):
                        closeness = 1.0 - (err / float(window_min))
                        amp = 1.0 - abs(float(uv_h[i]) - float(ms_h[j]))
                        score = float(closeness) + float(amp_weight) * float(amp)
                        ns = base_s + score
                        nc = base_c + 1
                        if better(ns, nc, float(dp_score[i + 1, j + 1]), int(dp_count[i + 1, j + 1])):
                            dp_score[i + 1, j + 1] = ns
                            dp_count[i + 1, j + 1] = nc
                            prev_i[i + 1, j + 1] = i
                            prev_j[i + 1, j + 1] = j
                            prev_move[i + 1, j + 1] = 3

        i = n
        j = m
        pairs: List[Tuple[int, int]] = []
        while i > 0 or j > 0:
            mv = int(prev_move[i, j])
            pi = int(prev_i[i, j])
            pj = int(prev_j[i, j])
            if mv == 0 or pi < 0 or pj < 0:
                break
            if mv == 3:
                pairs.append((i - 1, j - 1))
            i, j = pi, pj
        pairs.reverse()
        return pairs

    def _auto_align_uv_ms(self) -> None:
        if self._active_uv_session() is None:
            messagebox.showerror("Auto-align", "Link a UV CSV to the active mzML session first.", parent=self)
            return
        if self._filtered_rts is None or self._filtered_tics is None or self._filtered_rts.size < 5:
            messagebox.showerror("Auto-align", "Load an mzML file first (MS1 TIC is required).", parent=self)
            return

        uv_x, uv_y = self._active_uv_xy()
        if uv_x is None or uv_y is None or uv_x.size < 5:
            messagebox.showerror("Auto-align", "UV chromatogram is empty.", parent=self)
            return

        uv_x0 = np.asarray(uv_x, dtype=float).copy()
        uv_y0 = np.asarray(uv_y, dtype=float).copy()
        ms_x0 = np.asarray(self._filtered_rts, dtype=float).copy()
        ms_y0 = np.asarray(self._filtered_tics, dtype=float).copy()
        off0 = float(self._uv_ms_rt_offset_min)

        self._show_busy("Auto-aligning UV↔MS…")

        result_box: Dict[str, Any] = {}

        def worker() -> None:
            try:
                uv_peaks_x, uv_peaks_h = self._pick_peaks_time_series(uv_x0, uv_y0, top_n=25)
                ms_peaks_x, ms_peaks_h = self._pick_peaks_time_series(ms_x0, ms_y0, top_n=25)
                if uv_peaks_x.size < 3 or ms_peaks_x.size < 3:
                    raise RuntimeError("Not enough peaks detected (need at least 3 in both UV and TIC).")

                pairs = self._match_peaks_monotonic(
                    uv_peaks_x,
                    uv_peaks_h,
                    ms_peaks_x,
                    ms_peaks_h,
                    offset_guess_min=off0,
                    window_min=1.5,
                )
                if len(pairs) < 3:
                    raise RuntimeError("Failed to find a stable peak alignment (need ≥3 matched peaks).")

                uv_anchor = np.asarray([float(uv_peaks_x[i]) for i, _j in pairs], dtype=float)
                ms_anchor = np.asarray([float(ms_peaks_x[j]) for _i, j in pairs], dtype=float)

                order = np.argsort(uv_anchor)
                uv_anchor = uv_anchor[order]
                ms_anchor = ms_anchor[order]

                keep = [0]
                for k in range(1, int(uv_anchor.size)):
                    if float(uv_anchor[k]) > float(uv_anchor[keep[-1]]) + 1e-9:
                        keep.append(k)
                uv_anchor = uv_anchor[np.asarray(keep, dtype=int)]
                ms_anchor = ms_anchor[np.asarray(keep, dtype=int)]

                if uv_anchor.size < 3:
                    raise RuntimeError("Alignment produced too few unique anchors.")

                deltas = ms_anchor - uv_anchor
                med_delta = float(np.median(deltas))
                mad = float(np.median(np.abs(deltas - med_delta)))

                result_box["uv_anchor"] = uv_anchor
                result_box["ms_anchor"] = ms_anchor
                result_box["n"] = int(uv_anchor.size)
                result_box["med_delta"] = float(med_delta)
                result_box["mad"] = float(mad)
            except Exception as exc:
                result_box["error"] = str(exc)

        def done() -> None:
            self._hide_busy()
            err = result_box.get("error")
            if err:
                messagebox.showerror("Auto-align", f"Auto-align failed:\n{err}", parent=self)
                return

            uv_anchor = result_box.get("uv_anchor")
            ms_anchor = result_box.get("ms_anchor")
            if uv_anchor is None or ms_anchor is None:
                messagebox.showerror("Auto-align", "Auto-align failed (no anchors returned).", parent=self)
                return

            self._uv_ms_align_uv_rts = np.asarray(uv_anchor, dtype=float)
            self._uv_ms_align_ms_rts = np.asarray(ms_anchor, dtype=float)
            self.uv_ms_align_enabled_var.set(True)

            if self._current_spectrum_meta is not None:
                try:
                    self._selected_rt_min = float(self._map_ms_to_uv_rt(float(self._current_spectrum_meta.rt_min)))
                except Exception:
                    pass
            if self._active_uv_session() is not None:
                self._plot_uv()
            self._update_status_current()

            n = int(result_box.get("n", 0))
            med_delta = float(result_box.get("med_delta", 0.0))
            mad = float(result_box.get("mad", 0.0))
            messagebox.showinfo(
                "Auto-align",
                f"Auto-align enabled.\n\nAnchors: {n}\nMedian offset: {med_delta:.3f} min\nOffset variation (MAD): {mad:.3f} min",
                parent=self,
            )

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        def poll() -> None:
            if t.is_alive():
                self.after(200, poll)
            else:
                done()

        self.after(200, poll)

    def _local_maxima_indices(
        self,
        rt: np.ndarray,
        signal: np.ndarray,
        *,
        center_rt: float,
        half_window_min: float,
        min_height_rel: float = 0.08,
    ) -> np.ndarray:
        """Return indices (into original arrays) of local maxima near center_rt.

        Uses a simple local-max condition (y[i] >= y[i-1] and y[i] > y[i+1])
        with a basic min-height threshold relative to the local window.
        """
        r = np.asarray(rt, dtype=float)
        s = np.asarray(signal, dtype=float)
        if r.size < 5 or s.size < 5:
            return np.asarray([], dtype=int)

        mask = np.isfinite(r) & np.isfinite(s)
        if not np.any(mask):
            return np.asarray([], dtype=int)
        r = r[mask]
        s = s[mask]
        if r.size < 5:
            return np.asarray([], dtype=int)

        # Ensure increasing RT for windowing
        if np.any(np.diff(r) < 0):
            order = np.argsort(r)
            r = r[order]
            s = s[order]

        c = float(center_rt)
        w = float(half_window_min)
        in_win = (r >= (c - w)) & (r <= (c + w))
        if np.count_nonzero(in_win) < 5:
            return np.asarray([], dtype=int)

        idx = np.where(in_win)[0]
        lo = int(idx[0])
        hi = int(idx[-1])
        if hi - lo < 4:
            return np.asarray([], dtype=int)

        seg = s[lo : hi + 1]
        if seg.size < 5:
            return np.asarray([], dtype=int)

        s_min = float(np.nanmin(seg))
        s_max = float(np.nanmax(seg))
        span = float(s_max - s_min)
        if not np.isfinite(span) or span <= 0:
            return np.asarray([], dtype=int)
        min_height = s_min + float(min_height_rel) * span

        mid = seg[1:-1]
        is_max = (mid >= seg[:-2]) & (mid > seg[2:]) & (mid >= float(min_height))
        local = np.where(is_max)[0] + 1
        if local.size == 0:
            return np.asarray([], dtype=int)

        return (local + lo).astype(int)

    def _nearest_apex_distance_and_crowding(
        self,
        rt: np.ndarray,
        signal: np.ndarray,
        *,
        anchor_rt: float,
        search_half_window_min: float,
        crowd_half_window_min: float,
    ) -> Tuple[Optional[float], int]:
        peaks = self._local_maxima_indices(
            rt,
            signal,
            center_rt=float(anchor_rt),
            half_window_min=float(search_half_window_min),
        )
        if peaks.size == 0:
            return None, 0
        r = np.asarray(rt, dtype=float)
        peak_rts = np.asarray([float(r[int(i)]) for i in peaks.tolist()], dtype=float)
        dists = np.abs(peak_rts - float(anchor_rt))
        apex_dist = float(np.min(dists)) if dists.size else None
        crowd = int(np.count_nonzero(dists <= float(crowd_half_window_min)))
        return apex_dist, crowd

    def _compute_confidence_for_uv_label(self, anchor_uv_rt: float, ms_rt: float) -> Dict[str, float]:
        """Compute confidence metadata for an MS→UV transferred label (0..100%)."""
        tau = 0.08

        uv_anchor = float(anchor_uv_rt)
        ms_scan = float(ms_rt)

        if bool(self.uv_ms_align_enabled_var.get()) and self._has_uv_ms_alignment():
            ms_pred = float(self._map_uv_to_ms_rt(float(uv_anchor)))
        else:
            ms_pred = float(uv_anchor) + float(self._uv_ms_rt_offset_min)

        rt_delta = abs(float(ms_pred) - float(ms_scan))
        rt_score = float(math.exp(-float(rt_delta) / float(tau)))

        uv_x, uv_y = self._active_uv_xy()
        uv_peak_score = 0.3
        crowd_count = 0
        if uv_x is not None and uv_y is not None and uv_x.size >= 5:
            apex_dist, crowd_count = self._nearest_apex_distance_and_crowding(
                uv_x,
                uv_y,
                anchor_rt=float(uv_anchor),
                search_half_window_min=0.25,
                crowd_half_window_min=0.10,
            )
            if apex_dist is not None:
                uv_peak_score = float(math.exp(-float(apex_dist) / 0.08))

        tic_peak_score = 0.5
        if self._filtered_rts is not None and self._filtered_tics is not None and self._filtered_rts.size >= 5:
            peaks = self._local_maxima_indices(
                self._filtered_rts,
                self._filtered_tics,
                center_rt=float(ms_scan),
                half_window_min=0.35,
            )
            if peaks.size > 0:
                peak_rts = np.asarray([float(self._filtered_rts[int(i)]) for i in peaks.tolist()], dtype=float)
                tic_dist = float(np.min(np.abs(peak_rts - float(ms_scan))))
                tic_peak_score = float(math.exp(-float(tic_dist) / 0.08))

        crowd_factor = 1.0
        if int(crowd_count) > 1:
            crowd_factor = float(0.85 ** (int(crowd_count) - 1))

        raw = 0.5 * float(rt_score) + 0.35 * float(uv_peak_score) + 0.15 * float(tic_peak_score)
        conf = 100.0 * float(raw) * float(crowd_factor)
        conf = max(0.0, min(100.0, float(conf)))
        return {
            "confidence": float(conf),
            "rt_delta_min": float(rt_delta),
            "uv_peak_score": float(uv_peak_score),
            "tic_peak_score": float(tic_peak_score),
        }

    def _format_uv_label_display_text(self, st: UVLabelState) -> str:
        base = str(getattr(st, "text", "") or "").strip()
        try:
            conf = float(getattr(st, "confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        return f"{base}  [{conf:.0f}%]"

    def _go_to_index(self, idx: int) -> None:
        if not self._filtered_meta:
            return
        self._show_spectrum_for_index(int(idx))

    def _go_last(self) -> None:
        if not self._filtered_meta:
            return
        self._show_spectrum_for_index(len(self._filtered_meta) - 1)

    def _step_spectrum(self, delta: int) -> None:
        if not self._filtered_meta:
            return
        if self._current_scan_index is None:
            self._show_spectrum_for_index(0)
            return
        self._show_spectrum_for_index(int(self._current_scan_index) + int(delta))

    def _jump_to_rt(self) -> None:
        if self._filtered_rts is None or self._filtered_rts.size == 0:
            return
        raw = (self._rt_jump_var.get() or "").strip()
        if not raw:
            return
        try:
            rt = float(raw)
        except Exception:
            messagebox.showerror("Invalid", "RT must be a number (minutes).", parent=self)
            return
        idx = int(np.argmin(np.abs(self._filtered_rts - float(rt))))
        self._show_spectrum_for_index(idx)

    # --- EIC (Extracted Ion Chromatogram) ---
    def _register_ms_position_listener(self, cb: Callable[[Optional[float]], None]) -> None:
        if cb not in self._ms_position_listeners:
            self._ms_position_listeners.append(cb)

    def _unregister_ms_position_listener(self, cb: Callable[[Optional[float]], None]) -> None:
        try:
            if cb in self._ms_position_listeners:
                self._ms_position_listeners.remove(cb)
        except Exception:
            pass

    def _register_active_session_listener(self, cb: Callable[[], None]) -> None:
        if cb not in self._active_session_listeners:
            self._active_session_listeners.append(cb)

    def _unregister_active_session_listener(self, cb: Callable[[], None]) -> None:
        try:
            if cb in self._active_session_listeners:
                self._active_session_listeners.remove(cb)
        except Exception:
            pass

    def _notify_ms_position_changed(self) -> None:
        rt_min: Optional[float] = None
        try:
            meta = self._current_spectrum_meta
            if meta is not None:
                rt_min = float(meta.rt_min)
        except Exception:
            rt_min = None
        for cb in list(self._ms_position_listeners):
            try:
                cb(rt_min)
            except Exception:
                continue

    def _notify_active_session_changed(self) -> None:
        for cb in list(self._active_session_listeners):
            try:
                cb()
            except Exception:
                continue

    def _open_sim_dialog(self) -> None:
        if self._index is None or self.mzml_path is None:
            messagebox.showinfo("EIC", "Open an mzML file first.", parent=self)
            return
        dlg = tk.Toplevel(self)
        dlg.title("EIC (Extracted Ion Chromatogram)")
        dlg.resizable(False, False)
        dlg.transient(self)

        pad = 10
        frm = ttk.Frame(dlg, padding=pad)
        frm.grid(row=0, column=0)

        # Defaults
        last = self._sim_last_params
        default_target = ""
        default_tol = "10"
        default_unit = "ppm"
        default_use_pol = True
        if last is not None:
            try:
                default_target = f"{float(last[0]):.6f}".rstrip("0").rstrip(".")
            except Exception:
                default_target = ""
            try:
                default_tol = f"{float(last[1]):g}"
            except Exception:
                default_tol = "10"
            try:
                default_unit = str(last[2] or "ppm")
            except Exception:
                default_unit = "ppm"
            try:
                default_use_pol = bool(last[3])
            except Exception:
                default_use_pol = True

        ttk.Label(frm, text="Target m/z").grid(row=0, column=0, sticky="w")
        target_var = tk.StringVar(value=default_target)
        target_ent = ttk.Entry(frm, textvariable=target_var, width=14)
        target_ent.grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Label(frm, text="Tolerance").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tol_var = tk.StringVar(value=default_tol)
        tol_ent = ttk.Entry(frm, textvariable=tol_var, width=14)
        tol_ent.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(8, 0))

        unit_var = tk.StringVar(value=default_unit)
        unit_frame = ttk.Frame(frm)
        unit_frame.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(4, 0))
        unit_ppm = ttk.Radiobutton(unit_frame, text="ppm", value="ppm", variable=unit_var)
        unit_da = ttk.Radiobutton(unit_frame, text="Da", value="Da", variable=unit_var)
        unit_ppm.pack(side=tk.LEFT)
        unit_da.pack(side=tk.LEFT, padx=(10, 0))

        use_pol_var = tk.BooleanVar(value=bool(default_use_pol))
        use_pol_cb = ttk.Checkbutton(frm, text="Use current polarity filter", variable=use_pol_var)
        use_pol_cb.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, sticky="e", pady=(12, 0))
        run_btn = ttk.Button(btns, text="Run", command=lambda: on_run())
        cancel_btn = ttk.Button(btns, text="Cancel", command=dlg.destroy)
        run_btn.grid(row=0, column=0, padx=(0, 8))
        cancel_btn.grid(row=0, column=1)

        def on_run() -> None:
            try:
                target_mz = float((target_var.get() or "").strip())
            except Exception:
                messagebox.showerror("Invalid", "Target m/z must be a number.", parent=dlg)
                return
            try:
                tol_value = float((tol_var.get() or "").strip())
            except Exception:
                messagebox.showerror("Invalid", "Tolerance must be a number.", parent=dlg)
                return
            if not np.isfinite(float(target_mz)):
                messagebox.showerror("Invalid", "Target m/z must be finite.", parent=dlg)
                return
            if not np.isfinite(float(tol_value)) or float(tol_value) <= 0:
                messagebox.showerror("Invalid", "Tolerance must be a positive finite number.", parent=dlg)
                return
            unit = (unit_var.get() or "ppm").strip()
            if unit.lower() not in ("ppm", "da"):
                unit = "ppm"
            use_pol = bool(use_pol_var.get())
            self._sim_last_params = (float(target_mz), float(tol_value), str(unit), bool(use_pol))
            try:
                dlg.destroy()
            except Exception:
                pass
            # Create the non-modal EIC window (it will compute/cached-load as needed)
            SIMWindow(self, target_mz=float(target_mz), tol_value=float(tol_value), tol_unit=str(unit), use_current_polarity=bool(use_pol))

        ToolTip.attach(target_ent, TOOLTIP_TEXT["sim_target_mz"])
        ToolTip.attach(tol_ent, TOOLTIP_TEXT["sim_tol_value"])
        ToolTip.attach(unit_ppm, TOOLTIP_TEXT["sim_tol_unit"])
        ToolTip.attach(unit_da, TOOLTIP_TEXT["sim_tol_unit"])
        ToolTip.attach(use_pol_cb, TOOLTIP_TEXT["sim_use_polarity"])

        try:
            target_ent.focus_set()
        except Exception:
            pass

    def _sim_cache_key(self, *, mzml_path: Path, polarity_filter: str, target_mz: float, tol_value: float, unit: str) -> Tuple[str, str, float, float, str]:
        return (
            str(Path(mzml_path).expanduser().resolve()),
            str(polarity_filter or "all"),
            float(round(float(target_mz), 4)),
            float(tol_value),
            str(unit or "ppm").strip().lower(),
        )

    def _run_sim_for_window(self, win: "SIMWindow") -> None:
        if self._index is None or self.mzml_path is None:
            try:
                win.set_data(mzml_path=Path(""), rts=np.asarray([], dtype=float), intensities=np.asarray([], dtype=float), polarity_filter="all")
            except Exception:
                pass
            return

        target_mz, tol_value, unit, use_pol = win.sim_params
        unit = (unit or "ppm").strip()
        pol_filter = str(self.polarity_var.get()) if bool(use_pol) else "all"
        mzml_path = Path(self.mzml_path)

        key = self._sim_cache_key(
            mzml_path=mzml_path,
            polarity_filter=pol_filter,
            target_mz=float(target_mz),
            tol_value=float(tol_value),
            unit=str(unit),
        )

        with self._sim_cache_lock:
            cached = self._sim_cache.get(key)
        if cached is not None:
            rts, ints = cached
            try:
                win.set_data(mzml_path=mzml_path, rts=rts, intensities=ints, polarity_filter=pol_filter)
            except Exception:
                pass
            return

        # Snapshot scan list for this run (avoids races if active session switches during compute).
        meta_list: List[SpectrumMeta] = []
        try:
            if bool(use_pol):
                meta_list = list(self._filtered_meta)
            else:
                ms1 = list(self._index.ms1) if self._index is not None else []
                ms1.sort(key=lambda m: float(m.rt_min))
                meta_list = ms1
        except Exception:
            meta_list = []

        if not meta_list:
            messagebox.showinfo("EIC", "No MS1 scans available (after filtering).", parent=self)
            try:
                win.set_data(mzml_path=mzml_path, rts=np.asarray([], dtype=float), intensities=np.asarray([], dtype=float), polarity_filter=pol_filter)
            except Exception:
                pass
            return

        self._sim_run_token += 1
        token = int(self._sim_run_token)
        try:
            setattr(win, "_sim_token", int(token))
        except Exception:
            pass
        self._show_busy("Computing EIC chromatogram…")

        # Prepare lightweight tuples for the worker
        scan_tuples: List[Tuple[str, float]] = []
        for m in meta_list:
            try:
                scan_tuples.append((str(m.spectrum_id), float(m.rt_min)))
            except Exception:
                continue

        def worker() -> None:
            reader = None
            try:
                reader = mzml.MzML(str(mzml_path))
                rts = np.asarray([rt for _sid, rt in scan_tuples], dtype=float)
                out = np.zeros((len(scan_tuples),), dtype=float)

                unit_l = str(unit or "ppm").strip().lower()
                tgt = float(target_mz)
                tol_v = float(tol_value)

                for i, (sid, _rt) in enumerate(scan_tuples):
                    try:
                        try:
                            spec = reader.get_by_id(str(sid))
                        except Exception:
                            spec = reader[str(sid)]
                        mz_array = spec.get("m/z array")
                        int_array = spec.get("intensity array")
                        if mz_array is None or int_array is None:
                            out[i] = 0.0
                            continue
                        mz_vals = np.asarray(mz_array, dtype=float)
                        int_vals = np.asarray(int_array, dtype=float)
                        if mz_vals.size == 0 or int_vals.size == 0:
                            out[i] = 0.0
                            continue
                        mask = np.isfinite(mz_vals) & np.isfinite(int_vals)
                        if not np.any(mask):
                            out[i] = 0.0
                            continue
                        mz_vals = mz_vals[mask]
                        int_vals = int_vals[mask]

                        if unit_l == "ppm":
                            tol_da = abs(float(tgt)) * (float(tol_v) * 1e-6)
                        else:
                            tol_da = float(tol_v)

                        hit = np.abs(mz_vals - float(tgt)) <= float(tol_da)
                        if not np.any(hit):
                            out[i] = 0.0
                            continue
                        out[i] = float(np.sum(int_vals[hit]))
                    except Exception:
                        out[i] = 0.0

                self.after(
                    0,
                    lambda: self._on_sim_ready(
                        token=token,
                        key=key,
                        mzml_path=mzml_path,
                        pol_filter=pol_filter,
                        rts=rts,
                        ints=out,
                        win=win,
                    ),
                )
            except Exception as exc:
                self.after(0, lambda: self._on_sim_error(token=token, err=exc))
            finally:
                try:
                    if reader is not None:
                        reader.close()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _on_sim_error(self, *, token: int, err: Exception) -> None:
        try:
            self._hide_busy()
        except Exception:
            pass
        messagebox.showerror("EIC", f"Failed to compute EIC chromatogram:\n\n{err}", parent=self)

    def _on_sim_ready(
        self,
        *,
        token: int,
        key: Tuple[str, str, float, float, str],
        mzml_path: Path,
        pol_filter: str,
        rts: np.ndarray,
        ints: np.ndarray,
        win: "SIMWindow",
    ) -> None:
        try:
            self._hide_busy()
        except Exception:
            pass
        # Best-effort cancellation (per window)
        try:
            if int(getattr(win, "_sim_token", token)) != int(token):
                return
        except Exception:
            pass

        rts = np.asarray(rts, dtype=float)
        ints = np.asarray(ints, dtype=float)
        with self._sim_cache_lock:
            self._sim_cache[key] = (rts, ints)
        try:
            win.set_data(mzml_path=Path(mzml_path), rts=rts, intensities=ints, polarity_filter=str(pol_filter))
        except Exception:
            pass

    def _mz_find_history_format(self, entry: Tuple[float, float, str, str, float]) -> str:
        mz_v, tol_v, unit, mode, min_int = entry
        unit = (unit or "ppm").strip()
        mode = (mode or "Nearest").strip()
        min_int = float(min_int)
        min_part = ""
        if min_int > 0:
            min_part = f"; I>={min_int:g}"
        if unit.lower() == "ppm":
            return f"m/z {mz_v:.4f} ± {tol_v:g} ppm ({mode}){min_part}"
        return f"m/z {mz_v:.4f} ± {tol_v:g} Da ({mode}){min_part}"

    def _mz_find_history_refresh_combobox(self) -> None:
        combo = getattr(self, "_mz_find_history_combo", None)
        if combo is None:
            return
        self._mz_find_history_map.clear()
        labels: List[str] = []
        for entry in list(self._mz_find_history)[:10]:
            lbl = self._mz_find_history_format(entry)
            labels.append(lbl)
            self._mz_find_history_map[lbl] = entry
        try:
            combo.configure(values=labels)
        except Exception:
            pass

    def _mz_find_history_on_select(self, _evt=None) -> None:
        key = (self._mz_find_history_var.get() or "").strip()
        if not key:
            return
        entry = self._mz_find_history_map.get(key)
        if entry is None:
            return
        mz_v, tol_v, unit, mode, min_int = entry
        self._mz_find_mz_var.set(f"{float(mz_v):.6f}".rstrip("0").rstrip("."))
        self._mz_find_tol_var.set(f"{float(tol_v):g}")
        self._mz_find_unit_var.set(str(unit))
        self._mz_find_mode_var.set(str(mode))
        self._mz_find_min_int_var.set(f"{float(min_int):g}")

    def _open_find_mz_dialog(self) -> None:
        if getattr(self, "_mz_find_dialog", None) is not None:
            try:
                self._mz_find_history_refresh_combobox()
                self._mz_find_dialog.deiconify()
                self._mz_find_dialog.lift()
                return
            except Exception:
                self._mz_find_dialog = None

        dlg = tk.Toplevel(self)
        dlg.title("Find m/z")
        dlg.resizable(False, False)
        dlg.transient(self)

        frm = ttk.Frame(dlg, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="History").grid(row=0, column=0, sticky="w")
        hist = ttk.Combobox(frm, textvariable=self._mz_find_history_var, state="readonly", width=34)
        hist.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(10, 0))
        self._mz_find_history_combo = hist
        self._mz_find_history_refresh_combobox()
        try:
            hist.bind("<<ComboboxSelected>>", self._mz_find_history_on_select, add=True)
        except Exception:
            pass

        ttk.Label(frm, text="m/z").grid(row=1, column=0, sticky="w", pady=(10, 0))
        mz_ent = ttk.Entry(frm, textvariable=self._mz_find_mz_var, width=14)
        mz_ent.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(10, 0))

        ttk.Label(frm, text="Tolerance").grid(row=2, column=0, sticky="w", pady=(8, 0))
        tol_ent = ttk.Entry(frm, textvariable=self._mz_find_tol_var, width=14)
        tol_ent.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(8, 0))

        unit = ttk.Combobox(frm, textvariable=self._mz_find_unit_var, values=["ppm", "Da"], state="readonly", width=8)
        unit.grid(row=2, column=2, sticky="w", padx=(10, 0), pady=(8, 0))

        ttk.Label(frm, text="Min intensity").grid(row=3, column=0, sticky="w", pady=(8, 0))
        minint_ent = ttk.Entry(frm, textvariable=self._mz_find_min_int_var, width=14)
        minint_ent.grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(8, 0))

        ttk.Label(frm, text="Mode").grid(row=4, column=0, sticky="w", pady=(8, 0))
        mode = ttk.Combobox(frm, textvariable=self._mz_find_mode_var, values=["Nearest", "Forward", "Backward"], state="readonly", width=12)
        mode.grid(row=4, column=1, sticky="w", padx=(10, 0), pady=(8, 0))

        buttons = ttk.Frame(frm)
        buttons.grid(row=5, column=0, columnspan=4, sticky="e", pady=(14, 0))
        find_btn = ttk.Button(buttons, text="Find", command=self._find_mz_jump)
        find_btn.grid(row=0, column=0, padx=(0, 8))
        close_btn = ttk.Button(buttons, text="Close", command=lambda: dlg.event_generate("<<CloseFindMz>>"))
        close_btn.grid(row=0, column=1)

        def _on_close() -> None:
            try:
                self._mz_find_dialog = None
            except Exception:
                pass
            try:
                dlg.destroy()
            except Exception:
                pass

        dlg.protocol("WM_DELETE_WINDOW", _on_close)
        try:
            dlg.bind("<<CloseFindMz>>", lambda e: (_on_close(), "break"), add=True)
        except Exception:
            pass
        self._mz_find_dialog = dlg

        ToolTip.attach(hist, TOOLTIP_TEXT["find_history"])
        ToolTip.attach(mz_ent, TOOLTIP_TEXT["find_target_mz"])
        ToolTip.attach(tol_ent, TOOLTIP_TEXT["find_tol"])
        ToolTip.attach(unit, TOOLTIP_TEXT["find_tol_unit"])
        ToolTip.attach(minint_ent, TOOLTIP_TEXT["find_min_int"])
        ToolTip.attach(mode, TOOLTIP_TEXT["find_mode"])
        ToolTip.attach(find_btn, TOOLTIP_TEXT["find_action"])
        ToolTip.attach(close_btn, TOOLTIP_TEXT["find_close"])
        try:
            mz_ent.focus_set()
            mz_ent.selection_range(0, tk.END)
            mz_ent.bind("<Return>", lambda e: (self._find_mz_jump(), "break"))
            tol_ent.bind("<Return>", lambda e: (self._find_mz_jump(), "break"))
            minint_ent.bind("<Return>", lambda e: (self._find_mz_jump(), "break"))
        except Exception:
            pass

    def _clear_mz_find_highlight(self) -> None:
        try:
            if self._mz_find_highlight_artist is not None:
                self._mz_find_highlight_artist.remove()
        except Exception:
            pass
        self._mz_find_highlight_artist = None
        self._mz_find_highlight_target_mz = None
        self._mz_find_highlight_tol_da = None

    def _draw_mz_find_highlight(self, peak_mz: float) -> None:
        if self._ax_spec is None:
            return
        self._clear_mz_find_highlight()
        try:
            self._mz_find_highlight_artist = self._ax_spec.axvline(float(peak_mz), color=ACCENT_ORANGE, lw=1.3, alpha=0.95)
        except Exception:
            self._mz_find_highlight_artist = None
        try:
            if self._canvas is not None:
                self._canvas.draw_idle()
        except Exception:
            pass

    def _mz_find_cache_key(self, target_mz: float, tol_value: float, tol_unit: str, min_intensity: float) -> Tuple[str, str, float, float, str, float]:
        mzml_path = str(self.mzml_path) if self.mzml_path is not None else ""
        pol = (self.polarity_var.get() or "all").strip()
        mz_r = float(round(float(target_mz), 4))
        unit = (tol_unit or "ppm").strip()
        return (mzml_path, str(pol), float(mz_r), float(tol_value), str(unit), float(min_intensity))

    def _mz_find_tol_da(self, target_mz: float, tol_value: float, tol_unit: str) -> float:
        unit = (tol_unit or "ppm").strip().lower()
        if unit == "ppm":
            return abs(float(target_mz)) * float(tol_value) * 1e-6
        return float(tol_value)

    def _mz_find_update_history(self, params: Tuple[float, float, str, str, float]) -> None:
        # unique by full tuple, most recent first
        try:
            self._mz_find_history = [p for p in self._mz_find_history if p != params]
        except Exception:
            pass
        self._mz_find_history.insert(0, params)
        self._mz_find_history = self._mz_find_history[:10]
        self._mz_find_history_refresh_combobox()
        # Keep the displayed selection aligned with the most recent entry.
        try:
            self._mz_find_history_var.set(self._mz_find_history_format(params))
        except Exception:
            pass

    def _mz_find_pick_target_index(
        self,
        matches: List[int],
        *,
        meta_list: List[SpectrumMeta],
        cur_idx: int,
        mode: str,
        same_params: bool,
        tol_da: float,
        target_mz: float,
        peak_cache: Dict[int, Tuple[float, float, float]],
    ) -> Optional[int]:
        if not matches:
            return None
        matches_sorted = sorted(set(int(i) for i in matches))
        cur_idx = int(max(0, min(cur_idx, len(meta_list) - 1))) if meta_list else int(cur_idx)

        # Determine whether current scan matches
        cur_matches = int(cur_idx) in set(matches_sorted)

        # Press-again cycling: if current matches and params identical to last => next match forward (wrap)
        if cur_matches and bool(same_params):
            pos = 0
            for j, mi in enumerate(matches_sorted):
                if mi == int(cur_idx):
                    pos = j
                    break
            next_pos = (pos + 1) % len(matches_sorted)
            return int(matches_sorted[next_pos])

        m = (mode or "Nearest").strip().lower()
        if m == "forward":
            for mi in matches_sorted:
                if int(mi) > int(cur_idx):
                    return int(mi)
            return int(matches_sorted[0])
        if m == "backward":
            for mi in reversed(matches_sorted):
                if int(mi) < int(cur_idx):
                    return int(mi)
            return int(matches_sorted[-1])

        # Nearest (RT distance); exclude current if it matches
        if not meta_list:
            return int(matches_sorted[0])
        cur_rt = float(meta_list[int(cur_idx)].rt_min) if 0 <= int(cur_idx) < len(meta_list) else float(meta_list[0].rt_min)

        candidates = [mi for mi in matches_sorted if (not cur_matches) or (mi != int(cur_idx))]
        if not candidates:
            return int(matches_sorted[0])

        best_i = None
        best_dist = None
        for mi in candidates:
            rt = float(meta_list[int(mi)].rt_min)
            dist = abs(float(rt) - float(cur_rt))
            if best_dist is None or dist < float(best_dist) or (abs(dist - float(best_dist)) < 1e-12 and (best_i is None or int(mi) > int(best_i))):
                best_dist = float(dist)
                best_i = int(mi)
        return int(best_i) if best_i is not None else None

    def _find_mz_jump(self) -> None:
        if self._index is None or not self._filtered_meta:
            messagebox.showinfo("Find m/z", "Open an mzML file first.", parent=self)
            return
        if self.mzml_path is None:
            messagebox.showinfo("Find m/z", "No active mzML session.", parent=self)
            return

        raw_mz = (self._mz_find_mz_var.get() or "").strip()
        raw_tol = (self._mz_find_tol_var.get() or "").strip()
        raw_minint = (self._mz_find_min_int_var.get() or "").strip()
        unit = (self._mz_find_unit_var.get() or "ppm").strip()
        mode = (self._mz_find_mode_var.get() or "Nearest").strip()

        try:
            target_mz = float(raw_mz)
            tol_value = float(raw_tol)
            min_int = float(raw_minint) if raw_minint else 0.0
        except Exception:
            messagebox.showerror("Invalid", "Enter numeric m/z, tolerance, and min intensity.", parent=self)
            return
        if tol_value <= 0:
            messagebox.showerror("Invalid", "Tolerance must be > 0.", parent=self)
            return
        if min_int < 0:
            messagebox.showerror("Invalid", "Min intensity must be >= 0.", parent=self)
            return
        if unit not in ("ppm", "Da"):
            unit = "ppm"
        if mode not in ("Nearest", "Forward", "Backward"):
            mode = "Nearest"

        params = (float(target_mz), float(tol_value), str(unit), str(mode), float(min_int))
        self._mz_find_update_history(params)

        same_params = (self._last_mz_find_params == params)
        if not same_params:
            self._last_mz_find_last_hit_idx = None
        self._last_mz_find_params = params

        key = self._mz_find_cache_key(target_mz, tol_value, unit, min_int)
        tol_da = self._mz_find_tol_da(target_mz, tol_value, unit)

        if key in self._mz_find_cache:
            matches = list(self._mz_find_cache.get(key) or [])
            peak_cache = dict(self._mz_find_peak_cache.get(key) or {})
            self._mz_find_finish_jump(target_mz, tol_value, unit, mode, tol_da, key, matches, peak_cache, same_params=same_params)
            return

        # Compute in worker thread and cache
        meta_list = list(self._filtered_meta)
        mzml_path = Path(self.mzml_path)

        self._show_busy("Searching scans for m/z…")

        def worker() -> None:
            try:
                matches: List[int] = []
                peak_info: Dict[int, Tuple[float, float, float]] = {}

                with mzml.MzML(str(mzml_path)) as reader:
                    for idx, meta in enumerate(meta_list):
                        spectrum_id = str(meta.spectrum_id)
                        try:
                            spectrum = reader.get_by_id(spectrum_id)
                        except Exception:
                            spectrum = reader[spectrum_id]
                        mz_array = spectrum.get("m/z array")
                        int_array = spectrum.get("intensity array")
                        if mz_array is None or int_array is None:
                            continue

                        mz_vals = np.asarray(mz_array, dtype=float)
                        int_vals = np.asarray(int_array, dtype=float)
                        if mz_vals.size == 0 or int_vals.size == 0:
                            continue

                        mask = np.isfinite(mz_vals) & np.isfinite(int_vals)
                        if not np.any(mask):
                            continue
                        mz_vals = mz_vals[mask]
                        int_vals = int_vals[mask]

                        if mz_vals.size < 1:
                            continue

                        # Ensure sorted for searchsorted
                        try:
                            if mz_vals.size > 2 and np.any(np.diff(mz_vals) < 0):
                                order = np.argsort(mz_vals)
                                mz_s = mz_vals[order]
                                int_s = int_vals[order]
                            else:
                                mz_s = mz_vals
                                int_s = int_vals
                        except Exception:
                            order = np.argsort(mz_vals)
                            mz_s = mz_vals[order]
                            int_s = int_vals[order]

                        t = float(target_mz)
                        i = int(np.searchsorted(mz_s, t))
                        cand: List[int] = []
                        if 0 <= i < mz_s.size:
                            cand.append(i)
                        if i - 1 >= 0:
                            cand.append(i - 1)
                        if i + 1 < mz_s.size:
                            cand.append(i + 1)
                        if not cand:
                            continue

                        # Choose the closest peak that also passes the intensity threshold
                        eligible: List[int] = []
                        for j in cand:
                            try:
                                if float(int_s[j]) >= float(min_int):
                                    eligible.append(int(j))
                            except Exception:
                                continue
                        if not eligible:
                            continue

                        best_j = min(eligible, key=lambda j: abs(float(mz_s[j]) - t))
                        err = abs(float(mz_s[best_j]) - t)
                        if float(err) <= float(tol_da):
                            matches.append(int(idx))
                            peak_info[int(idx)] = (float(mz_s[best_j]), float(int_s[best_j]), float(err))

                def done() -> None:
                    self._hide_busy()
                    self._mz_find_cache[key] = sorted(set(int(i) for i in matches))
                    self._mz_find_peak_cache[key] = dict(peak_info)
                    if not self._mz_find_cache[key]:
                        messagebox.showinfo("Find m/z", "No scan contains m/z within tolerance.", parent=self)
                        return
                    self._mz_find_finish_jump(target_mz, tol_value, unit, mode, tol_da, key, self._mz_find_cache[key], self._mz_find_peak_cache[key], same_params=same_params)

                self.after(0, done)
            except Exception as exc:
                self.after(0, lambda: (self._hide_busy(), messagebox.showerror("Error", f"Find m/z failed:\n{exc}", parent=self)))

        threading.Thread(target=worker, daemon=True).start()

    def _mz_find_finish_jump(
        self,
        target_mz: float,
        tol_value: float,
        unit: str,
        mode: str,
        tol_da: float,
        key: Tuple[str, str, float, float, str, float],
        matches: List[int],
        peak_cache: Dict[int, Tuple[float, float, float]],
        *,
        same_params: bool,
    ) -> None:
        if not matches:
            messagebox.showinfo("Find m/z", "No scan contains m/z within tolerance.", parent=self)
            return

        cur_idx = int(self._current_scan_index) if self._current_scan_index is not None else 0
        meta_list = list(self._filtered_meta)

        target_idx = self._mz_find_pick_target_index(
            list(matches),
            meta_list=meta_list,
            cur_idx=cur_idx,
            mode=str(mode),
            same_params=bool(same_params),
            tol_da=float(tol_da),
            target_mz=float(target_mz),
            peak_cache=peak_cache,
        )
        if target_idx is None:
            messagebox.showinfo("Find m/z", "No scan contains m/z within tolerance.", parent=self)
            return

        self._show_spectrum_for_index(int(target_idx))
        self._last_mz_find_last_hit_idx = int(target_idx)

        # Optional highlight + status
        info = peak_cache.get(int(target_idx))
        if info is not None:
            peak_mz, peak_int, err_da = info
            self._draw_mz_find_highlight(float(peak_mz))
            try:
                meta = self._filtered_meta[int(target_idx)]
                unit_s = (unit or "ppm").strip()
                tol_s = f"{float(tol_value):g} {unit_s}"
                min_s = ""
                try:
                    if float(key[5]) > 0:
                        min_s = f"; I>={float(key[5]):g}"
                except Exception:
                    min_s = ""
                self._set_status(
                    f"Found m/z {float(target_mz):.4f} ± {tol_s}{min_s} at RT={float(meta.rt_min):.4f} min (closest peak {float(peak_mz):.4f}, intensity {float(peak_int):.3g}, err {float(err_da):.4g} Da)"
                )
            except Exception:
                pass

    def _export_spectrum_csv(self) -> None:
        if self._current_spectrum_meta is None or self._current_spectrum_mz is None or self._current_spectrum_int is None:
            messagebox.showerror("No spectrum", "Load a spectrum first (click TIC/UV).", parent=self)
            return
        meta = self._current_spectrum_meta
        stem = "spectrum_export"
        if self.mzml_path is not None:
            stem = f"{self.mzml_path.stem}_rt{meta.rt_min:.3f}_ms1"

        path = filedialog.asksaveasfilename(
            title="Export spectrum",
            defaultextension=".csv",
            initialfile=f"{stem}.csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        mz = np.asarray(self._current_spectrum_mz, dtype=float)
        inten = np.asarray(self._current_spectrum_int, dtype=float)
        labels_by_key = self._collect_labels_for_export(mz, inten)

        labels_col: List[str] = []
        for mz_v in mz.tolist():
            k = self._mz_key(float(mz_v))
            items = labels_by_key.get(float(k), [])
            if items:
                labels_col.append(" | ".join([f"{kind}:{txt}" for kind, txt in items]))
            else:
                labels_col.append("")

        df = pd.DataFrame(
            {
                "mz": mz,
                "intensity": inten,
                "labels": labels_col,
                "rt_min": float(meta.rt_min),
                "spectrum_id": str(meta.spectrum_id),
                "polarity": str(meta.polarity or ""),
            }
        )

        try:
            df.to_csv(path, index=False)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to write CSV:\n{exc}", parent=self)
            return
        messagebox.showinfo("Exported", f"Saved:\n{path}", parent=self)

    def _export_overlay_tic_csv(self) -> None:
        if not self._is_overlay_active():
            messagebox.showinfo("Overlay", "Overlay mode is not active.", parent=self)
            return
        ids = self._overlay_dataset_ids()
        if len(ids) < 2:
            messagebox.showinfo("Overlay", "Select at least two datasets for overlay.", parent=self)
            return

        pol = str(self.polarity_var.get())
        rts_by_sid: Dict[str, np.ndarray] = {}
        tics_by_sid: Dict[str, np.ndarray] = {}
        for sid in ids:
            _meta, rts, tics = self._overlay_meta_for_session(str(sid), pol)
            rts_by_sid[str(sid)] = np.asarray(rts, dtype=float)
            tics_by_sid[str(sid)] = np.asarray(tics, dtype=float)

        if not any(r.size for r in rts_by_sid.values()):
            messagebox.showinfo("Overlay", "No TIC data available for overlay.", parent=self)
            return

        try:
            union_rts = np.unique(np.concatenate([r for r in rts_by_sid.values() if r.size]))
        except Exception:
            union_rts = np.asarray([], dtype=float)

        if union_rts.size == 0:
            messagebox.showinfo("Overlay", "No TIC data available for overlay.", parent=self)
            return

        name_map = self._overlay_display_names(ids)
        data: Dict[str, Any] = {"rt_min": union_rts}
        for sid in ids:
            rts = np.asarray(rts_by_sid.get(str(sid), np.asarray([], dtype=float)), dtype=float)
            tics = np.asarray(tics_by_sid.get(str(sid), np.asarray([], dtype=float)), dtype=float)
            if rts.size == 0 or tics.size == 0:
                y = np.full_like(union_rts, np.nan, dtype=float)
            else:
                try:
                    y = np.interp(union_rts, rts, tics, left=np.nan, right=np.nan)
                except Exception:
                    y = np.full_like(union_rts, np.nan, dtype=float)
            col_name = str(name_map.get(str(sid), str(sid)))
            data[col_name] = y

        path = filedialog.asksaveasfilename(
            title="Export overlay TIC",
            defaultextension=".csv",
            initialfile="overlay_tic.csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to write overlay TIC:\n{exc}", parent=self)
            return
        messagebox.showinfo("Exported", f"Saved:\n{path}", parent=self)

    def _export_overlay_spectra(self) -> None:
        if not self._is_overlay_active():
            messagebox.showinfo("Overlay", "Overlay mode is not active.", parent=self)
            return
        ids = self._overlay_dataset_ids()
        if len(ids) < 2:
            messagebox.showinfo("Overlay", "Select at least two datasets for overlay.", parent=self)
            return

        rt = self._overlay_selected_ms_rt
        if rt is None and self._current_spectrum_meta is not None:
            rt = float(self._current_spectrum_meta.rt_min)
        if rt is None:
            messagebox.showinfo("Overlay", "Select an RT first (click TIC/UV).", parent=self)
            return

        stem = f"overlay_spectra_rt{float(rt):.3f}.csv"
        path = filedialog.asksaveasfilename(
            title="Export overlay spectra",
            defaultextension=".csv",
            initialfile=stem,
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        base = Path(path).with_suffix("")
        name_map = self._overlay_display_names(ids)
        skipped: List[str] = []

        for sid in ids:
            got = self._get_spectrum_for_rt(str(sid), float(rt))
            if got is None:
                skipped.append(str(name_map.get(str(sid), str(sid))))
                continue
            meta, mz_vals, int_vals, _dt = got
            safe_name = self._safe_filename(str(name_map.get(str(sid), str(sid))))
            out_path = str(base) + f"_{safe_name}.csv"
            df = pd.DataFrame(
                {
                    "mz": np.asarray(mz_vals, dtype=float),
                    "intensity": np.asarray(int_vals, dtype=float),
                    "rt_min": float(meta.rt_min),
                    "spectrum_id": str(meta.spectrum_id),
                    "polarity": str(meta.polarity or ""),
                }
            )
            try:
                df.to_csv(out_path, index=False)
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to export {safe_name}:\n{exc}", parent=self)
                return

        msg = f"Saved overlay spectra with base:\n{base}"
        if skipped:
            msg += f"\n\nSkipped (no scan near RT):\n" + "\n".join(skipped)
        messagebox.showinfo("Exported", msg, parent=self)

    def _safe_filename(self, name: str) -> str:
        raw = "".join(ch for ch in str(name) if ch.isalnum() or ch in ("-", "_", " ")).strip()
        return (raw or "dataset").replace(" ", "_")

    def _snapshot_labeling_settings(self) -> LabelingSettings:
        def _get_float(var, default: float) -> float:
            try:
                return float(var.get())
            except Exception:
                return float(default)

        def _get_int(var, default: int) -> int:
            try:
                return int(var.get())
            except Exception:
                return int(default)

        def _get_bool(var, default: bool) -> bool:
            try:
                return bool(var.get())
            except Exception:
                return bool(default)

        def _get_str(var, default: str) -> str:
            try:
                return str(var.get())
            except Exception:
                return str(default)

        return LabelingSettings(
            annotate_peaks=_get_bool(self.annotate_peaks_var, True),
            annotate_min_rel=max(0.0, min(1.0, _get_float(self.annotate_min_rel_var, 0.0))),
            annotate_top_n=max(0, _get_int(self.annotate_top_n_var, 0)),
            poly_enabled=_get_bool(self.poly_enabled_var, False),
            poly_monomers_text=_get_str(self.poly_monomers_text_var, ""),
            poly_charges_text=_get_str(self.poly_charges_var, "1"),
            poly_max_dp=max(1, min(200, _get_int(self.poly_max_dp_var, 12))),
            poly_bond_delta=_get_float(self.poly_bond_delta_var, 0.0),
            poly_extra_delta=_get_float(self.poly_extra_delta_var, 0.0),
            poly_adduct_mass=_get_float(self.poly_adduct_mass_var, 1.007276),
            poly_decarb_enabled=_get_bool(self.poly_decarb_enabled_var, False),
            poly_oxid_enabled=_get_bool(self.poly_oxid_enabled_var, False),
            poly_cluster_enabled=_get_bool(self.poly_cluster_enabled_var, False),
            poly_cluster_adduct_mass=_get_float(self.poly_cluster_adduct_mass_var, -1.007276),
            poly_adduct_na=_get_bool(self.poly_adduct_na_var, False),
            poly_adduct_k=_get_bool(self.poly_adduct_k_var, False),
            poly_adduct_cl=_get_bool(self.poly_adduct_cl_var, False),
            poly_adduct_formate=_get_bool(self.poly_adduct_formate_var, False),
            poly_adduct_acetate=_get_bool(self.poly_adduct_acetate_var, False),
            poly_tol_value=_get_float(self.poly_tol_value_var, 0.02),
            poly_tol_unit=_get_str(self.poly_tol_unit_var, "Da"),
            poly_min_rel_int=max(0.0, min(1.0, _get_float(self.poly_min_rel_int_var, 0.01))),
        )

    def _collect_label_occurrences_for_spectrum(
        self,
        spectrum_id: str,
        meta: Optional[SpectrumMeta],
        mz_vals: np.ndarray,
        int_vals: np.ndarray,
        *,
        custom_labels_by_spectrum: Optional[Dict[str, List[CustomLabel]]] = None,
        spec_label_overrides: Optional[Dict[str, Dict[Tuple[str, float], Optional[str]]]] = None,
        settings: Optional[LabelingSettings] = None,
    ) -> List[Tuple[float, str, str, float, Optional[float]]]:
        """Return [(mz_key, kind, text, mz_actual, intensity_or_None), ...] as the GUI would label."""
        sid = str(spectrum_id) if spectrum_id else "__no_spectrum__"
        overrides_all = spec_label_overrides if spec_label_overrides is not None else self._spec_label_overrides
        overrides = overrides_all.get(sid, {})
        custom_all = custom_labels_by_spectrum if custom_labels_by_spectrum is not None else self._custom_labels_by_spectrum
        custom_items = custom_all.get(sid, [])

        st = settings if settings is not None else self._snapshot_labeling_settings()
        out: List[Tuple[float, str, str, float, Optional[float]]] = []
        seen: set[Tuple[float, str, str]] = set()

        def add(mz_v: float, kind: str, text: str, intensity: Optional[float]) -> None:
            mz_k = float(self._mz_key(float(mz_v)))
            kind = str(kind)
            text = (text or "").strip()
            if not text:
                return
            key = (mz_k, kind, text)
            if key in seen:
                return
            seen.add(key)
            out.append((mz_k, kind, text, float(mz_v), None if intensity is None else float(intensity)))

        # Auto labels
        if bool(st.annotate_peaks):
            if mz_vals.size and int_vals.size:
                max_int = float(np.max(int_vals)) if int_vals.size else 0.0
                if max_int > 0:
                    min_rel = float(st.annotate_min_rel)
                    mask = int_vals >= (min_rel * max_int)
                    mz_f = mz_vals[mask]
                    int_f = int_vals[mask]
                    if mz_f.size:
                        top_n = int(st.annotate_top_n)
                        if top_n > 0 and int_f.size > top_n:
                            order = np.argsort(int_f)[::-1][:top_n]
                            mz_f = mz_f[order]
                            int_f = int_f[order]
                        for mz_v, inten_v in zip(mz_f.tolist(), int_f.tolist()):
                            mz_key = float(self._mz_key(float(mz_v)))
                            ov_key = ("auto", float(mz_key))
                            ov = overrides.get(ov_key)
                            if ov is None and ov_key in overrides:
                                continue
                            text = str(ov) if isinstance(ov, str) else f"{float(mz_v):.4f}"
                            add(float(mz_v), "auto", text, float(inten_v))

        # Custom labels
        if custom_items and mz_vals.size and int_vals.size:
            for item in custom_items:
                if not (item.label or "").strip():
                    continue
                mz_target = float(item.mz)
                if bool(item.snap_to_nearest_peak):
                    found = self._find_nearest_peak(mz_vals, int_vals, mz_target)
                    if found is None:
                        continue
                    mz_use, inten_use = found
                    add(float(mz_use), "custom", str(item.label).strip(), float(inten_use))
                else:
                    add(float(mz_target), "custom", str(item.label).strip(), None)

        # Polymer labels
        if bool(st.poly_enabled):
            order_mz = np.argsort(mz_vals)
            mz_s = mz_vals[order_mz]
            int_s = int_vals[order_mz]
            pol = (meta.polarity if meta is not None else None)
            best_by_peak = self._compute_polymer_best_by_peak_sorted(mz_s, int_s, polarity=pol, settings=st)
            kind_order = ["poly", "ox", "decarb", "oxdecarb", "2m"]
            for _pi, kinds in best_by_peak.items():
                for knd in kind_order:
                    if knd not in kinds:
                        continue
                    _err, label, mz_act, inten_act = kinds[knd]
                    mz_key = float(self._mz_key(float(mz_act)))
                    ov_key = (str(knd), float(mz_key))
                    ov = overrides.get(ov_key)
                    if ov is None and ov_key in overrides:
                        continue
                    text = str(ov) if isinstance(ov, str) else str(label)
                    add(float(mz_act), str(knd), text, float(inten_act))

        order_map = {"poly": 0, "ox": 1, "decarb": 2, "oxdecarb": 3, "2m": 4, "custom": 5, "auto": 6}
        out.sort(key=lambda it: (float(it[0]), order_map.get(it[1], 99), it[1], it[2]))
        return out

    def _collect_labels_for_spectrum(
        self,
        spectrum_id: str,
        meta: Optional[SpectrumMeta],
        mz_vals: np.ndarray,
        int_vals: np.ndarray,
        *,
        custom_labels_by_spectrum: Optional[Dict[str, List[CustomLabel]]] = None,
        spec_label_overrides: Optional[Dict[str, Dict[Tuple[str, float], Optional[str]]]] = None,
        settings: Optional[LabelingSettings] = None,
    ) -> Dict[float, List[Tuple[str, str]]]:
        """Return mapping mz_key -> [(kind, label_text), ...] reflecting what the GUI would label."""
        occ = self._collect_label_occurrences_for_spectrum(
            spectrum_id,
            meta,
            mz_vals,
            int_vals,
            custom_labels_by_spectrum=custom_labels_by_spectrum,
            spec_label_overrides=spec_label_overrides,
            settings=settings,
        )
        out: Dict[float, List[Tuple[str, str]]] = {}
        order_map = {"poly": 0, "ox": 1, "decarb": 2, "oxdecarb": 3, "2m": 4, "custom": 5, "auto": 6}
        for mz_key, kind, text, _mz_act, _inten in occ:
            out.setdefault(float(mz_key), []).append((str(kind), str(text)))
        for k in list(out.keys()):
            out[k].sort(key=lambda it: (order_map.get(it[0], 99), it[0], it[1]))
        return out

    def _collect_labels_for_export(self, mz_vals: np.ndarray, int_vals: np.ndarray) -> Dict[float, List[Tuple[str, str]]]:
        meta = self._current_spectrum_meta
        spectrum_id = str(meta.spectrum_id) if meta is not None else "__no_spectrum__"
        return self._collect_labels_for_spectrum(spectrum_id, meta, mz_vals, int_vals)

    def _export_all_labels_xlsx(self) -> None:
        if self._index is None or not self._filtered_meta:
            messagebox.showinfo("Export", "Open an mzML file first.", parent=self)
            return
        if self.mzml_path is None:
            messagebox.showinfo("Export", "No active mzML session.", parent=self)
            return

        mzml_path = Path(self.mzml_path)
        stem = mzml_path.stem
        path = filedialog.asksaveasfilename(
            title="Export all labels",
            defaultextension=".xlsx",
            initialfile=f"{stem}_all_labels.xlsx",
            filetypes=[("Excel Workbook", "*.xlsx"), ("All files", "*.*")],
        )
        if not path:
            return

        # Snapshot everything needed for worker thread (no Tk calls in worker)
        meta_list = list(self._filtered_meta)
        try:
            custom_snapshot = copy.deepcopy(self._custom_labels_by_spectrum)
        except Exception:
            custom_snapshot = dict(self._custom_labels_by_spectrum)
        try:
            overrides_snapshot = copy.deepcopy(self._spec_label_overrides)
        except Exception:
            overrides_snapshot = dict(self._spec_label_overrides)
        settings = self._snapshot_labeling_settings()

        out_path = str(path)

        self._show_busy("Exporting all labels…")

        def worker() -> None:
            try:
                rows: List[Dict[str, Any]] = []

                def get_spectrum_by_id(reader: mzml.MzML, spectrum_id: str) -> Dict[str, Any]:
                    try:
                        return reader.get_by_id(spectrum_id)
                    except Exception:
                        return reader[spectrum_id]

                with mzml.MzML(str(mzml_path)) as reader:
                    for meta in meta_list:
                        spectrum_id = str(meta.spectrum_id)
                        spectrum = get_spectrum_by_id(reader, spectrum_id)
                        mz_array = spectrum.get("m/z array")
                        int_array = spectrum.get("intensity array")
                        if mz_array is None or int_array is None:
                            continue

                        mz_vals = np.asarray(mz_array, dtype=float)
                        int_vals = np.asarray(int_array, dtype=float)
                        if mz_vals.size == 0 or int_vals.size == 0:
                            continue

                        mask = np.isfinite(mz_vals) & np.isfinite(int_vals)
                        if not np.any(mask):
                            continue
                        mz_vals = mz_vals[mask]
                        int_vals = int_vals[mask]

                        occ = self._collect_label_occurrences_for_spectrum(
                            spectrum_id,
                            meta,
                            mz_vals,
                            int_vals,
                            custom_labels_by_spectrum=custom_snapshot,
                            spec_label_overrides=overrides_snapshot,
                            settings=settings,
                        )
                        if not occ:
                            continue

                        for mz_key, kind, text, mz_act, inten in occ:
                            rows.append(
                                {
                                    "file_name": mzml_path.name,
                                    "spectrum_id": spectrum_id,
                                    "rt_min": float(meta.rt_min),
                                    "polarity": str(meta.polarity or ""),
                                    "label_kind": str(kind),
                                    "label_text": str(text),
                                    "mz": float(mz_act),
                                    "intensity": (None if inten is None else float(inten)),
                                    "mz_key": float(mz_key),
                                }
                            )

                if not rows:
                    self.after(0, lambda: (self._hide_busy(), messagebox.showinfo("Export", "No labels found to export.", parent=self)))
                    return

                df = pd.DataFrame(rows)
                try:
                    df.sort_values(by=["rt_min", "mz"], inplace=True, ascending=[True, True])
                except Exception:
                    pass

                # Write Excel (requires openpyxl)
                try:
                    import openpyxl  # noqa: F401
                    from openpyxl.utils import get_column_letter
                except Exception as exc:
                    raise RuntimeError(f"openpyxl is required for Excel export. Install it and retry.\n\n{exc}")

                with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                    sheet = "labels"
                    df.to_excel(writer, index=False, sheet_name=sheet)
                    try:
                        ws = writer.sheets[sheet]
                        ws.freeze_panes = "A2"
                        ws.auto_filter.ref = ws.dimensions

                        # Best-effort column autosize (cap work for huge sheets)
                        max_rows = min(int(len(df)), 2000)
                        for i, col in enumerate(df.columns, start=1):
                            values = [str(col)]
                            try:
                                values.extend(["" if v is None else str(v) for v in df[col].iloc[:max_rows].tolist()])
                            except Exception:
                                pass
                            width = min(60, max(10, max((len(s) for s in values), default=10)))
                            ws.column_dimensions[get_column_letter(i)].width = float(width)
                    except Exception:
                        pass

                self.after(
                    0,
                    lambda: (
                        self._hide_busy(),
                        messagebox.showinfo("Exported", f"Saved:\n{out_path}\n\nRows: {len(df)}", parent=self),
                    ),
                )
            except Exception as exc:
                self.after(0, lambda: (self._hide_busy(), messagebox.showerror("Error", f"Failed to export Excel:\n{exc}", parent=self)))

        threading.Thread(target=worker, daemon=True).start()

    def _parse_optional_float(self, raw: str) -> Optional[float]:
        raw = (raw or "").strip()
        if not raw:
            return None
        return float(raw)

    def _trace_global_max(self, y: Optional[np.ndarray]) -> float:
        if y is None or y.size == 0:
            return 0.0
        try:
            return float(np.max(y))
        except Exception:
            return 0.0

    def _snap_index_to_local_max(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_clicked: float,
        *,
        max_rt_delta_min: float = 0.35,
        max_steps: int = 500,
    ) -> Optional[int]:
        """Return index of a nearby local maximum (peak apex) near x_clicked."""
        if x.size == 0 or y.size == 0 or x.size != y.size:
            return None

        i0 = int(np.argmin(np.abs(x - float(x_clicked))))
        i = i0
        steps = 0

        while steps < max_steps:
            steps += 1
            yi = float(y[i])
            left = float(y[i - 1]) if i > 0 else float("-inf")
            right = float(y[i + 1]) if i < (y.size - 1) else float("-inf")

            # Stop at local maximum (or plateau)
            if left <= yi and right <= yi:
                break

            # Move to the neighbor with higher intensity
            next_i = (i - 1) if left >= right else (i + 1)

            # Don't drift too far from where the user clicked
            if abs(float(x[next_i]) - float(x[i0])) > float(max_rt_delta_min):
                break
            i = next_i

        return int(i)

    def _accept_apex(self, apex_y: float, y_global_max: float) -> bool:
        # Avoid selecting baseline/noise.
        if y_global_max <= 0:
            return False
        return float(apex_y) >= (0.01 * float(y_global_max))

    def _apply_plot_style_to_axes(self, fig: Optional[Figure], axes: Sequence[Any]) -> None:
        bg = (self._matplotlib_bg_var.get() or "").strip()
        try:
            if not bg:
                bg = "#f5f5f5"
            mcolors.to_rgba(bg)
        except Exception:
            bg = "#f5f5f5"

        # Figure background
        try:
            if fig is not None:
                fig.patch.set_facecolor(bg)
        except Exception:
            pass

        title_fs = int(self.title_fontsize_var.get())
        label_fs = int(self.label_fontsize_var.get())
        tick_fs = int(self.tick_fontsize_var.get())

        for ax in [a for a in axes if a is not None]:
            try:
                ax.set_facecolor("#ffffff")
            except Exception:
                pass

            try:
                ax.title.set_fontsize(title_fs)
                ax.xaxis.label.set_fontsize(label_fs)
                ax.yaxis.label.set_fontsize(label_fs)
                ax.tick_params(axis="both", which="major", labelsize=tick_fs)
            except Exception:
                pass

            # Brand colors for text/spines
            try:
                ax.title.set_color(PRIMARY_TEAL)
                ax.xaxis.label.set_color(TEXT_DARK)
                ax.yaxis.label.set_color(TEXT_DARK)
                ax.tick_params(axis="both", colors=TEXT_MUTED)
                for spine in ax.spines.values():
                    spine.set_color(DIVIDER)
            except Exception:
                pass

            # Linear scale with scientific-notation tick labels (powers of 10)
            try:
                ax.set_yscale("linear")
                sci = ScalarFormatter(useMathText=True)
                sci.set_scientific(True)
                sci.set_powerlimits((0, 0))
                sci.set_useOffset(False)
                ax.yaxis.set_major_formatter(sci)
            except Exception:
                pass

            try:
                ax.grid(True, which="major", color=DIVIDER, alpha=0.35)
            except Exception:
                pass

    def _apply_plot_style(self) -> None:
        if self._ax_tic is None or self._ax_spec is None:
            return
        axes = [ax for ax in [self._ax_tic, self._ax_spec, self._ax_uv] if ax is not None]
        self._apply_plot_style_to_axes(self._fig, axes)

        # Ensure watermark stays behind plots
        try:
            self._draw_mpl_watermark()
        except Exception:
            pass

        # TIC limits
        if self._ax_tic is not None:
            xmin = self._parse_optional_float(self.tic_xlim_min_var.get())
            xmax = self._parse_optional_float(self.tic_xlim_max_var.get())
            ymin = self._parse_optional_float(self.tic_ylim_min_var.get())
            ymax = self._parse_optional_float(self.tic_ylim_max_var.get())
            if xmin is not None or xmax is not None:
                self._ax_tic.set_xlim(left=xmin, right=xmax)
            else:
                self._ax_tic.autoscale(enable=True, axis="x", tight=False)

            if ymin is not None or ymax is not None:
                self._ax_tic.set_ylim(bottom=ymin, top=ymax)
            else:
                # Auto: 0..max peak
                top = None
                if self._filtered_tics is not None and self._filtered_tics.size:
                    top = float(np.max(self._filtered_tics))
                # Leave headroom so annotations aren't clipped
                self._ax_tic.set_ylim(bottom=0.0, top=((top if top and top > 0 else 1.0) * 1.10))

        # Spectrum limits
        if self._ax_spec is not None:
            xmin = self._parse_optional_float(self.spec_xlim_min_var.get())
            xmax = self._parse_optional_float(self.spec_xlim_max_var.get())
            ymin = self._parse_optional_float(self.spec_ylim_min_var.get())
            ymax = self._parse_optional_float(self.spec_ylim_max_var.get())
            if xmin is not None or xmax is not None:
                self._ax_spec.set_xlim(left=xmin, right=xmax)
            else:
                self._ax_spec.autoscale(enable=True, axis="x", tight=False)

            if ymin is not None or ymax is not None:
                self._ax_spec.set_ylim(bottom=ymin, top=ymax)
            else:
                # Auto: 0..max peak
                top = None
                if self._current_spectrum_int is not None and self._current_spectrum_int.size:
                    top = float(np.max(self._current_spectrum_int))
                # Leave headroom so annotations aren't clipped
                self._ax_spec.set_ylim(bottom=0.0, top=((top if top and top > 0 else 1.0) * 1.35))

        # UV limits
        if self._ax_uv is not None:
            xmin = self._parse_optional_float(self.uv_xlim_min_var.get())
            xmax = self._parse_optional_float(self.uv_xlim_max_var.get())
            ymin = self._parse_optional_float(self.uv_ylim_min_var.get())
            ymax = self._parse_optional_float(self.uv_ylim_max_var.get())
            if xmin is not None or xmax is not None:
                self._ax_uv.set_xlim(left=xmin, right=xmax)
            else:
                self._ax_uv.autoscale(enable=True, axis="x", tight=False)

            if ymin is not None or ymax is not None:
                self._ax_uv.set_ylim(bottom=ymin, top=ymax)
            else:
                _x, y = self._active_uv_xy()
                if y is None or y.size == 0:
                    self._ax_uv.autoscale(enable=True, axis="y", tight=False)
                else:
                    y_min = float(np.min(y))
                    y_max = float(np.max(y))
                    if y_max == y_min:
                        y_max = y_min + 1.0
                    headroom = 0.08 * (y_max - y_min)
                    self._ax_uv.set_ylim(bottom=(y_min - headroom), top=(y_max + headroom))

    def _apply_matplotlib_bg(self, fig: Optional[Figure], canvas: Optional[Any]) -> None:
        if fig is None:
            return
        bg = (self._matplotlib_bg_var.get() or "").strip()
        try:
            if not bg:
                bg = "#f5f5f5"
            mcolors.to_rgba(bg)
        except Exception:
            bg = "#f5f5f5"
        try:
            fig.patch.set_facecolor(bg)
        except Exception:
            pass
        try:
            if canvas is not None:
                canvas.draw_idle()
        except Exception:
            pass

    def _apply_matplotlib_bg_current(self) -> None:
        tab = self._active_module_name().strip().lower()
        if tab == "lcms":
            self._apply_matplotlib_bg(self._fig, self._canvas)
            self._apply_plot_style()
            return
        if tab == "ftir":
            view = getattr(self, "_ftir_view", None)
            fig = getattr(view, "_fig", None) if view is not None else None
            canvas = getattr(view, "_canvas", None) if view is not None else None
            self._apply_matplotlib_bg(fig, canvas)
            try:
                if view is not None:
                    view._redraw()
            except Exception:
                pass
            return
        if tab == "plate reader":
            view = getattr(self, "_plate_reader_view", None)
            fig = getattr(view, "_fig", None) if view is not None else None
            canvas = getattr(view, "_canvas", None) if view is not None else None
            self._apply_matplotlib_bg(fig, canvas)
            return
        if tab == "data studio":
            view = getattr(self, "_data_studio_view", None)
            fig = getattr(view, "_fig", None) if view is not None else None
            canvas = getattr(view, "_canvas", None) if view is not None else None
            self._apply_matplotlib_bg(fig, canvas)
            return

    def _open_graph_settings(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Graph Settings")
        dlg.resizable(False, False)
        dlg.transient(self)

        pad = 10
        frm = ttk.Frame(dlg, padding=pad)
        frm.grid(row=0, column=0)

        labels = ttk.LabelFrame(frm, text="Titles & axis labels", padding=pad)
        labels.grid(row=0, column=0, sticky="ew")

        ttk.Label(labels, text="TIC title").grid(row=0, column=0, sticky="w")
        tic_title_ent = ttk.Entry(labels, textvariable=self.tic_title_var, width=34)
        tic_title_ent.grid(row=0, column=1, padx=(8, 0))
        ttk.Label(labels, text="X").grid(row=0, column=2, padx=(10, 0), sticky="e")
        tic_x_ent = ttk.Entry(labels, textvariable=self.tic_xlabel_var, width=22)
        tic_x_ent.grid(row=0, column=3, padx=(8, 0))
        ttk.Label(labels, text="Y").grid(row=0, column=4, padx=(10, 0), sticky="e")
        tic_y_ent = ttk.Entry(labels, textvariable=self.tic_ylabel_var, width=16)
        tic_y_ent.grid(row=0, column=5, padx=(8, 0))

        ttk.Label(labels, text="Spectrum title").grid(row=1, column=0, sticky="w", pady=(6, 0))
        spec_title_ent = ttk.Entry(labels, textvariable=self.spec_title_var, width=34)
        spec_title_ent.grid(row=1, column=1, padx=(8, 0), pady=(6, 0))
        ttk.Label(labels, text="X").grid(row=1, column=2, padx=(10, 0), sticky="e", pady=(6, 0))
        spec_x_ent = ttk.Entry(labels, textvariable=self.spec_xlabel_var, width=22)
        spec_x_ent.grid(row=1, column=3, padx=(8, 0), pady=(6, 0))
        ttk.Label(labels, text="Y").grid(row=1, column=4, padx=(10, 0), sticky="e", pady=(6, 0))
        spec_y_ent = ttk.Entry(labels, textvariable=self.spec_ylabel_var, width=16)
        spec_y_ent.grid(row=1, column=5, padx=(8, 0), pady=(6, 0))

        ttk.Label(labels, text="UV title").grid(row=2, column=0, sticky="w", pady=(6, 0))
        uv_title_ent = ttk.Entry(labels, textvariable=self.uv_title_var, width=34)
        uv_title_ent.grid(row=2, column=1, padx=(8, 0), pady=(6, 0))
        ttk.Label(labels, text="X").grid(row=2, column=2, padx=(10, 0), sticky="e", pady=(6, 0))
        uv_x_ent = ttk.Entry(labels, textvariable=self.uv_xlabel_var, width=22)
        uv_x_ent.grid(row=2, column=3, padx=(8, 0), pady=(6, 0))
        ttk.Label(labels, text="Y").grid(row=2, column=4, padx=(10, 0), sticky="e", pady=(6, 0))
        uv_y_ent = ttk.Entry(labels, textvariable=self.uv_ylabel_var, width=16)
        uv_y_ent.grid(row=2, column=5, padx=(8, 0), pady=(6, 0))

        fonts = ttk.LabelFrame(frm, text="Fonts", padding=pad)
        fonts.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        ttk.Label(fonts, text="Title font size").grid(row=0, column=0, sticky="w")
        title_fs = ttk.Spinbox(fonts, from_=6, to=30, textvariable=self.title_fontsize_var, width=6)
        title_fs.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(fonts, text="Axis label font size").grid(row=1, column=0, sticky="w", pady=(6, 0))
        label_fs = ttk.Spinbox(fonts, from_=6, to=30, textvariable=self.label_fontsize_var, width=6)
        label_fs.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        ttk.Label(fonts, text="Tick font size").grid(row=2, column=0, sticky="w", pady=(6, 0))
        tick_fs = ttk.Spinbox(fonts, from_=6, to=30, textvariable=self.tick_fontsize_var, width=6)
        tick_fs.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        limits = ttk.LabelFrame(frm, text="Axis ranges (leave blank = auto)", padding=pad)
        limits.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        # TIC limits
        ttk.Label(limits, text="TIC x(min,max)").grid(row=0, column=0, sticky="w")
        tic_xmin = ttk.Entry(limits, textvariable=self.tic_xlim_min_var, width=10)
        tic_xmin.grid(row=0, column=1, padx=(8, 4))
        tic_xmax = ttk.Entry(limits, textvariable=self.tic_xlim_max_var, width=10)
        tic_xmax.grid(row=0, column=2)

        ttk.Label(limits, text="TIC y(min,max)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tic_ymin = ttk.Entry(limits, textvariable=self.tic_ylim_min_var, width=10)
        tic_ymin.grid(row=1, column=1, padx=(8, 4), pady=(6, 0))
        tic_ymax = ttk.Entry(limits, textvariable=self.tic_ylim_max_var, width=10)
        tic_ymax.grid(row=1, column=2, pady=(6, 0))

        # Spectrum limits
        ttk.Label(limits, text="Spectrum x(min,max)").grid(row=2, column=0, sticky="w", pady=(10, 0))
        spec_xmin = ttk.Entry(limits, textvariable=self.spec_xlim_min_var, width=10)
        spec_xmin.grid(row=2, column=1, padx=(8, 4), pady=(10, 0))
        spec_xmax = ttk.Entry(limits, textvariable=self.spec_xlim_max_var, width=10)
        spec_xmax.grid(row=2, column=2, pady=(10, 0))

        ttk.Label(limits, text="Spectrum y(min,max)").grid(row=3, column=0, sticky="w", pady=(6, 0))
        spec_ymin = ttk.Entry(limits, textvariable=self.spec_ylim_min_var, width=10)
        spec_ymin.grid(row=3, column=1, padx=(8, 4), pady=(6, 0))
        spec_ymax = ttk.Entry(limits, textvariable=self.spec_ylim_max_var, width=10)
        spec_ymax.grid(row=3, column=2, pady=(6, 0))

        # UV limits
        ttk.Label(limits, text="UV x(min,max)").grid(row=4, column=0, sticky="w", pady=(10, 0))
        uv_xmin = ttk.Entry(limits, textvariable=self.uv_xlim_min_var, width=10)
        uv_xmin.grid(row=4, column=1, padx=(8, 4), pady=(10, 0))
        uv_xmax = ttk.Entry(limits, textvariable=self.uv_xlim_max_var, width=10)
        uv_xmax.grid(row=4, column=2, pady=(10, 0))

        ttk.Label(limits, text="UV y(min,max)").grid(row=5, column=0, sticky="w", pady=(6, 0))
        uv_ymin = ttk.Entry(limits, textvariable=self.uv_ylim_min_var, width=10)
        uv_ymin.grid(row=5, column=1, padx=(8, 4), pady=(6, 0))
        uv_ymax = ttk.Entry(limits, textvariable=self.uv_ylim_max_var, width=10)
        uv_ymax.grid(row=5, column=2, pady=(6, 0))

        def on_reset() -> None:
            self.title_fontsize_var.set(12)
            self.label_fontsize_var.set(10)
            self.tick_fontsize_var.set(9)

            self.tic_title_var.set("TIC (MS1)")
            self.tic_xlabel_var.set("Retention time (min)")
            self.tic_ylabel_var.set("Intensity")

            self.spec_title_var.set("Spectrum (MS1)")
            self.spec_xlabel_var.set("m/z")
            self.spec_ylabel_var.set("Intensity")

            self.uv_title_var.set("UV chromatogram")
            self.uv_xlabel_var.set("Retention time (min)")
            self.uv_ylabel_var.set("Signal")

            for v in [
                self.tic_xlim_min_var,
                self.tic_xlim_max_var,
                self.tic_ylim_min_var,
                self.tic_ylim_max_var,
                self.spec_xlim_min_var,
                self.spec_xlim_max_var,
                self.spec_ylim_min_var,
                self.spec_ylim_max_var,
                self.uv_xlim_min_var,
                self.uv_xlim_max_var,
                self.uv_ylim_min_var,
                self.uv_ylim_max_var,
            ]:
                v.set("")

            self._redraw_all()

        def on_apply() -> None:
            try:
                # Validate numeric fields (optional)
                for raw in [
                    self.tic_xlim_min_var.get(),
                    self.tic_xlim_max_var.get(),
                    self.tic_ylim_min_var.get(),
                    self.tic_ylim_max_var.get(),
                    self.spec_xlim_min_var.get(),
                    self.spec_xlim_max_var.get(),
                    self.spec_ylim_min_var.get(),
                    self.spec_ylim_max_var.get(),
                    self.uv_xlim_min_var.get(),
                    self.uv_xlim_max_var.get(),
                    self.uv_ylim_min_var.get(),
                    self.uv_ylim_max_var.get(),
                ]:
                    self._parse_optional_float(raw)

            except Exception:
                messagebox.showerror(
                    "Invalid value",
                    "Axis limits must be numbers (or blank).",
                    parent=dlg,
                )
                return

            self._redraw_all()

        buttons = ttk.Frame(frm)
        buttons.grid(row=3, column=0, sticky="e", pady=(10, 0))
        apply_btn = ttk.Button(buttons, text="Apply", command=on_apply)
        apply_btn.grid(row=0, column=0, padx=(0, 8))
        reset_btn = ttk.Button(buttons, text="Reset", command=on_reset)
        reset_btn.grid(row=0, column=1, padx=(0, 8))
        close_btn = ttk.Button(buttons, text="Close", command=dlg.destroy)
        close_btn.grid(row=0, column=2)

        # Tooltips
        ToolTip.attach(tic_title_ent, TOOLTIP_TEXT["gs_tic_title"])
        ToolTip.attach(tic_x_ent, TOOLTIP_TEXT["gs_tic_xlabel"])
        ToolTip.attach(tic_y_ent, TOOLTIP_TEXT["gs_tic_ylabel"])
        ToolTip.attach(spec_title_ent, TOOLTIP_TEXT["gs_spec_title"])
        ToolTip.attach(spec_x_ent, TOOLTIP_TEXT["gs_spec_xlabel"])
        ToolTip.attach(spec_y_ent, TOOLTIP_TEXT["gs_spec_ylabel"])
        ToolTip.attach(uv_title_ent, TOOLTIP_TEXT["gs_uv_title"])
        ToolTip.attach(uv_x_ent, TOOLTIP_TEXT["gs_uv_xlabel"])
        ToolTip.attach(uv_y_ent, TOOLTIP_TEXT["gs_uv_ylabel"])

        ToolTip.attach(title_fs, TOOLTIP_TEXT["gs_title_fs"])
        ToolTip.attach(label_fs, TOOLTIP_TEXT["gs_label_fs"])
        ToolTip.attach(tick_fs, TOOLTIP_TEXT["gs_tick_fs"])

        for wdg in [
            tic_xmin,
            tic_xmax,
            tic_ymin,
            tic_ymax,
            spec_xmin,
            spec_xmax,
            spec_ymin,
            spec_ymax,
            uv_xmin,
            uv_xmax,
            uv_ymin,
            uv_ymax,
        ]:
            ToolTip.attach(wdg, TOOLTIP_TEXT["gs_limits"])

        ToolTip.attach(apply_btn, TOOLTIP_TEXT["gs_apply"])
        ToolTip.attach(reset_btn, TOOLTIP_TEXT["gs_reset"])
        ToolTip.attach(close_btn, TOOLTIP_TEXT["gs_close"])

    def _open_annotation_settings(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Peak Annotations")
        dlg.resizable(False, False)
        dlg.transient(self)

        pad = 10
        frm = ttk.Frame(dlg, padding=pad)
        frm.grid(row=0, column=0)

        cb_annot = ttk.Checkbutton(frm, text="Annotate spectrum peaks with m/z", variable=self.annotate_peaks_var)
        cb_annot.grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(frm, text="Top N peaks:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        topn = ttk.Spinbox(frm, from_=0, to=200, textvariable=self.annotate_top_n_var, width=8)
        topn.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frm, text="Min intensity (fraction of max):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        minrel = ttk.Entry(frm, textvariable=self.annotate_min_rel_var, width=10)
        minrel.grid(row=2, column=1, sticky="w", pady=(8, 0))

        cb_drag = ttk.Checkbutton(frm, text="Enable dragging labels with mouse", variable=self.drag_annotations_var)
        cb_drag.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Separator(frm, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 6))

        cb_uv = ttk.Checkbutton(
            frm,
            text="Transfer top MS peaks to UV labels at selected RT",
            variable=self.uv_label_from_ms_var,
        )
        cb_uv.grid(row=5, column=0, columnspan=2, sticky="w")

        ttk.Label(frm, text="How many peaks (2–3):").grid(row=6, column=0, sticky="w", pady=(8, 0))
        uvn = ttk.Combobox(
            frm,
            textvariable=self.uv_label_from_ms_top_n_var,
            values=[2, 3],
            state="readonly",
            width=6,
        )
        uvn.grid(row=6, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frm, text="Min UV label confidence (%):").grid(row=7, column=0, sticky="w", pady=(8, 0))
        uvconf = ttk.Spinbox(frm, from_=0, to=100, increment=1, textvariable=self.uv_label_min_conf_var, width=8)
        uvconf.grid(row=7, column=1, sticky="w", pady=(8, 0))

        def on_apply() -> None:
            try:
                top_n = int(self.annotate_top_n_var.get())
                if top_n < 0:
                    raise ValueError
                min_rel = float(self.annotate_min_rel_var.get())
                if not (0.0 <= min_rel <= 1.0):
                    raise ValueError
                uv_top_n = int(self.uv_label_from_ms_top_n_var.get())
                if uv_top_n not in (2, 3):
                    raise ValueError
                uv_min_conf = float(self.uv_label_min_conf_var.get())
                if not (0.0 <= uv_min_conf <= 100.0):
                    raise ValueError
            except Exception:
                messagebox.showerror(
                    "Invalid value",
                    "Top N must be >= 0, min intensity must be between 0 and 1, UV transfer peaks must be 2 or 3, and min UV confidence must be 0..100.",
                    parent=dlg,
                )
                return

            self._redraw_spectrum_only()
            # Refresh UV to show/hide stored labels; also add labels for the current spectrum if enabled.
            self._maybe_store_uv_ms_labels_for_current_spectrum(anchor_rt_min=None)
            self._plot_uv()

        buttons = ttk.Frame(frm)
        buttons.grid(row=8, column=0, columnspan=2, sticky="e", pady=(10, 0))
        apply_btn = ttk.Button(buttons, text="Apply", command=on_apply)
        apply_btn.grid(row=0, column=0, padx=(0, 8))
        close_btn = ttk.Button(buttons, text="Close", command=dlg.destroy)
        close_btn.grid(row=0, column=1)

        ToolTip.attach(cb_annot, TOOLTIP_TEXT["annotate_enable"])
        ToolTip.attach(topn, TOOLTIP_TEXT["annotate_top_n"])
        ToolTip.attach(minrel, TOOLTIP_TEXT["annotate_min_rel"])
        ToolTip.attach(cb_drag, TOOLTIP_TEXT["annotate_drag"])
        ToolTip.attach(cb_uv, TOOLTIP_TEXT["uv_transfer_labels"])
        ToolTip.attach(uvn, TOOLTIP_TEXT["uv_transfer_howmany"])
        ToolTip.attach(apply_btn, TOOLTIP_TEXT["pa_apply"])
        ToolTip.attach(close_btn, TOOLTIP_TEXT["pa_close"])

    def _open_custom_labels(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Custom Labels")
        dlg.resizable(False, False)
        dlg.transient(self)

        pad = 10
        frm = ttk.Frame(dlg, padding=pad)
        frm.grid(row=0, column=0)

        ttk.Label(frm, text="Labels are drawn only on the currently displayed spectrum.").grid(
            row=0, column=0, columnspan=4, sticky="w"
        )

        lst = tk.Listbox(frm, height=7, width=62)
        lst.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        def _current_spectrum_key() -> str:
            meta = self._current_spectrum_meta
            if meta is None:
                return "__no_spectrum__"
            return str(meta.spectrum_id)

        def refresh_list() -> None:
            lst.delete(0, tk.END)
            for item in self._custom_labels_by_spectrum.get(_current_spectrum_key(), []):
                snap = "snap" if item.snap_to_nearest_peak else "free"
                lst.insert(tk.END, f"{item.label} @ {item.mz:.4f} ({snap})")

        refresh_list()

        ttk.Label(frm, text="Label").grid(row=2, column=0, sticky="w", pady=(10, 0))
        label_var = tk.StringVar(value="")
        label_ent = ttk.Entry(frm, textvariable=label_var, width=28)
        label_ent.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        ttk.Label(frm, text="m/z").grid(row=2, column=2, sticky="e", padx=(12, 0), pady=(10, 0))
        mz_var = tk.StringVar(value="")
        mz_ent = ttk.Entry(frm, textvariable=mz_var, width=12)
        mz_ent.grid(row=2, column=3, sticky="w", padx=(8, 0), pady=(10, 0))

        snap_var = tk.BooleanVar(value=True)
        snap_cb = ttk.Checkbutton(frm, text="Snap to nearest peak", variable=snap_var)
        snap_cb.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        def on_add() -> None:
            label = (label_var.get() or "").strip()
            if not label:
                messagebox.showerror("Invalid", "Label cannot be empty.", parent=dlg)
                return
            try:
                mz = float((mz_var.get() or "").strip())
            except Exception:
                messagebox.showerror("Invalid", "m/z must be a number.", parent=dlg)
                return
            key = _current_spectrum_key()
            self._custom_labels_by_spectrum.setdefault(key, []).append(
                CustomLabel(label=label, mz=float(mz), snap_to_nearest_peak=bool(snap_var.get()))
            )
            refresh_list()
            self._redraw_spectrum_only()

        def on_remove() -> None:
            sel = lst.curselection()
            if not sel:
                return
            idx = int(sel[0])
            key = _current_spectrum_key()
            items = self._custom_labels_by_spectrum.get(key, [])
            if 0 <= idx < len(items):
                items.pop(idx)
                if not items:
                    self._custom_labels_by_spectrum.pop(key, None)
                refresh_list()
                self._redraw_spectrum_only()

        def on_clear() -> None:
            self._custom_labels_by_spectrum.pop(_current_spectrum_key(), None)
            refresh_list()
            self._redraw_spectrum_only()

        buttons = ttk.Frame(frm)
        buttons.grid(row=4, column=0, columnspan=4, sticky="e", pady=(12, 0))
        add_btn = ttk.Button(buttons, text="Add", command=on_add)
        add_btn.grid(row=0, column=0, padx=(0, 8))
        rm_btn = ttk.Button(buttons, text="Remove", command=on_remove)
        rm_btn.grid(row=0, column=1, padx=(0, 8))
        clear_btn = ttk.Button(buttons, text="Clear all", command=on_clear)
        clear_btn.grid(row=0, column=2, padx=(0, 8))
        close_btn = ttk.Button(buttons, text="Close", command=dlg.destroy)
        close_btn.grid(row=0, column=3)

        ToolTip.attach(lst, TOOLTIP_TEXT["cl_list"])
        ToolTip.attach(label_ent, TOOLTIP_TEXT["cl_label"])
        ToolTip.attach(mz_ent, TOOLTIP_TEXT["cl_mz"])
        ToolTip.attach(snap_cb, TOOLTIP_TEXT["cl_snap"])
        ToolTip.attach(add_btn, TOOLTIP_TEXT["cl_add"])
        ToolTip.attach(rm_btn, TOOLTIP_TEXT["cl_remove"])
        ToolTip.attach(clear_btn, TOOLTIP_TEXT["cl_clear"])
        ToolTip.attach(close_btn, TOOLTIP_TEXT["cl_close"])

    def _open_polymer_match(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Polymer / Reaction Match")
        dlg.resizable(False, False)
        dlg.transient(self)

        pad = 10
        frm = ttk.Frame(dlg, padding=pad)
        frm.grid(row=0, column=0)

        pm_enable = ttk.Checkbutton(frm, text="Enable polymer/reaction matching on spectrum", variable=self.poly_enabled_var)
        pm_enable.grid(row=0, column=0, columnspan=4, sticky="w")

        ttk.Label(frm, text="Monomers (one per line: name,mass or name mass or mass)").grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        txt = tk.Text(frm, width=62, height=5)
        txt.grid(row=2, column=0, columnspan=4, sticky="ew")
        txt.insert("1.0", self.poly_monomers_text_var.get())

        presets = {
            "Dehydration (-H2O)": -18.010565,
            "None (0)": 0.0,
        }
        extra_presets = {
            "None (0)": 0.0,
            "Decarboxylation (-CO2)": -43.989829,
            "Dehydration (-H2O)": -18.010565,
        }

        ttk.Label(frm, text="Per-bond delta (polymerization)").grid(row=3, column=0, sticky="w", pady=(10, 0))
        bond_choice = tk.StringVar(value="Dehydration (-H2O)")
        bond_combo = ttk.Combobox(frm, textvariable=bond_choice, values=list(presets.keys()) + ["Custom"], state="readonly", width=24)
        bond_combo.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(10, 0))
        ttk.Label(frm, text="Custom (Da)").grid(row=3, column=2, sticky="e", padx=(12, 0), pady=(10, 0))
        bond_custom = tk.StringVar(value=str(self.poly_bond_delta_var.get()))
        bond_entry = ttk.Entry(frm, textvariable=bond_custom, width=12)
        bond_entry.grid(row=3, column=3, sticky="w", padx=(8, 0), pady=(10, 0))

        ttk.Label(frm, text="Extra delta (once per chain)").grid(row=4, column=0, sticky="w", pady=(6, 0))
        extra_choice = tk.StringVar(value="None (0)")
        extra_combo = ttk.Combobox(frm, textvariable=extra_choice, values=list(extra_presets.keys()) + ["Custom"], state="readonly", width=24)
        extra_combo.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Label(frm, text="Custom (Da)").grid(row=4, column=2, sticky="e", padx=(12, 0), pady=(6, 0))
        extra_custom = tk.StringVar(value=str(self.poly_extra_delta_var.get()))
        extra_entry = ttk.Entry(frm, textvariable=extra_custom, width=12)
        extra_entry.grid(row=4, column=3, sticky="w", padx=(8, 0), pady=(6, 0))

        pol = self._current_spectrum_meta.polarity if self._current_spectrum_meta is not None else None

        ttk.Label(frm, text="H adduct mass (Da)").grid(row=5, column=0, sticky="w", pady=(10, 0))
        adduct_var = tk.StringVar(value=str(self.poly_adduct_mass_var.get()))
        adduct_ent = ttk.Entry(frm, textvariable=adduct_var, width=12)
        adduct_ent.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        # Extra adducts to match in addition to H
        extra_frame = ttk.Frame(frm)
        extra_frame.grid(row=5, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=(10, 0))
        ttk.Label(extra_frame, text="Also match:").pack(side=tk.LEFT)
        if pol == "negative":
            cb_cl = ttk.Checkbutton(extra_frame, text="+Cl", variable=self.poly_adduct_cl_var)
            cb_cl.pack(side=tk.LEFT, padx=(10, 0))
            cb_formate = ttk.Checkbutton(extra_frame, text="+HCOO", variable=self.poly_adduct_formate_var)
            cb_formate.pack(side=tk.LEFT, padx=(10, 0))
            cb_ac = ttk.Checkbutton(extra_frame, text="+Ac", variable=self.poly_adduct_acetate_var)
            cb_ac.pack(side=tk.LEFT, padx=(10, 0))
        else:
            cb_na = ttk.Checkbutton(extra_frame, text="+Na", variable=self.poly_adduct_na_var)
            cb_na.pack(side=tk.LEFT, padx=(10, 0))
            cb_k = ttk.Checkbutton(extra_frame, text="+K", variable=self.poly_adduct_k_var)
            cb_k.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(frm, text="Charges (comma-separated)").grid(row=6, column=2, sticky="e", padx=(12, 0), pady=(6, 0))
        charges_ent = ttk.Entry(frm, textvariable=self.poly_charges_var, width=12)
        charges_ent.grid(row=6, column=3, sticky="w", padx=(8, 0), pady=(6, 0))

        ttk.Separator(frm, orient="horizontal").grid(row=7, column=0, columnspan=4, sticky="ew", pady=10)

        cluster_adduct_var = tk.StringVar(value=str(self.poly_cluster_adduct_mass_var.get()))
        decarb_cb = ttk.Checkbutton(
            frm,
            text="Also match decarboxylation products (−CO2)",
            variable=self.poly_decarb_enabled_var,
        )
        decarb_cb.grid(row=8, column=0, columnspan=3, sticky="w")

        oxid_cb = ttk.Checkbutton(
            frm,
            text="Also match oxidation products (+O)",
            variable=self.poly_oxid_enabled_var,
        )
        oxid_cb.grid(row=9, column=0, columnspan=3, sticky="w")

        cluster_cb = ttk.Checkbutton(frm, text="Enable noncovalent polymer dimers (2M−H)", variable=self.poly_cluster_enabled_var)
        cluster_cb.grid(row=10, column=0, columnspan=3, sticky="w")

        ttk.Label(frm, text="Cluster H adduct mass (Da)").grid(row=11, column=0, sticky="w")
        cluster_adduct_ent = ttk.Entry(frm, textvariable=cluster_adduct_var, width=12)
        cluster_adduct_ent.grid(row=11, column=1, sticky="w", padx=(8, 0))
        ttk.Label(frm, text="(uses the same extra adducts as above)").grid(row=11, column=2, columnspan=2, sticky="w", padx=(12, 0))

        ttk.Label(frm, text="Max total monomers (DP)").grid(row=12, column=0, sticky="w", pady=(6, 0))
        dp_spin = ttk.Spinbox(frm, from_=1, to=200, textvariable=self.poly_max_dp_var, width=8)
        dp_spin.grid(row=12, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        ttk.Label(frm, text="Tolerance").grid(row=12, column=2, sticky="e", padx=(12, 0), pady=(6, 0))
        tol_val = tk.StringVar(value=str(self.poly_tol_value_var.get()))
        tol_ent = ttk.Entry(frm, textvariable=tol_val, width=10)
        tol_ent.grid(row=12, column=3, sticky="w", padx=(8, 0), pady=(6, 0))
        tol_da_rb = ttk.Radiobutton(frm, text="Da", value="Da", variable=self.poly_tol_unit_var)
        tol_da_rb.grid(row=13, column=3, sticky="w")
        tol_ppm_rb = ttk.Radiobutton(frm, text="ppm", value="ppm", variable=self.poly_tol_unit_var)
        tol_ppm_rb.grid(row=13, column=3, sticky="e")

        ttk.Label(frm, text="Min peak intensity (fraction of max)").grid(row=14, column=0, sticky="w", pady=(10, 0))
        minrel = tk.StringVar(value=str(self.poly_min_rel_int_var.get()))
        minrel_ent = ttk.Entry(frm, textvariable=minrel, width=12)
        minrel_ent.grid(row=14, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        def sync_bond(*_) -> None:
            choice = bond_choice.get()
            if choice in presets:
                bond_custom.set(str(presets[choice]))
        def sync_extra(*_) -> None:
            choice = extra_choice.get()
            if choice in extra_presets:
                extra_custom.set(str(extra_presets[choice]))

        bond_combo.bind("<<ComboboxSelected>>", sync_bond)
        extra_combo.bind("<<ComboboxSelected>>", sync_extra)
        sync_bond()
        sync_extra()

        def on_reset() -> None:
            self.poly_enabled_var.set(False)
            self.poly_monomers_text_var.set("")
            txt.delete("1.0", tk.END)
            bond_choice.set("Dehydration (-H2O)")
            bond_custom.set(str(-18.010565))
            extra_choice.set("None (0)")
            extra_custom.set("0.0")
            adduct_var.set("1.007276")
            self.poly_adduct_na_var.set(False)
            self.poly_adduct_k_var.set(False)
            self.poly_adduct_cl_var.set(False)
            self.poly_adduct_formate_var.set(False)
            self.poly_adduct_acetate_var.set(False)
            self.poly_decarb_enabled_var.set(False)
            self.poly_oxid_enabled_var.set(False)
            self.poly_cluster_enabled_var.set(False)
            cluster_adduct_var.set(str(-1.007276))
            self.poly_charges_var.set("1")
            self.poly_max_dp_var.set(12)
            tol_val.set("0.02")
            self.poly_tol_unit_var.set("Da")
            minrel.set("0.01")
            self._redraw_spectrum_only()

        def on_apply() -> None:
            try:
                self.poly_monomers_text_var.set(txt.get("1.0", tk.END).strip())
                self.poly_bond_delta_var.set(float(bond_custom.get().strip()))
                self.poly_extra_delta_var.set(float(extra_custom.get().strip()))
                self.poly_adduct_mass_var.set(float(adduct_var.get().strip()))
                self.poly_cluster_adduct_mass_var.set(float(cluster_adduct_var.get().strip()))
                self.poly_tol_value_var.set(float(tol_val.get().strip()))
                self.poly_min_rel_int_var.set(float(minrel.get().strip()))
            except Exception:
                messagebox.showerror("Invalid", "One or more numeric fields are invalid.", parent=dlg)
                return
            self._redraw_spectrum_only()

        def on_match_region() -> None:
            on_apply()
            try:
                if not bool(self.poly_enabled_var.get()):
                    self.poly_enabled_var.set(True)
            except Exception:
                pass
            self._region_force_poly_match = True
            if self._tic_region_active_rt is None:
                messagebox.showinfo("Polymer Match", "Select a TIC region first (drag on the TIC).", parent=dlg)
                return
            try:
                a, b = self._tic_region_active_rt
            except Exception:
                messagebox.showinfo("Polymer Match", "Select a TIC region first (drag on the TIC).", parent=dlg)
                return
            self._compute_region_summed_spectrum(float(a), float(b))

        buttons = ttk.Frame(frm)
        buttons.grid(row=15, column=0, columnspan=4, sticky="e", pady=(12, 0))
        apply_btn = ttk.Button(buttons, text="Apply", command=on_apply)
        apply_btn.grid(row=0, column=0, padx=(0, 8))
        region_btn = ttk.Button(buttons, text="Match Region", command=on_match_region)
        region_btn.grid(row=0, column=1, padx=(0, 8))
        reset_btn = ttk.Button(buttons, text="Reset", command=on_reset)
        reset_btn.grid(row=0, column=2, padx=(0, 8))
        close_btn = ttk.Button(buttons, text="Close", command=dlg.destroy)
        close_btn.grid(row=0, column=3)

        # Tooltips
        ToolTip.attach(pm_enable, TOOLTIP_TEXT["pm_enable"])
        ToolTip.attach(txt, TOOLTIP_TEXT["pm_monomers"])
        ToolTip.attach(bond_combo, TOOLTIP_TEXT["pm_bond_combo"])
        ToolTip.attach(bond_entry, TOOLTIP_TEXT["pm_bond_custom"])
        ToolTip.attach(extra_combo, TOOLTIP_TEXT["pm_extra_combo"])
        ToolTip.attach(extra_entry, TOOLTIP_TEXT["pm_extra_custom"])
        ToolTip.attach(adduct_ent, TOOLTIP_TEXT["pm_adduct"])
        ToolTip.attach(extra_frame, TOOLTIP_TEXT["pm_extra_adducts"])
        ToolTip.attach(charges_ent, TOOLTIP_TEXT["pm_charges"])
        ToolTip.attach(decarb_cb, "Also match decarboxylation products (−CO2).")
        ToolTip.attach(oxid_cb, "Also match oxidation products (+O).")
        ToolTip.attach(cluster_cb, "Also match noncovalent dimers (2M−H) and other selected adducts.")
        ToolTip.attach(cluster_adduct_ent, "Adduct mass used for cluster (dimer) matching.")
        ToolTip.attach(dp_spin, TOOLTIP_TEXT["pm_dp"])
        ToolTip.attach(tol_ent, TOOLTIP_TEXT["pm_tol"])
        ToolTip.attach(tol_da_rb, TOOLTIP_TEXT["pm_tol_unit"])
        ToolTip.attach(tol_ppm_rb, TOOLTIP_TEXT["pm_tol_unit"])
        ToolTip.attach(minrel_ent, TOOLTIP_TEXT["pm_minrel"])
        ToolTip.attach(apply_btn, TOOLTIP_TEXT["pm_apply"])
        ToolTip.attach(region_btn, TOOLTIP_TEXT["pm_match_region"])
        ToolTip.attach(reset_btn, TOOLTIP_TEXT["pm_reset"])
        ToolTip.attach(close_btn, TOOLTIP_TEXT["pm_close"])

    def _show_busy(self, message: str) -> None:
        if self._busy_dialog is not None:
            return
        dlg = tk.Toplevel(self)
        dlg.title("Working")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text=message, padding=10).grid(row=0, column=0, sticky="w")
        bar = ttk.Progressbar(dlg, mode="indeterminate", length=360)
        bar.grid(row=1, column=0, padx=10, pady=(0, 10))
        bar.start(12)
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)

        self._busy_dialog = dlg
        self._busy_bar = bar

    def _hide_busy(self) -> None:
        if self._busy_bar is not None:
            try:
                self._busy_bar.stop()
            except Exception:
                pass
        if self._busy_dialog is not None:
            try:
                self._busy_dialog.grab_release()
            except Exception:
                pass
            self._busy_dialog.destroy()
        self._busy_dialog = None
        self._busy_bar = None

    def _open_mzml(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an mzML file",
            filetypes=[("mzML files", "*.mzML"), ("All files", "*.*")],
        )
        if not path:
            return
        self._add_mzml_paths([Path(path)], make_first_active=(self._active_session_id is None))

    def _open_uv_csv(self) -> None:
        # Backward-compatible wrapper (Ctrl+U now uses _open_uv_csv_single).
        self._open_uv_csv_single()

    def _plot_uv(self) -> None:
        if self._ax_uv is None or self._canvas is None:
            return

        prev_lims = self._capture_axes_limits_map([self._ax_tic, self._ax_spec, self._ax_uv])

        if self._is_overlay_active() and bool(self._overlay_show_uv_var.get()):
            self._plot_uv_overlay()
            return

        self._ax_uv.clear()
        # Any previously created UV label artists are no longer valid after clear().
        self._uv_annotations = []
        self._uv_ann_key_by_objid = {}
        self._uv_rt_marker = None
        base_title = (self.uv_title_var.get() or "UV chromatogram").strip()
        uv_sess = self._active_uv_session()
        suffix = (" — " + uv_sess.path.name) if uv_sess is not None else ""
        self._ax_uv.set_title(f"{base_title}{suffix}")
        self._ax_uv.set_xlabel(self.uv_xlabel_var.get())
        self._ax_uv.set_ylabel(self.uv_ylabel_var.get())

        x, y = self._active_uv_xy()

        if x is None or y is None or x.size == 0:
            self._ax_uv.text(0.5, 0.5, "No UV linked to active mzML", ha="center", va="center", transform=self._ax_uv.transAxes)
            self._apply_plot_style()
            self._canvas.draw()
            return

        self._ax_uv.plot(x, y, linewidth=1, color=PRIMARY_TEAL)

        # Selected RT marker (if available)
        if self._selected_rt_min is not None:
            try:
                uv_i = int(np.argmin(np.abs(x - float(self._selected_rt_min))))
                x0 = float(x[uv_i])
                y0 = float(y[uv_i])
                self._uv_rt_marker = self._ax_uv.scatter(
                    [x0],
                    [y0],
                    s=36,
                    color=ACCENT_ORANGE,
                    edgecolors=TEXT_DARK,
                    linewidths=0.4,
                )
            except Exception:
                self._uv_rt_marker = None

        if bool(self.uv_label_from_ms_var.get()):
            self._draw_uv_ms_peak_labels()

        self._apply_plot_style()
        self._restore_axes_limits_map(prev_lims)
        self._canvas.draw()

    def _save_axis_image(self, ax, default_stem: str) -> None:
        if self._fig is None or self._canvas is None or ax is None:
            return

        initial = f"{default_stem}.png"
        path = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            initialfile=initial,
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            # Ensure renderer exists
            self._canvas.draw()
            renderer = self._canvas.get_renderer()
            bbox = ax.get_tightbbox(renderer).expanded(1.02, 1.08)
            bbox_inches = bbox.transformed(self._fig.dpi_scale_trans.inverted())
            self._fig.savefig(path, bbox_inches=bbox_inches)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save plot:\n{exc}")
            return

        messagebox.showinfo("Saved", f"Saved:\n{path}")

    def _reset_spectrum_view(self) -> None:
        if self._ax_spec is None or self._canvas is None:
            return
        meta = self._current_spectrum_meta
        mz = self._current_spectrum_mz
        inten = self._current_spectrum_int
        if meta is None or mz is None or inten is None:
            return
        self._plot_spectrum(meta, mz, inten)

    def _capture_axes_limits(self, ax) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if ax is None:
            return None
        try:
            return (ax.get_xlim(), ax.get_ylim())
        except Exception:
            return None

    def _restore_axes_limits(self, ax, lim: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
        if ax is None or lim is None:
            return
        try:
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[1])
        except Exception:
            pass

    def _capture_axes_limits_map(self, axes: Sequence[Any]) -> Dict[Any, Tuple[Tuple[float, float], Tuple[float, float]]]:
        out: Dict[Any, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        for ax in list(axes or []):
            if ax is None:
                continue
            try:
                out[ax] = (ax.get_xlim(), ax.get_ylim())
            except Exception:
                continue
        return out

    def _restore_axes_limits_map(self, lims: Dict[Any, Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
        for ax, lim in list((lims or {}).items()):
            try:
                ax.set_xlim(lim[0])
                ax.set_ylim(lim[1])
            except Exception:
                continue

    def _save_tic_plot(self) -> None:
        if self._ax_tic is None:
            return
        if not self._is_overlay_active():
            if self._filtered_rts is None or self._filtered_tics is None or self._filtered_rts.size == 0:
                messagebox.showerror("No TIC", "Load an mzML file first.", parent=self)
                return
        stem = "tic"
        if self._is_overlay_active():
            stem = "overlay_tic"
        elif self.mzml_path is not None:
            stem = f"{self.mzml_path.stem}_tic"
        ExportEditor(self, kind="tic", default_stem=stem, tooltip_text=TOOLTIP_TEXT)

    def _save_spectrum_plot(self) -> None:
        if self._ax_spec is None:
            return
        if self._current_spectrum_meta is None or self._current_spectrum_mz is None or self._current_spectrum_int is None:
            messagebox.showerror("No spectrum", "Load a spectrum first (click TIC/UV).", parent=self)
            return
        stem = "spectrum"
        if self._is_overlay_active():
            stem = "overlay_spectrum"
        elif self.mzml_path is not None:
            stem = f"{self.mzml_path.stem}_spectrum_rt{self._current_spectrum_meta.rt_min:.3f}"
        ExportEditor(self, kind="spectrum", default_stem=stem, tooltip_text=TOOLTIP_TEXT)

    def _save_uv_plot(self) -> None:
        if self._ax_uv is None:
            return
        uv_sess = self._active_uv_session()
        if uv_sess is None and not (self._is_overlay_active() and bool(self._overlay_show_uv_var.get())):
            messagebox.showerror("No UV linked", "Link a UV CSV to the active mzML session first.", parent=self)
            return
        stem = "uv_chromatogram"
        if self._is_overlay_active() and bool(self._overlay_show_uv_var.get()):
            stem = "overlay_uv"
        elif uv_sess is not None:
            stem = f"{uv_sess.path.stem}_uv"
        ExportEditor(self, kind="uv", default_stem=stem, tooltip_text=TOOLTIP_TEXT)

    def _open_tic_window(self) -> None:
        if self._ax_tic is None:
            return
        if not self._is_overlay_active():
            if self._filtered_rts is None or self._filtered_tics is None or self._filtered_rts.size == 0:
                messagebox.showerror("No TIC", "Load an mzML file first.", parent=self)
                return
        stem = "tic"
        if self._is_overlay_active():
            stem = "overlay_tic"
        elif self.mzml_path is not None:
            stem = f"{self.mzml_path.stem}_tic"
        ExportEditor(self, kind="tic", default_stem=stem, tooltip_text=TOOLTIP_TEXT)

    def _open_spectrum_window(self) -> None:
        if self._ax_spec is None:
            return
        if self._current_spectrum_meta is None or self._current_spectrum_mz is None or self._current_spectrum_int is None:
            messagebox.showerror("No spectrum", "Load a spectrum first (click TIC/UV).", parent=self)
            return
        stem = "spectrum"
        if self._is_overlay_active():
            stem = "overlay_spectrum"
        elif self.mzml_path is not None:
            stem = f"{self.mzml_path.stem}_spectrum_rt{self._current_spectrum_meta.rt_min:.3f}"
        ExportEditor(self, kind="spectrum", default_stem=stem, tooltip_text=TOOLTIP_TEXT)

    def _open_uv_window(self) -> None:
        if self._ax_uv is None:
            return
        uv_sess = self._active_uv_session()
        if uv_sess is None and not (self._is_overlay_active() and bool(self._overlay_show_uv_var.get())):
            messagebox.showerror("No UV linked", "Link a UV CSV to the active mzML session first.", parent=self)
            return
        stem = "uv_chromatogram"
        if self._is_overlay_active() and bool(self._overlay_show_uv_var.get()):
            stem = "overlay_uv"
        elif uv_sess is not None:
            stem = f"{uv_sess.path.stem}_uv"
        ExportEditor(self, kind="uv", default_stem=stem, tooltip_text=TOOLTIP_TEXT)

    def _on_loaded(self, mzml_path: Path, idx: MzMLTICIndex) -> None:
        self._hide_busy()

        # Close previous reader
        try:
            if self._reader is not None:
                self._reader.close()
        except Exception:
            pass

        self.mzml_path = mzml_path
        self._index = idx
        self._reader = mzml.MzML(str(mzml_path))

        # New mzML: clear any transferred UV labels (RT alignment may change).
        try:
            sid = getattr(self, "_active_session_id", None)
            if sid and sid in getattr(self, "_sessions", {}):
                self._sessions[sid].uv_labels_by_uv_id.clear()
        except Exception:
            pass
        self._sync_active_uv_id()

        self._status.configure(text=f"Loaded: {mzml_path.name} (MS1 spectra: {len(idx.ms1)})")
        self._refresh_tic()
        self._update_status_current()

    def _on_load_error(self, exc: Exception) -> None:
        self._hide_busy()
        messagebox.showerror("Error", str(exc))
        self._status.configure(text="Open an mzML file to begin")
        self._update_status_current()

    def _is_overlay_active(self) -> bool:
        return bool(self._overlay_session is not None and len(self._overlay_session.dataset_ids) >= 2)

    def _overlay_dataset_ids(self) -> List[str]:
        if self._overlay_session is not None and self._overlay_session.dataset_ids:
            ids = [str(i) for i in self._overlay_session.dataset_ids if str(i) in self._sessions]
            return [sid for sid in self._session_order if sid in ids]
        # Fallback: selected in workspace list
        ids = [sid for sid, sess in self._sessions.items() if bool(getattr(sess, "overlay_selected", False))]
        return [sid for sid in self._session_order if sid in ids]

    def _overlay_selected_ids_from_flags(self) -> List[str]:
        ids = [sid for sid, sess in self._sessions.items() if bool(getattr(sess, "overlay_selected", False))]
        return [sid for sid in self._session_order if sid in ids]

    def _refresh_overlay_view(self) -> None:
        if not self._is_overlay_active():
            try:
                if self._overlay_legend_frame is not None:
                    self._overlay_legend_frame.grid_remove()
            except Exception:
                pass
            if self._overlay_session is None:
                return
            return

        # Update overlay session settings from UI vars
        try:
            self._overlay_session = OverlaySession(
                dataset_ids=list(self._overlay_session.dataset_ids),
                mode=str(self._overlay_mode_var.get()),
                colors=dict(self._overlay_session.colors),
                persist=bool(self._overlay_persist_var.get()),
                show_uv=bool(self._overlay_show_uv_var.get()),
                stack_spectra=bool(self._overlay_stack_spectra_var.get()),
                show_labels_all=bool(self._overlay_show_labels_all_var.get()),
                multi_drag=bool(self._overlay_multi_drag_var.get()),
                active_dataset_id=str(self._overlay_active_dataset_id or self._active_session_id or "") or None,
            )
        except Exception:
            pass

        try:
            self._apply_overlay_color_scheme(ids=self._overlay_dataset_ids())
        except Exception:
            pass

        try:
            if self._overlay_legend_frame is not None:
                self._overlay_legend_frame.grid()
        except Exception:
            pass

        self._refresh_overlay_legend()
        self._refresh_overlay_tic()
        self._plot_uv()

        # Keep current RT selection if possible
        rt = self._overlay_selected_ms_rt
        if rt is None and self._current_spectrum_meta is not None:
            rt = float(self._current_spectrum_meta.rt_min)
        if rt is not None:
            self._plot_overlay_spectrum_for_rt(float(rt))

    def _overlay_find_nearest_dataset_id_by_rt(self, rt_min: float) -> Optional[str]:
        best_sid = None
        best_dt = None
        pol = str(self.polarity_var.get())
        for sid in self._overlay_dataset_ids():
            meta, rts, _tics = self._overlay_meta_for_session(str(sid), pol)
            if rts is None or rts.size == 0:
                continue
            i = int(np.argmin(np.abs(rts - float(rt_min))))
            dt = float(abs(float(rts[i]) - float(rt_min)))
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_sid = str(sid)
        return best_sid

    def _overlay_meta_for_session(self, session_id: str, pol: str) -> Tuple[List[SpectrumMeta], np.ndarray, np.ndarray]:
        key = (str(session_id), str(pol))
        if key in self._overlay_tic_cache:
            return self._overlay_tic_cache[key]
        sess = self._sessions.get(str(session_id))
        if sess is None or sess.index is None:
            empty = ([], np.asarray([], dtype=float), np.asarray([], dtype=float))
            self._overlay_tic_cache[key] = empty
            return empty
        if pol == "all":
            meta = list(sess.index.ms1)
        else:
            meta = [m for m in sess.index.ms1 if m.polarity == pol]
        meta.sort(key=lambda m: m.rt_min)
        rts = np.asarray([m.rt_min for m in meta], dtype=float) if meta else np.asarray([], dtype=float)
        tics = np.asarray([m.tic for m in meta], dtype=float) if meta else np.asarray([], dtype=float)
        out = (meta, rts, tics)
        self._overlay_tic_cache[key] = out
        return out

    def _refresh_overlay_legend(self) -> None:
        tree = self._overlay_legend_tree
        if tree is None:
            return
        try:
            for iid in list(tree.get_children("")):
                tree.delete(iid)
        except Exception:
            pass

        ids = self._overlay_dataset_ids()
        if not ids:
            return

        name_map = self._overlay_display_names(ids)
        for sid in ids:
            sess = self._sessions.get(str(sid))
            if sess is None:
                continue
            nm = str(name_map.get(str(sid), sess.display_name))

            status = str(self._overlay_status_by_sid.get(str(sid), ""))
            pol = str(sess.polarity_summary or "unknown")
            ms1 = str(sess.ms1_count)
            color_mark = "■"
            tag = f"ovr_{str(sid)}"
            try:
                tree.tag_configure(tag, foreground=str(self._ensure_overlay_color(str(sid))))
            except Exception:
                pass
            try:
                tree.insert("", "end", iid=str(sid), values=(color_mark, nm, ms1, pol, status), tags=(tag,))
            except Exception:
                continue

        try:
            if self._active_session_id and tree.exists(str(self._active_session_id)):
                tree.selection_set(str(self._active_session_id))
        except Exception:
            pass

        try:
            tree.bind("<<TreeviewSelect>>", lambda _e: self._set_active_overlay_from_legend(), add=True)
        except Exception:
            pass

    def _overlay_display_names(self, ids: Sequence[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        names: List[str] = []
        for sid in ids:
            sess = self._sessions.get(str(sid))
            if sess is None:
                continue
            names.append(str(sess.display_name))
        counts: Dict[str, int] = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1

        used: Dict[str, int] = {}
        for sid in ids:
            sess = self._sessions.get(str(sid))
            if sess is None:
                continue
            nm = str(sess.display_name)
            if counts.get(nm, 0) > 1:
                used[nm] = used.get(nm, 0) + 1
                nm = f"{nm} ({used[nm]})"
            out[str(sid)] = nm
        return out

    def _set_active_overlay_from_legend(self) -> None:
        tree = self._overlay_legend_tree
        if tree is None:
            return
        try:
            sel = tree.selection()
            sid = str(sel[0]) if sel else ""
        except Exception:
            sid = ""
        if not sid or sid not in self._sessions:
            return
        if sid != self._active_session_id:
            self._set_active_session(str(sid))

    def _refresh_overlay_tic(self) -> None:
        if self._ax_tic is None or self._canvas is None:
            return

        prev_lims = self._capture_axes_limits_map([self._ax_tic, self._ax_spec, self._ax_uv])

        self._tic_line = None
        try:
            if self._tic_marker is not None:
                self._tic_marker.remove()
        except Exception:
            pass
        self._tic_marker = None

        ids = self._overlay_dataset_ids()
        if len(ids) < 2:
            return

        pol = str(self.polarity_var.get())
        mode = str(self._overlay_mode_var.get() or "Stacked")

        if len(ids) > 8 and mode == "Stacked":
            mode = "Normalized"
            try:
                self._overlay_mode_var.set(mode)
            except Exception:
                pass
            self._warn("Overlay: many datasets selected; switching to Normalized mode for readability.")

        self._ax_tic.clear()
        base_title = (self.tic_title_var.get() or "TIC (MS1)").strip()
        self._ax_tic.set_title(f"{base_title} — overlay ({mode}) | polarity: {pol}")
        self._ax_tic.set_xlabel(self.tic_xlabel_var.get())
        self._ax_tic.set_ylabel(self.tic_ylabel_var.get())

        # Reset status map
        self._overlay_status_by_sid = {}

        # Keep filtered_meta for active dataset (for navigation)
        if self._active_session_id in ids:
            meta, rts, tics = self._overlay_meta_for_session(str(self._active_session_id), pol)
            self._filtered_meta = list(meta)
            self._filtered_rts = np.asarray(rts, dtype=float) if rts is not None else None
            self._filtered_tics = np.asarray(tics, dtype=float) if tics is not None else None

        # Determine offsets/normalization
        max_global = 0.0
        per_max: Dict[str, float] = {}
        for sid in ids:
            _meta, rts, tics = self._overlay_meta_for_session(str(sid), pol)
            if tics is None or tics.size == 0:
                per_max[str(sid)] = 0.0
                continue
            m = float(np.max(tics)) if tics.size else 0.0
            per_max[str(sid)] = m
            max_global = max(max_global, m)

        offset_step = 0.12 * max_global if max_global > 0 else 1.0

        for i, sid in enumerate(ids):
            meta, rts, tics = self._overlay_meta_for_session(str(sid), pol)
            if rts is None or tics is None or rts.size == 0 or tics.size == 0:
                self._overlay_status_by_sid[str(sid)] = "no MS1 TIC"
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

            col = self._ensure_overlay_color(str(sid))
            self._ax_tic.plot(rts, y, linewidth=1.0, color=col, alpha=0.85)

            # Selected RT marker for each dataset
            if self._overlay_selected_ms_rt is not None:
                rt = float(self._overlay_selected_ms_rt)
                j = int(np.argmin(np.abs(rts - rt)))
                dt = float(abs(float(rts[j]) - rt))
                if dt <= float(self._overlay_rt_tolerance_min):
                    self._ax_tic.scatter([rts[j]], [y[j]], s=28, color=col, edgecolors=TEXT_DARK, linewidths=0.3)
                else:
                    self._overlay_status_by_sid[str(sid)] = "no scan near RT"

        self._apply_plot_style()
        try:
            self._restore_axes_limits_map(prev_lims)
            self._canvas.draw()
        except Exception:
            pass

        self._refresh_overlay_legend()

    def _plot_uv_overlay(self) -> None:
        if self._ax_uv is None or self._canvas is None:
            return

        prev_lims = self._capture_axes_limits_map([self._ax_tic, self._ax_spec, self._ax_uv])

        self._ax_uv.clear()
        self._uv_annotations = []
        self._uv_ann_key_by_objid = {}
        self._uv_rt_marker = None

        base_title = (self.uv_title_var.get() or "UV chromatogram").strip()
        self._ax_uv.set_title(f"{base_title} — overlay")
        self._ax_uv.set_xlabel(self.uv_xlabel_var.get())
        self._ax_uv.set_ylabel(self.uv_ylabel_var.get())

        ids = self._overlay_dataset_ids()
        if not ids:
            self._ax_uv.text(0.5, 0.5, "No overlay datasets", ha="center", va="center", transform=self._ax_uv.transAxes)
            self._apply_plot_style()
            self._canvas.draw()
            return

        any_uv = False
        for sid in ids:
            sess = self._sessions.get(str(sid))
            if sess is None:
                continue
            uv_id = sess.linked_uv_id
            if not uv_id or str(uv_id) not in self._uv_sessions:
                continue
            uv_sess = self._uv_sessions[str(uv_id)]
            x = np.asarray(uv_sess.rt_min, dtype=float)
            y = np.asarray(uv_sess.signal, dtype=float)
            if x.size == 0 or y.size == 0:
                continue
            any_uv = True
            col = self._ensure_overlay_color(str(sid))
            self._ax_uv.plot(x, y, linewidth=1, color=col, alpha=0.85)

        if not any_uv:
            self._ax_uv.text(0.5, 0.5, "No UV linked for overlay datasets", ha="center", va="center", transform=self._ax_uv.transAxes)
            self._apply_plot_style()
            self._canvas.draw()
            return

        if self._selected_rt_min is not None:
            # Mark selected UV RT on active dataset if available
            try:
                uv = self._active_uv_session()
                if uv is not None:
                    x = np.asarray(uv.rt_min, dtype=float)
                    y = np.asarray(uv.signal, dtype=float)
                    if x.size > 0:
                        uv_i = int(np.argmin(np.abs(x - float(self._selected_rt_min))))
                        x0 = float(x[uv_i])
                        y0 = float(y[uv_i])
                        self._uv_rt_marker = self._ax_uv.scatter(
                            [x0],
                            [y0],
                            s=36,
                            color=ACCENT_ORANGE,
                            edgecolors=TEXT_DARK,
                            linewidths=0.4,
                        )
            except Exception:
                self._uv_rt_marker = None

        if bool(self.uv_label_from_ms_var.get()):
            self._draw_uv_ms_peak_labels()

        self._apply_plot_style()
        self._restore_axes_limits_map(prev_lims)
        self._canvas.draw()

    def _get_overlay_reader(self, session_id: str) -> Optional[mzml.MzML]:
        if session_id == self._active_session_id:
            return self._reader
        if session_id in self._overlay_readers:
            return self._overlay_readers[session_id]
        sess = self._sessions.get(str(session_id))
        if sess is None:
            return None
        try:
            rdr = mzml.MzML(str(sess.path))
        except Exception:
            return None
        self._overlay_readers[session_id] = rdr
        return rdr

    def _get_spectrum_for_rt(self, session_id: str, target_rt: float) -> Optional[Tuple[SpectrumMeta, np.ndarray, np.ndarray, float]]:
        pol = str(self.polarity_var.get())
        meta, rts, _tics = self._overlay_meta_for_session(str(session_id), pol)
        if rts is None or rts.size == 0 or not meta:
            return None
        i = int(np.argmin(np.abs(rts - float(target_rt))))
        dt = float(abs(float(rts[i]) - float(target_rt)))
        if dt > float(self._overlay_rt_tolerance_min):
            return None
        m = meta[int(i)]
        rdr = self._get_overlay_reader(str(session_id))
        if rdr is None:
            self._overlay_status_by_sid[str(session_id)] = "spectrum load failed"
            try:
                self._set_status(f"Overlay: failed to open mzML for {session_id}")
            except Exception:
                pass
            return None
        try:
            try:
                spectrum = rdr.get_by_id(str(m.spectrum_id))
            except Exception:
                spectrum = rdr[str(m.spectrum_id)]
        except Exception:
            self._overlay_status_by_sid[str(session_id)] = "spectrum load failed"
            try:
                self._set_status(f"Overlay: failed to load spectrum for {session_id}")
            except Exception:
                pass
            return None
        mz_array = spectrum.get("m/z array")
        int_array = spectrum.get("intensity array")
        if mz_array is None or int_array is None:
            return None
        mz_vals = np.asarray(mz_array, dtype=float)
        int_vals = np.asarray(int_array, dtype=float)
        return m, mz_vals, int_vals, dt

    def _plot_overlay_spectrum_for_rt(self, target_rt: float) -> None:
        if self._ax_spec is None or self._canvas is None:
            return
        ids = self._overlay_dataset_ids()
        if len(ids) < 2:
            return

        self._ax_spec.clear()
        self._clear_spectrum_annotations()

        mode = str(self._overlay_mode_var.get() or "Stacked")
        base_title = (self.spec_title_var.get() or "Spectrum (MS1)").strip()
        self._ax_spec.set_title(f"{base_title} — overlay at RT={float(target_rt):.4f} min")
        self._ax_spec.set_xlabel(self.spec_xlabel_var.get())
        self._ax_spec.set_ylabel(self.spec_ylabel_var.get())

        max_global = 0.0
        spectra: List[Tuple[str, SpectrumMeta, np.ndarray, np.ndarray]] = []
        for sid in ids:
            got = self._get_spectrum_for_rt(str(sid), float(target_rt))
            if got is None:
                if not self._overlay_status_by_sid.get(str(sid)):
                    self._overlay_status_by_sid[str(sid)] = "no scan near RT"
                continue
            meta, mz_vals, int_vals, _dt = got
            spectra.append((str(sid), meta, mz_vals, int_vals))
            if int_vals.size:
                max_global = max(max_global, float(np.max(int_vals)))

        if not spectra:
            self._ax_spec.text(0.5, 0.5, "No spectra near selected RT", ha="center", va="center", transform=self._ax_spec.transAxes)
            self._apply_plot_style()
            self._canvas.draw()
            return

        offset_step = 0.12 * max_global if max_global > 0 else 1.0
        stack = bool(self._overlay_stack_spectra_var.get())

        # Choose active dataset
        active_sid = str(self._active_session_id or "")
        if active_sid not in ids and ids:
            active_sid = str(ids[0])

        try:
            pol = str(self.polarity_var.get())
            meta_list, rts, _tics = self._overlay_meta_for_session(str(active_sid), pol)
            if rts is not None and rts.size:
                k = int(np.argmin(np.abs(rts - float(target_rt))))
                self._current_scan_index = int(k)
        except Exception:
            pass

        active_set = False
        for i, (sid, meta, mz_vals, int_vals) in enumerate(spectra):
            col = self._ensure_overlay_color(str(sid))
            y = np.asarray(int_vals, dtype=float)
            if mode in ("Normalized", "Percent of max"):
                denom = float(np.max(y)) if y.size else 0.0
                if denom > 0:
                    y = y / denom
                if mode == "Percent of max":
                    y = y * 100.0
            if stack:
                y = y + float(i) * float(offset_step)
            self._ax_spec.vlines(mz_vals, 0.0 + (float(i) * float(offset_step) if stack else 0.0), y, linewidth=0.7, color=col, alpha=0.8)

            # Apply labels if active or "show all" enabled
            if str(sid) == active_sid:
                self._current_spectrum_meta = meta
                self._current_spectrum_mz = mz_vals
                self._current_spectrum_int = int_vals
                active_set = True
                if bool(self.annotate_peaks_var.get()):
                    self._annotate_peaks(mz_vals, int_vals)
                self._apply_custom_labels(mz_vals, int_vals)
                self._apply_polymer_matches(mz_vals, int_vals)
            elif bool(self._overlay_show_labels_all_var.get()):
                self._apply_labels_for_session(str(sid), meta, mz_vals, int_vals)

        if not active_set:
            self._current_spectrum_meta = None
            self._current_spectrum_mz = None
            self._current_spectrum_int = None

        self._apply_plot_style()
        self._canvas.draw()

        self._refresh_overlay_legend()

    def _apply_labels_for_session(self, session_id: str, meta: SpectrumMeta, mz_vals: np.ndarray, int_vals: np.ndarray) -> None:
        if session_id not in self._sessions:
            return
        sess = self._sessions[session_id]
        # Temporarily swap session label state
        prev_custom = self._custom_labels_by_spectrum
        prev_overrides = self._spec_label_overrides
        prev_meta = self._current_spectrum_meta
        prev_mz = self._current_spectrum_mz
        prev_int = self._current_spectrum_int
        try:
            self._custom_labels_by_spectrum = sess.custom_labels_by_spectrum
            self._spec_label_overrides = sess.spec_label_overrides
            self._current_spectrum_meta = meta
            self._current_spectrum_mz = mz_vals
            self._current_spectrum_int = int_vals
            self._apply_custom_labels(mz_vals, int_vals)
            self._apply_polymer_matches(mz_vals, int_vals)
        except Exception:
            pass
        finally:
            self._custom_labels_by_spectrum = prev_custom
            self._spec_label_overrides = prev_overrides
            self._current_spectrum_meta = prev_meta
            self._current_spectrum_mz = prev_mz
            self._current_spectrum_int = prev_int

    def _start_overlay_selected(self) -> None:
        ids = [sid for sid in self._overlay_dataset_ids() if sid in self._sessions]
        if len(ids) < 2:
            messagebox.showinfo("Overlay", "Select at least two mzML files for overlay.", parent=self)
            return

        colors: Dict[str, str] = {}
        scheme = str(self._overlay_scheme_var.get() or "").strip()
        if scheme and scheme != "Manual (per-dataset)":
            scheme_colors = self._overlay_colors_for_scheme(scheme, len(ids))
            for sid, col in zip(ids, scheme_colors):
                sess = self._sessions.get(str(sid))
                if sess is not None:
                    try:
                        sess.overlay_color = str(col)
                    except Exception:
                        pass
        for sid in ids:
            colors[str(sid)] = self._ensure_overlay_color(str(sid))
            try:
                self._refresh_ws_tree_row(str(sid))
            except Exception:
                pass

        self._overlay_prev_active_session_id = self._active_session_id
        if self._active_session_id not in ids:
            self._set_active_session(str(ids[0]))

        self._overlay_active_dataset_id = self._active_session_id
        self._overlay_session = OverlaySession(
            dataset_ids=list(ids),
            mode=str(self._overlay_mode_var.get()),
            colors=dict(colors),
            persist=bool(self._overlay_persist_var.get()),
            show_uv=bool(self._overlay_show_uv_var.get()),
            stack_spectra=bool(self._overlay_stack_spectra_var.get()),
            show_labels_all=bool(self._overlay_show_labels_all_var.get()),
            multi_drag=bool(self._overlay_multi_drag_var.get()),
            active_dataset_id=str(self._overlay_active_dataset_id) if self._overlay_active_dataset_id else None,
        )
        if self._current_spectrum_meta is not None:
            try:
                self._overlay_selected_ms_rt = float(self._current_spectrum_meta.rt_min)
            except Exception:
                self._overlay_selected_ms_rt = None
        self._refresh_overlay_view()
        self._log("INFO", f"Overlay started with {len(ids)} datasets")

    def _clear_overlay(self) -> None:
        # Close overlay readers
        for sid, rdr in list(self._overlay_readers.items()):
            try:
                rdr.close()
            except Exception:
                pass
            self._overlay_readers.pop(sid, None)

        self._overlay_session = None
        self._overlay_selected_ms_rt = None
        self._overlay_status_by_sid = {}
        try:
            if self._overlay_legend_frame is not None:
                self._overlay_legend_frame.grid_remove()
        except Exception:
            pass

        prev = self._overlay_prev_active_session_id
        self._overlay_prev_active_session_id = None
        if prev and prev in self._sessions:
            self._set_active_session(str(prev))
        else:
            self._refresh_tic()
            self._plot_uv()
        self._log("INFO", "Overlay cleared")

    def _refresh_tic(self) -> None:
        if self._is_overlay_active():
            self._refresh_overlay_tic()
            return
        if self._index is None or self._canvas is None:
            return

        prev_lims = self._capture_axes_limits_map([self._ax_tic, self._ax_spec, self._ax_uv])

        pol = self.polarity_var.get()
        prev_pol = str(getattr(self, "_last_polarity_filter", "all"))
        self._last_polarity_filter = str(pol)

        # Polarity filter changes affect which scans are included; clear region selection.
        if self._tic_region_active_rt is not None and str(prev_pol) != str(pol):
            try:
                self._clear_tic_region_selection(restore_scan=False)
            except Exception:
                pass
        if pol == "all":
            meta = list(self._index.ms1)
        else:
            meta = [m for m in self._index.ms1 if m.polarity == pol]

        meta.sort(key=lambda m: m.rt_min)
        self._filtered_meta = meta
        self._filtered_rts = np.asarray([m.rt_min for m in meta], dtype=float) if meta else None
        self._filtered_tics = np.asarray([m.tic for m in meta], dtype=float) if meta else None

        if self._ax_tic is not None:
            self._ax_tic.clear()
            base_title = (self.tic_title_var.get() or "TIC (MS1)").strip()
            self._ax_tic.set_title(f"{base_title} — polarity: {pol}")
            self._ax_tic.set_xlabel(self.tic_xlabel_var.get())
            self._ax_tic.set_ylabel(self.tic_ylabel_var.get())

        if not meta:
            if self._ax_tic is not None:
                self._ax_tic.text(0.5, 0.5, "No MS1 spectra for this polarity", ha="center", va="center", transform=self._ax_tic.transAxes)
            self._tic_line = None
            self._tic_marker = None
            try:
                self._canvas.draw()
            except Exception:
                pass
            self._current_scan_index = None
            self._current_spectrum_meta = None
            self._current_spectrum_mz = None
            self._current_spectrum_int = None
            self._update_status_current()
            return

        # Keep current selection if possible
        sel_idx = int(self._current_scan_index) if self._current_scan_index is not None else 0
        sel_idx = max(0, min(sel_idx, len(meta) - 1))

        if self._ax_tic is not None:
            (line,) = self._ax_tic.plot(self._filtered_rts, self._filtered_tics, linewidth=1, color=PRIMARY_TEAL)
            self._tic_line = line

            # Reset marker
            if self._tic_marker is not None:
                try:
                    self._tic_marker.remove()
                except Exception:
                    pass
            self._tic_marker = self._ax_tic.scatter(
                [meta[sel_idx].rt_min],
                [meta[sel_idx].tic],
                s=34,
                color=ACCENT_ORANGE,
                edgecolors=TEXT_DARK,
                linewidths=0.4,
            )

            # Re-draw active region span (if any)
            try:
                self._draw_tic_region_span()
            except Exception:
                pass

            self._apply_plot_style()
            try:
                self._restore_axes_limits_map(prev_lims)
                self._canvas.draw()
            except Exception:
                pass

        # Load/refresh current spectrum (plots only if spectrum panel is visible)
        self._show_spectrum_for_index(sel_idx)
        self._update_status_current()

    def _redraw_all(self) -> None:
        # Redraw TIC and current spectrum using current settings
        self._refresh_tic()
        self._plot_uv()

    def _redraw_spectrum_only(self) -> None:
        if self._is_overlay_active():
            rt = self._overlay_selected_ms_rt
            if rt is None and self._current_spectrum_meta is not None:
                rt = float(self._current_spectrum_meta.rt_min)
            if rt is not None:
                self._plot_overlay_spectrum_for_rt(float(rt))
            return
        if self._current_spectrum_meta is None or self._current_spectrum_mz is None or self._current_spectrum_int is None:
            return
        self._plot_spectrum(self._current_spectrum_meta, self._current_spectrum_mz, self._current_spectrum_int)

    def _on_plot_click(self, event) -> None:
        try:
            if bool(getattr(event, "dblclick", False)):
                if event is not None and event.inaxes == self._ax_spec:
                    self._reset_spectrum_view()
                return
        except Exception:
            pass
        try:
            if event is not None and int(getattr(event, "button", 0) or 0) == 1:
                nav = getattr(self, "_mpl_nav", None)
                if nav is not None and bool(getattr(nav, "is_box_pending")()):
                    return
        except Exception:
            pass
        # TIC region selection (click + drag)
        if (
            bool(getattr(self, "tic_region_select_var", None) and self.tic_region_select_var.get())
            and self._ax_tic is not None
            and event.inaxes == self._ax_tic
            and int(getattr(event, "button", 0) or 0) == 1
        ):
            # Don't interfere with pan/zoom tools
            try:
                if self._toolbar is not None and bool(getattr(self._toolbar, "mode", "")):
                    return
            except Exception:
                pass
            if event.xdata is None:
                return
            self._tic_region_dragging = True
            self._tic_region_start_rt = float(event.xdata)
            self._tic_region_end_rt = float(event.xdata)
            self._set_tic_region_span(float(event.xdata), float(event.xdata))
            return

        # Edit/delete a label with right-click or double-click
        if event.inaxes in [self._ax_spec, self._ax_uv] and (event.button == 3 or bool(getattr(event, "dblclick", False))):
            is_shift = False
            try:
                k = str(getattr(event, "key", "") or "").lower()
                is_shift = ("shift" in k)
            except Exception:
                is_shift = False
            if event.inaxes == self._ax_spec and self._spectrum_annotations:
                for ann in self._spectrum_annotations:
                    try:
                        contains, _ = ann.contains(event)
                    except Exception:
                        contains = False
                    if contains:
                        if bool(is_shift) and event.button == 3:
                            self._open_label_explanation(ann)
                        else:
                            self._open_label_editor(ann)
                        return
            if event.inaxes == self._ax_uv and self._uv_annotations:
                for ann in self._uv_annotations:
                    try:
                        contains, _ = ann.contains(event)
                    except Exception:
                        contains = False
                    if contains:
                        if bool(is_shift) and event.button == 3:
                            self._open_label_explanation(ann)
                        else:
                            self._open_label_editor(ann)
                        return

        # Start dragging an existing annotation (spectrum axis) if enabled
        if (
            bool(self.drag_annotations_var.get())
            and self._ax_spec is not None
            and event.inaxes == self._ax_spec
            and self._spectrum_annotations
        ):
            for ann in self._spectrum_annotations:
                try:
                    contains, _ = ann.contains(event)
                except Exception:
                    contains = False
                if contains:
                    self._active_annotation = ann
                    self._active_annotation_ax = self._ax_spec
                    return

        # Start dragging an existing annotation (UV axis) if enabled
        if (
            bool(self.drag_annotations_var.get())
            and self._ax_uv is not None
            and event.inaxes == self._ax_uv
            and self._uv_annotations
        ):
            for ann in self._uv_annotations:
                try:
                    contains, _ = ann.contains(event)
                except Exception:
                    contains = False
                if contains:
                    self._active_annotation = ann
                    self._active_annotation_ax = self._ax_uv
                    return

        if self._is_overlay_active():
            if event.xdata is None:
                return
            if event.inaxes not in [self._ax_tic, self._ax_uv]:
                return

            # TIC click in overlay mode
            if event.inaxes == self._ax_tic:
                target_rt = float(event.xdata)
                self._overlay_selected_ms_rt = float(target_rt)
                closest_sid = self._overlay_find_nearest_dataset_id_by_rt(float(target_rt))
                if closest_sid and closest_sid != self._active_session_id:
                    self._set_active_session(str(closest_sid))
                self._refresh_overlay_tic()
                self._plot_overlay_spectrum_for_rt(float(target_rt))
                # Optional: transfer top MS peaks to UV labels for this RT pick
                uv_anchor_rt = float(self._map_ms_to_uv_rt(float(target_rt)))
                self._selected_rt_min = float(uv_anchor_rt)
                self._maybe_store_uv_ms_labels_for_current_spectrum(anchor_rt_min=float(uv_anchor_rt))
                if self._overlay_show_uv_var.get():
                    self._plot_uv()
                return

            # UV click in overlay mode
            x, y = self._active_uv_xy()
            if x is None or y is None or x.size == 0:
                return
            uv_i = int(np.argmin(np.abs(x - float(event.xdata))))
            uv_rt = float(x[uv_i])
            ms_target_rt = float(self._map_uv_to_ms_rt(float(uv_rt)))
            self._overlay_selected_ms_rt = float(ms_target_rt)
            self._refresh_overlay_tic()
            self._plot_overlay_spectrum_for_rt(float(ms_target_rt))
            self._selected_rt_min = float(uv_rt)
            self._maybe_store_uv_ms_labels_for_current_spectrum(anchor_rt_min=float(uv_rt))
            self._plot_uv()
            self._update_status_current()
            return

        if self._filtered_rts is None or self._filtered_tics is None:
            return
        if event.xdata is None:
            return

        if event.inaxes not in [self._ax_tic, self._ax_uv]:
            return

        # TIC click: select nearest RT (no snapping to peak apex)
        if event.inaxes == self._ax_tic:
            nearest_idx = int(np.argmin(np.abs(self._filtered_rts - float(event.xdata))))
            self._show_spectrum_for_index(nearest_idx)
            # Optional: transfer top MS peaks to UV labels for this RT pick
            # Anchor on UV time mapped from MS time
            ms_rt = float(self._filtered_meta[int(nearest_idx)].rt_min)
            uv_anchor_rt = float(self._map_ms_to_uv_rt(float(ms_rt)))
            self._selected_rt_min = float(uv_anchor_rt)
            self._maybe_store_uv_ms_labels_for_current_spectrum(anchor_rt_min=float(uv_anchor_rt))
            # Ensure UV marker updates even if transfer is off
            if self._active_uv_session() is not None:
                self._plot_uv()
            return

        # UV click: select nearest UV RT, then map to nearest mzML RT (no snapping to UV/TIC apex)
        x, y = self._active_uv_xy()
        if x is None or y is None or x.size == 0:
            return

        uv_i = int(np.argmin(np.abs(x - float(event.xdata))))
        uv_rt = float(x[uv_i])
        # Map UV -> MS using auto-align if enabled (fallback: fixed offset).
        ms_target_rt = float(self._map_uv_to_ms_rt(float(uv_rt)))
        nearest_scan = int(np.argmin(np.abs(self._filtered_rts - float(ms_target_rt))))
        self._show_spectrum_for_index(int(nearest_scan))

        # Keep UV marker anchored to the clicked UV RT (not the nearest MS RT back-projection).
        self._selected_rt_min = float(uv_rt)
        # Optional: transfer top MS peaks to UV labels for this RT pick (UV-anchored)
        self._maybe_store_uv_ms_labels_for_current_spectrum(anchor_rt_min=float(uv_rt))
        self._plot_uv()
        self._update_status_current()

    def _on_plot_motion(self, event) -> None:
        # Update TIC region span while dragging
        if bool(getattr(self, "_tic_region_dragging", False)) and self._ax_tic is not None and event.inaxes == self._ax_tic:
            if event.xdata is None:
                return
            if self._tic_region_start_rt is None:
                return
            self._tic_region_end_rt = float(event.xdata)
            self._set_tic_region_span(float(self._tic_region_start_rt), float(self._tic_region_end_rt))
            return

        if self._active_annotation is None:
            return
        if self._active_annotation_ax is None or event.inaxes != self._active_annotation_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Move the label; arrow stays anchored to the peak.
        try:
            self._active_annotation.set_position((float(event.xdata), float(event.ydata)))
        except Exception:
            return

        if (
            self._is_overlay_active()
            and bool(self._overlay_multi_drag_var.get())
            and self._active_annotation_ax == self._ax_spec
        ):
            try:
                txt = str(self._active_annotation.get_text())
            except Exception:
                txt = ""
            if txt:
                for ann in list(self._spectrum_annotations):
                    if ann is self._active_annotation:
                        continue
                    try:
                        if str(ann.get_text()) == txt:
                            ann.set_position((float(event.xdata), float(event.ydata)))
                    except Exception:
                        continue

        # Persist UV label new position (so it survives UV redraw)
        if self._active_annotation_ax == self._ax_uv:
            key = self._uv_ann_key_by_objid.get(id(self._active_annotation))
            if key is not None:
                uv_rt, j = key
                labels_by_uvrt = self._active_uv_labels_by_uvrt(create=False)
                items = labels_by_uvrt.get(float(uv_rt))
                if items is not None and 0 <= int(j) < len(items):
                    items[int(j)].xytext = (float(event.xdata), float(event.ydata))
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _on_plot_release(self, event) -> None:
        # Finish TIC region selection
        if bool(getattr(self, "_tic_region_dragging", False)):
            self._tic_region_dragging = False
            if self._tic_region_start_rt is None or self._tic_region_end_rt is None:
                return

            rt0 = float(self._tic_region_start_rt)
            rt1 = float(self._tic_region_end_rt)
            if abs(rt1 - rt0) < 1e-3:
                # Treat as a regular click (select nearest scan) and clear span.
                try:
                    self._clear_tic_region_selection()
                except Exception:
                    pass
                try:
                    if self._filtered_rts is not None and event is not None and event.xdata is not None:
                        nearest_idx = int(np.argmin(np.abs(self._filtered_rts - float(event.xdata))))
                        self._show_spectrum_for_index(nearest_idx)
                except Exception:
                    pass
                return

            a = min(float(rt0), float(rt1))
            b = max(float(rt0), float(rt1))
            self._tic_region_active_rt = (float(a), float(b))
            self._set_tic_region_clear_btn_state(True)
            self._set_tic_region_span(float(a), float(b))
            self._compute_region_summed_spectrum(float(a), float(b))
            return

        self._active_annotation = None
        self._active_annotation_ax = None

    def _on_tic_region_select_changed(self) -> None:
        # If the user turns off the mode mid-drag, cancel the drag.
        if not bool(self.tic_region_select_var.get()):
            self._tic_region_dragging = False

    def _set_tic_region_clear_btn_state(self, enabled: bool) -> None:
        btn = getattr(self, "_tic_region_clear_btn", None)
        if btn is None:
            return
        try:
            btn.configure(state=("normal" if bool(enabled) else "disabled"))
        except Exception:
            pass

    def _clear_tic_region_selection(self, *, restore_scan: bool = True) -> None:
        # Clear span + state. Keep current scan marker as-is.
        self._tic_region_dragging = False
        self._tic_region_start_rt = None
        self._tic_region_end_rt = None
        self._tic_region_active_rt = None
        self._set_tic_region_clear_btn_state(False)
        if self._tic_region_span_artist is not None:
            try:
                self._tic_region_span_artist.remove()
            except Exception:
                pass
        self._tic_region_span_artist = None
        try:
            if self._canvas is not None:
                self._canvas.draw_idle()
        except Exception:
            pass

        # If we were viewing a region spectrum, restore the currently selected scan spectrum.
        try:
            if bool(restore_scan) and self._current_scan_index is not None and self._filtered_meta:
                self._show_spectrum_for_index(int(self._current_scan_index))
        except Exception:
            pass

    def _draw_tic_region_span(self) -> None:
        if self._ax_tic is None:
            return
        if self._tic_region_active_rt is None:
            return
        a, b = self._tic_region_active_rt
        self._set_tic_region_span(float(a), float(b))

    def _set_tic_region_span(self, rt_a: float, rt_b: float) -> None:
        if self._ax_tic is None:
            return
        a = float(min(rt_a, rt_b))
        b = float(max(rt_a, rt_b))
        if self._tic_region_span_artist is not None:
            try:
                self._tic_region_span_artist.remove()
            except Exception:
                pass
            self._tic_region_span_artist = None
        try:
            self._tic_region_span_artist = self._ax_tic.axvspan(float(a), float(b), facecolor=ACCENT_ORANGE, alpha=0.15)
        except Exception:
            self._tic_region_span_artist = None
        try:
            if self._canvas is not None:
                self._canvas.draw_idle()
        except Exception:
            pass

    def _compute_region_summed_spectrum(self, rt_a: float, rt_b: float) -> None:
        if self.mzml_path is None or self._index is None:
            return
        if not self._filtered_meta:
            return

        mzml_path = Path(self.mzml_path)
        pol_filter = str(self.polarity_var.get())

        a = float(min(rt_a, rt_b))
        b = float(max(rt_a, rt_b))

        # Snapshot scan IDs within range (uses filtered list, so polarity is respected).
        scan_ids: List[str] = []
        for m in list(self._filtered_meta):
            try:
                rt = float(m.rt_min)
                if rt < a or rt > b:
                    continue
                scan_ids.append(str(m.spectrum_id))
            except Exception:
                continue

        if not scan_ids:
            messagebox.showinfo("TIC region", "No MS1 scans in the selected region.", parent=self)
            return

        self._show_busy("Summing spectra in RT region…")

        # Bin width for region sum (Da)
        bin_w = 0.01

        def worker() -> None:
            reader = None
            try:
                reader = mzml.MzML(str(mzml_path))
                sums: Dict[int, float] = {}

                for sid in scan_ids:
                    try:
                        try:
                            spec = reader.get_by_id(str(sid))
                        except Exception:
                            spec = reader[str(sid)]
                        mz_array = spec.get("m/z array")
                        int_array = spec.get("intensity array")
                        if mz_array is None or int_array is None:
                            continue
                        mz_vals = np.asarray(mz_array, dtype=float)
                        int_vals = np.asarray(int_array, dtype=float)
                        if mz_vals.size == 0 or int_vals.size == 0:
                            continue
                        mask = np.isfinite(mz_vals) & np.isfinite(int_vals) & (int_vals > 0)
                        if not np.any(mask):
                            continue
                        mz_vals = mz_vals[mask]
                        int_vals = int_vals[mask]

                        idxs = np.rint(mz_vals / float(bin_w)).astype(np.int64)
                        if idxs.size == 0:
                            continue
                        uniq, inv = np.unique(idxs, return_inverse=True)
                        per = np.bincount(inv, weights=int_vals)
                        for u, v in zip(uniq.tolist(), per.tolist()):
                            k = int(u)
                            sums[k] = float(sums.get(k, 0.0)) + float(v)
                    except Exception:
                        continue

                if not sums:
                    mz_out = np.asarray([], dtype=float)
                    int_out = np.asarray([], dtype=float)
                else:
                    keys = np.asarray(sorted(sums.keys()), dtype=np.int64)
                    mz_out = keys.astype(float) * float(bin_w)
                    int_out = np.asarray([float(sums[int(k)]) for k in keys.tolist()], dtype=float)

                self.after(0, lambda: self._on_region_spectrum_ready(rt_a=a, rt_b=b, pol_filter=pol_filter, mz=mz_out, inten=int_out))
            except Exception as exc:
                self.after(0, lambda: self._on_region_spectrum_error(exc))
            finally:
                try:
                    if reader is not None:
                        reader.close()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _on_region_spectrum_error(self, exc: Exception) -> None:
        try:
            self._hide_busy()
        except Exception:
            pass
        messagebox.showerror("TIC region", f"Failed to sum region spectrum:\n\n{exc}", parent=self)

    def _on_region_spectrum_ready(self, *, rt_a: float, rt_b: float, pol_filter: str, mz: np.ndarray, inten: np.ndarray) -> None:
        try:
            self._hide_busy()
        except Exception:
            pass
        mz = np.asarray(mz, dtype=float)
        inten = np.asarray(inten, dtype=float)
        if mz.size == 0 or inten.size == 0:
            messagebox.showinfo("TIC region", "Region sum produced an empty spectrum.", parent=self)
            return
        if self._ax_spec is None:
            return

        pol: Optional[str] = None
        if str(pol_filter) in ("positive", "negative"):
            pol = str(pol_filter)

        sid = f"__region__{float(rt_a):.4f}-{float(rt_b):.4f}|pol={pol_filter}"
        meta = SpectrumMeta(
            spectrum_id=str(sid),
            rt_min=float((float(rt_a) + float(rt_b)) / 2.0),
            tic=float(np.sum(inten)) if inten.size else 0.0,
            polarity=pol,
            ms_level=1,
        )

        # Show region spectrum without changing current scan navigation index.
        self._current_spectrum_meta = meta
        self._current_spectrum_mz = mz
        self._current_spectrum_int = inten
        self._plot_spectrum(meta, mz, inten)

        # If requested, force polymer matching on the region spectrum.
        try:
            if bool(self._region_force_poly_match):
                self._region_force_poly_match = False
                try:
                    if not bool(self.poly_enabled_var.get()):
                        self.poly_enabled_var.set(True)
                except Exception:
                    pass
                spectrum_id = str(meta.spectrum_id)
                has_poly = any(
                    (k[0] == "poly" and str(k[1]) == spectrum_id)
                    for k in (self._spec_ann_key_by_objid.values() if isinstance(self._spec_ann_key_by_objid, dict) else [])
                    if isinstance(k, tuple) and len(k) >= 2
                )
                if not has_poly:
                    self._apply_polymer_matches(mz, inten)
                    if self._canvas is not None:
                        self._canvas.draw_idle()
        except Exception:
            pass

    def _mz_key(self, mz: float) -> float:
        try:
            return float(round(float(mz), 4))
        except Exception:
            return float(mz)

    def _open_label_editor(self, ann) -> None:
        """Edit/delete a clicked label in spectrum or UV."""
        if ann is None:
            return

        # Determine mapping
        uv_key = self._uv_ann_key_by_objid.get(id(ann))
        spec_key = self._spec_ann_key_by_objid.get(id(ann))

        if uv_key is None and spec_key is None:
            return

        current_text = ""
        if uv_key is not None:
            try:
                uv_rt, j = uv_key
                labels_by_uvrt = self._active_uv_labels_by_uvrt(create=False)
                items = labels_by_uvrt.get(float(uv_rt), [])
                if 0 <= int(j) < len(items):
                    current_text = str(getattr(items[int(j)], "text", ""))
            except Exception:
                current_text = ""
        if not current_text:
            try:
                current_text = str(ann.get_text())
                # If this was a UV label that includes a confidence suffix, strip it for editing.
                if "  [" in current_text and current_text.rstrip().endswith("%]"):
                    current_text = current_text.split("  [", 1)[0].rstrip()
            except Exception:
                current_text = ""

        dlg = tk.Toplevel(self)
        dlg.title("Edit label")
        dlg.resizable(False, False)
        dlg.transient(self)

        pad = 10
        frm = ttk.Frame(dlg, padding=pad)
        frm.grid(row=0, column=0)

        ttk.Label(frm, text="Label text").grid(row=0, column=0, sticky="w")
        text_var = tk.StringVar(value=current_text)
        ent = ttk.Entry(frm, textvariable=text_var, width=54)
        ent.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        try:
            ent.focus_set()
            ent.selection_range(0, tk.END)
        except Exception:
            pass

        def do_apply() -> None:
            new_text = (text_var.get() or "").strip()
            if not new_text:
                messagebox.showerror("Invalid", "Label cannot be empty (use Delete to remove).", parent=dlg)
                return

            if uv_key is not None:
                uv_rt, j = uv_key
                labels_by_uvrt = self._active_uv_labels_by_uvrt(create=True)
                items = labels_by_uvrt.get(float(uv_rt), [])
                if 0 <= int(j) < len(items):
                    items[int(j)].text = new_text
                    self._plot_uv()
                dlg.destroy()
                return

            if spec_key is not None:
                kind = str(spec_key[0])
                spectrum_id = str(spec_key[1])
                if kind == "custom":
                    _kind, _sid, idx = spec_key
                    items = self._custom_labels_by_spectrum.get(spectrum_id, [])
                    if 0 <= int(idx) < len(items):
                        old = items[int(idx)]
                        items[int(idx)] = CustomLabel(label=new_text, mz=float(old.mz), snap_to_nearest_peak=bool(old.snap_to_nearest_peak))
                        self._redraw_spectrum_only()
                    dlg.destroy()
                    return

                if kind == "auto":
                    _kind, _sid, mz_key = spec_key
                    self._spec_label_overrides.setdefault(spectrum_id, {})[("auto", float(mz_key))] = new_text
                    self._redraw_spectrum_only()
                    dlg.destroy()
                    return

                if kind == "poly":
                    _kind, _sid, poly_kind, mz_key = spec_key
                    self._spec_label_overrides.setdefault(spectrum_id, {})[(str(poly_kind), float(mz_key))] = new_text
                    self._redraw_spectrum_only()
                    dlg.destroy()
                    return

        def do_delete() -> None:
            if uv_key is not None:
                uv_rt, j = uv_key
                labels_by_uvrt = self._active_uv_labels_by_uvrt(create=True)
                items = labels_by_uvrt.get(float(uv_rt), [])
                if 0 <= int(j) < len(items):
                    items.pop(int(j))
                    if not items:
                        labels_by_uvrt.pop(float(uv_rt), None)
                    self._plot_uv()
                dlg.destroy()
                return

            if spec_key is not None:
                kind = str(spec_key[0])
                spectrum_id = str(spec_key[1])
                if kind == "custom":
                    _kind, _sid, idx = spec_key
                    items = self._custom_labels_by_spectrum.get(spectrum_id, [])
                    if 0 <= int(idx) < len(items):
                        items.pop(int(idx))
                        if not items:
                            self._custom_labels_by_spectrum.pop(spectrum_id, None)
                        self._redraw_spectrum_only()
                    dlg.destroy()
                    return

                if kind == "auto":
                    _kind, _sid, mz_key = spec_key
                    self._spec_label_overrides.setdefault(spectrum_id, {})[("auto", float(mz_key))] = None
                    self._redraw_spectrum_only()
                    dlg.destroy()
                    return

                if kind == "poly":
                    _kind, _sid, poly_kind, mz_key = spec_key
                    self._spec_label_overrides.setdefault(spectrum_id, {})[(str(poly_kind), float(mz_key))] = None
                    self._redraw_spectrum_only()
                    dlg.destroy()
                    return

        ttk.Button(frm, text="Apply", command=do_apply).grid(row=2, column=0, sticky="e", pady=(10, 0), padx=(0, 8))
        ttk.Button(frm, text="Explain…", command=lambda: self._open_label_explanation(ann)).grid(
            row=2, column=1, sticky="e", pady=(10, 0), padx=(0, 8)
        )
        ttk.Button(frm, text="Delete", command=do_delete).grid(row=2, column=2, sticky="e", pady=(10, 0), padx=(0, 8))
        ttk.Button(frm, text="Cancel", command=dlg.destroy).grid(row=2, column=3, sticky="e", pady=(10, 0))

    def _open_label_explanation(self, ann) -> None:
        if ann is None:
            return
        text = self._build_label_explanation_text(ann)

        win = getattr(self, "_label_explain_win", None)
        if win is not None:
            try:
                if bool(win.winfo_exists()):
                    win.deiconify()
                    win.lift()
                    try:
                        win.focus_force()
                    except Exception:
                        pass
                    try:
                        win.set_content(text)
                    except Exception:
                        pass
                    return
            except Exception:
                pass

        win = LabelExplanationWindow(self, title="Explain label")
        self._label_explain_win = win
        try:
            win.set_content(text)
        except Exception:
            pass

    def _build_label_explanation_text(self, ann) -> str:
        # Determine mapping
        uv_key = self._uv_ann_key_by_objid.get(id(ann))
        spec_key = self._spec_ann_key_by_objid.get(id(ann))

        def safe_get_text() -> str:
            try:
                t = str(ann.get_text() or "")
            except Exception:
                return ""
            # Strip UV confidence suffix for readability.
            if "  [" in t and t.rstrip().endswith("%]"):
                try:
                    return t.split("  [", 1)[0].rstrip()
                except Exception:
                    return t
            return t

        label_text = safe_get_text()

        lines: List[str] = []
        lines.append("Explain this label")
        lines.append("=" * 60)
        lines.append(f"Label text: {label_text}")

        # UV label explanation
        if uv_key is not None:
            try:
                uv_rt, j = uv_key
            except Exception:
                uv_rt, j = (None, None)

            st: Optional[UVLabelState] = None
            try:
                if uv_rt is not None and j is not None:
                    labels_by_uvrt = self._active_uv_labels_by_uvrt(create=False)
                    items = labels_by_uvrt.get(float(uv_rt), [])
                    if 0 <= int(j) < len(items):
                        st = items[int(j)]
            except Exception:
                st = None

            lines.append("Label type: UV label")
            if uv_rt is not None:
                lines.append(f"UV RT anchor: {float(uv_rt):.4f} min")
                try:
                    ms_rt = float(self._map_uv_to_ms_rt(float(uv_rt)))
                    lines.append(f"Linked MS RT (current offset/auto-align): {ms_rt:.4f} min")
                except Exception:
                    pass

            if st is not None:
                conf = float(getattr(st, "confidence", 0.0) or 0.0)
                rt_delta = float(getattr(st, "rt_delta_min", 0.0) or 0.0)
                uv_score = float(getattr(st, "uv_peak_score", 0.0) or 0.0)
                tic_score = float(getattr(st, "tic_peak_score", 0.0) or 0.0)
                if conf > 0.0 or rt_delta != 0.0 or uv_score != 0.0 or tic_score != 0.0:
                    lines.append("Label origin: UV-transferred (MS→UV)")
                    lines.append(f"Confidence: {conf:.1f}%")
                    lines.append(f"RT delta (UV vs mapped MS): {rt_delta:+.4f} min")
                    lines.append(f"UV peak score: {uv_score:.3f}")
                    lines.append(f"TIC peak score: {tic_score:.3f}")
                else:
                    lines.append("Label origin: UV label (manual/unknown)")

            try:
                min_conf = float(self.uv_label_min_conf_var.get())
                lines.append(f"UV min-confidence filter: {min_conf:g}%")
            except Exception:
                pass

            return "\n".join(lines) + "\n"

        # Spectrum label explanation
        if spec_key is None:
            lines.append("Label type: (unknown)")
            return "\n".join(lines) + "\n"

        kind = str(spec_key[0])
        spectrum_id = str(spec_key[1]) if len(spec_key) > 1 else ""
        lines.append(f"Label type: {('custom' if kind == 'custom' else ('auto' if kind == 'auto' else 'polymer'))}")

        # Observed point
        try:
            mz_obs = float(ann.xy[0])
            inten_obs = float(ann.xy[1])
            lines.append(f"Observed m/z: {mz_obs:.6f}")
            lines.append(f"Observed intensity: {inten_obs:.6g}")
        except Exception:
            mz_obs = None
            inten_obs = None

        # RT/meta lookup
        meta_obj: Optional[SpectrumMeta] = None
        try:
            if self._current_spectrum_meta is not None and str(self._current_spectrum_meta.spectrum_id) == spectrum_id:
                meta_obj = self._current_spectrum_meta
            else:
                for m in (self._filtered_meta or []):
                    if str(m.spectrum_id) == spectrum_id:
                        meta_obj = m
                        break
        except Exception:
            meta_obj = None

        if spectrum_id:
            lines.append(f"spectrum_id: {spectrum_id}")
        if meta_obj is not None:
            lines.append(f"RT (min): {float(meta_obj.rt_min):.4f}")
            lines.append(f"Polarity: {str(meta_obj.polarity or '')}")

        # Grab arrays (best-effort)
        mz_vals = None
        int_vals = None
        try:
            if self._current_spectrum_meta is not None and str(self._current_spectrum_meta.spectrum_id) == spectrum_id:
                mz_vals = (None if self._current_spectrum_mz is None else np.asarray(self._current_spectrum_mz, dtype=float))
                int_vals = (None if self._current_spectrum_int is None else np.asarray(self._current_spectrum_int, dtype=float))
        except Exception:
            mz_vals, int_vals = (None, None)
        if mz_vals is None or int_vals is None:
            try:
                spec = self._get_spectrum_by_id(spectrum_id)
                mz_array = spec.get("m/z array")
                int_array = spec.get("intensity array")
                if mz_array is not None and int_array is not None:
                    mz_vals = np.asarray(mz_array, dtype=float)
                    int_vals = np.asarray(int_array, dtype=float)
            except Exception:
                mz_vals, int_vals = (None, None)

        # Explain by kind
        if kind == "auto":
            lines.append("")
            lines.append("Auto label details")
            lines.append("-" * 60)
            try:
                top_n = int(self.annotate_top_n_var.get())
            except Exception:
                top_n = 0
            try:
                min_rel = float(self.annotate_min_rel_var.get())
            except Exception:
                min_rel = 0.0
            lines.append(f"Criteria: top-N={top_n} (0 means no limit), min_rel={min_rel:g}")

            if mz_vals is not None and int_vals is not None and int_vals.size:
                try:
                    max_int = float(np.max(int_vals))
                    thr = float(min_rel) * float(max_int)
                    lines.append(f"Spectrum max intensity: {max_int:.6g}")
                    lines.append(f"Min intensity threshold: {thr:.6g}")
                    if inten_obs is not None and max_int > 0:
                        lines.append(f"Observed rel intensity: {(float(inten_obs)/max_int):.4f}")
                except Exception:
                    pass
            lines.append("Result: passed (label is drawn)")

            try:
                _kind, _sid, mz_key = spec_key
                mz_key_f = float(mz_key)
                overrides = self._spec_label_overrides.get(spectrum_id, {})
                ov_key = ("auto", mz_key_f)
                if ov_key in overrides:
                    ov = overrides.get(ov_key)
                    if ov is None:
                        lines.append("Override state: suppressed (would not be drawn)")
                    else:
                        lines.append(f"Override state: text overridden to: {str(ov)}")
                else:
                    lines.append("Override state: none")
            except Exception:
                pass

            return "\n".join(lines) + "\n"

        if kind == "custom":
            lines.append("")
            lines.append("Custom label details")
            lines.append("-" * 60)
            try:
                _kind, _sid, idx = spec_key
                idx = int(idx)
            except Exception:
                idx = -1
            item: Optional[CustomLabel] = None
            try:
                items = self._custom_labels_by_spectrum.get(spectrum_id, [])
                if 0 <= idx < len(items):
                    item = items[idx]
            except Exception:
                item = None
            if item is not None:
                lines.append(f"Target m/z: {float(item.mz):.6f}")
                lines.append(f"Snap rule: {'nearest peak' if bool(item.snap_to_nearest_peak) else 'no snap'}")
                if bool(item.snap_to_nearest_peak) and mz_vals is not None and int_vals is not None and mz_vals.size and int_vals.size:
                    try:
                        found = self._find_nearest_peak(mz_vals, int_vals, float(item.mz))
                        if found is not None:
                            mz_use, inten_use = found
                            lines.append(f"Nearest peak m/z: {float(mz_use):.6f}")
                            lines.append(f"Nearest peak intensity: {float(inten_use):.6g}")
                            lines.append(f"Nearest-peak distance: {abs(float(mz_use) - float(item.mz)):.6f} Da")
                    except Exception:
                        pass
            else:
                lines.append("Custom label source: not found (index changed?)")

            return "\n".join(lines) + "\n"

        # Polymer label (spec_key is ("poly", spectrum_id, poly_kind, mz_key))
        poly_kind = "poly"
        mz_key_val: Optional[float] = None
        try:
            _kind, _sid, poly_kind, mz_key_val = spec_key
            poly_kind = str(poly_kind)
            mz_key_val = float(mz_key_val)
        except Exception:
            poly_kind = "poly"
            mz_key_val = None

        lines.append("")
        lines.append("Polymer label details")
        lines.append("-" * 60)
        lines.append(f"Polymer kind: {poly_kind}")

        # Overrides
        try:
            if mz_key_val is not None:
                overrides = self._spec_label_overrides.get(spectrum_id, {})
                ov_key = (str(poly_kind), float(mz_key_val))
                if ov_key in overrides:
                    ov = overrides.get(ov_key)
                    if ov is None:
                        lines.append("Override state: suppressed (would not be drawn)")
                    else:
                        lines.append(f"Override state: text overridden to: {str(ov)}")
                else:
                    lines.append("Override state: none")
        except Exception:
            pass

        # Best-effort recomputation for predicted m/z + error (current settings)
        if mz_vals is not None and int_vals is not None and mz_vals.size and int_vals.size and mz_obs is not None:
            try:
                order = np.argsort(mz_vals)
                mz_s = np.asarray(mz_vals[order], dtype=float)
                int_s = np.asarray(int_vals[order], dtype=float)
                peak_i = int(np.argmin(np.abs(mz_s - float(mz_obs))))
                pol = (meta_obj.polarity if meta_obj is not None else (self._current_spectrum_meta.polarity if self._current_spectrum_meta is not None else None))
                st = self._snapshot_labeling_settings()

                info = self._explain_polymer_best_match_for_peak(mz_s, int_s, peak_i, str(poly_kind), polarity=pol, settings=st)
                if info is not None:
                    lines.append(f"Predicted m/z: {float(info['mz_pred']):.6f}")
                    lines.append(f"Tolerance: {float(info['tol_value']):g} {str(info['tol_unit'])}")
                    lines.append(f"Absolute error: {float(info['abs_err']):.6f} Da")
                    lines.append(f"Charge: z={int(info['z'])}")
                    lines.append(f"Adduct: {str(info['adduct_label']) or '(default)'} ({float(info['adduct_mass']):+.6f})")
                    lines.append(f"Min rel intensity: {float(info['min_rel']):g}")
                    lines.append(f"Composition: {str(info['composition'])}")
                    lines.append("Best match at this peak: yes")
                else:
                    lines.append("Best match at this peak: not found (with current settings)")
            except Exception:
                lines.append("Best match at this peak: (error computing)")

        return "\n".join(lines) + "\n"

    def _explain_polymer_best_match_for_peak(
        self,
        mz_s: np.ndarray,
        int_s: np.ndarray,
        peak_i: int,
        kind: str,
        *,
        polarity: Optional[str],
        settings: LabelingSettings,
    ) -> Optional[Dict[str, Any]]:
        st = settings
        if not bool(st.poly_enabled):
            return None

        monomers = self._parse_monomers(st.poly_monomers_text)
        if not monomers:
            return None

        charges = self._parse_charges(st.poly_charges_text)
        if not charges:
            charges = [1]

        max_dp = max(1, min(200, int(st.poly_max_dp)))

        bond_delta = float(st.poly_bond_delta)
        extra_delta = float(st.poly_extra_delta)
        adduct_mass = float(st.poly_adduct_mass)
        decarb_enabled = bool(st.poly_decarb_enabled)
        oxid_enabled = bool(st.poly_oxid_enabled)
        cluster_enabled = bool(st.poly_cluster_enabled)
        cluster_adduct_mass = float(st.poly_cluster_adduct_mass)

        pol = polarity
        if pol in ("positive", "negative"):
            h = 1.007276
            sign = 1.0 if pol == "positive" else -1.0
            if abs(abs(adduct_mass) - h) <= 0.01:
                adduct_mass = sign * abs(adduct_mass)
            if abs(abs(cluster_adduct_mass) - h) <= 0.01:
                cluster_adduct_mass = sign * abs(cluster_adduct_mass)

        compat = bool(str(os.environ.get("LAB_GUI_POLYMER_COMPAT", "")).strip())

        try:
            info = poly_match.explain_best_match_for_peak_sorted(
                mz_s,
                int_s,
                peak_i=int(peak_i),
                target_kind=str(kind),
                monomer_names=[n for n, _m in monomers],
                monomer_masses=[m for _n, m in monomers],
                charges=list(charges),
                max_dp=int(max_dp),
                bond_delta=float(bond_delta),
                extra_delta=float(extra_delta),
                polarity=pol,
                base_adduct_mass=float(adduct_mass),
                enable_decarb=bool(decarb_enabled),
                enable_oxid=bool(oxid_enabled),
                enable_cluster=bool(cluster_enabled),
                cluster_adduct_mass=float(cluster_adduct_mass),
                enable_na=bool(st.poly_adduct_na),
                enable_k=bool(st.poly_adduct_k),
                enable_cl=bool(st.poly_adduct_cl),
                enable_formate=bool(st.poly_adduct_formate),
                enable_acetate=bool(getattr(st, "poly_adduct_acetate", False)),
                tol_value=float(st.poly_tol_value),
                tol_unit=str(st.poly_tol_unit or "Da"),
                min_rel_int=float(st.poly_min_rel_int),
                allow_variant_combo=True,
                compatibility_mode=bool(compat),
            )
            return (None if info is None else dict(info))
        except poly_match.PolymerSearchTooLarge:
            return None
        except Exception:
            return None

    def _get_spectrum_by_id(self, spectrum_id: str) -> Dict[str, Any]:
        if self._reader is None:
            raise RuntimeError("No mzML reader available")
        try:
            return self._reader.get_by_id(spectrum_id)
        except Exception:
            # Fallback: __getitem__ by id
            return self._reader[spectrum_id]

    def _show_spectrum_for_index(self, idx: int) -> None:
        if self._canvas is None:
            return
        if not self._filtered_meta:
            return
        idx = max(0, min(idx, len(self._filtered_meta) - 1))

        meta = self._filtered_meta[idx]
        self._current_scan_index = int(idx)
        # When the selection is driven by the MS chromatogram/navigation,
        # show the corresponding UV time (auto-align if enabled; otherwise fixed offset).
        self._selected_rt_min = float(self._map_ms_to_uv_rt(float(meta.rt_min)))

        # Overlay mode: select RT and draw overlaid spectra across datasets.
        if self._is_overlay_active():
            self._overlay_selected_ms_rt = float(meta.rt_min)
            self._plot_overlay_spectrum_for_rt(float(meta.rt_min))
            self._refresh_overlay_tic()
            self._plot_uv()
            self._update_status_current()
            self._notify_ms_position_changed()
            return

        # Update marker
        if self._tic_marker is not None:
            try:
                self._tic_marker.remove()
            except Exception:
                pass
        if self._ax_tic is not None:
            self._tic_marker = self._ax_tic.scatter([meta.rt_min], [meta.tic], s=35)

        try:
            spectrum = self._get_spectrum_by_id(meta.spectrum_id)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load spectrum:\n{exc}")
            return

        mz_array = spectrum.get("m/z array")
        int_array = spectrum.get("intensity array")
        if mz_array is None or int_array is None:
            messagebox.showerror("Error", "Selected spectrum has no m/z or intensity array.")
            return

        mz_vals = np.asarray(mz_array, dtype=float)
        int_vals = np.asarray(int_array, dtype=float)

        self._current_spectrum_meta = meta
        self._current_spectrum_mz = mz_vals
        self._current_spectrum_int = int_vals
        if self._ax_spec is not None:
            self._plot_spectrum(meta, mz_vals, int_vals)
        # Update UV marker (if UV panel is visible)
        self._plot_uv()
        self._update_status_current()

        # Notify any non-modal windows that track the current MS RT (e.g., EIC)
        self._notify_ms_position_changed()

        # Note: UV label transfer is triggered by user RT clicks (not here),
        # so the initial auto-loaded spectrum doesn't clutter the UV plot.

    def _top_peaks_by_intensity(self, mz_vals: np.ndarray, int_vals: np.ndarray, top_n: int) -> List[float]:
        if mz_vals.size == 0 or int_vals.size == 0 or top_n <= 0:
            return []
        mz = np.asarray(mz_vals, dtype=float)
        inten = np.asarray(int_vals, dtype=float)
        mask = np.isfinite(mz) & np.isfinite(inten) & (inten > 0)
        if not np.any(mask):
            return []
        mz = mz[mask]
        inten = inten[mask]
        if mz.size == 0:
            return []
        n = int(min(int(top_n), int(mz.size)))
        order = np.argsort(inten)[::-1][:n]
        return [float(v) for v in mz[order].tolist()]

    def _top_peaks_sorted_indices(self, mz_vals_sorted: np.ndarray, int_vals_sorted: np.ndarray, top_n: int) -> List[int]:
        if mz_vals_sorted.size == 0 or int_vals_sorted.size == 0 or top_n <= 0:
            return []
        mz_s = np.asarray(mz_vals_sorted, dtype=float)
        int_s = np.asarray(int_vals_sorted, dtype=float)
        mask = np.isfinite(mz_s) & np.isfinite(int_s) & (int_s > 0)
        if not np.any(mask):
            return []
        mz_s = mz_s[mask]
        int_s = int_s[mask]
        n = int(min(int(top_n), int(mz_s.size)))
        top_idx = np.argsort(int_s)[::-1][:n]
        return [int(i) for i in top_idx.tolist()]

    def _format_uv_id_label(self, label: str) -> str:
        s = (label or "").strip()
        if " z=" in s:
            s = s.split(" z=", 1)[0].rstrip()
        return s

    def _compute_polymer_best_by_peak_sorted(
        self,
        mz_s: np.ndarray,
        int_s: np.ndarray,
        *,
        polarity: Optional[str] = None,
        settings: Optional[LabelingSettings] = None,
    ) -> Dict[int, Dict[str, Tuple[float, str, float, float]]]:
        """Compute polymer/decarb/2M matches; keys are indices into the given m/z-sorted arrays."""
        st = settings if settings is not None else self._snapshot_labeling_settings()
        if not bool(st.poly_enabled):
            return {}
        if mz_s.size == 0 or int_s.size == 0:
            return {}

        monomers = self._parse_monomers(st.poly_monomers_text)
        if not monomers:
            return {}

        charges = self._parse_charges(st.poly_charges_text)
        if not charges:
            charges = [1]

        max_dp = max(1, min(200, int(st.poly_max_dp)))

        bond_delta = float(st.poly_bond_delta)
        extra_delta = float(st.poly_extra_delta)
        adduct_mass = float(st.poly_adduct_mass)
        decarb_enabled = bool(st.poly_decarb_enabled)
        oxid_enabled = bool(st.poly_oxid_enabled)
        cluster_enabled = bool(st.poly_cluster_enabled)
        cluster_adduct_mass = float(st.poly_cluster_adduct_mass)

        pol = polarity
        if pol is None and self._current_spectrum_meta is not None:
            pol = self._current_spectrum_meta.polarity

        if pol in ("positive", "negative"):
            h = 1.007276
            sign = 1.0 if pol == "positive" else -1.0
            if abs(abs(adduct_mass) - h) <= 0.01:
                adduct_mass = sign * abs(adduct_mass)
            if abs(abs(cluster_adduct_mass) - h) <= 0.01:
                cluster_adduct_mass = sign * abs(cluster_adduct_mass)

        compat = bool(str(os.environ.get("LAB_GUI_POLYMER_COMPAT", "")).strip())

        try:
            return poly_match.compute_polymer_best_by_peak_sorted(
                mz_s,
                int_s,
                monomer_names=[n for n, _m in monomers],
                monomer_masses=[m for _n, m in monomers],
                charges=list(charges),
                max_dp=int(max_dp),
                bond_delta=float(bond_delta),
                extra_delta=float(extra_delta),
                polarity=pol,
                base_adduct_mass=float(adduct_mass),
                enable_decarb=bool(decarb_enabled),
                enable_oxid=bool(oxid_enabled),
                enable_cluster=bool(cluster_enabled),
                cluster_adduct_mass=float(cluster_adduct_mass),
                enable_na=bool(st.poly_adduct_na),
                enable_k=bool(st.poly_adduct_k),
                enable_cl=bool(st.poly_adduct_cl),
                enable_formate=bool(st.poly_adduct_formate),
                enable_acetate=bool(getattr(st, "poly_adduct_acetate", False)),
                tol_value=float(st.poly_tol_value),
                tol_unit=str(st.poly_tol_unit or "Da"),
                min_rel_int=float(st.poly_min_rel_int),
                allow_variant_combo=True,
                compatibility_mode=bool(compat),
            )
        except poly_match.PolymerSearchTooLarge:
            return {}
        except Exception:
            return {}

    def _maybe_store_uv_ms_labels_for_current_spectrum(self, *, anchor_rt_min: Optional[float]) -> None:
        if not bool(self.uv_label_from_ms_var.get()):
            return
        x, y = self._active_uv_xy()
        if x is None or y is None or x.size == 0:
            return
        if self._current_spectrum_mz is None or self._current_spectrum_int is None:
            return

        meta = self._current_spectrum_meta
        if anchor_rt_min is None:
            if meta is None:
                return
            # Default to UV RT corresponding to the current MS scan.
            anchor_rt_min = float(self._map_ms_to_uv_rt(float(meta.rt_min)))
        if meta is None:
            return
        ms_rt = float(meta.rt_min)

        try:
            top_n = int(self.uv_label_from_ms_top_n_var.get())
        except Exception:
            top_n = 3
        if top_n not in (2, 3):
            top_n = 3

        mz_vals = np.asarray(self._current_spectrum_mz, dtype=float)
        int_vals = np.asarray(self._current_spectrum_int, dtype=float)
        if mz_vals.size == 0 or int_vals.size == 0:
            return

        order_mz = np.argsort(mz_vals)
        mz_s = mz_vals[order_mz]
        int_s = int_vals[order_mz]

        top_peak_is = self._top_peaks_sorted_indices(mz_s, int_s, top_n)
        if not top_peak_is:
            return

        pol = (self._current_spectrum_meta.polarity if self._current_spectrum_meta is not None else None)
        best_by_peak = self._compute_polymer_best_by_peak_sorted(mz_s, int_s, polarity=pol)

        kind_order = ["poly", "ox", "decarb", "oxdecarb", "2m"]
        texts: List[str] = []
        for peak_i in top_peak_is:
            kinds = best_by_peak.get(int(peak_i))
            if not kinds:
                continue
            label = None
            for k in kind_order:
                if k in kinds:
                    label = kinds[k][1]
                    break
            if label is None:
                label = next(iter(kinds.values()))[1]
            texts.append(self._format_uv_id_label(str(label)))

        # Include any snapped custom labels that land on the same top peaks
        meta = self._current_spectrum_meta
        spec_key = str(meta.spectrum_id) if meta is not None else "__no_spectrum__"
        custom_items = self._custom_labels_by_spectrum.get(spec_key, [])
        if custom_items:
            for item in custom_items:
                if not item.label or not bool(item.snap_to_nearest_peak):
                    continue
                found = self._find_nearest_peak(mz_s, int_s, float(item.mz))
                if found is None:
                    continue
                mz_use, _inten_use = found
                snapped_i = int(np.argmin(np.abs(mz_s - float(mz_use))))
                if snapped_i in top_peak_is:
                    texts.append(str(item.label).strip())

        # de-dup
        seen: set[str] = set()
        texts = [t for t in (t.strip() for t in texts) if t and not (t in seen or seen.add(t))]
        if not texts:
            return

        # Snap anchor to the nearest UV x point (stable dict keys, exact plotted x).
        uv_i = int(np.argmin(np.abs(x - float(anchor_rt_min))))
        uv_rt = float(x[uv_i])

        conf_meta = self._compute_confidence_for_uv_label(float(uv_rt), float(ms_rt))

        labels_by_uvrt = self._active_uv_labels_by_uvrt(create=True)
        prev_states = labels_by_uvrt.get(uv_rt, [])
        prev_xy_by_text = {st.text: st.xytext for st in prev_states}

        y0 = float(y[uv_i])
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        span = (y_max - y_min) if (y_max != y_min) else max(1.0, abs(y_max))
        y_step = 0.06 * span

        states: List[UVLabelState] = []
        for j, text in enumerate(texts[:top_n]):
            if text in prev_xy_by_text:
                xytext = prev_xy_by_text[text]
            else:
                xytext = (float(uv_rt), float(y0 + (j + 1) * y_step))
            states.append(
                UVLabelState(
                    text=str(text),
                    xytext=(float(xytext[0]), float(xytext[1])),
                    confidence=float(conf_meta.get("confidence", 0.0)),
                    rt_delta_min=float(conf_meta.get("rt_delta_min", 0.0)),
                    uv_peak_score=float(conf_meta.get("uv_peak_score", 0.0)),
                    tic_peak_score=float(conf_meta.get("tic_peak_score", 0.0)),
                )
            )

        labels_by_uvrt[uv_rt] = states

        # Redraw UV to show updated labels.
        self._plot_uv()

    def _draw_uv_ms_peak_labels(self) -> None:
        if self._ax_uv is None:
            return
        x, y = self._active_uv_xy()
        if x is None or y is None or x.size == 0:
            return
        labels_by_uvrt = self._active_uv_labels_by_uvrt(create=False)
        if not labels_by_uvrt:
            return

        # Remove any previously created UV annotation artists
        for ann in getattr(self, "_uv_annotations", []):
            try:
                ann.remove()
            except Exception:
                pass
        self._uv_annotations = []
        self._uv_ann_key_by_objid = {}

        fs = max(6, int(self.tick_fontsize_var.get()) - 1)
        try:
            min_conf = float(self.uv_label_min_conf_var.get())
        except Exception:
            min_conf = 0.0
        min_conf = max(0.0, min(100.0, float(min_conf)))
        for uv_rt, states in sorted(labels_by_uvrt.items(), key=lambda kv: float(kv[0])):
            uv_i = int(np.argmin(np.abs(x - float(uv_rt))))
            x0 = float(x[uv_i])
            y0 = float(y[uv_i])

            drawn = 0
            for j, st in enumerate(list(states)):
                if drawn >= 3:
                    break
                conf = float(getattr(st, "confidence", 0.0) or 0.0)
                if float(conf) < float(min_conf):
                    continue
                display_text = self._format_uv_label_display_text(st)
                ann = self._ax_uv.annotate(
                    str(display_text),
                    xy=(float(x0), float(y0)),
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
                self._uv_annotations.append(ann)
                self._uv_ann_key_by_objid[id(ann)] = (float(uv_rt), int(j))
                drawn += 1

    def _plot_spectrum(self, meta: SpectrumMeta, mz_vals: np.ndarray, int_vals: np.ndarray) -> None:
        if self._ax_spec is None or self._canvas is None:
            return

        self._ax_spec.clear()
        self._clear_spectrum_annotations()
        base_title = (self.spec_title_var.get() or "Spectrum (MS1)").strip()
        self._ax_spec.set_title(f"{base_title} at RT={meta.rt_min:.4f} min | {meta.polarity or 'unknown'} | id={meta.spectrum_id}")
        self._ax_spec.set_xlabel(self.spec_xlabel_var.get())
        self._ax_spec.set_ylabel(self.spec_ylabel_var.get())

        # Stick plot
        self._ax_spec.vlines(mz_vals, 0.0, int_vals, linewidth=0.8, color=PRIMARY_TEAL)

        if bool(self.annotate_peaks_var.get()):
            self._annotate_peaks(mz_vals, int_vals)

        # Apply custom labels and polymer matching (if enabled)
        self._apply_custom_labels(mz_vals, int_vals)
        self._apply_polymer_matches(mz_vals, int_vals)

        self._apply_plot_style()
        self._canvas.draw()

    def _annotate_peaks(self, mz_vals: np.ndarray, int_vals: np.ndarray) -> None:
        if self._ax_spec is None:
            return
        if mz_vals.size == 0 or int_vals.size == 0:
            return

        # Filter peaks by relative intensity threshold
        max_int = float(np.max(int_vals)) if int_vals.size else 0.0
        if max_int <= 0:
            return

        try:
            min_rel = float(self.annotate_min_rel_var.get())
        except Exception:
            min_rel = 0.0
        min_rel = max(0.0, min(1.0, min_rel))
        mask = int_vals >= (min_rel * max_int)

        mz_f = mz_vals[mask]
        int_f = int_vals[mask]
        if mz_f.size == 0:
            return

        # Take top N by intensity (if N > 0)
        top_n = int(self.annotate_top_n_var.get())
        if top_n > 0 and int_f.size > top_n:
            order = np.argsort(int_f)[::-1][:top_n]
            mz_f = mz_f[order]
            int_f = int_f[order]

        # Leader line + draggable label (optional)
        y_offset = 0.06 * max_int
        meta = self._current_spectrum_meta
        spectrum_id = str(meta.spectrum_id) if meta is not None else "__no_spectrum__"
        overrides = self._spec_label_overrides.get(spectrum_id, {})
        for mz_v, inten_v in zip(mz_f.tolist(), int_f.tolist()):
            inten_v = float(inten_v)
            y_text = inten_v + y_offset
            mz_key = self._mz_key(float(mz_v))
            ov_key = ("auto", float(mz_key))
            ov = overrides.get(ov_key)
            if ov is None and ov_key in overrides:
                continue
            ann = self._ax_spec.annotate(
                (str(ov) if isinstance(ov, str) else f"{float(mz_v):.4f}"),
                xy=(float(mz_v), float(inten_v)),
                xytext=(float(mz_v), float(y_text)),
                textcoords="data",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=max(6, int(self.tick_fontsize_var.get()) - 1),
                arrowprops={"arrowstyle": "-", "lw": 0.8},
                clip_on=True,
            )
            # Enable click-to-drag hit testing
            try:
                ann.set_picker(True)
            except Exception:
                pass
            self._spectrum_annotations.append(ann)
            self._spec_ann_key_by_objid[id(ann)] = ("auto", spectrum_id, float(mz_key))

    def _clear_spectrum_annotations(self) -> None:
        # Remove any previously created annotations (auto/custom/polymer)
        for ann in getattr(self, "_spectrum_annotations", []):
            try:
                ann.remove()
            except Exception:
                pass
        self._spectrum_annotations = []
        self._spec_ann_key_by_objid = {}

    def _find_nearest_peak(self, mz_vals: np.ndarray, int_vals: np.ndarray, target_mz: float) -> Optional[Tuple[float, float]]:
        if mz_vals.size == 0:
            return None
        # Conservative snapping: treat as "nearest" using a small default window.
        # This keeps the UX stable but avoids missing peaks in unsorted arrays.
        hit = poly_match.find_best_peak_match(mz_vals, int_vals, float(target_mz), tol_da=0.05, tol_ppm=None)
        if hit is None:
            # Fallback: nearest in Da (no window) if tolerance misses.
            order = np.argsort(mz_vals)
            mz_s = mz_vals[order]
            int_s = int_vals[order]
            i = int(np.searchsorted(mz_s, float(target_mz)))
            candidates = []
            if 0 <= i < mz_s.size:
                candidates.append(i)
            if i - 1 >= 0:
                candidates.append(i - 1)
            if i + 1 < mz_s.size:
                candidates.append(i + 1)
            if not candidates:
                return None
            best_i = min(candidates, key=lambda j: abs(float(mz_s[j]) - float(target_mz)))
            return float(mz_s[best_i]), float(int_s[best_i])
        return float(hit.matched_mz), float(hit.intensity)

    def _apply_custom_labels(self, mz_vals: np.ndarray, int_vals: np.ndarray) -> None:
        if self._ax_spec is None:
            return
        meta = self._current_spectrum_meta
        key = str(meta.spectrum_id) if meta is not None else "__no_spectrum__"
        items = self._custom_labels_by_spectrum.get(key, [])
        if not items:
            return
        if mz_vals.size == 0 or int_vals.size == 0:
            return

        max_int = float(np.max(int_vals)) if int_vals.size else 0.0
        if max_int <= 0:
            return

        y_offset = 0.08 * max_int
        for idx, item in enumerate(list(items)):
            mz_target = float(item.mz)
            label = str(item.label)
            if not label:
                continue

            if bool(item.snap_to_nearest_peak):
                found = self._find_nearest_peak(mz_vals, int_vals, mz_target)
                if found is None:
                    continue
                mz_use, inten_use = found
            else:
                mz_use = mz_target
                inten_use = 0.0

            y_text = (inten_use if inten_use > 0 else max_int * 0.05) + y_offset
            ann = self._ax_spec.annotate(
                label,
                xy=(float(mz_use), float(inten_use if inten_use > 0 else 0.0)),
                xytext=(float(mz_use), float(y_text)),
                textcoords="data",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=max(6, int(self.tick_fontsize_var.get()) - 1),
                arrowprops={"arrowstyle": "-", "lw": 0.9},
                clip_on=True,
            )
            try:
                ann.set_picker(True)
            except Exception:
                pass
            self._spectrum_annotations.append(ann)
            self._spec_ann_key_by_objid[id(ann)] = ("custom", key, int(idx))

    def _parse_charges(self, raw: str) -> List[int]:
        vals: List[int] = []
        for part in (raw or "").replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                z = int(part)
            except Exception:
                continue
            if z > 0:
                vals.append(z)
        # de-dup, keep order
        out: List[int] = []
        for z in vals:
            if z not in out:
                out.append(z)
        return out

    def _parse_monomers(self, raw: str) -> List[Tuple[str, float]]:
        lines = (raw or "").splitlines()
        monomers: List[Tuple[str, float]] = []
        auto_i = 1
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            # allow: name,mass  OR  name mass  OR  mass
            name = ""
            mass_s = ""
            if "," in ln:
                parts = [p.strip() for p in ln.split(",") if p.strip()]
                if len(parts) >= 2:
                    name, mass_s = parts[0], parts[1]
                elif len(parts) == 1:
                    mass_s = parts[0]
            else:
                parts = ln.split()
                if len(parts) >= 2:
                    name, mass_s = " ".join(parts[:-1]), parts[-1]
                elif len(parts) == 1:
                    mass_s = parts[0]

            try:
                mass = float(mass_s)
            except Exception:
                continue
            if not name:
                name = f"M{auto_i}"
                auto_i += 1
            monomers.append((name, float(mass)))

        return monomers

    def _apply_polymer_matches(self, mz_vals: np.ndarray, int_vals: np.ndarray) -> None:
        if self._ax_spec is None:
            return
        if not bool(self.poly_enabled_var.get()):
            return
        if mz_vals.size == 0 or int_vals.size == 0:
            return

        monomers = self._parse_monomers(self.poly_monomers_text_var.get())
        if not monomers:
            return

        charges = self._parse_charges(self.poly_charges_var.get())
        if not charges:
            charges = [1]

        try:
            max_dp = int(self.poly_max_dp_var.get())
        except Exception:
            max_dp = 12
        max_dp = max(1, min(200, int(max_dp)))

        bond_delta = float(self.poly_bond_delta_var.get())
        extra_delta = float(self.poly_extra_delta_var.get())
        adduct_mass = float(self.poly_adduct_mass_var.get())
        decarb_enabled = bool(self.poly_decarb_enabled_var.get())
        oxid_enabled = bool(self.poly_oxid_enabled_var.get())
        cluster_enabled = bool(self.poly_cluster_enabled_var.get())
        cluster_adduct_mass = float(self.poly_cluster_adduct_mass_var.get())

        pol = None
        if self._current_spectrum_meta is not None:
            pol = self._current_spectrum_meta.polarity

        if pol in ("positive", "negative"):
            h = 1.007276
            sign = 1.0 if pol == "positive" else -1.0
            if abs(abs(adduct_mass) - h) <= 0.01:
                adduct_mass = sign * abs(adduct_mass)
            if abs(abs(cluster_adduct_mass) - h) <= 0.01:
                cluster_adduct_mass = sign * abs(cluster_adduct_mass)

        tol_value = float(self.poly_tol_value_var.get())
        tol_unit = (self.poly_tol_unit_var.get() or "Da").strip()

        max_int = float(np.max(int_vals)) if int_vals.size else 0.0
        if max_int <= 0:
            return
        min_rel = float(self.poly_min_rel_int_var.get())
        min_rel = max(0.0, min(1.0, float(min_rel)))

        order = np.argsort(mz_vals)
        mz_s = mz_vals[order]
        int_s = int_vals[order]

        compat = bool(str(os.environ.get("LAB_GUI_POLYMER_COMPAT", "")).strip())

        try:
            best_by_peak = poly_match.compute_polymer_best_by_peak_sorted(
                mz_s,
                int_s,
                monomer_names=[n for n, _m in monomers],
                monomer_masses=[m for _n, m in monomers],
                charges=list(charges),
                max_dp=int(max_dp),
                bond_delta=float(bond_delta),
                extra_delta=float(extra_delta),
                polarity=pol,
                base_adduct_mass=float(adduct_mass),
                enable_decarb=bool(decarb_enabled),
                enable_oxid=bool(oxid_enabled),
                enable_cluster=bool(cluster_enabled),
                cluster_adduct_mass=float(cluster_adduct_mass),
                enable_na=bool(self.poly_adduct_na_var.get()),
                enable_k=bool(self.poly_adduct_k_var.get()),
                enable_cl=bool(self.poly_adduct_cl_var.get()),
                enable_formate=bool(self.poly_adduct_formate_var.get()),
                enable_acetate=bool(self.poly_adduct_acetate_var.get()),
                tol_value=float(tol_value),
                tol_unit=str(tol_unit or "Da"),
                min_rel_int=float(min_rel),
                allow_variant_combo=True,
                compatibility_mode=bool(compat),
            )
        except poly_match.PolymerSearchTooLarge as exc:
            # Graceful warning: do not crash or hang the UI.
            try:
                messagebox.showwarning("Polymer Match", str(exc), parent=self)
            except Exception:
                pass
            return
        except Exception as exc:
            try:
                messagebox.showerror("Polymer Match", f"Polymer matching failed:\n{exc}", parent=self)
            except Exception:
                pass
            return

        if not best_by_peak:
            return

        y_offset = 0.10 * max_int
        kind_order = ["poly", "ox", "decarb", "oxdecarb", "2m"]
        meta = self._current_spectrum_meta
        spectrum_id = str(meta.spectrum_id) if meta is not None else "__no_spectrum__"
        overrides = self._spec_label_overrides.get(spectrum_id, {})
        for _peak_i, kinds in best_by_peak.items():
            # Stack labels upward so multiple reactions remain visible.
            items: List[Tuple[str, float, str, float, float]] = []
            for knd in kind_order:
                if knd in kinds:
                    _err, label, mz_act, inten_act = kinds[knd]
                    items.append((str(knd), float(_err), str(label), float(mz_act), float(inten_act)))
            if not items:
                for knd, (_err, label, mz_act, inten_act) in kinds.items():
                    items.append((str(knd), float(_err), str(label), float(mz_act), float(inten_act)))

            for j, (knd, _err, label, mz_act, inten_act) in enumerate(items):
                mz_key = self._mz_key(float(mz_act))
                ov_key = (str(knd), float(mz_key))
                ov = overrides.get(ov_key)
                if ov is None and ov_key in overrides:
                    continue
                if isinstance(ov, str):
                    label = ov
                y_text = float(inten_act) + y_offset * (1.0 + float(j))
                ann = self._ax_spec.annotate(
                    label,
                    xy=(float(mz_act), float(inten_act)),
                    xytext=(float(mz_act), float(y_text)),
                    textcoords="data",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=max(6, int(self.tick_fontsize_var.get()) - 1),
                    arrowprops={"arrowstyle": "-", "lw": 1.0},
                    clip_on=True,
                )
                try:
                    ann.set_picker(True)
                except Exception:
                    pass
                self._spectrum_annotations.append(ann)
                self._spec_ann_key_by_objid[id(ann)] = ("poly", spectrum_id, str(knd), float(mz_key))


def run_overlay_self_checks() -> Dict[str, object]:
    """Lightweight, dev-only overlay checks (no file I/O)."""
    results: Dict[str, object] = {"ok": True, "checks": {}}

    # 1) Union RT grid length
    rts1 = np.asarray([1.0, 2.0, 3.0], dtype=float)
    rts2 = np.asarray([1.1, 2.1, 3.1], dtype=float)
    union = np.unique(np.concatenate([rts1, rts2]))
    results["checks"]["union_rt_len"] = int(union.size)
    if int(union.size) != 6:
        results["ok"] = False

    # 2) Nearest RT selection
    target = 2.05
    idx1 = int(np.argmin(np.abs(rts1 - float(target))))
    idx2 = int(np.argmin(np.abs(rts2 - float(target))))
    results["checks"]["nearest_idx_1"] = int(idx1)
    results["checks"]["nearest_idx_2"] = int(idx2)
    if idx1 != 1 or idx2 != 1:
        results["ok"] = False

    # 3) Overlay export headers
    names = ["A", "B"]
    headers = ["rt_min"] + names
    results["checks"]["export_headers"] = headers
    if headers[0] != "rt_min" or len(headers) != 3:
        results["ok"] = False

    return results


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
