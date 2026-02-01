from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SpectrumMeta:
    spectrum_id: str
    rt_min: float
    tic: float
    polarity: Optional[str]  # 'positive' | 'negative' | None
    ms_level: int


@dataclass(frozen=True)
class CustomLabel:
    label: str
    mz: float
    snap_to_nearest_peak: bool = True

@dataclass(frozen=True)
class OverlaySession:
    dataset_ids: List[str]
    mode: str
    colors: Dict[str, str]
    persist: bool = False
    show_uv: bool = True
    stack_spectra: bool = False
    show_labels_all: bool = False
    multi_drag: bool = False
    active_dataset_id: Optional[str] = None


@dataclass(frozen=True)
class LabelingSettings:
    annotate_peaks: bool
    annotate_min_rel: float
    annotate_top_n: int

    poly_enabled: bool
    poly_monomers_text: str
    poly_charges_text: str
    poly_max_dp: int
    poly_bond_delta: float
    poly_extra_delta: float
    poly_adduct_mass: float
    poly_decarb_enabled: bool
    poly_oxid_enabled: bool
    poly_cluster_enabled: bool
    poly_cluster_adduct_mass: float
    poly_adduct_na: bool
    poly_adduct_k: bool
    poly_adduct_cl: bool
    poly_adduct_formate: bool
    poly_adduct_acetate: bool
    poly_tol_value: float
    poly_tol_unit: str
    poly_min_rel_int: float


@dataclass
class UVLabelState:
    text: str
    xytext: Tuple[float, float]
    confidence: float = 0.0
    rt_delta_min: float = 0.0
    uv_peak_score: float = 0.0
    tic_peak_score: float = 0.0


@dataclass
class MzMLSession:
    session_id: str
    path: Path
    # NOTE: intentionally typed as Any to avoid importing IO classes here.
    index: Any
    load_order: int
    display_name: str

    custom_labels_by_spectrum: Dict[str, Any]
    spec_label_overrides: Dict[Any, Any]

    ms1_count: int
    rt_min: Optional[float]
    rt_max: Optional[float]
    polarity_summary: str

    last_selected_rt_min: Optional[float] = None
    last_scan_index: Optional[int] = None
    last_polarity_filter: Optional[str] = None

    linked_uv_id: Optional[str] = None
    uv_labels_by_uv_id: Dict[str, Any] = field(default_factory=dict)

    # Overlay UI metadata (LCMS only)
    overlay_color: str = ""
    overlay_selected: bool = False


@dataclass
class UVSession:
    uv_id: str
    path: Path
    rt_min: np.ndarray
    signal: np.ndarray
    xcol: str
    ycol: str
    n_points: int
    rt_min_range: Tuple[float, float]
    load_order: int
    import_warnings: List[str] = field(default_factory=list)


@dataclass
class LCMSDataset:
    """Workspace-level representation of an LCMS dataset.

    The LCMS implementation still uses `MzMLSession` internally; this dataset record
    provides a stable list + active selection for the app workspace.
    """

    session_id: str
    mzml_path: Path
    uv_csv_path: Optional[Path] = None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_ms_level(spectrum: Dict[str, Any]) -> Optional[int]:
    value = spectrum.get("ms level")
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _extract_rt_minutes(spectrum: Dict[str, Any], rt_unit: str) -> Optional[float]:
    scan_list = spectrum.get("scanList") or {}
    scans = scan_list.get("scan") or []
    if not scans:
        return None

    scan0 = scans[0]
    scan_start_time = scan0.get("scan start time")
    rt = _safe_float(scan_start_time)
    if rt is None:
        return None

    if rt_unit == "minutes":
        return rt
    if rt_unit == "seconds":
        return rt / 60.0
    return None


def _extract_polarity(spectrum: Dict[str, Any]) -> Optional[str]:
    # Common CV params appear as boolean-ish keys in pyteomics.
    if spectrum.get("positive scan") is not None:
        return "positive"
    if spectrum.get("negative scan") is not None:
        return "negative"

    scan_list = spectrum.get("scanList") or {}
    scans = scan_list.get("scan") or []
    if scans:
        scan0 = scans[0]
        if scan0.get("positive scan") is not None:
            return "positive"
        if scan0.get("negative scan") is not None:
            return "negative"

    return None


def _spectrum_id(spectrum: Dict[str, Any]) -> str:
    sid = spectrum.get("id")
    if sid is None:
        sid = spectrum.get("index")
    return str(sid)
