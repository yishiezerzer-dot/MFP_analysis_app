from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


FTIRDatasetKey = Tuple[str, str]  # (workspace_id, dataset_id)


@dataclass
class FTIRBondAnnotation:
    dataset_id: str
    text: str
    x_cm1: float
    y_value: float
    xytext: Tuple[float, float]
    show_vline: bool = False
    line_color: str = "#444444"
    text_color: str = "#111111"
    fontsize: int = 9
    rotation: int = 0
    preset_id: Optional[str] = None


@dataclass
class FTIRDataset:
    id: str
    name: str
    path: Optional[Path] = None

    # Full-resolution arrays
    x_full: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    y_full: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))

    # Display arrays (decimated for rendering when needed)
    x_disp: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    y_disp: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))

    y_mode: str = "absorbance"
    x_units: Optional[str] = None
    y_units: Optional[str] = None
    loaded_at_utc: Optional[str] = None

    # Peak picking (per dataset)
    peak_settings: Dict[str, Any] = field(default_factory=dict)
    peaks: List[Dict[str, Any]] = field(default_factory=list)
    peak_label_overrides: Dict[str, str] = field(default_factory=dict)
    peak_suppressed: set[str] = field(default_factory=set)
    peak_label_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    bond_annotations: List[FTIRBondAnnotation] = field(default_factory=list)


@dataclass
class FTIRWorkspace:
    id: str
    name: str
    datasets: List[FTIRDataset] = field(default_factory=list)
    active_dataset_id: Optional[str] = None
    # Per-workspace line color for FTIR plotting. If None, a stable default is derived from workspace id.
    line_color: Optional[str] = None

    bond_annotations: List[FTIRBondAnnotation] = field(default_factory=list)


@dataclass
class StyleState:
    # Minimal style state for now (kept for future expansion).
    linewidth: float = 1.2


@dataclass
class OverlayGroup:
    group_id: str
    name: str
    members: List[FTIRDatasetKey] = field(default_factory=list)  # ordered
    active_member: Optional[FTIRDatasetKey] = None
    per_member_style: Dict[FTIRDatasetKey, StyleState] = field(default_factory=dict)
    created_at: float = 0.0
