from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DataStudioDataset:
    dataset_id: str
    path: Path
    display_name: str
    sheet_name: Optional[str] = None
    header_row: int = 0
    columns: Dict[str, str] = field(default_factory=dict)
    schema_hash: str = ""


@dataclass
class DataStudioPlotDef:
    plot_id: str
    dataset_id: str
    x_col: Optional[str] = None
    y_cols: List[str] = field(default_factory=list)
    plot_type: str = "Line"
    options: Dict[str, object] = field(default_factory=dict)
    last_validated_schema_hash: str = ""


@dataclass
class DataStudioWorkspace:
    datasets: Dict[str, DataStudioDataset] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    plot_defs: Dict[str, DataStudioPlotDef] = field(default_factory=dict)
    active_plot_id: Optional[str] = None
    active_id: Optional[str] = None
    overlay_ids: List[str] = field(default_factory=list)
    preferred_axes_by_dataset: Dict[str, Tuple[Optional[str], List[str]]] = field(default_factory=dict)
