from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MicroscopyDataset:
    id: str
    display_name: str
    file_path: str
    workspace_id: str
    created_at: str
    notes: str
    output_dir: str
    last_macro_run: Optional[str] = None


@dataclass
class MicroscopyWorkspace:
    id: str
    name: str
    datasets: List[MicroscopyDataset] = field(default_factory=list)
