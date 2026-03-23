from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class AppAIContext:
    """Compact, read-only app context for AI assistant prompting."""

    active_module: str
    loaded_filenames: List[str]
    module_summary: str

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Return a small JSON-serializable payload for prompts."""
        return {
            "active_module": self.active_module,
            "loaded_filenames": list(self.loaded_filenames),
            "module_summary": self.module_summary,
        }


def build_ai_context(app: Any, workspace: Any, *, max_files: int = 8) -> AppAIContext:
    """Build safe read-only context from current app state.

    This intentionally excludes large arrays/dataframes/raw analysis payloads.
    """
    active_module = _safe_active_module_name(app)
    mod_key = active_module.strip().lower()

    if mod_key == "lcms":
        filenames, summary = _context_lcms(workspace)
    elif mod_key == "ftir":
        filenames, summary = _context_ftir(workspace)
    elif mod_key == "plate reader":
        filenames, summary = _context_plate_reader(workspace)
    elif mod_key == "data studio":
        filenames, summary = _context_data_studio(app)
    else:
        filenames = _collect_generic_filenames(workspace)
        summary = "General app context is available, but no module-specific summary was found."

    clipped = filenames[: max(1, int(max_files))]
    return AppAIContext(active_module=active_module, loaded_filenames=clipped, module_summary=summary)


def _safe_active_module_name(app: Any) -> str:
    """Implement the `_safe_active_module_name` behavior for this module.

    Text-only documentation note: modify internal logic here to change behavior.
    """
    fn = getattr(app, "_active_module_name", None)
    if callable(fn):
        try:
            name = str(fn() or "").strip()
            if name:
                return name
        except Exception:
            pass

    nb = getattr(app, "_module_notebook", None)
    if nb is not None:
        try:
            tab_id = nb.select()
            text = str(nb.tab(tab_id, "text") or "").strip()
            if text:
                return text
        except Exception:
            pass

    return "LCMS"


def _context_lcms(workspace: Any) -> tuple[List[str], str]:
    """Implement the `_context_lcms` behavior for this module.

    Text-only documentation note: modify internal logic here to change behavior.
    """
    datasets = list(getattr(workspace, "lcms_datasets", []) or [])
    active_id = str(getattr(workspace, "active_lcms", "") or "")

    filenames = []
    active_name = ""
    for ds in datasets:
        path = getattr(ds, "mzml_path", None)
        name = _name_from_path(path)
        if name:
            filenames.append(name)
        if active_id and str(getattr(ds, "session_id", "")) == active_id:
            active_name = name or active_name

    if not datasets:
        summary = "LCMS module is open; no mzML datasets are loaded."
    else:
        focus = active_name or filenames[0] if filenames else "(unknown file)"
        summary = f"LCMS has {len(datasets)} dataset(s) loaded. Active session: {focus}."

    return filenames, summary


def _context_ftir(workspace: Any) -> tuple[List[str], str]:
    """Implement the `_context_ftir` behavior for this module.

    Text-only documentation note: modify internal logic here to change behavior.
    """
    datasets = list(getattr(workspace, "ftir_datasets", []) or [])
    active_id = str(getattr(workspace, "active_ftir_id", "") or "")

    filenames = []
    active_name = ""
    for ds in datasets:
        path = getattr(ds, "path", None)
        name = _name_from_path(path) or str(getattr(ds, "name", "") or "")
        if name:
            filenames.append(name)
        if active_id and str(getattr(ds, "id", "")) == active_id:
            active_name = name or active_name

    if not datasets:
        summary = "FTIR module is open; no FTIR datasets are loaded."
    else:
        focus = active_name or filenames[0] if filenames else "(unknown dataset)"
        summary = f"FTIR has {len(datasets)} dataset(s) loaded. Active dataset: {focus}."

    return filenames, summary


def _context_plate_reader(workspace: Any) -> tuple[List[str], str]:
    """Implement the `_context_plate_reader` behavior for this module.

    Text-only documentation note: modify internal logic here to change behavior.
    """
    datasets = list(getattr(workspace, "plate_reader_datasets", []) or [])
    active_id = str(getattr(workspace, "active_plate_reader_id", "") or "")

    filenames = []
    active_name = ""
    for ds in datasets:
        display = str(getattr(ds, "display_name", "") or "").strip()
        path_name = _name_from_path(getattr(ds, "path", None))
        name = display or path_name or str(getattr(ds, "name", "") or "")
        if name:
            filenames.append(name)
        if active_id and str(getattr(ds, "id", "")) == active_id:
            active_name = name or active_name

    if not datasets:
        summary = "Plate Reader module is open; no datasets are loaded."
    else:
        focus = active_name or filenames[0] if filenames else "(unknown dataset)"
        summary = f"Plate Reader has {len(datasets)} dataset(s) loaded. Active dataset: {focus}."

    return filenames, summary


def _context_data_studio(app: Any) -> tuple[List[str], str]:
    """Implement the `_context_data_studio` behavior for this module.

    Text-only documentation note: modify internal logic here to change behavior.
    """
    view = getattr(app, "_data_studio_view", None)
    ws = getattr(view, "_ws", None)
    if ws is None:
        return [], "Data Studio module is open; workspace details are unavailable."

    datasets_map = getattr(ws, "datasets", {}) or {}
    order = list(getattr(ws, "order", []) or [])
    active_id = str(getattr(ws, "active_id", "") or "")
    active_plot_id = str(getattr(ws, "active_plot_id", "") or "")
    plot_defs = getattr(ws, "plot_defs", {}) or {}

    filenames: List[str] = []
    active_name = ""
    for dataset_id in order:
        ds = datasets_map.get(dataset_id)
        if ds is None:
            continue
        display = str(getattr(ds, "display_name", "") or "").strip()
        path_name = _name_from_path(getattr(ds, "path", None))
        name = display or path_name or str(dataset_id)
        filenames.append(name)
        if active_id and str(dataset_id) == active_id:
            active_name = name or active_name

    if not filenames:
        return [], "Data Studio module is open; no table datasets are loaded."

    plot_type = ""
    if active_plot_id and active_plot_id in plot_defs:
        try:
            plot_type = str(getattr(plot_defs.get(active_plot_id), "plot_type", "") or "")
        except Exception:
            plot_type = ""

    focus = active_name or filenames[0]
    if plot_type:
        summary = f"Data Studio has {len(filenames)} dataset(s). Active dataset: {focus}. Active plot type: {plot_type}."
    else:
        summary = f"Data Studio has {len(filenames)} dataset(s). Active dataset: {focus}."

    return filenames, summary


def _collect_generic_filenames(workspace: Any) -> List[str]:
    """Implement the `_collect_generic_filenames` behavior for this module.

    Text-only documentation note: modify internal logic here to change behavior.
    """
    out: List[str] = []

    for ds in list(getattr(workspace, "lcms_datasets", []) or []):
        name = _name_from_path(getattr(ds, "mzml_path", None))
        if name:
            out.append(name)

    for ds in list(getattr(workspace, "ftir_datasets", []) or []):
        name = _name_from_path(getattr(ds, "path", None)) or str(getattr(ds, "name", "") or "")
        if name:
            out.append(name)

    for ds in list(getattr(workspace, "plate_reader_datasets", []) or []):
        name = _name_from_path(getattr(ds, "path", None)) or str(getattr(ds, "name", "") or "")
        if name:
            out.append(name)

    return out


def _name_from_path(value: Any) -> str:
    """Implement the `_name_from_path` behavior for this module.

    Text-only documentation note: modify internal logic here to change behavior.
    """
    if value is None:
        return ""
    try:
        return Path(value).name
    except Exception:
        try:
            s = str(value)
            return Path(s).name if s else ""
        except Exception:
            return ""


# TODO: Add per-module interpreter helpers once module snapshot APIs are formalized.
# TODO: Include microscopy context after a stable shared snapshot contract is added.
