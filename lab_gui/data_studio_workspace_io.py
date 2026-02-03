from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from lab_gui.data_studio_model import DataStudioDataset, DataStudioPlotDef, DataStudioWorkspace


def encode_workspace(ws: DataStudioWorkspace) -> Dict[str, Any]:
    datasets: List[Dict[str, Any]] = []
    for ds_id in ws.order:
        ds = ws.datasets.get(ds_id)
        if ds is None:
            continue
        datasets.append(
            {
                "id": str(ds.dataset_id),
                "path": str(ds.path),
                "display_name": str(ds.display_name),
                "sheet_name": (None if ds.sheet_name is None else str(ds.sheet_name)),
                "header_row": int(ds.header_row),
                "columns": dict(ds.columns or {}),
                "schema_hash": str(ds.schema_hash or ""),
            }
        )

    plot_defs: List[Dict[str, Any]] = []
    for pid, pd in (ws.plot_defs or {}).items():
        plot_defs.append(
            {
                "id": str(pd.plot_id or pid),
                "dataset_id": str(pd.dataset_id),
                "x_col": (None if pd.x_col is None else str(pd.x_col)),
                "y_cols": list(pd.y_cols or []),
                "plot_type": str(pd.plot_type or "Line"),
                "options": dict(pd.options or {}),
                "last_validated_schema_hash": str(pd.last_validated_schema_hash or ""),
            }
        )

    preferred_axes: Dict[str, Any] = {}
    for ds_id, axes in (ws.preferred_axes_by_dataset or {}).items():
        if not axes:
            continue
        x_col, y_cols = axes
        preferred_axes[str(ds_id)] = {
            "x_col": (None if x_col is None else str(x_col)),
            "y_cols": list(y_cols or []),
        }

    return {
        "schema_version": 1,
        "kind": "data_studio_workspace",
        "datasets": datasets,
        "plot_defs": plot_defs,
        "active_plot_id": (None if ws.active_plot_id is None else str(ws.active_plot_id)),
        "preferred_axes_by_dataset": preferred_axes,
    }


def decode_workspace(payload: Dict[str, Any]) -> Tuple[DataStudioWorkspace, List[str]]:
    ws = DataStudioWorkspace()
    errors: List[str] = []

    rows = payload.get("datasets") or []
    if not isinstance(rows, list):
        rows = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        p_str = str(row.get("path", "") or "")
        if not p_str:
            continue
        p = Path(p_str)
        if not p.exists():
            errors.append(f"Missing file: {p}")
            continue
        ds_id = str(row.get("id", "") or "") or None
        if not ds_id:
            errors.append(f"Missing dataset id for {p}")
            continue
        display_name = str(row.get("display_name", "") or p.name)
        ds = DataStudioDataset(
            dataset_id=ds_id,
            path=p,
            display_name=display_name,
            sheet_name=(None if row.get("sheet_name") in (None, "") else str(row.get("sheet_name"))),
            header_row=int(row.get("header_row") or 0),
            columns=dict(row.get("columns") or {}),
            schema_hash=str(row.get("schema_hash") or ""),
        )
        ws.datasets[ds_id] = ds
        ws.order.append(ds_id)

    pd_rows = payload.get("plot_defs") or []
    if not isinstance(pd_rows, list):
        pd_rows = []
    for row in pd_rows:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id", "") or "")
        ds_id = str(row.get("dataset_id", "") or "")
        if not pid or not ds_id:
            continue
        pd = DataStudioPlotDef(
            plot_id=pid,
            dataset_id=ds_id,
            x_col=(None if row.get("x_col") in (None, "") else str(row.get("x_col"))),
            y_cols=list(row.get("y_cols") or []),
            plot_type=str(row.get("plot_type") or "Line"),
            options=dict(row.get("options") or {}),
            last_validated_schema_hash=str(row.get("last_validated_schema_hash") or ""),
        )
        ws.plot_defs[pid] = pd

    preferred = payload.get("preferred_axes_by_dataset") or {}
    if isinstance(preferred, dict):
        for ds_id, axes in preferred.items():
            if not isinstance(axes, dict):
                continue
            ws.preferred_axes_by_dataset[str(ds_id)] = (
                (None if axes.get("x_col") in (None, "") else str(axes.get("x_col"))),
                list(axes.get("y_cols") or []),
            )

    ws.active_plot_id = (None if payload.get("active_plot_id") in (None, "") else str(payload.get("active_plot_id")))
    if not ws.active_plot_id and ws.plot_defs:
        ws.active_plot_id = next(iter(ws.plot_defs.keys()))

    if ws.order and ws.active_id is None:
        ws.active_id = ws.order[0]

    return ws, errors
