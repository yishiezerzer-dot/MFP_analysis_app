from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np


def list_excel_sheets(path: Path) -> List[str]:
    try:
        xf = pd.ExcelFile(path)
        return [str(s) for s in (xf.sheet_names or [])]
    except Exception:
        return []


def read_plate_file(
    path: Path,
    *,
    sheet_name: Optional[str] = None,
    header_row: Optional[int] = 0,
) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()

    header = None if header_row is None else int(header_row)

    if suf in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(p, sheet_name=sheet_name or 0, header=header)

    # Default: treat as delimited text.
    # Use python engine so it can autodetect separator reasonably.
    return pd.read_csv(p, header=header, sep=None, engine="python")


def coerce_numeric_matrix(
    df: pd.DataFrame,
    *,
    row_indices: Sequence[int],
    columns: Sequence[str],
) -> Tuple[np.ndarray, float]:
    """Coerce a selected row/column slice to float matrix.

    Returns:
        (values, nan_ratio)
    """

    # Rows are selected by position in the UI (Row 1 == first displayed row),
    # so always interpret them as positional indices.
    rows = [int(i) for i in (row_indices or []) if int(i) >= 0 and int(i) < int(df.shape[0])]

    # Columns may be strings in the UI even when df.columns are ints (e.g. header=None).
    # Build a robust lookup by stringifying the actual column labels.
    col_lookup = {str(c): c for c in df.columns}
    cols: List[object] = []
    for c in (columns or []):
        if c in df.columns:
            cols.append(c)
        else:
            key = str(c)
            if key in col_lookup:
                cols.append(col_lookup[key])

    # De-duplicate while preserving order
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]

    if not rows or not cols:
        return np.zeros((0, 0), dtype=float), 1.0

    # Use iloc for rows to avoid depending on df.index labels.
    block = df.iloc[rows][cols].copy()
    for c in cols:
        try:
            block[c] = pd.to_numeric(block[c], errors="coerce")
        except Exception:
            block[c] = np.nan

    arr = block.to_numpy(dtype=float)
    if arr.size == 0:
        return arr, 1.0

    nan_ratio = float(np.isnan(arr).sum()) / float(arr.size)
    return arr, nan_ratio


def preview_dataframe(df: pd.DataFrame, *, max_rows: int = 50) -> Tuple[List[str], List[List[str]]]:
    cols = [str(c) for c in df.columns]
    n = min(int(max_rows), int(df.shape[0]))

    rows: List[List[str]] = []
    for i in range(n):
        row = []
        for c in df.columns:
            try:
                v = df.iloc[i][c]
            except Exception:
                v = ""
            if pd.isna(v):
                row.append("")
            else:
                row.append(str(v))
        rows.append(row)

    return cols, rows
