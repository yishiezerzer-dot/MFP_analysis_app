from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def get_sheet_names(path: Path) -> List[str]:
    try:
        xls = pd.ExcelFile(str(path))
        return list(xls.sheet_names)
    except Exception:
        return []


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        try:
            out[col] = pd.to_numeric(out[col], errors="ignore")
        except Exception:
            continue
    return out


def _replace_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            try:
                out[col] = out[col].astype(str).str.replace(",", ".", regex=False)
            except Exception:
                continue
    return out


def load_table(
    path: Path,
    *,
    sheet_name: Optional[str] = None,
    header_row: int = 0,
    decimal_comma: bool = False,
    auto_cast: bool = True,
) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    ext = str(path.suffix).lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(str(path), sheet_name=sheet_name or 0, header=int(header_row))
    else:
        df = pd.read_csv(str(path), sep=None, engine="python", header=int(header_row))

    if decimal_comma:
        df = _replace_decimal_commas(df)
    if auto_cast:
        df = _coerce_numeric(df)
    return df


def numeric_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                out.append(str(col))
        except Exception:
            continue
    return out


def column_types_summary(df: pd.DataFrame) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for col in df.columns:
        try:
            out.append((str(col), str(df[col].dtype)))
        except Exception:
            out.append((str(col), "unknown"))
    return out


def column_type_map(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in df.columns:
        try:
            out[str(col)] = str(df[col].dtype)
        except Exception:
            out[str(col)] = "unknown"
    return out


def schema_hash_from_columns(cols: Dict[str, str]) -> str:
    try:
        parts = [f"{k}:{cols.get(k, '')}" for k in sorted(cols.keys())]
        raw = "|".join(parts).encode("utf-8", errors="ignore")
        return hashlib.sha256(raw).hexdigest()[:16]
    except Exception:
        return ""


def normalize_series(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "Min-Max":
        vmin = float(np.nanmin(values)) if values.size else 0.0
        vmax = float(np.nanmax(values)) if values.size else 0.0
        if vmax == vmin:
            return np.asarray(values, dtype=float)
        return (values - vmin) / (vmax - vmin)
    if mode == "Z-score":
        mu = float(np.nanmean(values)) if values.size else 0.0
        sd = float(np.nanstd(values)) if values.size else 1.0
        if sd == 0:
            return np.asarray(values, dtype=float)
        return (values - mu) / sd
    return np.asarray(values, dtype=float)
