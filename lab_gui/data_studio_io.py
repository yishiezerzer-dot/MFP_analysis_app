from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd


def get_sheet_names(path: Path) -> List[str]:
    try:
        xls = pd.ExcelFile(str(path))
        return [str(s) for s in list(xls.sheet_names)]
    except Exception:
        return []


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        try:
            # Pandas supports errors="ignore" at runtime, but some type stubs only allow "raise"/"coerce".
            out[col] = pd.to_numeric(out[col], errors="ignore")  # type: ignore[arg-type]
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


def apply_transform_steps(df: pd.DataFrame, steps: List[Dict[str, Any]]) -> pd.DataFrame:
    """Apply ordered transform steps to a DataFrame (non-destructive)."""
    out: pd.DataFrame = df.copy()
    warnings: List[str] = []

    def _warn(msg: str) -> None:
        warnings.append(str(msg))

    for step in (steps or []):
        if not isinstance(step, dict):
            continue
        stype = str(step.get("type") or "").strip()
        cols = list(step.get("columns") or [])

        try:
            if stype == "select_columns":
                mode = str(step.get("mode") or "keep")
                cols_set = [c for c in cols if c in out.columns]
                if not cols_set and cols:
                    _warn("select_columns: no matching columns")
                if mode == "drop":
                    out = cast(pd.DataFrame, out.drop(columns=[c for c in cols if c in out.columns], errors="ignore"))
                else:
                    out = cast(pd.DataFrame, out.loc[:, cols_set]) if cols_set else out

            elif stype == "rename":
                mapping = dict(step.get("mapping") or {})
                if mapping:
                    missing = [k for k in mapping.keys() if k not in out.columns]
                    if missing:
                        _warn(f"rename: missing columns {missing}")
                    out = cast(pd.DataFrame, out.rename(columns={k: v for k, v in mapping.items() if k in out.columns}))

            elif stype == "to_numeric":
                errs_raw = str(step.get("errors") or "coerce").strip().lower()
                errs = "raise" if errs_raw == "raise" else "coerce"
                for c in cols:
                    if c not in out.columns:
                        _warn(f"to_numeric: missing column {c}")
                        continue
                    out[c] = pd.to_numeric(out[c], errors=errs)

            elif stype == "fillna":
                val = step.get("value")
                for c in cols:
                    if c not in out.columns:
                        _warn(f"fillna: missing column {c}")
                        continue
                    if str(val).lower() == "mean":
                        out[c] = out[c].fillna(pd.to_numeric(out[c], errors="coerce").mean())
                    elif str(val).lower() == "ffill":
                        out[c] = out[c].fillna(method="ffill")
                    else:
                        out[c] = out[c].fillna(val)

            elif stype == "normalize":
                mode = str(step.get("mode") or "minmax").lower()
                for c in cols:
                    if c not in out.columns:
                        _warn(f"normalize: missing column {c}")
                        continue
                    arr = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)
                    if mode == "zscore":
                        out[c] = normalize_series(arr, "Z-score")
                    else:
                        out[c] = normalize_series(arr, "Min-Max")

            elif stype == "baseline":
                method = str(step.get("method") or "first")
                rng = step.get("range")
                for c in cols:
                    if c not in out.columns:
                        _warn(f"baseline: missing column {c}")
                        continue
                    series = pd.to_numeric(out[c], errors="coerce")
                    baseline_val = None
                    if method == "mean_range" and isinstance(rng, list) and len(rng) == 2:
                        try:
                            start = int(rng[0])
                            end = int(rng[1])
                            baseline_val = series.iloc[start : end + 1].mean()
                        except Exception:
                            baseline_val = series.mean()
                    else:
                        try:
                            baseline_val = float(series.iloc[0])
                        except Exception:
                            baseline_val = series.mean()
                    out[c] = series - float(baseline_val or 0.0)

            elif stype == "log":
                base = float(step.get("base") or 10.0)
                offset = float(step.get("offset") or 0.0)
                base = base if base > 0 else 10.0
                for c in cols:
                    if c not in out.columns:
                        _warn(f"log: missing column {c}")
                        continue
                    series = pd.to_numeric(out[c], errors="coerce")
                    out[c] = np.log(series + offset) / np.log(base)

            elif stype == "rolling_mean":
                window = int(step.get("window") or 5)
                center = bool(step.get("center", True))
                for c in cols:
                    if c not in out.columns:
                        _warn(f"rolling_mean: missing column {c}")
                        continue
                    series = pd.to_numeric(out[c], errors="coerce")
                    out[c] = series.rolling(window=window, center=center, min_periods=1).mean()

        except Exception as exc:
            _warn(f"{stype}: {exc}")
            continue

    out.attrs["transform_warnings"] = warnings
    return out
