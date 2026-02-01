from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pyteomics import mzml

from .lcms_model import SpectrumMeta, _extract_ms_level, _extract_polarity, _extract_rt_minutes, _spectrum_id


class LCMSLoadError(Exception):
    pass


class UVLoadError(Exception):
    pass


class MzMLTICIndex:
    """Minimal MS1 index used by the GUI.

    Builds a list of MS1 spectra metadata (RT, TIC, polarity).

    UI-free: does not import tkinter and does not show dialogs.
    """

    def __init__(self, mzml_path: Path, *, rt_unit: str = "minutes") -> None:
        self.path = Path(mzml_path).expanduser().resolve()
        self.rt_unit = str(rt_unit)
        self.ms1: List[SpectrumMeta] = []
        self.stats: Dict[str, Any] = {}

    def build(self) -> None:
        ms1: List[SpectrumMeta] = []
        stats: Dict[str, Any] = {
            "total_spectra": 0,
            "ms1_kept": 0,
            "skipped_non_ms1": 0,
            "skipped_no_rt": 0,
            "skipped_no_intensity": 0,
            "skipped_error": 0,
            "fatal_error": None,
        }

        try:
            reader = mzml.MzML(str(self.path))
        except Exception as exc:
            stats["fatal_error"] = f"Failed to open mzML: {exc!r}"
            self.ms1 = []
            self.stats = stats
            return

        try:
            with reader:
                try:
                    for spectrum in reader:
                        stats["total_spectra"] += 1
                        try:
                            ms_level = _extract_ms_level(spectrum)
                            if ms_level != 1:
                                stats["skipped_non_ms1"] += 1
                                continue

                            rt_min = _extract_rt_minutes(spectrum, rt_unit=self.rt_unit)
                            if rt_min is None:
                                stats["skipped_no_rt"] += 1
                                continue

                            inten = spectrum.get("intensity array")
                            if inten is None:
                                stats["skipped_no_intensity"] += 1
                                continue

                            pol = _extract_polarity(spectrum)
                            try:
                                tic = float(np.sum(np.asarray(inten, dtype=float)))
                            except Exception:
                                tic = 0.0

                            ms1.append(
                                SpectrumMeta(
                                    spectrum_id=_spectrum_id(spectrum),
                                    rt_min=float(rt_min),
                                    tic=float(tic),
                                    polarity=pol,
                                    ms_level=1,
                                )
                            )
                            stats["ms1_kept"] += 1
                        except Exception as exc:
                            stats["skipped_error"] += 1
                            if stats.get("fatal_error") is None:
                                stats["fatal_error"] = f"Error while parsing spectrum: {exc!r}"
                            continue
                except Exception as exc:
                    stats["fatal_error"] = f"mzML iteration aborted: {exc!r}"
        except Exception as exc:
            stats["fatal_error"] = f"mzML read failed: {exc!r}"

        ms1.sort(key=lambda m: float(m.rt_min))
        self.ms1 = ms1
        self.stats = stats


def preview_dataframe_rows(df: pd.DataFrame, *, n: int = 10) -> List[Tuple[Any, ...]]:
    out: List[Tuple[Any, ...]] = []
    try:
        for i in range(min(int(n), int(df.shape[0]))):
            out.append(tuple(df.iloc[i].tolist()))
    except Exception:
        return []
    return out


def infer_uv_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer UV time/signal columns from headers.

    Returns dict: cols, xcol, ycol, unit_guess, low_conf, reason, x_scores, y_scores.
    """
    if int(df.shape[1]) < 2:
        raise UVLoadError("CSV must have at least 2 columns (time, signal).")

    cols = [str(c) for c in df.columns]

    def score_x(name: str) -> int:
        n = name.lower()
        score = 0
        for k in ["rt", "retention", "time", "minute", "min", "second", "sec"]:
            if k in n:
                score += 1
        return score

    def score_y(name: str) -> int:
        n = name.lower()
        score = 0
        for k in ["uv", "abs", "absorb", "au", "signal", "intensity", "counts"]:
            if k in n:
                score += 1
        return score

    x_scores = {c: score_x(c) for c in cols}
    x_best = max(cols, key=lambda c: x_scores.get(c, 0))
    y_candidates = [c for c in cols if c != x_best]
    y_scores = {c: score_y(c) for c in y_candidates}
    y_best = max(y_candidates, key=lambda c: y_scores.get(c, 0)) if y_candidates else x_best

    # Confidence heuristic: require a positive score and a clear winner.
    x_sorted = sorted((v, k) for k, v in x_scores.items())
    y_sorted = sorted((v, k) for k, v in y_scores.items())
    x_top = x_sorted[-1][0] if x_sorted else 0
    x_2nd = x_sorted[-2][0] if len(x_sorted) >= 2 else -1
    y_top = y_sorted[-1][0] if y_sorted else 0
    y_2nd = y_sorted[-2][0] if len(y_sorted) >= 2 else -1
    low_conf = (x_top <= 0) or (y_top <= 0) or (x_top == x_2nd and x_top > 0) or (y_top == y_2nd and y_top > 0)

    unit_guess = "minutes"
    xname = str(x_best).lower()
    if ("sec" in xname) or ("second" in xname):
        unit_guess = "seconds"
    elif ("min" in xname) or ("minute" in xname):
        unit_guess = "minutes"

    reason = ""
    if low_conf:
        reason = f"Heuristic scores were ambiguous (x best={x_best} score={x_scores.get(x_best, 0)}, y best={y_best} score={y_scores.get(y_best, 0)})."

    return {
        "cols": cols,
        "xcol": str(x_best),
        "ycol": str(y_best),
        "unit_guess": str(unit_guess),
        "low_conf": bool(low_conf),
        "reason": str(reason),
        "x_scores": dict(x_scores),
        "y_scores": dict(y_scores),
    }


def parse_uv_arrays(df: pd.DataFrame, *, xcol: str, ycol: str, unit_guess: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], List[str]]:
    """Convert UV DataFrame columns into sorted, de-duplicated arrays.

    Returns: (rt_min, signal, rt_range, import_warnings)
    """
    import_warnings: List[str] = []

    try:
        x = pd.to_numeric(df[str(xcol)], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[str(ycol)], errors="coerce").to_numpy(dtype=float)
    except Exception as exc:
        raise UVLoadError(f"Failed to parse columns '{xcol}'/'{ycol}': {exc}")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if int(x.size) == 0:
        raise UVLoadError("No numeric data found in the selected CSV columns.")

    # Always store minutes.
    if str(unit_guess).lower().startswith("sec"):
        x = x / 60.0
    else:
        try:
            if float(np.nanmax(x)) > 500.0:
                import_warnings.append("Time values look large; if this CSV is in seconds, choose 'seconds' in UV import settings.")
        except Exception:
            pass

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Handle duplicate RTs (average signal)
    try:
        if x.size > 1:
            ux, inv = np.unique(x, return_inverse=True)
            if ux.size != x.size:
                sumy = np.bincount(inv, weights=y)
                cnt = np.bincount(inv)
                y = sumy / np.maximum(1.0, cnt)
                x = ux
                import_warnings.append("Duplicate RT values detected; averaged signal for identical RTs.")
    except Exception:
        pass

    rt_min = np.asarray(x, dtype=float)
    signal = np.asarray(y, dtype=float)

    try:
        rt_range = (float(np.min(rt_min)), float(np.max(rt_min)))
    except Exception:
        rt_range = (float("nan"), float("nan"))

    return rt_min, signal, rt_range, import_warnings
