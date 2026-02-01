from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class FTIRLoadError(Exception):
    pass


def _try_parse_float_pair(line: str) -> Optional[Tuple[float, float]]:
    s = (line or "").strip()
    if not s:
        return None
    # Quick reject for obvious non-data lines
    if any(ch.isalpha() for ch in s):
        return None
    # Common delimiters: comma/semicolon/tab/space
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    elif ";" in s:
        parts = [p.strip() for p in s.split(";")]
    else:
        parts = s.split()
    if len(parts) < 2:
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        return x, y
    except Exception:
        return None


def _parse_ftir_xy_only(path_str: str) -> Tuple[List[float], List[float], Dict[str, str]]:
    """Parse FTIR exports and return ONLY numeric XY pairs.

    Supports JASCO-style metadata + `XYDATA` blocks (CSV), and generic text/CSV
    containing numeric pairs. Designed to be safe to run in a subprocess.
    """

    # Avoid resolve() here; on some Windows/OneDrive setups it can block.
    p = Path(path_str).expanduser()
    meta: Dict[str, str] = {}
    xs: List[float] = []
    ys: List[float] = []

    # First pass: look for XYDATA marker and parse only the numeric block.
    in_xy = False
    with p.open("r", errors="ignore") as fh:
        for line_i, raw in enumerate(fh):
            # Cooperative yield so Tk doesn't appear frozen on Windows when parsing in threads.
            # (Pure-Python parsing can otherwise starve the UI via the GIL.)
            if line_i and (line_i % 500 == 0):
                time.sleep(0)
            line = raw.strip("\r\n")
            if not in_xy:
                if line.strip().upper() == "XYDATA":
                    in_xy = True
                    continue
                # Collect KEY,VALUE metadata lines (best effort)
                if "," in line:
                    try:
                        k, v = line.split(",", 1)
                        k = str(k).strip().upper()
                        v = str(v).strip()
                        if k:
                            meta[k] = v
                    except Exception:
                        pass
                continue

            pair = _try_parse_float_pair(line)
            if pair is None:
                # Ignore occasional non-data lines; do not stop early.
                continue
            x, y = pair
            xs.append(float(x))
            ys.append(float(y))

    if len(xs) >= 5:
        return xs, ys, meta

    # Fallback: no XYDATA marker (or empty) — scan for numeric pairs anywhere.
    xs.clear()
    ys.clear()
    with p.open("r", errors="ignore") as fh:
        for line_i, raw in enumerate(fh):
            if line_i and (line_i % 500 == 0):
                time.sleep(0)
            pair = _try_parse_float_pair(raw)
            if pair is None:
                continue
            x, y = pair
            xs.append(float(x))
            ys.append(float(y))

    return xs, ys, meta


def _parse_ftir_xy_numpy(path_str: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """Parse FTIR exports and return numeric XY as numpy arrays.

    This prefers numpy's C-based parsing (releases the GIL) to avoid starving Tk
    when running in background threads.
    """

    p = Path(path_str).expanduser()
    meta: Dict[str, str] = {}
    xy_idx: Optional[int] = None

    # Lightweight scan for metadata + XYDATA marker (best effort).
    try:
        with p.open("r", errors="ignore") as fh:
            for i, raw in enumerate(fh):
                if i and (i % 2000 == 0):
                    time.sleep(0)
                s = (raw or "").strip("\r\n")
                if s.strip().upper() == "XYDATA":
                    xy_idx = int(i)
                    break
                if "," in s:
                    try:
                        k, v = s.split(",", 1)
                        k = str(k).strip().upper()
                        v = str(v).strip()
                        if k:
                            meta[k] = v
                    except Exception:
                        pass
                if i > 5000:
                    break
    except Exception:
        xy_idx = None

    # Guess delimiter from a few lines after XYDATA.
    delim: Optional[str] = None
    if xy_idx is not None:
        try:
            samples: List[str] = []
            with p.open("r", errors="ignore") as fh2:
                for i, raw in enumerate(fh2):
                    if i <= int(xy_idx):
                        continue
                    s = (raw or "").strip()
                    if not s:
                        continue
                    samples.append(s)
                    if len(samples) >= 8:
                        break
            counts = {",": 0, ";": 0, "\t": 0}
            for s in samples:
                for d in counts:
                    counts[d] += int(s.count(d))
            best = max(counts, key=lambda k: counts[k])
            delim = (str(best) if int(counts[best]) > 0 else None)
        except Exception:
            delim = None

    def _load(skip: int, delimiter: Optional[str]) -> np.ndarray:
        # genfromtxt tolerates mixed/header lines (yields NaNs) and runs in C.
        try:
            cw = np.lib._iotools.ConversionWarning  # type: ignore[attr-defined]
        except Exception:
            cw = Warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", cw)
            arr = np.genfromtxt(
                str(p),
                delimiter=delimiter,
                skip_header=int(skip),
                usecols=(0, 1),
                invalid_raise=False,
                encoding=None,
            )
        if arr.ndim == 1:
            arr = np.atleast_2d(arr)
        return np.asarray(arr, dtype=float)

    # Prefer numeric block after XYDATA when present.
    try:
        if xy_idx is not None:
            arr = _load(int(xy_idx) + 1, delim)
        else:
            # Generic fallback: parse whole file (header rows become NaN, filtered out below).
            arr = _load(0, None)
    except Exception:
        # Last resort: use the pure-Python parser.
        xs, ys, meta2 = _parse_ftir_xy_only(path_str)
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        return x, y, (meta2 or meta)

    if arr.shape[1] < 2:
        raise FTIRLoadError("File must contain at least two numeric columns")
    x = np.asarray(arr[:, 0], dtype=float)
    y = np.asarray(arr[:, 1], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    return x, y, meta


def _ftir_parse_for_executor(path_str: str) -> Tuple[str, Optional[List[float]], Optional[List[float]], Dict[str, str], Optional[str]]:
    """Subprocess entrypoint: returns (path, xs, ys, meta, error_text)."""
    try:
        xs, ys, meta = _parse_ftir_xy_only(path_str)
        if len(xs) < 5:
            return path_str, None, None, meta, "No usable numeric XY data found"
        return path_str, xs, ys, meta, None
    except Exception as exc:
        return path_str, None, None, {}, str(exc)
