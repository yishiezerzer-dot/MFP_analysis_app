from __future__ import annotations

"""FTIR analysis utilities (no GUI, no plotting).

This module is intentionally dependency-light. If SciPy is available, peak picking
uses `scipy.signal.find_peaks`; otherwise it falls back to a simple local-maxima
finder with an approximate prominence estimate.

Conventions
- `wn` is wavenumber in cm^-1 (or any monotonic x-units).
- `mode="absorbance"`: peaks are local maxima.
- `mode="transmittance"`: absorption bands are local minima, so we pick peaks on
  `-y` (but return the original `y` at the selected `wn`).

All functions tolerate lists/arrays, NaNs/Infs, and unsorted x.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import math

import numpy as np


def preprocess_spectrum(
    wn: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    mode: str = "absorbance",
    smoothing_window: int = 0,
    poly_order: int = 2,
    baseline: str = "none",
    normalize: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess an FTIR spectrum for peak picking.

    This function does NOT pick peaks; it only returns processed arrays.

    Steps (in order):
      1) sanitize + sort by `wn`
      2) optional smoothing
      3) optional baseline correction (polyfit)
      4) optional normalization (max/area)

    Args:
        wn: x array (wavenumber).
        y: signal array.
        mode: "absorbance" or "transmittance" (used only for sensible defaults;
            this function returns processed y in the original orientation).
        smoothing_window: Savitzky-Golay (SciPy) or moving-average window size.
            Use 0/1 to disable.
        poly_order: For Savitzky-Golay (if SciPy available). Must be < window.
        baseline: "none" or "polyfit".
        normalize: "none", "max", or "area".

    Returns:
        (wn_sorted, y_processed)

    Notes:
        - Non-finite points are dropped.
        - If the result is empty, returns empty arrays.
        - Baseline polyfit uses degree=min(3, n-1).
    """

    x, y0 = _sanitize_xy(wn, y)
    if x.size == 0:
        return x, y0

    y_out = np.asarray(y0, dtype=float)

    # --- smoothing ---
    w = int(smoothing_window or 0)
    if w >= 3:
        # Make odd for SavGol
        if (w % 2) == 0:
            w += 1
        if w >= 3 and w <= int(y_out.size):
            y_out = _smooth(y_out, window=w, poly_order=int(poly_order or 0))

    # --- baseline correction ---
    b = (baseline or "none").strip().lower()
    if b == "polyfit":
        y_out = _baseline_polyfit(x, y_out)

    # --- normalization ---
    nrm = (normalize or "none").strip().lower()
    if nrm == "max":
        y_out = _normalize_max(y_out)
    elif nrm == "area":
        y_out = _normalize_area(x, y_out)

    return x, np.asarray(y_out, dtype=float)


def _smooth(y: np.ndarray, *, window: int, poly_order: int) -> np.ndarray:
    """Smooth with Savitzky-Golay if available else moving average."""

    yy = np.asarray(y, dtype=float)
    if yy.size < 3:
        return yy

    w = int(window)
    if w < 3:
        return yy
    if (w % 2) == 0:
        w += 1
    if w > int(yy.size):
        w = int(yy.size) if (int(yy.size) % 2 == 1) else int(yy.size) - 1
    if w < 3:
        return yy

    po = int(poly_order or 0)
    po = max(0, po)
    po = min(po, w - 1)
    po = min(po, 5)  # prevent absurd degrees

    try:
        from scipy.signal import savgol_filter  # type: ignore

        try:
            return np.asarray(savgol_filter(yy, window_length=w, polyorder=po, mode="interp"), dtype=float)
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: moving average (edge-padded)
    k = w
    pad = k // 2
    ypad = np.pad(yy, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=float) / float(k)
    return np.convolve(ypad, kernel, mode="valid").astype(float)


def _baseline_polyfit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    n = int(yy.size)
    if n < 4:
        return yy

    deg = min(3, n - 1)
    try:
        coeff = np.polyfit(xx, yy, deg=int(deg))
        base = np.polyval(coeff, xx)
        return (yy - base).astype(float)
    except Exception:
        return yy


def _normalize_max(y: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=float)
    try:
        m = float(np.nanmax(np.abs(yy)))
        if not math.isfinite(m) or m <= 0.0:
            return yy
        return (yy / m).astype(float)
    except Exception:
        return yy


def _normalize_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    try:
        area = float(np.trapz(np.abs(yy), xx))
        if not math.isfinite(area) or area <= 0.0:
            return yy
        return (yy / area).astype(float)
    except Exception:
        return yy


@dataclass(frozen=True)
class FTIRPeak:
    """A picked FTIR peak.

    Notes:
    - `y` is the original signal value at the picked `wn`.
    - `prominence` is always positive and refers to the prominence in the peak-
      picking direction (absorbance=maxima; transmittance=minima via `-y`).
    """

    wn: float
    y: float
    prominence: float
    left_base_wn: Optional[float] = None
    right_base_wn: Optional[float] = None
    width_cm1: Optional[float] = None


def format_peak_label(peak: FTIRPeak, *, fmt: str = "{wn:.1f}") -> str:
    """Format a label for a peak.

    `fmt` is a `str.format` template with fields: `wn`, `y`, `prominence`.

    Examples:
        format_peak_label(p) -> "1720.5"
        format_peak_label(p, fmt="{wn:.0f}") -> "1720"
        format_peak_label(p, fmt="{wn:.0f} ({prominence:.3g})")
    """

    try:
        return fmt.format(wn=float(peak.wn), y=float(peak.y), prominence=float(peak.prominence))
    except Exception:
        # Safe fallback
        try:
            return f"{float(peak.wn):.1f}"
        except Exception:
            return str(peak.wn)


def pick_peaks(
    wn: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    mode: str = "absorbance",
    min_prominence: float = 0.01,
    min_height: Optional[float] = None,
    min_distance_cm1: float = 8.0,
    top_n: int = 0,
) -> List[FTIRPeak]:
    """Pick peaks from an FTIR spectrum.

    Args:
        wn: X axis (wavenumber). Can be unsorted.
        y: Signal values.
        mode: "absorbance" (maxima) or "transmittance" (minima).
        min_prominence: Minimum prominence (in y units) in the peak-picking direction.
        min_height: Optional minimum height threshold in the peak-picking direction.
        min_distance_cm1: Minimum separation between returned peaks, in x units.
            This is enforced in *cm^-1*, not in sample indices.
        top_n: If > 0, keep only the top N peaks by prominence (after distance filtering).

    Returns:
        A list of `FTIRPeak`, sorted by increasing `wn`.

    Edge handling:
        - Empty input, all-NaN, or constant signals return an empty list.
        - Non-finite pairs are removed.
        - If `wn` is not strictly monotonic, points are sorted by `wn`.
    """

    x, y0 = _sanitize_xy(wn, y)
    if x.size < 3:
        return []

    mode_norm = (mode or "absorbance").strip().lower()
    if mode_norm not in ("absorbance", "transmittance"):
        mode_norm = "absorbance"

    # Peak-picking signal direction
    y_pick = y0 if mode_norm == "absorbance" else -y0

    # Quick exit for constant-ish signals
    try:
        if float(np.nanmax(y_pick) - np.nanmin(y_pick)) == 0.0:
            return []
    except Exception:
        return []

    candidates = _pick_candidates(x, y0, y_pick, min_prominence=min_prominence, min_height=min_height)
    if not candidates:
        return []

    # Enforce min distance in x-units (cm^-1) using prominence-first greedy selection.
    selected = _enforce_min_distance(candidates, min_distance_cm1=float(min_distance_cm1 or 0.0))

    # Keep top N by prominence if requested.
    if int(top_n or 0) > 0:
        selected = sorted(selected, key=lambda p: float(p.prominence), reverse=True)[: int(top_n)]

    # Present in ascending wavenumber order
    selected = sorted(selected, key=lambda p: float(p.wn))
    return selected


def _sanitize_xy(wn: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(wn, dtype=float)
    yv = np.asarray(y, dtype=float)
    if x.shape != yv.shape:
        # Attempt a minimal reshape/flatten match
        x = np.ravel(x)
        yv = np.ravel(yv)
        n = min(int(x.size), int(yv.size))
        x = x[:n]
        yv = yv[:n]

    mask = np.isfinite(x) & np.isfinite(yv)
    x = x[mask]
    yv = yv[mask]

    if x.size == 0:
        return x, yv

    # Sort by x to make distances meaningful
    try:
        order = np.argsort(x)
        x = x[order]
        yv = yv[order]
    except Exception:
        pass

    return x, yv


def _pick_candidates(
    x: np.ndarray,
    y_orig: np.ndarray,
    y_pick: np.ndarray,
    *,
    min_prominence: float,
    min_height: Optional[float],
) -> List[FTIRPeak]:
    # Prefer SciPy if available.
    peaks = _pick_candidates_scipy(x, y_orig, y_pick, min_prominence=min_prominence, min_height=min_height)
    if peaks is not None:
        return peaks

    return _pick_candidates_fallback(x, y_orig, y_pick, min_prominence=min_prominence, min_height=min_height)


def _pick_candidates_scipy(
    x: np.ndarray,
    y_orig: np.ndarray,
    y_pick: np.ndarray,
    *,
    min_prominence: float,
    min_height: Optional[float],
) -> Optional[List[FTIRPeak]]:
    try:
        from scipy.signal import find_peaks  # type: ignore
    except Exception:
        return None

    kwargs = {}
    if min_prominence is not None:
        kwargs["prominence"] = float(min_prominence)
    if min_height is not None:
        kwargs["height"] = float(min_height)

    try:
        idx, props = find_peaks(y_pick, **kwargs)
    except Exception:
        return []

    if idx is None or len(idx) == 0:
        return []

    prominences = props.get("prominences")
    left_bases = props.get("left_bases")
    right_bases = props.get("right_bases")

    out: List[FTIRPeak] = []
    for j, i in enumerate(idx):
        try:
            ii = int(i)
            wn0 = float(x[ii])
            y0 = float(y_orig[ii])

            prom = None
            try:
                if prominences is not None:
                    prom = float(prominences[j])
            except Exception:
                prom = None
            if prom is None:
                # Approximate
                prom = float(max(0.0, _approx_prominence(ii, y_pick)))

            if prom < float(min_prominence or 0.0):
                continue

            lb = None
            rb = None
            width = None
            try:
                if left_bases is not None:
                    lb = float(x[int(left_bases[j])])
                if right_bases is not None:
                    rb = float(x[int(right_bases[j])])
                if lb is not None and rb is not None and math.isfinite(lb) and math.isfinite(rb):
                    width = float(abs(rb - lb))
            except Exception:
                lb = rb = width = None

            out.append(FTIRPeak(wn=wn0, y=y0, prominence=float(prom), left_base_wn=lb, right_base_wn=rb, width_cm1=width))
        except Exception:
            continue

    return out


def _pick_candidates_fallback(
    x: np.ndarray,
    y_orig: np.ndarray,
    y_pick: np.ndarray,
    *,
    min_prominence: float,
    min_height: Optional[float],
) -> List[FTIRPeak]:
    out: List[FTIRPeak] = []

    n = int(x.size)
    if n < 3:
        return out

    h = None if min_height is None else float(min_height)
    mp = float(min_prominence or 0.0)

    # Simple local maxima in y_pick
    for i in range(1, n - 1):
        yp = float(y_pick[i])
        if not (yp > float(y_pick[i - 1]) and yp >= float(y_pick[i + 1])):
            continue
        if h is not None and yp < h:
            continue

        prom, lb_i, rb_i = _approx_prominence_with_bases(i, y_pick)
        if prom < mp:
            continue

        lb = float(x[lb_i]) if lb_i is not None else None
        rb = float(x[rb_i]) if rb_i is not None else None
        width = None
        try:
            if lb is not None and rb is not None:
                width = float(abs(rb - lb))
        except Exception:
            width = None

        out.append(
            FTIRPeak(
                wn=float(x[i]),
                y=float(y_orig[i]),
                prominence=float(prom),
                left_base_wn=lb,
                right_base_wn=rb,
                width_cm1=width,
            )
        )

    return out


def _approx_prominence(i: int, y: np.ndarray) -> float:
    prom, _, _ = _approx_prominence_with_bases(i, y)
    return float(prom)


def _approx_prominence_with_bases(i: int, y: np.ndarray) -> Tuple[float, Optional[int], Optional[int]]:
    """Approximate prominence using a simple valley search.

    This is not identical to SciPy's prominence, but is monotonic and works
    reasonably for FTIR spectra.
    """

    n = int(y.size)
    if i <= 0 or i >= n - 1:
        return 0.0, None, None

    peak = float(y[i])

    # Search left until the signal rises above the peak; track minimum.
    left_min = peak
    left_base = i
    j = i
    while j > 0:
        j -= 1
        v = float(y[j])
        if v < left_min:
            left_min = v
            left_base = j
        if v > peak:
            break

    # Search right similarly.
    right_min = peak
    right_base = i
    k = i
    while k < n - 1:
        k += 1
        v = float(y[k])
        if v < right_min:
            right_min = v
            right_base = k
        if v > peak:
            break

    baseline = max(left_min, right_min)
    prom = max(0.0, peak - baseline)
    return float(prom), int(left_base), int(right_base)


def _enforce_min_distance(peaks: List[FTIRPeak], min_distance_cm1: float) -> List[FTIRPeak]:
    md = float(min_distance_cm1 or 0.0)
    if md <= 0.0 or len(peaks) <= 1:
        return list(peaks)

    # Greedy selection: highest prominence first, then reject too-close peaks.
    ordered = sorted(peaks, key=lambda p: float(p.prominence), reverse=True)
    chosen: List[FTIRPeak] = []

    for p in ordered:
        wn0 = float(p.wn)
        ok = True
        for q in chosen:
            if abs(wn0 - float(q.wn)) < md:
                ok = False
                break
        if ok:
            chosen.append(p)

    return chosen


# -------------------- manual self-checks --------------------


def _self_check_empty() -> None:
    assert pick_peaks([], []) == []
    assert pick_peaks([1.0, 2.0], [0.0, 1.0]) == []


def _self_check_simple_absorbance() -> None:
    x = np.linspace(1000, 2000, 2001)
    y = np.exp(-0.5 * ((x - 1500) / 10.0) ** 2) + 0.02 * np.exp(-0.5 * ((x - 1700) / 20.0) ** 2)
    peaks = pick_peaks(x, y, mode="absorbance", min_prominence=0.01, min_distance_cm1=20.0)
    assert len(peaks) >= 1
    # Strongest should be near 1500
    p0 = sorted(peaks, key=lambda p: p.prominence, reverse=True)[0]
    assert abs(p0.wn - 1500) < 2.0


def _self_check_transmittance_minima() -> None:
    x = np.linspace(1000, 2000, 2001)
    # transmittance dips
    y = 1.0 - 0.3 * np.exp(-0.5 * ((x - 1600) / 12.0) ** 2)
    peaks = pick_peaks(x, y, mode="transmittance", min_prominence=0.05, min_distance_cm1=10.0)
    assert len(peaks) >= 1
    p0 = peaks[0]
    assert abs(p0.wn - 1600) < 2.0
    # y should be the original (dip), not inverted
    assert p0.y < 1.0


def _self_check_nan_handling() -> None:
    x = np.array([1000, 1001, np.nan, 1003, 1004, 1005], dtype=float)
    y = np.array([0.0, 1.0, 2.0, np.nan, 1.0, 0.0], dtype=float)
    peaks = pick_peaks(x, y, min_prominence=0.1)
    assert isinstance(peaks, list)


def run_self_checks(*, verbose: bool = True) -> None:
    """Run lightweight self-checks (manual, no pytest needed)."""

    checks = [
        _self_check_empty,
        _self_check_simple_absorbance,
        _self_check_transmittance_minima,
        _self_check_nan_handling,
    ]

    for fn in checks:
        fn()
        if verbose:
            print(f"ok: {fn.__name__}")

    if verbose:
        print("All FTIR self-checks passed")


if __name__ == "__main__":
    run_self_checks(verbose=True)
