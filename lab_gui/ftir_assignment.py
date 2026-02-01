from __future__ import annotations

"""FTIR peak assignment (library-based suggestions).

Pure functions only:
- no Tkinter
- no file I/O
- deterministic scoring

The goal is to provide *candidate* functional-group/bond assignments per peak.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math


def assign_ftir_peaks(
    peaks: Sequence[Dict[str, Any]],
    library: Sequence[Dict[str, Any]],
    spectrum_context: Optional[Dict[str, Any]] = None,
    *,
    top_n: int = 3,
    min_score: float = 35.0,
) -> List[Dict[str, Any]]:
    """Assign FTIR peak candidates from a correlation library.

    Args:
        peaks: list of dicts: {"wn": float, "height": float|None, "width": float|None,
            "prominence": float|None, "sharpness": float|None}
        library: list of correlation entries (dicts). Each entry must have:
            id, range_cm1=(min,max), label, typical_shape, typical_intensity,
            notes, context_hints={positive:[...], negative:[...]}
        spectrum_context: optional extra info (ignored by default; reserved).
        top_n: number of candidates to return per peak.
        min_score: minimum score to include; if none exceed, return top-1 with low confidence.

    Returns:
        List of assignments, one per input peak, each:
            {"wn": ..., "peak_metrics": {...}, "candidates": [{id,label,score,reasons}...]}

    Scoring overview (0–100):
        - Base score: closeness to range center (only if within range).
        - Shape: match inferred peak shape vs typical_shape.
        - Intensity: match inferred intensity vs typical_intensity.
        - Context: supporting/conflicting peaks.
    """

    peaks_norm = [_normalize_peak(p) for p in (peaks or [])]
    library_norm = [_normalize_entry(e) for e in (library or [])]

    # Use spectrum-level stats for intensity inference.
    heights = [p["height"] for p in peaks_norm if _finite(p["height"]) is True]
    proms = [p["prominence"] for p in peaks_norm if _finite(p["prominence"]) is True]
    max_height = max(heights) if heights else None
    max_prom = max(proms) if proms else None

    wn_all = [p["wn"] for p in peaks_norm]

    out: List[Dict[str, Any]] = []
    for i, p in enumerate(peaks_norm):
        wn = float(p["wn"])
        peak_shape = _infer_peak_shape(width=p.get("width"), sharpness=p.get("sharpness"))
        peak_intensity = _infer_peak_intensity(p, max_height=max_height, max_prom=max_prom)

        scored: List[Dict[str, Any]] = []
        for entry in library_norm:
            score, reasons = _score_entry(
                wn=wn,
                entry=entry,
                peak_shape=peak_shape,
                peak_intensity=peak_intensity,
                other_wns=wn_all,
                self_index=i,
            )
            scored.append(
                {
                    "id": entry["id"],
                    "label": entry["label"],
                    "score": float(score),
                    "reasons": reasons,
                }
            )

        scored_sorted = sorted(scored, key=lambda c: float(c.get("score", 0.0)), reverse=True)
        keep = [c for c in scored_sorted if float(c.get("score", 0.0)) >= float(min_score)]
        keep = keep[: max(1, int(top_n or 0) or 1)]

        if not keep:
            # Always return at least one.
            keep = scored_sorted[:1]
            if keep:
                keep[0]["reasons"] = list(keep[0].get("reasons") or []) + ["low confidence"]

        # Clamp/round scores to 0–100.
        for c in keep:
            c["score"] = float(_clamp(float(c.get("score", 0.0)), 0.0, 100.0))

        out.append(
            {
                "wn": wn,
                "peak_metrics": {
                    "wn": wn,
                    "height": p.get("height"),
                    "width": p.get("width"),
                    "prominence": p.get("prominence"),
                    "sharpness": p.get("sharpness"),
                    "shape": peak_shape,
                    "intensity": peak_intensity,
                },
                "candidates": keep,
            }
        )

    return out


# ------------------------- normalization helpers -------------------------


def _normalize_peak(p: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(p or {})
    wn = _to_float(d.get("wn"), default=float("nan"))
    if not _finite(wn):
        raise ValueError("Each peak must have a finite 'wn' (cm^-1)")

    height = _to_float(d.get("height"), default=None)
    width = _to_float(d.get("width"), default=None)
    prominence = _to_float(d.get("prominence"), default=None)
    sharpness = _to_float(d.get("sharpness"), default=None)

    # Treat negative/zero width as missing.
    if _finite(width) and float(width) <= 0:
        width = None

    return {
        "wn": float(wn),
        "height": float(height) if _finite(height) else None,
        "width": float(width) if _finite(width) else None,
        "prominence": float(prominence) if _finite(prominence) else None,
        "sharpness": float(sharpness) if _finite(sharpness) else None,
    }


def _normalize_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(e or {})
    lo, hi = d.get("range_cm1") or (None, None)
    lo_f = _to_float(lo, default=float("nan"))
    hi_f = _to_float(hi, default=float("nan"))
    if not (_finite(lo_f) and _finite(hi_f)):
        raise ValueError(f"Invalid range_cm1 in entry {d.get('id')}")
    lo_v, hi_v = (float(lo_f), float(hi_f))
    if lo_v > hi_v:
        lo_v, hi_v = hi_v, lo_v

    ctx = dict(d.get("context_hints") or {})
    pos = list(ctx.get("positive") or [])
    neg = list(ctx.get("negative") or [])

    return {
        "id": str(d.get("id") or "").strip(),
        "range_cm1": (lo_v, hi_v),
        "label": str(d.get("label") or "").strip(),
        "typical_shape": [str(s).strip().lower() for s in (d.get("typical_shape") or [])],
        "typical_intensity": [str(s).strip().lower() for s in (d.get("typical_intensity") or [])],
        "notes": str(d.get("notes") or "").strip(),
        "context_hints": {"positive": pos, "negative": neg},
    }


# ------------------------------ scoring ---------------------------------


def _score_entry(
    *,
    wn: float,
    entry: Dict[str, Any],
    peak_shape: str,
    peak_intensity: str,
    other_wns: Sequence[float],
    self_index: int,
) -> Tuple[float, List[str]]:
    (lo, hi) = entry["range_cm1"]
    center = (float(lo) + float(hi)) / 2.0
    half = max(1e-9, (float(hi) - float(lo)) / 2.0)

    reasons: List[str] = []

    # Base closeness score only if within range.
    within = (float(lo) <= float(wn) <= float(hi))
    if within:
        closeness = 1.0 - min(1.0, abs(float(wn) - center) / half)
        base = 70.0 * closeness
        reasons.append(f"within {lo:.0f}–{hi:.0f} cm^-1")
    else:
        base = 0.0

    score = float(base)

    # Shape adjustment
    ts = set(entry.get("typical_shape") or [])
    if peak_shape and ts:
        if peak_shape in ts:
            score += 10.0
            reasons.append(f"shape matches ({peak_shape})")
        else:
            score -= 6.0
            reasons.append(f"shape mismatch ({peak_shape} vs {', '.join(sorted(ts))})")

    # Intensity adjustment
    ti = set(entry.get("typical_intensity") or [])
    if peak_intensity and ti:
        if "variable" in ti:
            score += 0.0
        elif peak_intensity in ti:
            score += 8.0
            reasons.append(f"intensity matches ({peak_intensity})")
        else:
            score -= 5.0
            reasons.append(f"intensity mismatch ({peak_intensity} vs {', '.join(sorted(ti))})")

    # Context adjustment (supporting/conflicting peaks)
    ctx = dict(entry.get("context_hints") or {})
    pos = list(ctx.get("positive") or [])
    neg = list(ctx.get("negative") or [])

    other = [float(w) for j, w in enumerate(other_wns) if j != int(self_index)]

    for pat in pos:
        r = pat.get("range_cm1") or (None, None)
        plo = _to_float(r[0] if isinstance(r, (list, tuple)) and len(r) >= 2 else None, default=None)
        phi = _to_float(r[1] if isinstance(r, (list, tuple)) and len(r) >= 2 else None, default=None)
        if not (_finite(plo) and _finite(phi)):
            continue
        lo2, hi2 = (float(plo), float(phi))
        if lo2 > hi2:
            lo2, hi2 = hi2, lo2
        if any((lo2 <= w <= hi2) for w in other):
            score += 7.0
            reasons.append(str(pat.get("text") or f"supporting peak in {lo2:.0f}–{hi2:.0f}").strip())

    for pat in neg:
        r = pat.get("range_cm1") or (None, None)
        nlo = _to_float(r[0] if isinstance(r, (list, tuple)) and len(r) >= 2 else None, default=None)
        nhi = _to_float(r[1] if isinstance(r, (list, tuple)) and len(r) >= 2 else None, default=None)
        if not (_finite(nlo) and _finite(nhi)):
            continue
        lo2, hi2 = (float(nlo), float(nhi))
        if lo2 > hi2:
            lo2, hi2 = hi2, lo2
        if any((lo2 <= w <= hi2) for w in other):
            score -= 10.0
            reasons.append(str(pat.get("text") or f"conflicting peak in {lo2:.0f}–{hi2:.0f}").strip())

    score = _clamp(score, 0.0, 100.0)
    return float(score), reasons


# ------------------------- peak feature inference ------------------------


def _infer_peak_shape(*, width: Optional[float], sharpness: Optional[float]) -> str:
    """Infer one of: sharp/medium/broad."""

    w = width if _finite(width) else None
    s = sharpness if _finite(sharpness) else None

    # Prefer width (cm^-1) if available.
    if w is not None:
        if w <= 15.0:
            return "sharp"
        if w <= 40.0:
            return "medium"
        return "broad"

    # Fallback to sharpness heuristic.
    if s is not None:
        if s >= 0.08:
            return "sharp"
        if s >= 0.03:
            return "medium"
        return "broad"

    return "medium"


def _infer_peak_intensity(p: Dict[str, Any], *, max_height: Optional[float], max_prom: Optional[float]) -> str:
    """Infer weak/medium/strong/variable from relative height/prominence."""

    # Prefer prominence, else height.
    prom = p.get("prominence")
    height = p.get("height")

    if _finite(prom) and _finite(max_prom) and float(max_prom) > 0:
        rel = float(prom) / float(max_prom)
    elif _finite(height) and _finite(max_height) and float(max_height) > 0:
        rel = float(height) / float(max_height)
    else:
        return "variable"

    if rel >= 0.66:
        return "strong"
    if rel >= 0.33:
        return "medium"
    return "weak"


# ------------------------------ utilities --------------------------------


def _to_float(x: Any, *, default: Any) -> Any:
    if x is None:
        return default
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))
