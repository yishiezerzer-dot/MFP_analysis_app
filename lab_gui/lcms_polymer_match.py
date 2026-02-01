from __future__ import annotations

from dataclasses import dataclass
from math import comb
import os

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# Keep constants consistent with existing app defaults/labels.
PROTON_MASS = 1.007276
NA_MASS = 22.989218
K_MASS = 38.963158
NH4_MASS = 18.033823
CL_MASS = 34.968853
FORMATE_MASS = 44.997655  # HCOO-
ACETATE_MASS = 59.013851  # CH3COO-

OXIDATION_MASS = 15.994915
CO2_LOSS_MASS = 43.989829


class PolymerSearchTooLarge(RuntimeError):
    pass


@dataclass(frozen=True)
class IonCandidate:
    mz: float
    label_suffix: str  # e.g. "[M+H]+"
    charge: int
    mass_delta: float


@dataclass(frozen=True)
class Variant:
    mass_delta: float
    tag: str  # e.g. "", "+O", "-CO2", "+O-CO2"


@dataclass(frozen=True)
class PeakMatch:
    matched_mz: float
    intensity: float
    abs_err: float
    ppm_err: float
    index: int  # index into the ORIGINAL (unsorted) input arrays


def estimate_num_compositions(n_monomers: int, max_dp: int, min_dp: int = 1) -> int:
    """Estimate how many compositions will be generated.

    Counts weak compositions (allowing zeros) for each total DP and sums.
    """
    n = int(n_monomers)
    if n <= 0:
        return 0
    lo = int(min_dp)
    hi = int(max_dp)
    if hi < lo:
        return 0
    lo = max(0, lo)
    hi = max(0, hi)
    # #solutions to c1+..+cn = k with ci>=0 is C(k+n-1, n-1)
    return int(sum(comb(int(k) + n - 1, n - 1) for k in range(lo, hi + 1)))


def generate_polymer_compositions(n_monomers: int, max_dp: int, min_dp: int = 1) -> Iterator[Tuple[int, ...]]:
    """Yield integer count vectors c[0..n-1] where min_dp <= sum(c) <= max_dp.

    This is a generator and is intended to be memory-safe.
    """
    n = int(n_monomers)
    if n <= 0:
        return
        yield  # pragma: no cover

    lo = int(min_dp)
    hi = int(max_dp)
    if hi < lo:
        return
        yield  # pragma: no cover

    lo = max(0, lo)
    hi = max(0, hi)

    def gen_for_total(total: int) -> Iterator[Tuple[int, ...]]:
        if n == 1:
            yield (int(total),)
            return

        prefix: List[int] = []

        def rec(i: int, remaining: int) -> Iterator[Tuple[int, ...]]:
            if i == n - 1:
                yield tuple(prefix + [int(remaining)])
                return
            for c in range(0, int(remaining) + 1):
                prefix.append(int(c))
                yield from rec(i + 1, int(remaining) - int(c))
                prefix.pop()

        yield from rec(0, int(total))

    for total in range(lo, hi + 1):
        if total <= 0:
            continue
        yield from gen_for_total(int(total))


def generate_variants(
    *,
    max_ox: int = 1,
    max_decarb: int = 1,
    allow_combo: bool = True,
) -> List[Variant]:
    max_ox = max(0, int(max_ox))
    max_decarb = max(0, int(max_decarb))
    out: List[Variant] = []
    for ox in range(0, max_ox + 1):
        for dec in range(0, max_decarb + 1):
            if not allow_combo and ox > 0 and dec > 0:
                continue
            delta = float(ox) * OXIDATION_MASS - float(dec) * CO2_LOSS_MASS

            tag_parts: List[str] = []
            if ox > 0:
                tag_parts.append("+O" if ox == 1 else f"+{ox}O")
            if dec > 0:
                tag_parts.append("-CO2" if dec == 1 else f"-{dec}CO2")
            tag = "".join(tag_parts)
            out.append(Variant(mass_delta=float(delta), tag=str(tag)))

    # De-dup (shouldn't be needed, but safe)
    seen: set[Tuple[float, str]] = set()
    uniq: List[Variant] = []
    for v in out:
        k = (round(float(v.mass_delta), 12), str(v.tag))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(v)
    return uniq


def _auto_sign_proton_like(delta_mass: float, polarity: Optional[str]) -> float:
    pol = (polarity or "").strip().lower()
    if pol not in ("positive", "negative"):
        return float(delta_mass)
    if abs(abs(float(delta_mass)) - PROTON_MASS) <= 0.01:
        sign = 1.0 if pol == "positive" else -1.0
        return float(sign) * abs(float(delta_mass))
    return float(delta_mass)


def build_default_adduct_deltas(
    *,
    polarity: Optional[str],
    base_adduct_mass: float,
    enable_na: bool,
    enable_k: bool,
    enable_cl: bool,
    enable_formate: bool,
    enable_nh4_default: bool = False,
    enable_acetate_default: bool = False,
) -> List[Tuple[str, float]]:
    """Return list of (label_for_existing_UI, mass_delta).

    The first entry is always the user-provided base adduct (label "").
    """
    pol = (polarity or "").strip().lower()

    base = _auto_sign_proton_like(float(base_adduct_mass), pol)
    out: List[Tuple[str, float]] = [("", float(base))]

    if pol == "negative":
        # Negative mode: only consider additional adducts when explicitly enabled.
        if enable_cl:
            out.append(("+Cl", float(CL_MASS)))
        if enable_formate:
            out.append(("+HCOO", float(FORMATE_MASS)))
        if enable_acetate_default:
            out.append(("+Ac", float(ACETATE_MASS)))
    else:
        # Positive/unknown: only consider additional adducts when explicitly enabled.
        if pol == "positive" and enable_nh4_default:
            out.append(("+NH4", float(NH4_MASS)))
        if enable_na:
            out.append(("+Na", float(NA_MASS)))
        if enable_k:
            out.append(("+K", float(K_MASS)))

    # De-dup by (label, delta)
    seen: set[Tuple[str, float]] = set()
    uniq: List[Tuple[str, float]] = []
    for lbl, dm in out:
        k = (str(lbl), round(float(dm), 9))
        if k in seen:
            continue
        seen.add(k)
        uniq.append((str(lbl), float(dm)))
    return uniq


def find_best_peak_match(
    mz_array,
    inten_array,
    target_mz: float,
    tol_da: Optional[float] = None,
    tol_ppm: Optional[float] = None,
) -> Optional[PeakMatch]:
    """Return best match in tolerance window.

    - considers all peaks within window
    - chooses best by smallest ppm error, then highest intensity
    - handles unsorted arrays (sorts safely)

    If both tol_da and tol_ppm are provided, uses the larger window in Da.
    """
    try:
        mz_vals = np.asarray(mz_array, dtype=float)
        int_vals = np.asarray(inten_array, dtype=float)
    except Exception:
        return None

    if mz_vals.size == 0 or int_vals.size == 0:
        return None

    mask = np.isfinite(mz_vals) & np.isfinite(int_vals)
    if not np.any(mask):
        return None
    mz_vals = mz_vals[mask]
    int_vals = int_vals[mask]

    if mz_vals.size == 0:
        return None

    t = float(target_mz)
    da1 = 0.0 if tol_da is None else float(abs(float(tol_da)))
    da2 = 0.0
    if tol_ppm is not None:
        try:
            da2 = abs(float(t)) * (float(abs(float(tol_ppm))) * 1e-6)
        except Exception:
            da2 = 0.0
    win = float(max(da1, da2))
    if win <= 0.0:
        return None

    # Sort if needed
    try:
        needs_sort = bool(mz_vals.size > 2 and np.any(np.diff(mz_vals) < 0))
    except Exception:
        needs_sort = True

    if needs_sort:
        order = np.argsort(mz_vals)
        mz_s = mz_vals[order]
        int_s = int_vals[order]
        orig_idx = order
    else:
        mz_s = mz_vals
        int_s = int_vals
        orig_idx = np.arange(mz_vals.size)

    left = int(np.searchsorted(mz_s, float(t - win), side="left"))
    right = int(np.searchsorted(mz_s, float(t + win), side="right"))
    if right <= left:
        return None

    mz_win = mz_s[left:right]
    int_win = int_s[left:right]
    abs_err = np.abs(mz_win - float(t))
    ppm_err = np.zeros_like(abs_err)
    if float(t) != 0.0:
        ppm_err = abs_err / abs(float(t)) * 1e6

    # Choose smallest ppm error, then highest intensity, then smallest abs error.
    # Use argsort with structured keys.
    try:
        best_local = int(
            np.lexsort(
                (
                    abs_err,  # tie-break
                    -int_win,  # intensity desc
                    ppm_err,  # primary
                )
            )[0]
        )
    except Exception:
        best_local = int(np.argmin(ppm_err))

    best_global = int(left + best_local)
    idx_orig = int(orig_idx[best_global])
    mz_match = float(mz_s[best_global])
    inten_match = float(int_s[best_global])
    abs_e = float(abs(mz_match - float(t)))
    ppm_e = float(0.0 if float(t) == 0.0 else (abs_e / abs(float(t)) * 1e6))

    return PeakMatch(matched_mz=mz_match, intensity=inten_match, abs_err=abs_e, ppm_err=ppm_e, index=idx_orig)


def _tol_to_da(*, mz_pred: float, tol_value: float, tol_unit: str) -> Tuple[Optional[float], Optional[float]]:
    unit = (tol_unit or "Da").strip().lower()
    if unit == "ppm":
        return None, float(tol_value)
    return float(tol_value), None


def _kind_for_variant(tag: str) -> str:
    t = str(tag or "")
    has_o = "+O" in t
    has_co2 = "CO2" in t and "-CO2" in t
    if not t:
        return "poly"
    if has_o and has_co2:
        return "oxdecarb"
    if has_o:
        return "ox"
    if has_co2:
        return "decarb"
    return "poly"


def compute_polymer_best_by_peak_sorted(
    mz_sorted: np.ndarray,
    int_sorted: np.ndarray,
    *,
    monomer_names: Sequence[str],
    monomer_masses: Sequence[float],
    charges: Sequence[int],
    max_dp: int,
    bond_delta: float,
    extra_delta: float,
    polarity: Optional[str],
    base_adduct_mass: float,
    enable_decarb: bool,
    enable_oxid: bool,
    enable_cluster: bool,
    cluster_adduct_mass: float,
    enable_na: bool,
    enable_k: bool,
    enable_cl: bool,
    enable_formate: bool,
    enable_acetate: bool = False,
    tol_value: float,
    tol_unit: str,
    min_rel_int: float,
    allow_variant_combo: bool = True,
    max_combinations_warn: int = 2_000_000,
    compatibility_mode: bool = False,
) -> Dict[int, Dict[str, Tuple[float, str, float, float]]]:
    """Compute polymer matches.

    Returns best_by_peak[peak_i][kind] = (abs_err, label, mz_act, intensity).
    peak_i is an index into the provided *sorted* arrays.

    NOTE: This function is pure (no Tk calls). Raise PolymerSearchTooLarge if
    the search is estimated to exceed the provided threshold.
    """
    mz_s = np.asarray(mz_sorted, dtype=float)
    int_s = np.asarray(int_sorted, dtype=float)
    if mz_s.size == 0 or int_s.size == 0:
        return {}

    # Guard: estimate combinations before enumerating.
    n = int(len(monomer_masses))
    if n <= 0:
        return {}
    est = estimate_num_compositions(n, int(max_dp), 1)
    if int(est) > int(max_combinations_warn):
        raise PolymerSearchTooLarge(
            f"Polymer search is too large (estimated {int(est):,} compositions).\n\n"
            "Tighten constraints: reduce Max DP, reduce monomer count, or disable variants/adducts."
        )

    max_int = float(np.max(int_s)) if int_s.size else 0.0
    if max_int <= 0.0:
        return {}

    # Threshold behavior: keep existing default behavior.
    rel = max(0.0, min(1.0, float(min_rel_int)))
    min_int = float(rel) * float(max_int)

    names = [str(nm) for nm in monomer_names]
    masses = [float(m) for m in monomer_masses]

    charges_use = [int(z) for z in charges if int(z) > 0]
    if not charges_use:
        charges_use = [1]

    max_dp_i = max(1, min(200, int(max_dp)))

    # Adduct list for label compatibility ("", "+Na", etc)
    poly_adducts = build_default_adduct_deltas(
        polarity=polarity,
        base_adduct_mass=float(base_adduct_mass),
        enable_na=bool(enable_na),
        enable_k=bool(enable_k),
        enable_cl=bool(enable_cl),
        enable_formate=bool(enable_formate),
        enable_acetate_default=bool(enable_acetate),
    )

    cluster_adducts = build_default_adduct_deltas(
        polarity=polarity,
        base_adduct_mass=float(cluster_adduct_mass),
        enable_na=bool(enable_na),
        enable_k=bool(enable_k),
        enable_cl=bool(enable_cl),
        enable_formate=bool(enable_formate),
        enable_acetate_default=bool(enable_acetate),
    )

    if compatibility_mode:
        # Old behavior: oxidation = -2H only, and no ox+decarb combos.
        variants = [Variant(0.0, "")]
        if enable_oxid:
            variants.append(Variant(-2.015650, "-2H"))
        if enable_decarb:
            variants.append(Variant(-CO2_LOSS_MASS, "-CO2"))
    else:
        variants = generate_variants(
            max_ox=(1 if enable_oxid else 0),
            max_decarb=(1 if enable_decarb else 0),
            allow_combo=bool(allow_variant_combo),
        )

    # For early pruning: compute a broad neutral mass window that could map into the scan m/z range.
    tol_da0, tol_ppm0 = _tol_to_da(mz_pred=float(np.median(mz_s)), tol_value=float(tol_value), tol_unit=str(tol_unit))
    # Conservative window: compute from the largest possible Da window across m/z range.
    mz_lo = float(np.min(mz_s))
    mz_hi = float(np.max(mz_s))
    # If tolerance is ppm, max Da window occurs at mz_hi.
    extra_win = 0.0
    if tol_ppm0 is not None:
        extra_win = abs(float(mz_hi)) * (float(tol_ppm0) * 1e-6)
    elif tol_da0 is not None:
        extra_win = float(abs(float(tol_da0)))
    mz_lo -= float(extra_win)
    mz_hi += float(extra_win)

    var_min = float(min(v.mass_delta for v in variants)) if variants else 0.0
    var_max = float(max(v.mass_delta for v in variants)) if variants else 0.0

    adduct_min = float(min(dm for _lbl, dm in poly_adducts)) if poly_adducts else 0.0
    adduct_max = float(max(dm for _lbl, dm in poly_adducts)) if poly_adducts else 0.0

    neutral_min_allowed = None
    neutral_max_allowed = None
    for z in charges_use:
        lo = float(mz_lo) * float(z) - (adduct_max + var_max)
        hi = float(mz_hi) * float(z) - (adduct_min + var_min)
        neutral_min_allowed = lo if neutral_min_allowed is None else min(neutral_min_allowed, lo)
        neutral_max_allowed = hi if neutral_max_allowed is None else max(neutral_max_allowed, hi)

    best_by_peak: Dict[int, Dict[str, Tuple[float, str, float, float]]] = {}
    best_ppm_by_peak_kind: Dict[Tuple[int, str], float] = {}

    def set_best(peak_i: int, kind: str, err: float, ppm_err: float, label: str, mz_act: float, inten: float) -> None:
        d = best_by_peak.get(int(peak_i))
        if d is None:
            d = {}
            best_by_peak[int(peak_i)] = d
        key = (int(peak_i), str(kind))
        prev = d.get(str(kind))
        prev_ppm = best_ppm_by_peak_kind.get(key)

        # Prefer smaller ppm error, then higher intensity, then smaller absolute error.
        if prev is None:
            d[str(kind)] = (float(err), str(label), float(mz_act), float(inten))
            best_ppm_by_peak_kind[key] = float(ppm_err)
            return

        prev_err, _prev_label, _prev_mz, prev_inten = prev
        better = False
        if prev_ppm is None or float(ppm_err) < float(prev_ppm) - 1e-12:
            better = True
        elif abs(float(ppm_err) - float(prev_ppm or 0.0)) < 1e-12 and float(inten) > float(prev_inten) + 1e-12:
            better = True
        elif abs(float(ppm_err) - float(prev_ppm or 0.0)) < 1e-12 and abs(float(inten) - float(prev_inten)) < 1e-12 and float(err) < float(prev_err):
            better = True

        if better:
            d[str(kind)] = (float(err), str(label), float(mz_act), float(inten))
            best_ppm_by_peak_kind[key] = float(ppm_err)

    def _confidence_score(*, ppm_err: float, inten: float, max_inten: float, tol_ppm_eff: float, ambiguity_hits: int) -> float:
        inten_norm = 0.0 if max_inten <= 0 else max(0.0, min(1.0, float(inten) / float(max_inten)))
        tol_ppm_eff = max(1e-9, float(tol_ppm_eff))
        ppm_norm = max(0.0, min(1.0, 1.0 - (float(ppm_err) / float(tol_ppm_eff))))
        score = 0.65 * ppm_norm + 0.35 * inten_norm
        hits = max(1, int(ambiguity_hits))
        penalty = 1.0 / (1.0 + 0.25 * float(hits - 1))
        return max(0.0, min(1.0, float(score) * float(penalty)))

    def _maybe_min_score() -> Optional[float]:
        raw = str(os.environ.get("LAB_GUI_POLYMER_MIN_SCORE", "")).strip()
        if not raw:
            return None
        try:
            v = float(raw)
        except Exception:
            return None
        return max(0.0, min(1.0, float(v)))

    min_score = _maybe_min_score()

    def _effective_tol_ppm(*, mz_pred: float, tol_da: Optional[float], tol_ppm: Optional[float]) -> float:
        if tol_ppm is not None:
            try:
                return max(1.0, float(abs(float(tol_ppm))))
            except Exception:
                return 1.0
        if tol_da is not None:
            try:
                mzp = float(mz_pred)
                if mzp != 0.0:
                    return max(1.0, abs(float(tol_da)) / abs(mzp) * 1e6)
            except Exception:
                return 1.0
        return 1.0

    def cluster_tag(adduct_label: str, adduct_mass_val: float) -> str:
        if adduct_label:
            return f"2M{adduct_label}"
        # numeric fallback with unicode minus
        if abs(float(adduct_mass_val) - (-PROTON_MASS)) <= 0.002:
            return "2M−H"
        if abs(float(adduct_mass_val) - (PROTON_MASS)) <= 0.002:
            return "2M+H"
        return f"2M{float(adduct_mass_val):+.4f}".replace("-", "−")

    # Enumerate compositions
    for counts in generate_polymer_compositions(n, max_dp_i, 1):
        dp = int(sum(counts))
        if dp <= 0 or dp > max_dp_i:
            continue

        monomer_mass_sum = 0.0
        parts: List[str] = []
        for i, c in enumerate(counts):
            ci = int(c)
            if ci <= 0:
                continue
            monomer_mass_sum += float(ci) * float(masses[i])
            parts.append(f"{ci}-{names[i]}")

        base_label = " + ".join(parts) if parts else "polymer"
        neutral_poly = float(monomer_mass_sum) + float(dp - 1) * float(bond_delta) + float(extra_delta)

        if neutral_min_allowed is not None and neutral_max_allowed is not None:
            if float(neutral_poly) < float(neutral_min_allowed) or float(neutral_poly) > float(neutral_max_allowed):
                continue

        # Covalent polymer + variants
        for v in variants:
            neutral_var = float(neutral_poly) + float(v.mass_delta)
            kind = _kind_for_variant(str(v.tag))
            tag_txt = ("" if not v.tag else f" {str(v.tag)}")

            for z in charges_use:
                for adduct_lbl, adduct_mass_val in poly_adducts:
                    mz_pred = (float(neutral_var) + float(adduct_mass_val)) / float(z)
                    tol_da, tol_ppm = _tol_to_da(mz_pred=float(mz_pred), tol_value=float(tol_value), tol_unit=str(tol_unit))

                    match = find_best_peak_match(mz_s, int_s, float(mz_pred), tol_da=tol_da, tol_ppm=tol_ppm)
                    if match is None:
                        continue
                    # match.index is into the (already-sorted) arrays because we passed mz_s/int_s (sorted).
                    peak_i = int(match.index)
                    if not (0 <= peak_i < int(mz_s.size)):
                        continue
                    inten_act = float(int_s[peak_i])
                    if float(inten_act) < float(min_int):
                        continue

                    # Optional confidence filter (default OFF): keep behavior identical unless env var is set.
                    if min_score is not None:
                        tol_ppm_eff = _effective_tol_ppm(mz_pred=float(mz_pred), tol_da=tol_da, tol_ppm=tol_ppm)
                        score = _confidence_score(
                            ppm_err=float(match.ppm_err),
                            inten=float(inten_act),
                            max_inten=float(max_int),
                            tol_ppm_eff=float(max(1.0, tol_ppm_eff)),
                            ambiguity_hits=1,
                        )
                        if float(score) < float(min_score):
                            continue

                    suffix = f" {adduct_lbl}" if adduct_lbl else ""
                    label = f"{base_label}{tag_txt}{suffix} z={int(z)}".strip()
                    set_best(
                        peak_i,
                        str(kind),
                        float(match.abs_err),
                        float(match.ppm_err),
                        str(label),
                        float(mz_s[peak_i]),
                        float(inten_act),
                    )

        # Cluster (2M...) based on the unmodified covalent polymer mass (keeps legacy behavior).
        if enable_cluster:
            neutral_dimer = 2.0 * float(neutral_poly)
            for z in charges_use:
                for adduct_lbl, adduct_mass_val in cluster_adducts:
                    mz_pred = (float(neutral_dimer) + float(adduct_mass_val)) / float(z)
                    tol_da, tol_ppm = _tol_to_da(mz_pred=float(mz_pred), tol_value=float(tol_value), tol_unit=str(tol_unit))
                    match = find_best_peak_match(mz_s, int_s, float(mz_pred), tol_da=tol_da, tol_ppm=tol_ppm)
                    if match is None:
                        continue
                    peak_i = int(match.index)
                    if not (0 <= peak_i < int(mz_s.size)):
                        continue
                    inten_act = float(int_s[peak_i])
                    if float(inten_act) < float(min_int):
                        continue

                    if min_score is not None:
                        tol_ppm_eff = _effective_tol_ppm(mz_pred=float(mz_pred), tol_da=tol_da, tol_ppm=tol_ppm)
                        score = _confidence_score(
                            ppm_err=float(match.ppm_err),
                            inten=float(inten_act),
                            max_inten=float(max_int),
                            tol_ppm_eff=float(max(1.0, tol_ppm_eff)),
                            ambiguity_hits=1,
                        )
                        if float(score) < float(min_score):
                            continue

                    tag = cluster_tag(str(adduct_lbl), float(adduct_mass_val))
                    z_suffix = f" z={int(z)}" if len(charges_use) > 1 else ""
                    label = f"{base_label} ({tag}){z_suffix}".strip()
                    set_best(
                        peak_i,
                        "2m",
                        float(match.abs_err),
                        float(match.ppm_err),
                        str(label),
                        float(mz_s[peak_i]),
                        float(inten_act),
                    )

    return best_by_peak


def explain_best_match_for_peak_sorted(
    mz_sorted: np.ndarray,
    int_sorted: np.ndarray,
    *,
    peak_i: int,
    target_kind: str,
    monomer_names: Sequence[str],
    monomer_masses: Sequence[float],
    charges: Sequence[int],
    max_dp: int,
    bond_delta: float,
    extra_delta: float,
    polarity: Optional[str],
    base_adduct_mass: float,
    enable_decarb: bool,
    enable_oxid: bool,
    enable_cluster: bool,
    cluster_adduct_mass: float,
    enable_na: bool,
    enable_k: bool,
    enable_cl: bool,
    enable_formate: bool,
    enable_acetate: bool = False,
    tol_value: float,
    tol_unit: str,
    min_rel_int: float,
    allow_variant_combo: bool = True,
    max_combinations_warn: int = 2_000_000,
    compatibility_mode: bool = False,
) -> Optional[Dict[str, object]]:
    """Detailed explanation helper.

    Returns a dict compatible with the existing App's polymer label explanation.
    """
    mz_s = np.asarray(mz_sorted, dtype=float)
    int_s = np.asarray(int_sorted, dtype=float)
    if mz_s.size == 0 or int_s.size == 0:
        return None
    if not (0 <= int(peak_i) < int(mz_s.size)):
        return None

    n = int(len(monomer_masses))
    if n <= 0:
        return None
    est = estimate_num_compositions(n, int(max_dp), 1)
    if int(est) > int(max_combinations_warn):
        raise PolymerSearchTooLarge(
            f"Polymer search is too large (estimated {int(est):,} compositions).\n\n"
            "Tighten constraints: reduce Max DP, reduce monomer count, or disable variants/adducts."
        )

    max_int = float(np.max(int_s)) if int_s.size else 0.0
    if max_int <= 0.0:
        return None
    rel = max(0.0, min(1.0, float(min_rel_int)))
    min_int = float(rel) * float(max_int)

    names = [str(nm) for nm in monomer_names]
    masses = [float(m) for m in monomer_masses]

    charges_use = [int(z) for z in charges if int(z) > 0]
    if not charges_use:
        charges_use = [1]
    max_dp_i = max(1, min(200, int(max_dp)))

    poly_adducts = build_default_adduct_deltas(
        polarity=polarity,
        base_adduct_mass=float(base_adduct_mass),
        enable_na=bool(enable_na),
        enable_k=bool(enable_k),
        enable_cl=bool(enable_cl),
        enable_formate=bool(enable_formate),
        enable_acetate_default=bool(enable_acetate),
    )
    cluster_adducts = build_default_adduct_deltas(
        polarity=polarity,
        base_adduct_mass=float(cluster_adduct_mass),
        enable_na=bool(enable_na),
        enable_k=bool(enable_k),
        enable_cl=bool(enable_cl),
        enable_formate=bool(enable_formate),
        enable_acetate_default=bool(enable_acetate),
    )

    if compatibility_mode:
        variants = [Variant(0.0, "")]
        if enable_oxid:
            variants.append(Variant(-2.015650, "-2H"))
        if enable_decarb:
            variants.append(Variant(-CO2_LOSS_MASS, "-CO2"))
    else:
        variants = generate_variants(
            max_ox=(1 if enable_oxid else 0),
            max_decarb=(1 if enable_decarb else 0),
            allow_combo=bool(allow_variant_combo),
        )

    target_kind_s = str(target_kind)
    best: Optional[Dict[str, object]] = None

    def consider_candidate(
        *,
        kind: str,
        composition: str,
        mz_pred: float,
        z: int,
        adduct_label: str,
        adduct_mass: float,
    ) -> None:
        nonlocal best
        tol_da, tol_ppm = _tol_to_da(mz_pred=float(mz_pred), tol_value=float(tol_value), tol_unit=str(tol_unit))
        match = find_best_peak_match(mz_s, int_s, float(mz_pred), tol_da=tol_da, tol_ppm=tol_ppm)
        if match is None:
            return
        if int(match.index) != int(peak_i):
            return
        inten_act = float(int_s[int(peak_i)])
        if float(inten_act) < float(min_int):
            return
        if str(kind) != target_kind_s:
            return

        mz_act = float(mz_s[int(peak_i)])
        abs_err = float(abs(float(mz_act) - float(mz_pred)))
        if best is None or abs_err < float(best.get("abs_err", 1e99)):
            best = {
                "kind": str(kind),
                "mz_pred": float(mz_pred),
                "mz_act": float(mz_act),
                "abs_err": float(abs_err),
                "ppm_err": float(0.0 if mz_pred == 0 else (abs_err / abs(float(mz_pred)) * 1e6)),
                "z": int(z),
                "adduct_label": str(adduct_label),
                "adduct_mass": float(adduct_mass),
                "tol_value": float(tol_value),
                "tol_unit": str(tol_unit or "Da"),
                "min_rel": float(min_rel_int),
                "composition": str(composition),
            }

    def cluster_tag(adduct_label: str, adduct_mass_val: float) -> str:
        if adduct_label:
            return f"2M{adduct_label}"
        if abs(float(adduct_mass_val) - (-PROTON_MASS)) <= 0.002:
            return "2M−H"
        if abs(float(adduct_mass_val) - (PROTON_MASS)) <= 0.002:
            return "2M+H"
        return f"2M{float(adduct_mass_val):+.4f}".replace("-", "−")

    for counts in generate_polymer_compositions(n, max_dp_i, 1):
        dp = int(sum(counts))
        if dp <= 0 or dp > max_dp_i:
            continue

        monomer_mass_sum = 0.0
        parts: List[str] = []
        for i, c in enumerate(counts):
            ci = int(c)
            if ci <= 0:
                continue
            monomer_mass_sum += float(ci) * float(masses[i])
            parts.append(f"{ci}-{names[i]}")

        base_label = " + ".join(parts) if parts else "polymer"
        neutral_poly = float(monomer_mass_sum) + float(dp - 1) * float(bond_delta) + float(extra_delta)

        for v in variants:
            neutral_var = float(neutral_poly) + float(v.mass_delta)
            kind = _kind_for_variant(str(v.tag))
            comp = f"{base_label}{('' if not v.tag else ' ' + str(v.tag))}".strip()
            for z in charges_use:
                for adduct_lbl, adduct_mass_val in poly_adducts:
                    mz_pred = (float(neutral_var) + float(adduct_mass_val)) / float(z)
                    consider_candidate(
                        kind=str(kind),
                        composition=str(comp),
                        mz_pred=float(mz_pred),
                        z=int(z),
                        adduct_label=str(adduct_lbl),
                        adduct_mass=float(adduct_mass_val),
                    )

        if enable_cluster:
            neutral_dimer = 2.0 * float(neutral_poly)
            for z in charges_use:
                for adduct_lbl, adduct_mass_val in cluster_adducts:
                    mz_pred = (float(neutral_dimer) + float(adduct_mass_val)) / float(z)
                    tag = cluster_tag(str(adduct_lbl), float(adduct_mass_val))
                    comp = f"{base_label} (2M)"
                    consider_candidate(
                        kind="2m",
                        composition=str(comp),
                        mz_pred=float(mz_pred),
                        z=int(z),
                        adduct_label=str(adduct_lbl),
                        adduct_mass=float(adduct_mass_val),
                    )

    return best


def run_polymer_self_checks() -> Dict[str, object]:
    """Lightweight internal self-checks for dev/debug."""
    results: Dict[str, object] = {"ok": True, "checks": {}}

    # 1) Compositions count
    comps = list(generate_polymer_compositions(2, 3, 1))
    results["checks"]["compositions_2monomers_dp3_count"] = len(comps)
    if len(comps) != 9:
        results["ok"] = False

    # 2) Adduct calculations sanity
    m = 100.0
    pos = (m + PROTON_MASS) / 1.0
    neg = (m - PROTON_MASS) / 1.0
    results["checks"]["pos_M_plus_H"] = pos
    results["checks"]["neg_M_minus_H"] = neg
    if abs(pos - 101.007276) > 1e-9 or abs(neg - 98.992724) > 1e-9:
        results["ok"] = False

    # 3) Matcher finds peaks in unsorted arrays
    mz = np.array([100.01, 100.00, 99.99], dtype=float)
    inten = np.array([20.0, 10.0, 50.0], dtype=float)
    hit = find_best_peak_match(mz, inten, 100.00, tol_da=0.02, tol_ppm=None)
    results["checks"]["unsorted_match_mz"] = (None if hit is None else hit.matched_mz)
    results["checks"]["unsorted_match_intensity"] = (None if hit is None else hit.intensity)
    # smallest ppm error should pick 100.00 even if intensity is lower
    if hit is None or abs(hit.matched_mz - 100.0) > 1e-9:
        results["ok"] = False

    # 4) NEG mode defaults include a -H candidate
    adducts_neg = build_default_adduct_deltas(
        polarity="negative",
        base_adduct_mass=1.007276,
        enable_na=False,
        enable_k=False,
        enable_cl=True,
        enable_formate=True,
    )
    results["checks"]["neg_has_minus_H"] = any(abs(dm - (-PROTON_MASS)) < 1e-9 for _lbl, dm in adducts_neg)
    if not results["checks"]["neg_has_minus_H"]:
        results["ok"] = False

    return results
