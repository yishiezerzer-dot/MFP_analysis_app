from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class PlateReaderDataset:
    id: str
    name: str
    path: Path
    # UI-facing name (editable). Defaults to filename.
    display_name: str = ""
    sheet_name: Optional[str] = None
    header_row: Optional[int] = 0
    imported_at_utc: str = ""

    # Cached in-memory copies (so switching datasets does not reload from disk).
    # We keep both variants to support the wizard "First row is headers" toggle.
    df_header0: Optional[pd.DataFrame] = None
    df_header_none: Optional[pd.DataFrame] = None

    # Optional, persisted column renames applied after import.
    # Key: original column name; Value: renamed column name.
    renamed_columns: Dict[str, str] = field(default_factory=dict)

    # Data Inspector mapping (optional, user-provided)
    well_col: Optional[str] = None
    group_col: Optional[str] = None
    time_col: Optional[str] = None
    concentration_col: Optional[str] = None
    value_col: Optional[str] = None

    last_preset_id: Optional[str] = None

    # --- MIC state ---
    # "wide" = plate format with step columns (e.g., 1..12)
    # "long" = tidy format with explicit concentration column
    mic_input_mode: Optional[str] = None  # 'wide' | 'long' | None

    # MIC-wide: detected step columns and per-step concentration mapping.
    mic_wide_step_columns: list[str] = field(default_factory=list)
    mic_wide_step_to_conc: Dict[str, float] = field(default_factory=dict)

    # MIC-wide: row/group assignment.
    mic_row_id_col: Optional[str] = None
    mic_row_to_group: Dict[str, str] = field(default_factory=dict)
    mic_control_group_name: str = "Control"
    mic_blank_group_name: Optional[str] = None

    # MIC-wide options.
    mic_normalize_to_control_at_zero: bool = True
    mic_log_x: bool = True

    # --- Wizard-driven analysis (new source of truth) ---
    wizard_last_analysis: Optional[str] = None  # currently only 'mic'
    wizard_mic_config: Optional["PlateReaderMICWizardConfig"] = None
    wizard_mic_result: Optional["PlateReaderMICWizardResult"] = None

    def current_df(self) -> Optional[pd.DataFrame]:
        """Return the cached dataframe matching header_row (no disk IO)."""
        if self.header_row is None:
            return self.df_header_none
        return self.df_header0

    def render_current_plot(self, ax) -> None:  # ax: matplotlib.axes.Axes
        ax.clear()
        if self.wizard_last_analysis == "mic" and self.wizard_mic_result is not None:
            self.wizard_mic_result.render(ax, config=self.wizard_mic_config)
            return

        # Placeholder when the dataset is loaded but no analysis was run yet.
        title_name = self.display_name or self.name
        if not title_name:
            try:
                title_name = str(self.path.name)
            except Exception:
                title_name = "(dataset)"
        try:
            ax.set_axis_off()
        except Exception:
            pass
        ax.set_title(f"Loaded: {title_name}")
        try:
            ax.text(
                0.5,
                0.5,
                "No analysis yet.\nClick Run.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        except Exception:
            pass


@dataclass
class PlateReaderAnalysisResult:
    preset_id: str
    processed: pd.DataFrame
    summary: pd.DataFrame
    plot_kind: str
    x_label: str
    y_label: str
    title: str


@dataclass
class PlateReaderMICWizardConfig:
    use_first_row_as_header: bool = True
    sample_rows: List[int] = field(default_factory=list)  # 0-based indices
    control_rows: List[int] = field(default_factory=list)  # 0-based indices
    concentration_columns: List[str] = field(default_factory=list)  # df column names in order

    # Tick labels shown on the X-axis for the selected columns (same length as concentration_columns).
    tick_labels: List[str] = field(default_factory=list)

    # Wizard convenience: when enabled, the wizard auto-fills tick labels as 1,2,4,8,...
    # based on the number of selected concentration columns.
    auto_tick_labels_power2: bool = True

    title: str = "MIC"
    x_label: str = "Concentration"
    y_label: str = "OD 600nm"

    plot_type: str = "bar"  # 'bar' | 'line' | 'scatter'
    control_style: str = "bars"  # 'bars' | 'line'

    invert_x: bool = False

    # Defaults: dark blue sample, black control.
    sample_color: str = "#286BAD"
    control_color: str = "#5E5959"

    # --- Plot style (editable post-run) ---
    line_width: float = 1.6
    marker_size: float = 6.0
    # Default: thinner bars.
    bar_width: float = 0.20
    capsize: float = 3.0
    errorbar_linewidth: float = 1.0

    title_fontsize: int = 12
    label_fontsize: int = 10
    tick_fontsize: int = 9

    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    grid_on: bool = False
    legend_on: bool = True


@dataclass
class PlateReaderMICWizardResult:
    concentrations: List[float] = field(default_factory=list)  # numeric x positions
    x_tick_labels: List[str] = field(default_factory=list)
    sample_mean: List[float] = field(default_factory=list)
    sample_std: List[float] = field(default_factory=list)
    control_mean: Optional[List[float]] = None
    control_std: Optional[List[float]] = None

    def render(self, ax, *, config: Optional[PlateReaderMICWizardConfig]) -> None:
        title = (config.title if config else "MIC")
        x_label = (config.x_label if config else "Concentration")
        y_label = (config.y_label if config else "OD 600nm")
        plot_type = (config.plot_type if config else "bar")
        control_style = (config.control_style if config else "bars")

        title_fs = int(config.title_fontsize) if config else 12
        label_fs = int(config.label_fontsize) if config else 10
        tick_fs = int(config.tick_fontsize) if config else 9

        line_w = float(config.line_width) if config else 1.6
        mark_s = float(config.marker_size) if config else 6.0
        bar_w = float(config.bar_width) if config else 0.65
        cap = float(config.capsize) if config else 3.0
        e_lw = float(config.errorbar_linewidth) if config else 1.0

        ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel(x_label, fontsize=label_fs)
        ax.set_ylabel(y_label, fontsize=label_fs)
        try:
            ax.tick_params(axis="both", labelsize=tick_fs)
        except Exception:
            pass

        if not self.concentrations:
            return

        x = np.arange(len(self.concentrations), dtype=float)
        labels = self.x_tick_labels or [str(c) for c in self.concentrations]

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        # Deterministic x-axis direction:
        # - By default, force a reasonable xlim that matches the bar positions.
        # - If the user explicitly set x_min/x_max, respect those, then apply inversion.
        user_set_xlim = bool(config is not None and (config.x_min is not None or config.x_max is not None))
        if not user_set_xlim:
            try:
                ax.set_xlim(-0.5, float(len(x) - 1) + 0.5)
            except Exception:
                pass

        if config is None:
            want_invert = False
        else:
            want_invert = bool(getattr(config, "invert_x", False))
        if want_invert:
            try:
                ax.invert_xaxis()
            except Exception:
                pass

        sample_y = np.asarray(self.sample_mean, dtype=float)
        sample_err = np.asarray(self.sample_std, dtype=float)

        sample_color = (getattr(config, "sample_color", "#1f77b4") if config else "#1f77b4")
        control_color = (getattr(config, "control_color", "#ff7f0e") if config else "#ff7f0e")

        has_control = bool(self.control_mean is not None and len(self.control_mean or []) == len(self.sample_mean))
        control_y = np.asarray(self.control_mean, dtype=float) if has_control else None
        control_err = np.asarray(self.control_std, dtype=float) if (has_control and self.control_std is not None) else None

        if plot_type == "bar":
            # Keep widths sane to avoid overlap.
            try:
                bar_w = max(0.05, min(0.9, float(bar_w)))
            except Exception:
                bar_w = 0.65
            if has_control and control_style == "bars":
                bw = min(float(bar_w), 0.45)
                ax.bar(
                    x - bw / 2,
                    sample_y,
                    width=bw,
                    yerr=sample_err,
                    capsize=cap,
                    error_kw={"elinewidth": e_lw},
                    color=sample_color,
                    label="Sample",
                )
                ax.bar(
                    x + bw / 2,
                    control_y,
                    width=bw,
                    yerr=control_err,
                    capsize=cap,
                    error_kw={"elinewidth": e_lw},
                    color=control_color,
                    label="Gentamicin",
                )
            else:
                ax.bar(
                    x,
                    sample_y,
                    width=bar_w,
                    yerr=sample_err,
                    capsize=cap,
                    error_kw={"elinewidth": e_lw},
                    color=sample_color,
                    label="Sample",
                )
                if has_control and control_style == "line":
                    ax.plot(
                        x,
                        control_y,
                        marker="o",
                        markersize=mark_s,
                        linewidth=line_w,
                        color=control_color,
                        label="Control",
                    )
        elif plot_type == "scatter":
            ax.errorbar(
                x,
                sample_y,
                yerr=sample_err,
                fmt="o",
                markersize=mark_s,
                linewidth=line_w,
                capsize=cap,
                elinewidth=e_lw,
                color=sample_color,
                label="Sample",
            )
            if has_control:
                ax.errorbar(
                    x,
                    control_y,
                    yerr=control_err,
                    fmt="o",
                    markersize=mark_s,
                    linewidth=line_w,
                    capsize=cap,
                    elinewidth=e_lw,
                    color=control_color,
                    label="Control",
                )
        else:  # line
            ax.errorbar(
                x,
                sample_y,
                yerr=sample_err,
                fmt="-o",
                markersize=mark_s,
                linewidth=line_w,
                capsize=cap,
                elinewidth=e_lw,
                color=sample_color,
                label="Sample",
            )
            if has_control:
                ax.errorbar(
                    x,
                    control_y,
                    yerr=control_err,
                    fmt="-o",
                    markersize=mark_s,
                    linewidth=line_w,
                    capsize=cap,
                    elinewidth=e_lw,
                    color=control_color,
                    label="Control",
                )

        if config is None or bool(getattr(config, "grid_on", False)):
            try:
                ax.grid(True, alpha=0.3)
            except Exception:
                pass

        if config is None or bool(getattr(config, "legend_on", True)):
            try:
                ax.legend(loc="best")
            except Exception:
                pass

        # Optional axis limits (best-effort).
        if config is not None:
            try:
                if config.x_min is not None or config.x_max is not None:
                    ax.set_xlim(left=config.x_min, right=config.x_max)
            except Exception:
                pass
            try:
                # Always keep Y baseline at 0.
                if config.y_max is not None:
                    try:
                        y_max = float(config.y_max)
                    except Exception:
                        y_max = None
                    if y_max is not None and y_max > 0:
                        ax.set_ylim(bottom=0.0, top=y_max)
                    else:
                        ax.set_ylim(bottom=0.0)
                else:
                    ax.set_ylim(bottom=0.0)
            except Exception:
                pass

            # If user set xlim, apply inversion *after* it so the toggle still works.
            if config.x_min is not None or config.x_max is not None:
                if bool(getattr(config, "invert_x", False)):
                    try:
                        ax.invert_xaxis()
                    except Exception:
                        pass


def ensure_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series([np.nan] * int(series.shape[0]))


def compute_mic(
    df: pd.DataFrame,
    *,
    concentration_col: str,
    value_col: str,
    group_col: Optional[str] = None,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute a simple MIC summary.

    Interpretation:
    - For each group (or the whole dataset if group_col is None), aggregate by concentration
      using mean(value).
    - MIC is the minimum concentration where mean(value) <= threshold.

    Returns:
        (processed_df, summary_df)
    """

    if concentration_col not in df.columns or value_col not in df.columns:
        raise ValueError("Selected columns not found in dataframe")

    work = df.copy()
    work["__conc"] = ensure_numeric(work[concentration_col])
    work["__value"] = ensure_numeric(work[value_col])

    if group_col and group_col in work.columns:
        work["__group"] = work[group_col].astype(str)
    else:
        work["__group"] = "All"

    work = work[np.isfinite(work["__conc"]) & np.isfinite(work["__value"])].copy()
    if work.empty:
        processed = pd.DataFrame(columns=["group", "concentration", "mean", "std", "n"])
        summary = pd.DataFrame(columns=["group", "MIC", "threshold", "n_concentrations"])
        return processed, summary

    agg = (
        work.groupby(["__group", "__conc"], dropna=False)["__value"]
        .agg([("mean", "mean"), ("std", "std"), ("n", "count")])
        .reset_index()
        .rename(columns={"__group": "group", "__conc": "concentration"})
    )

    rows = []
    for g, sub in agg.groupby("group", dropna=False):
        sub2 = sub.sort_values("concentration")
        mic_val: Optional[float] = None
        try:
            hits = sub2[sub2["mean"] <= float(threshold)]
            if not hits.empty:
                mic_val = float(hits.iloc[0]["concentration"])
        except Exception:
            mic_val = None

        rows.append(
            {
                "group": str(g),
                "MIC": (np.nan if mic_val is None else mic_val),
                "threshold": float(threshold),
                "n_concentrations": int(sub2.shape[0]),
            }
        )

    summary = pd.DataFrame(rows)
    return agg, summary


def compute_xy_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generic plot helper.

    Returns:
        processed_df: tidy numeric df with columns [group, x, y]
        summary_df: empty dataframe (reserved for future)
    """

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Selected columns not found in dataframe")

    work = df.copy()
    work["__x"] = ensure_numeric(work[x_col])
    work["__y"] = ensure_numeric(work[y_col])

    if group_col and group_col in work.columns:
        work["__group"] = work[group_col].astype(str)
    else:
        work["__group"] = "All"

    work = work[np.isfinite(work["__x"]) & np.isfinite(work["__y"])].copy()
    processed = work[["__group", "__x", "__y"]].rename(columns={"__group": "group", "__x": "x", "__y": "y"})

    summary = pd.DataFrame(columns=[])
    return processed, summary


def detect_mic_wide_step_columns(df: pd.DataFrame, *, min_numeric_cols: int = 8) -> list[str]:
    """Detect MIC-wide step columns like '1'..'12'.

    Returns numeric-looking column names sorted by numeric value.
    """

    cols = [str(c) for c in df.columns]
    numeric: list[tuple[int, str]] = []
    for c in cols:
        s = str(c).strip()
        if not s:
            continue
        if s.isdigit():
            try:
                numeric.append((int(s), c))
            except Exception:
                continue

    numeric.sort(key=lambda t: t[0])
    out = [c for _, c in numeric]
    if len(out) < int(min_numeric_cols):
        return []
    return out


def default_mic_wide_step_to_conc(
    step_columns: list[str],
    *,
    highest_conc: float = 1024.0,
    dilution_factor: float = 2.0,
    n_drug_steps: int = 11,
    control_step_conc: float = 0.0,
) -> Dict[str, float]:
    """Generate a default step->concentration mapping.

    Assumes step 1 is highest concentration, then serial dilution.
    """

    if not step_columns:
        return {}

    # Sort by numeric step if possible
    def step_key(s: str) -> int:
        try:
            return int(str(s).strip())
        except Exception:
            return 10**9

    steps = sorted([str(s) for s in step_columns], key=step_key)
    mapping: Dict[str, float] = {}

    n_drug = max(1, int(n_drug_steps))
    if n_drug > len(steps):
        n_drug = len(steps)

    for i, step in enumerate(steps[:n_drug]):
        # i=0 -> highest
        try:
            mapping[step] = float(highest_conc) / (float(dilution_factor) ** float(i))
        except Exception:
            mapping[step] = float("nan")

    # Control step: default the last step to 0 if not covered by drug steps
    if steps:
        ctrl_step = steps[-1]
        if ctrl_step not in mapping:
            mapping[ctrl_step] = float(control_step_conc)

    return mapping


def build_mic_wide_long(
    df: pd.DataFrame,
    *,
    step_columns: list[str],
    step_to_conc: Dict[str, float],
    row_id_col: Optional[str],
    row_to_group: Dict[str, str],
) -> pd.DataFrame:
    """Build a long/tidy dataframe from a MIC-wide plate.

    Output columns: row_id, group, step, concentration, value
    """

    if not step_columns:
        return pd.DataFrame(columns=["row_id", "group", "step", "concentration", "value"])

    # Map row_id -> df index
    row_id_to_index: Dict[str, int] = {}
    if row_id_col and row_id_col in df.columns:
        for i in range(int(df.shape[0])):
            try:
                rid = str(df.iloc[i][row_id_col])
            except Exception:
                rid = str(i + 1)
            if rid in row_id_to_index:
                continue
            row_id_to_index[rid] = int(i)
    else:
        for i in range(int(df.shape[0])):
            row_id_to_index[str(i + 1)] = int(i)

    rows: list[dict[str, object]] = []
    for rid, grp in dict(row_to_group or {}).items():
        if rid not in row_id_to_index:
            continue
        i = row_id_to_index[rid]
        for step in step_columns:
            if step not in df.columns:
                continue
            if step not in step_to_conc:
                continue
            try:
                conc = float(step_to_conc[step])
            except Exception:
                conc = float("nan")
            try:
                val = df.at[i, step]
            except Exception:
                val = np.nan
            rows.append({"row_id": str(rid), "group": str(grp), "step": str(step), "concentration": conc, "value": val})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["row_id", "group", "step", "concentration", "value"])

    out["concentration"] = pd.to_numeric(out["concentration"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[np.isfinite(out["concentration"]) & np.isfinite(out["value"])].copy()
    return out


def compute_mic_wide(
    df: pd.DataFrame,
    *,
    step_columns: list[str],
    step_to_conc: Dict[str, float],
    row_id_col: Optional[str],
    row_to_group: Dict[str, str],
    control_group: str,
    blank_group: Optional[str] = None,
    threshold_percent: float = 10.0,
    normalize_to_control_at_zero: bool = True,
    blank_subtract: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute MIC from MIC-wide format.

    Returns:
        processed_df: per-(group, concentration) mean/sd/n + growth_percent
        summary_df: per-group MIC call + notes
    """

    long_df = build_mic_wide_long(
        df,
        step_columns=step_columns,
        step_to_conc=step_to_conc,
        row_id_col=row_id_col,
        row_to_group=row_to_group,
    )

    processed_cols = ["group", "concentration", "mean", "std", "n", "growth_percent"]
    if long_df.empty:
        processed = pd.DataFrame(columns=processed_cols)
        summary = pd.DataFrame(columns=["group", "MIC", "control_ref", "threshold_percent", "notes"])
        return processed, summary

    work = long_df.copy()

    # Optional blank subtraction (simple overall blank mean)
    blank_mean = 0.0
    blank_notes = ""
    if blank_subtract and blank_group:
        try:
            b = work[work["group"] == str(blank_group)]
            if not b.empty:
                blank_mean = float(b["value"].mean())
                work["value"] = work["value"] - blank_mean
                blank_notes = f"blank_subtracted_mean={blank_mean:.6g}"
        except Exception:
            pass

    agg = (
        work.groupby(["group", "concentration"], dropna=False)["value"]
        .agg([("mean", "mean"), ("std", "std"), ("n", "count")])
        .reset_index()
    )

    # Reference for growth normalization
    control_ref = np.nan
    notes_extra: list[str] = []
    try:
        ctrl = agg[(agg["group"] == str(control_group)) & (agg["concentration"] == 0.0)]
        if not ctrl.empty:
            control_ref = float(ctrl.iloc[0]["mean"])
    except Exception:
        control_ref = np.nan

    if not np.isfinite(control_ref) or control_ref == 0.0:
        # Best-effort fallback: use minimum concentration for control group as reference
        try:
            ctrl2 = agg[agg["group"] == str(control_group)].sort_values("concentration")
            if not ctrl2.empty:
                control_ref = float(ctrl2.iloc[0]["mean"])
                notes_extra.append("control_ref_used_min_concentration")
        except Exception:
            pass

    # Growth percent
    if normalize_to_control_at_zero:
        denom = control_ref
        agg["growth_percent"] = (100.0 * agg["mean"] / denom) if np.isfinite(denom) and denom != 0.0 else np.nan
    else:
        # Normalize each group to its own concentration==0 mean (if present)
        agg["growth_percent"] = np.nan
        for g, sub in agg.groupby("group", dropna=False):
            denom = np.nan
            try:
                z = sub[sub["concentration"] == 0.0]
                if not z.empty:
                    denom = float(z.iloc[0]["mean"])
            except Exception:
                denom = np.nan
            if np.isfinite(denom) and denom != 0.0:
                idx = agg["group"] == g
                agg.loc[idx, "growth_percent"] = 100.0 * agg.loc[idx, "mean"] / denom

    # MIC calling
    thr = float(threshold_percent)
    groups_all = [str(g) for g in sorted(set(work["group"].astype(str).tolist()))]
    sample_groups = [g for g in groups_all if g != str(control_group) and (blank_group is None or g != str(blank_group))]

    max_tested = float(np.nanmax(agg["concentration"]))
    min_positive = float(np.nanmin(agg[agg["concentration"] > 0.0]["concentration"])) if (agg["concentration"] > 0.0).any() else np.nan

    summary_rows: list[dict[str, object]] = []
    for g in sample_groups:
        sub = agg[agg["group"] == g].copy()
        sub = sub[np.isfinite(sub["concentration"]) & np.isfinite(sub["growth_percent"])].sort_values("concentration")
        mic_val: object

        # Only consider >0 concentrations for MIC
        sub_pos = sub[sub["concentration"] > 0.0]
        hits = sub_pos[sub_pos["growth_percent"] <= thr]
        if hits.empty:
            mic_val = f"> {max_tested:g}" if np.isfinite(max_tested) else "> highest tested"
        else:
            try:
                mic_val = float(hits.iloc[0]["concentration"])
            except Exception:
                mic_val = float("nan")

        notes = []
        if blank_notes:
            notes.append(blank_notes)
        notes.extend(notes_extra)
        if not np.isfinite(min_positive):
            notes.append("no_positive_concentrations")

        summary_rows.append(
            {
                "group": g,
                "MIC": mic_val,
                "control_ref": (np.nan if not np.isfinite(control_ref) else control_ref),
                "threshold_percent": thr,
                "notes": ";".join([n for n in notes if n]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    processed = agg.sort_values(["group", "concentration"]).reset_index(drop=True)
    return processed, summary


def compute_mic_wide_od(
    df: pd.DataFrame,
    *,
    step_columns: list[str],
    step_to_conc: Dict[str, float],
    row_id_col: Optional[str],
    row_to_group: Dict[str, str],
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute MIC from MIC-wide format using raw OD threshold.

    Interpretation:
    - Convert wide -> long using step_to_conc and row_to_group.
    - Aggregate mean OD per (group, concentration).
    - MIC is the minimum concentration where mean(OD) <= threshold.

    Returns:
        processed_df: [group, concentration, mean, std, n]
        summary_df: [group, MIC, threshold, n_concentrations]
    """

    long_df = build_mic_wide_long(
        df,
        step_columns=step_columns,
        step_to_conc=step_to_conc,
        row_id_col=row_id_col,
        row_to_group=row_to_group,
    )

    if long_df.empty:
        processed = pd.DataFrame(columns=["group", "concentration", "mean", "std", "n"])
        summary = pd.DataFrame(columns=["group", "MIC", "threshold", "n_concentrations"])
        return processed, summary

    agg = (
        long_df.groupby(["group", "concentration"], dropna=False)["value"]
        .agg([("mean", "mean"), ("std", "std"), ("n", "count")])
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    thr = float(threshold)
    for g, sub in agg.groupby("group", dropna=False):
        sub2 = sub.sort_values("concentration")
        mic_val: Optional[float] = None
        try:
            hits = sub2[sub2["mean"] <= thr]
            if not hits.empty:
                mic_val = float(hits.iloc[0]["concentration"])
        except Exception:
            mic_val = None

        rows.append(
            {
                "group": str(g),
                "MIC": (np.nan if mic_val is None else mic_val),
                "threshold": thr,
                "n_concentrations": int(sub2.shape[0]),
            }
        )

    summary = pd.DataFrame(rows)
    processed = agg.sort_values(["group", "concentration"]).reset_index(drop=True)
    return processed, summary
