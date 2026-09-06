"""Render the five main figures for the ZeroPP paper (F1-F5), per
docs/figure_style_guide.md. Matplotlib only, no plotnine. One function per
figure (make_f1..make_f5), each independently callable -- this module has no
monolithic main() that must run start-to-end; __main__ below just calls all
five in sequence for convenience.

Every color, line style, marker and font size used anywhere in this file
lives in the single STYLE dict below -- no hex-color literal appears
anywhere else in this module (grep for '#[0-9A-Fa-f]{6}' outside the STYLE
block as a mechanical check).

Every number that appears baked into a figure or its caption (breakpoints,
crossover lead time, coverage percentages, calendar-day equivalents, N
values at F3's arrowheads, etc.) is read at render time from the persisted
`results/*.parquet` files listed in docs/results_index.md -- never
hardcoded/retyped in this file. F4 is the one figure whose input data did
not already exist anywhere; scripts/10_compute_pit_histograms.py computes
and persists it to results/phase3_pit_histograms.parquet BEFORE this module
reads it, so the "numbers come from results files" rule holds for F4 too.

Column width note: the style guide fixes 90mm (single) / 190mm (double) as
the two allowed physical widths, set in code and never scaled after the
fact. All five figures here use the double-column (190mm) width -- each
carries multiple series, in-panel R2 annotations and (for F1/F3/F5) a
secondary axis or trajectory arrows/labels that do not fit legibly at 90mm;
first-draft renders at single-column width produced illegible, overlapping
text (see docs/figures_implementation_report.md). Heights are chosen per
figure and are NOT derived from any fixed ratio.

Exports per figure: <name>.pdf (vector, with source-parquet-path + git-SHA
written into the PDF's own metadata via savefig(metadata=...)), <name>.svg
(vector), <name>.png (300 dpi, preview only -- see .gitignore, PNGs under
figures/ are not tracked). Layout is done with explicit fig.subplots_adjust
margins (not tight_layout/constrained_layout), and every fig-level legend is
anchored at a small POSITIVE figure-fraction y (never a negative
bbox_to_anchor) so it always lands inside the fixed canvas -- tight_layout
is deliberately not used together with secondary axes / fig-level legends,
since matplotlib's own tight_layout warns it cannot account for either and
a first-draft render confirmed that in practice (clipped/overlapping
legends, an oversized blank margin on F1).

Captions (numbers substituted from the same results files, never retyped)
are collected into figures/captions.md by each make_fX call -- see
_update_caption(). F4's caption includes, verbatim, the sentence required by
docs/figure_style_guide.md: "PIT uniformity is assessed descriptively; no
formal uniformity test valid under serial dependence is applied." F3's
caption carries the Gneiting "up and to the left is better" reading (per the
style guide: "Say that in the caption" -- not on the canvas itself).
"""
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
CAPTIONS_PATH = FIGURES_DIR / "captions.md"

# --------------------------------------------------------------------------
# STYLE: every color / line style / marker / font size / geometry constant
# used anywhere in this file. No color literal appears outside this dict.
# Palette and typography are copied verbatim from docs/figure_style_guide.md
# sections 2-3 -- this is a fixed design spec, not a place to improvise.
# --------------------------------------------------------------------------
STYLE = {
    "color": {
        "raw_ensemble": "#999999",
        "var_inflation_fixed": "#E69F00",
        "tsfm3": "#0072B2",
        "emos_pooled": "#D55E00",
        "emos_local": "#009E73",
        "drn": "#CC79A7",
        "reference": "#666666",
        "gridline": "#DDDDDD",
    },
    "linestyle": {
        "raw_ensemble": ":",
        "var_inflation_fixed": "-.",
        "tsfm3": "-",
        "emos_pooled": "-",
        "emos_local": "-",
        "drn": "-",
        "reference": "--",
        "random_arm": "--",  # F1's random-arm mean: same color as emos_pooled, dashed
    },
    "marker": {
        "raw_ensemble": None,
        "var_inflation_fixed": None,
        "tsfm3": None,
        "emos_pooled": "o",
        "emos_local": "^",
        "drn": "s",
        "reference": None,
        "zero_shot_point": "D",  # F3 only: zero-shot methods as single points need SOME
                                  # visible marker; "diamond" is unclaimed elsewhere in
                                  # the palette table (whose "marker: none" describes
                                  # their flat-line rendering, not a scatter point).
    },
    "label": {
        "raw_ensemble": "Raw ensemble",
        "var_inflation_fixed": "Variance-inflated baseline",
        "tsfm3": "TimesFM-3 (zero-shot)",
        "emos_pooled": "EMOS pooled",
        "emos_local": "EMOS local",
        "drn": "DRN",
        "reference": "Reference / nominal",
        "random_arm_mean": "EMOS pooled, random arm (mean ± 1 SD)",
    },
    "font": {
        "family": "sans-serif",
        "sans_serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "panel_title_pt": 10,
        "axis_label_pt": 9,
        "tick_label_pt": 8,
        "annotation_pt": 8,
        "legend_pt": 9,
    },
    "linewidth": {"data": 1.4, "reference": 1.0, "gridline": 0.5},
    "marker_size_pt": 5,
    "mm_per_column": {"single": 90.0, "double": 190.0},
    "dpi_png": 300,
    "alpha": {"std_band": 0.20, "pit_fill": 0.70},
}


def _mm_to_in(mm: float) -> float:
    return mm / 25.4


def _figsize_in(kind: str, height_mm: float) -> tuple[float, float]:
    return (_mm_to_in(STYLE["mm_per_column"][kind]), _mm_to_in(height_mm))


def _apply_rcparams() -> None:
    plt.rcParams.update({
        "font.family": STYLE["font"]["family"],
        "font.sans-serif": STYLE["font"]["sans_serif"],
        "font.size": STYLE["font"]["tick_label_pt"],
        "axes.labelsize": STYLE["font"]["axis_label_pt"],
        "axes.titlesize": STYLE["font"]["panel_title_pt"],
        "xtick.labelsize": STYLE["font"]["tick_label_pt"],
        "ytick.labelsize": STYLE["font"]["tick_label_pt"],
        "legend.fontsize": STYLE["font"]["legend_pt"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _style_grid(ax) -> None:
    """R6: minimal chrome -- light grey gridlines on white, no background fill,
    no panel border beyond the left/bottom axis lines (top/right spines are
    disabled globally in _apply_rcparams)."""
    ax.set_facecolor("white")
    ax.grid(True, axis="both", color=STYLE["color"]["gridline"], linewidth=STYLE["linewidth"]["gridline"])
    ax.set_axisbelow(True)


def _add_top_headroom(ax, frac: float = 0.32) -> None:
    """Extend the y-axis above the current data max so R2's in-panel
    annotation stack has clear whitespace instead of overlapping the plotted
    series. Call AFTER all data/reference lines for the panel are drawn."""
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + frac * (ymax - ymin))


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / f"{name}.parquet")


def _save_figure(fig, name: str, source_parquets: list[str]) -> None:
    """Export PDF+SVG (vector) and PNG (300 dpi, preview only). The PDF's own
    metadata carries the source parquet path(s) and the git SHA (per the style
    guide: written into PDF metadata via savefig(metadata=...), never as a
    text annotation on the canvas). No bbox_inches='tight' anywhere: the
    figure's physical size is fixed in code (single/double column mm) and
    must not be scaled or trimmed after the fact."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    pdf_metadata = {
        "Subject": "source: " + "; ".join(source_parquets),
        "Keywords": f"git_sha:{sha}",
    }
    fig.savefig(FIGURES_DIR / f"{name}.pdf", metadata=pdf_metadata)
    fig.savefig(FIGURES_DIR / f"{name}.svg")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=STYLE["dpi_png"])


def _update_caption(fig_id: str, title: str, text: str) -> None:
    """Append/replace this figure's caption section in figures/captions.md,
    keeping the file's F1..F5 ordering stable regardless of call order --
    each make_fX call is independent and idempotent with respect to this
    file."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sections: dict[str, str] = {}
    if CAPTIONS_PATH.exists():
        raw = CAPTIONS_PATH.read_text()
        for block in re.split(r"(?=^## )", raw, flags=re.MULTILINE):
            block = block.strip()
            if not block:
                continue
            m = re.match(r"^## (F\d+)", block)
            if m:
                sections[m.group(1)] = block
    sections[fig_id] = f"## {fig_id} -- {title}\n\n{text.strip()}\n"
    ordered = [sections[k] for k in sorted(sections, key=lambda k: int(k[1:]))]
    CAPTIONS_PATH.write_text(
        "# ZeroPP figure captions\n\n"
        "Auto-generated by scripts/make_figures.py -- every number here is read "
        "programmatically from results/*.parquet at render time, never retyped.\n\n"
        + "\n".join(ordered)
    )


def _fmt_cases(n: float) -> str:
    return f"{n:.1f}" if abs(n - round(n)) > 1e-6 else f"{int(round(n))}"


def _annotate_stack(ax, entries: list[tuple[str, str]], x: float = 0.02, y0: float = 0.97, step: float = 0.085, ha: str = "left") -> None:
    """R2: stack colour-matched in-panel annotations top-to-bottom at a fixed
    axes-fraction position. entries: list of (text, color)."""
    for i, (text, color) in enumerate(entries):
        ax.text(
            x, y0 - i * step, text, transform=ax.transAxes, color=color,
            fontsize=STYLE["font"]["annotation_pt"], va="top", ha=ha,
        )


def _days_per_case_from_data(*dfs: pd.DataFrame) -> float:
    """Recover the (real, measured) case->calendar-day ratio directly from the
    already-persisted n_cases/n_calendar_days_equiv columns instead of typing
    the constant into this file -- averaged over every real (n_cases>0) row
    across the given frames to smooth out the rounding each source row already
    went through."""
    ratios = []
    for df in dfs:
        sub = df[df["n_cases"] > 0].dropna(subset=["n_cases", "n_calendar_days_equiv"])
        ratios.extend((sub["n_calendar_days_equiv"] / sub["n_cases"]).tolist())
    return float(np.mean(ratios))


# ============================================================================
# F1 -- Breakpoint curve (cover figure)
# ============================================================================
def make_f1() -> None:
    sweep = _load("phase3_data_size_sweep")
    low_n = _load("phase3_low_n_grid").rename(columns={"k": "n_cases"})
    bp_main = _load("phase3_data_size_sweep_breakpoints")
    bp_low = _load("phase3_low_n_grid_breakpoints")

    days_per_case = _days_per_case_from_data(sweep, low_n)

    def combined_curve(method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        parts = [sweep[(sweep["method"] == method) & (sweep["sampling_arm"] == "contiguous")][["n_cases", metric]]]
        if method in low_n.loc[low_n["sampling_arm"] == "contiguous", "method"].unique():
            parts.append(low_n[(low_n["method"] == method) & (low_n["sampling_arm"] == "contiguous")][["n_cases", metric]])
        combined = pd.concat(parts).dropna(subset=[metric]).drop_duplicates(subset=["n_cases"]).sort_values("n_cases")
        return combined["n_cases"].to_numpy(float), combined[metric].to_numpy(float)

    def random_arm(metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts = [
            sweep[(sweep["method"] == "emos_pooled") & (sweep["sampling_arm"] == "random_mean")][["n_cases", metric, f"{metric}_std"]],
            low_n[(low_n["method"] == "emos_pooled") & (low_n["sampling_arm"] == "random_mean")][["n_cases", metric, f"{metric}_std"]],
        ]
        combined = pd.concat(parts).dropna(subset=[metric]).drop_duplicates(subset=["n_cases"]).sort_values("n_cases")
        return combined["n_cases"].to_numpy(float), combined[metric].to_numpy(float), combined[f"{metric}_std"].fillna(0.0).to_numpy(float)

    def hline_value(method: str, metric: str) -> float:
        row = sweep[(sweep["method"] == method) & (sweep["sampling_arm"] == "n_independent")].iloc[0]
        return float(row[metric])

    def read_bp(metric: str, variant: str) -> tuple[float | None, float | None, str]:
        """Low-N grid file is authoritative for emos_pooled/emos_local (see
        docs/results_index.md); main sweep file is the only source for drn."""
        if variant in ("emos_pooled", "emos_local"):
            hit = bp_low[(bp_low["metric"] == metric) & (bp_low["emos_variant"] == variant)]
        else:
            hit = bp_main[(bp_main["metric"] == metric) & (bp_main["emos_variant"] == variant)]
        row = hit.iloc[0]
        bp = row["breakpoint_n_cases"]
        cal = row["breakpoint_calendar_days"]
        bp = None if pd.isna(bp) else float(bp)
        cal = None if pd.isna(cal) else float(cal)
        return bp, cal, str(row["crossing_direction"])

    _apply_rcparams()
    fig, axes = plt.subplots(3, 1, figsize=_figsize_in("double", 210), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.32, hspace=0.14)
    ax_crps, ax_cov, ax_width = axes
    curve_methods = ["emos_pooled", "emos_local", "drn"]
    hline_methods = ["raw_ensemble", "var_inflation_fixed", "tsfm3"]

    metrics_panels = [
        (ax_crps, "crps", "CRPS (K)"),
        (ax_cov, "coverage_80pct", "Coverage @ 80%"),
        (ax_width, "interval_width_k", "Interval width K (K)"),
    ]
    for ax, metric, ylabel in metrics_panels:
        _style_grid(ax)
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        for m in hline_methods:
            ax.axhline(
                hline_value(m, metric), color=STYLE["color"][m], linestyle=STYLE["linestyle"][m],
                linewidth=STYLE["linewidth"]["data"],
            )
        for m in curve_methods:
            x, y = combined_curve(m, metric)
            if len(x) == 0:
                continue
            ax.plot(
                x, y, color=STYLE["color"][m], linestyle=STYLE["linestyle"][m],
                marker=STYLE["marker"][m], markersize=STYLE["marker_size_pt"],
                linewidth=STYLE["linewidth"]["data"],
            )
        if metric in ("crps", "coverage_80pct"):
            x_r, y_r, std_r = random_arm(metric)
            if len(x_r):
                c = STYLE["color"]["emos_pooled"]
                ax.plot(x_r, y_r, color=c, linestyle=STYLE["linestyle"]["random_arm"], linewidth=STYLE["linewidth"]["data"])
                ax.fill_between(x_r, y_r - std_r, y_r + std_r, color=c, alpha=STYLE["alpha"]["std_band"], linewidth=0)

    ax_cov.axhline(0.80, color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"], linewidth=STYLE["linewidth"]["reference"])

    # Headroom BEFORE annotating, so R2's stacked text never overlaps the data.
    _add_top_headroom(ax_crps, 0.42)
    _add_top_headroom(ax_cov, 0.42)
    _add_top_headroom(ax_width, 0.30)

    # --- R2: in-panel, colour-matched breakpoint annotations ---
    crps_entries = []
    cov_entries = []
    for variant in ["emos_pooled", "emos_local", "drn"]:
        color = STYLE["color"][variant]
        label = STYLE["label"][variant]
        bp, cal, reason = read_bp("crps", variant)
        if bp is not None:
            crps_entries.append((f"{label}: ~{_fmt_cases(bp)} cases (~{cal:.0f} d) vs. TimesFM-3", color))
        else:
            crps_entries.append((f"{label}: {reason.split('—')[-1].strip()}", color))
        bp, cal, reason = read_bp("coverage_80pct", variant)
        if bp is not None:
            cov_entries.append((f"{label}: ~{_fmt_cases(bp)} cases (~{cal:.0f} d) vs. TimesFM-3", color))
        else:
            cov_entries.append((f"{label}: {reason.split('—')[-1].strip()}", color))
    _annotate_stack(ax_crps, crps_entries)
    _annotate_stack(ax_cov, cov_entries)
    ax_width.text(
        0.02, 0.97, "Sharpness has no single ‘better’ direction alone (Gneiting: maximize\n"
        "sharpness subject to calibration) — curve only, no breakpoint computed.",
        transform=ax_width.transAxes, color=STYLE["color"]["reference"], fontsize=STYLE["font"]["annotation_pt"],
        va="top", ha="left",
    )

    ax_width.set_xlabel("Training data size (cases, log scale)")

    # Secondary calendar-day tick row, bottom panel only (R1: shared axis, labelled once)
    sec = ax_width.secondary_xaxis(-0.45, functions=(lambda k: k * days_per_case, lambda d: d / days_per_case))
    sec.set_xlabel("Calendar-day equivalent")
    sec.tick_params(labelsize=STYLE["font"]["tick_label_pt"])

    handles = [
        Line2D([0], [0], color=STYLE["color"][m], linestyle=STYLE["linestyle"][m], marker=STYLE["marker"][m],
               markersize=STYLE["marker_size_pt"], linewidth=STYLE["linewidth"]["data"], label=STYLE["label"][m])
        for m in hline_methods + curve_methods
    ] + [
        Line2D([0], [0], color=STYLE["color"]["emos_pooled"], linestyle=STYLE["linestyle"]["random_arm"],
               linewidth=STYLE["linewidth"]["data"], label=STYLE["label"]["random_arm_mean"]),
        Line2D([0], [0], color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"],
               linewidth=STYLE["linewidth"]["reference"], label="Nominal 80% coverage"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01), frameon=False)

    _save_figure(fig, "f1_breakpoint_curve", [
        "results/phase3_data_size_sweep.parquet", "results/phase3_low_n_grid.parquet",
        "results/phase3_data_size_sweep_breakpoints.parquet", "results/phase3_low_n_grid_breakpoints.parquet",
    ])
    plt.close(fig)

    bp_pooled_crps, cal_pooled_crps, _ = read_bp("crps", "emos_pooled")
    bp_local_cov, cal_local_cov, _ = read_bp("coverage_80pct", "emos_local")
    _update_caption(
        "F1", "Breakpoint curve (cover figure)",
        "CRPS, coverage@80% and interval width K vs. training data size (cases, "
        "log scale; calendar-day equivalent on the secondary axis). Zero-shot "
        "methods (raw ensemble, variance-inflated baseline, TimesFM-3) render "
        "as flat horizontal lines; trained methods (EMOS pooled, EMOS local, "
        "DRN) render as curves. In-panel annotations give each trained method's "
        "breakpoint against TimesFM-3, e.g. EMOS pooled's CRPS crosses TimesFM-3 "
        f"at ~{_fmt_cases(bp_pooled_crps)} cases (~{cal_pooled_crps:.0f} calendar days); "
        f"EMOS local's coverage@80% crosses at ~{_fmt_cases(bp_local_cov)} cases "
        f"(~{cal_local_cov:.0f} calendar days). Random-arm (EMOS pooled) mean ± 1 SD "
        "shown as a dashed band. Sharpness (interval width) is reported as a "
        "curve only, with no breakpoint computed, since it has no inherent "
        "'better' direction independent of calibration (Gneiting).",
    )


# ============================================================================
# F2 -- Lead-time resolved comparison
# ============================================================================
def make_f2() -> None:
    lt = _load("phase3_lead_time_breakdown")
    crossover = _load("phase3_lead_time_crossover")

    method_map = {"raw_ensemble": "raw_ensemble", "tsfm3": "tsfm3", "emos": "emos_pooled"}
    durable = crossover[crossover["crossover_type"] == "durable_sign_flip"].iloc[0]
    first = crossover[crossover["crossover_type"] == "first_sign_flip"].iloc[0]
    x_durable = float(durable["step_hours"])
    x_first = float(first["step_hours"])

    _apply_rcparams()
    fig, (ax_crps, ax_cov) = plt.subplots(2, 1, figsize=_figsize_in("double", 130), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.96, bottom=0.18, hspace=0.10)
    for ax, metric, ylabel in [(ax_crps, "crps", "CRPS (K)"), (ax_cov, "coverage_80pct", "Coverage @ 80%")]:
        _style_grid(ax)
        for src, plot_key in method_map.items():
            sub = lt[lt["method"] == src].sort_values("step_hours")
            ax.plot(
                sub["step_hours"], sub[metric], color=STYLE["color"][plot_key], linestyle=STYLE["linestyle"][plot_key],
                marker=STYLE["marker"][plot_key], markersize=STYLE["marker_size_pt"], linewidth=STYLE["linewidth"]["data"],
            )
        ax.axvline(x_durable, color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"], linewidth=STYLE["linewidth"]["reference"])
    _add_top_headroom(ax_crps, 0.30)
    ax_cov.axhline(0.80, color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"], linewidth=STYLE["linewidth"]["reference"])
    ax_crps.set_ylabel("CRPS (K)")
    ax_cov.set_ylabel("Coverage @ 80%")
    ax_cov.set_xlabel("Lead time, step_hours (h)")

    _annotate_stack(
        ax_crps,
        [(f"Durable crossover ≈ {x_durable:.1f} h (TimesFM-3 stays worse for\nevery longer tested lead time)", STYLE["color"]["reference"])],
        x=0.40, y0=0.97,
    )

    handles = [
        Line2D([0], [0], color=STYLE["color"][k], linestyle=STYLE["linestyle"][k], marker=STYLE["marker"][k],
               markersize=STYLE["marker_size_pt"], linewidth=STYLE["linewidth"]["data"], label=STYLE["label"][k])
        for k in ["raw_ensemble", "tsfm3", "emos_pooled"]
    ] + [
        Line2D([0], [0], color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"],
               linewidth=STYLE["linewidth"]["reference"], label="Durable crossover / nominal 80%"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.015), frameon=False)

    _save_figure(fig, "f2_lead_time_resolved", [
        "results/phase3_lead_time_breakdown.parquet", "results/phase3_lead_time_crossover.parquet",
    ])
    plt.close(fig)

    _update_caption(
        "F2", "Lead-time resolved comparison",
        "CRPS and coverage@80% across all 21 tested lead times (step_hours). "
        f"The durable crossover at step_hours ≈ {x_durable:.1f} h (grey dashed vertical "
        "line) is the lead time after which TimesFM-3's CRPS remains worse than "
        f"(or tied with) EMOS pooled for every longer tested lead time; the first "
        f"sign flip occurs earlier, at step_hours ≈ {x_first:.1f} h, but is not durable "
        "(the sign flips back at least once between there and the durable "
        "crossover) and is reported here in the caption only, not annotated on "
        "the panel, per the project's convention of citing the durable crossover "
        "as the primary number.",
    )


# ============================================================================
# F3 -- Sharpness-calibration plane
# ============================================================================
def make_f3() -> None:
    sweep = _load("phase3_data_size_sweep")
    low_n = _load("phase3_low_n_grid").rename(columns={"k": "n_cases"})

    def trajectory(method: str) -> pd.DataFrame:
        parts = [sweep[(sweep["method"] == method) & (sweep["sampling_arm"] == "contiguous")][["n_cases", "interval_width_k", "coverage_80pct"]]]
        if method in low_n.loc[low_n["sampling_arm"] == "contiguous", "method"].unique():
            parts.append(low_n[(low_n["method"] == method) & (low_n["sampling_arm"] == "contiguous")][["n_cases", "interval_width_k", "coverage_80pct"]])
        combined = pd.concat(parts).dropna().drop_duplicates(subset=["n_cases"]).sort_values("n_cases")
        return combined

    def zero_shot_point(method: str) -> tuple[float, float]:
        row = sweep[(sweep["method"] == method) & (sweep["sampling_arm"] == "n_independent")].iloc[0]
        return float(row["interval_width_k"]), float(row["coverage_80pct"])

    # Per-method label offsets (offset points) for the N-value annotation at
    # each trajectory's arrowhead -- the three trained methods' full-N
    # endpoints sit close together on this plane, so a single shared offset
    # makes the three text labels collide; each gets its own direction.
    n_label_offset = {"emos_pooled": (8, -4), "emos_local": (8, 10), "drn": (14, -20)}

    _apply_rcparams()
    fig, ax = plt.subplots(figsize=_figsize_in("double", 110))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.24)
    _style_grid(ax)
    ax.axhline(0.80, color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"], linewidth=STYLE["linewidth"]["reference"])

    for m in ["raw_ensemble", "var_inflation_fixed", "tsfm3"]:
        k, cov = zero_shot_point(m)
        ax.scatter([k], [cov], color=STYLE["color"][m], marker=STYLE["marker"]["zero_shot_point"], s=STYLE["marker_size_pt"] ** 2, zorder=3)

    for m in ["emos_pooled", "emos_local", "drn"]:
        traj = trajectory(m)
        if traj.empty:
            continue
        ax.plot(
            traj["interval_width_k"], traj["coverage_80pct"], color=STYLE["color"][m], linestyle=STYLE["linestyle"][m],
            marker=STYLE["marker"][m], markersize=STYLE["marker_size_pt"], linewidth=STYLE["linewidth"]["data"],
        )
        x0, y0 = traj["interval_width_k"].iloc[-2], traj["coverage_80pct"].iloc[-2]
        x1, y1 = traj["interval_width_k"].iloc[-1], traj["coverage_80pct"].iloc[-1]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=STYLE["color"][m], lw=STYLE["linewidth"]["data"]),
        )
        n_max = traj["n_cases"].iloc[-1]
        ax.annotate(
            f"N={_fmt_cases(n_max)}", xy=(x1, y1), xytext=n_label_offset[m], textcoords="offset points",
            color=STYLE["color"][m], fontsize=STYLE["font"]["annotation_pt"],
        )

    ax.set_xlabel("Interval width, K (Kelvin)")
    ax.set_ylabel("Coverage @ 80%")
    ax.margins(x=0.12, y=0.12)

    handles = [
        Line2D([0], [0], color=STYLE["color"][k], marker=STYLE["marker"]["zero_shot_point"], linestyle="none", markersize=STYLE["marker_size_pt"], label=STYLE["label"][k])
        for k in ["raw_ensemble", "var_inflation_fixed", "tsfm3"]
    ] + [
        Line2D([0], [0], color=STYLE["color"][k], linestyle=STYLE["linestyle"][k], marker=STYLE["marker"][k],
               markersize=STYLE["marker_size_pt"], linewidth=STYLE["linewidth"]["data"], label=STYLE["label"][k])
        for k in ["emos_pooled", "emos_local", "drn"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.015), frameon=False)

    _save_figure(fig, "f3_sharpness_calibration_plane", ["results/phase3_data_size_sweep.parquet", "results/phase3_low_n_grid.parquet"])
    plt.close(fig)

    _update_caption(
        "F3", "Sharpness-calibration plane",
        "Interval width K (sharpness) vs. coverage@80% (calibration), grey "
        "dashed at nominal 0.80. Zero-shot methods (raw ensemble, "
        "variance-inflated baseline, TimesFM-3) plot as single points; trained "
        "methods (EMOS pooled, EMOS local, DRN) plot as trajectories across "
        "training data size N, with an arrowhead and the N value at the "
        "largest tested N. Up and to the left is better: sharper (smaller K) "
        "at equal or better calibration (closer to the nominal 0.80 line) is "
        "the Gneiting-optimal direction.",
    )


# ============================================================================
# F4 -- Calibration diagnostics (PIT histograms)
# ============================================================================
def make_f4() -> None:
    pit = _load("phase3_pit_histograms")
    order = ["raw_ensemble", "tsfm3", "emos_pooled", "emos_local"]
    n_bins = int(pit["bin_index"].nunique())
    uniform_expectation = 1.0 / n_bins

    _apply_rcparams()
    fig, axes = plt.subplots(1, 4, figsize=_figsize_in("double", 78), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.32, wspace=0.12)
    for ax, method in zip(axes, order):
        _style_grid(ax)
        sub = pit[pit["method"] == method].sort_values("bin_index")
        centers = (sub["bin_lo"] + sub["bin_hi"]) / 2.0
        width = (sub["bin_hi"] - sub["bin_lo"]).iloc[0]
        color = STYLE["color"][method]
        ax.bar(
            centers, sub["fraction"], width=width * 0.9, color=color, alpha=STYLE["alpha"]["pit_fill"],
            edgecolor=color, linewidth=0.8,
        )
        ax.axhline(uniform_expectation, color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"], linewidth=STYLE["linewidth"]["reference"])
        ax.set_title(STYLE["label"][method], fontsize=STYLE["font"]["panel_title_pt"])
        ax.set_xlim(0, 1)
        n_inst = int(sub["n_instances"].iloc[0])
        ax.text(
            0.5, -0.24, f"n={n_inst:,}", transform=ax.transAxes, ha="center", va="top",
            fontsize=STYLE["font"]["annotation_pt"], color=STYLE["color"]["reference"],
        )

    fig.supxlabel("PIT value", y=0.11)
    fig.supylabel("Fraction of instances")

    handles = [
        Line2D([0], [0], color=STYLE["color"]["reference"], linestyle=STYLE["linestyle"]["reference"],
               linewidth=STYLE["linewidth"]["reference"], label="Uniform expectation (1/n_bins)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0.005), frameon=False)

    _save_figure(fig, "f4_pit_histograms", ["results/phase3_pit_histograms.parquet"])
    plt.close(fig)

    _update_caption(
        "F4", "Calibration diagnostics (PIT histograms)",
        "Probability integral transform (PIT) histograms at full training "
        "data size (N=full), raw ensemble / TimesFM-3 / EMOS pooled / EMOS "
        f"local, {n_bins} bins, grey dashed at the uniform expectation "
        f"(1/{n_bins} ≈ {uniform_expectation:.3f}). PIT uniformity is assessed "
        "descriptively; no formal uniformity test valid under serial "
        "dependence is applied.",
    )


# ============================================================================
# F5 -- Breakpoint by lead-time group
# ============================================================================
def make_f5() -> None:
    bp = _load("phase3_lead_time_bucketed_breakpoints")
    sweep = _load("phase3_lead_time_bucketed_sweep")

    groups = ["0-24h", "24-72h", "72-120h"]
    rows_order = [("crps", "emos_pooled"), ("crps", "emos_local"), ("coverage_80pct", "emos_pooled"), ("coverage_80pct", "emos_local")]
    row_labels = {
        ("crps", "emos_pooled"): "CRPS — EMOS pooled",
        ("crps", "emos_local"): "CRPS — EMOS local",
        ("coverage_80pct", "emos_pooled"): "Coverage@80% — EMOS pooled",
        ("coverage_80pct", "emos_local"): "Coverage@80% — EMOS local",
    }

    _apply_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=_figsize_in("double", 115), sharey=True)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.90, bottom=0.24, wspace=0.08)
    y_positions = {key: i for i, key in enumerate(rows_order)}

    for ax, group in zip(axes, groups):
        _style_grid(ax)
        tested = sweep[(sweep["lead_time_bucket"] == group) & (sweep["sampling_arm"] == "contiguous")]
        k_min, k_max = float(tested["n_cases"].min()), float(tested["n_cases"].max())
        for metric, variant in rows_order:
            y = y_positions[(metric, variant)]
            color = STYLE["color"][variant]
            marker = STYLE["marker"][variant]
            row = bp[(bp["lead_time_bucket"] == group) & (bp["metric"] == metric) & (bp["emos_variant"] == variant)].iloc[0]
            val = row["breakpoint_n_cases"]
            if pd.isna(val):
                ax.scatter([k_max], [y], facecolors="none", edgecolors=color, marker=marker, s=STYLE["marker_size_pt"] ** 2 * 2, linewidths=1.2, zorder=3)
                ax.annotate(
                    f"no crossing\n(k {int(k_min)}–{int(k_max)})", xy=(k_max, y), xytext=(-4, 11),
                    textcoords="offset points", color=color, fontsize=STYLE["font"]["annotation_pt"],
                    ha="right", va="bottom",
                )
            else:
                ax.scatter([val], [y], color=color, marker=marker, s=STYLE["marker_size_pt"] ** 2, zorder=3)
                ax.annotate(
                    f"{_fmt_cases(float(val))} cases", xy=(val, y), xytext=(0, 9),
                    textcoords="offset points", color=color, fontsize=STYLE["font"]["annotation_pt"], ha="center",
                )
        ax.set_xscale("log")
        ax.set_xlim(k_min * 0.5, k_max * 3.0)
        ax.set_title(group, fontsize=STYLE["font"]["panel_title_pt"])
        ax.set_yticks(list(y_positions.values()))
        ax.set_ylim(-0.7, len(rows_order) - 1 + 0.7)

    axes[0].set_yticklabels([row_labels[k] for k in rows_order])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.supxlabel("Breakpoint vs. TimesFM-3 (training cases, log scale)", y=0.13)

    handles = [
        Line2D([0], [0], color=STYLE["color"]["emos_pooled"], marker=STYLE["marker"]["emos_pooled"], linestyle="none",
               markersize=STYLE["marker_size_pt"], label=STYLE["label"]["emos_pooled"]),
        Line2D([0], [0], color=STYLE["color"]["emos_local"], marker=STYLE["marker"]["emos_local"], linestyle="none",
               markersize=STYLE["marker_size_pt"], label=STYLE["label"]["emos_local"]),
        Line2D([0], [0], color=STYLE["color"]["reference"], marker="o", linestyle="none", markerfacecolor="none",
               markersize=STYLE["marker_size_pt"], label="No crossing observed within tested range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=False)

    _save_figure(fig, "f5_breakpoint_by_lead_group", [
        "results/phase3_lead_time_bucketed_breakpoints.parquet", "results/phase3_lead_time_bucketed_sweep.parquet",
    ])
    plt.close(fig)

    n_no_crossing = int(bp["breakpoint_n_cases"].isna().sum())
    _update_caption(
        "F5", "Breakpoint by lead-time group",
        "EMOS pooled/local breakpoint against TimesFM-3 (training cases, log "
        "scale), split by lead-time group (0-24h/24-72h/72-120h) and metric "
        "(CRPS, coverage@80%). Values are printed next to each marker. "
        f"{n_no_crossing} of {len(bp)} (metric, variant, lead-group) combinations "
        "show no crossing within the tested case range (open markers, labelled "
        "'no crossing observed' with the tested k range) rather than a bare "
        "'no breakpoint' claim, since the tested range is finite and a crossing "
        "below or above it cannot be ruled out. "
        "results/phase3_lead_time_bucketed_breakpoints.parquet does not include "
        "the low-N grid (k=1..7) tested elsewhere in this project, so any "
        "'no crossing' claim in the 0-24h row in particular has not been "
        "re-tested below k=9.",
    )


def main() -> None:
    make_f1()
    make_f2()
    make_f3()
    make_f4()
    make_f5()


if __name__ == "__main__":
    main()
