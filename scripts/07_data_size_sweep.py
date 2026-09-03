"""Training-data-size breakpoint curve (RQ2): refit EMOS at increasing amounts of
reforecast training data, under two sampling arms and two pooling variants; TimesFM-3
and raw ensemble are zero-shot / N-independent and reused as flat reference lines from
the already-persisted Phase 2 results.

"N days" is operationalized as k = round(N * 209 / 365) (year_idx, time_idx) issue-date
pairs out of the full 20-year x 209-date reforecast archive. Two arms pick WHICH k pairs:
  - contiguous (primary): the k pairs closest in time to the test period, working
    backward — simulates a real newly-deployed station's limited, chronologically
    contiguous history (small N sees roughly one season, not all four).
  - random (secondary, 5 seeds, mean +/- std reported): k pairs drawn uniformly at
    random from the full archive — gives artificial full-season coverage even at small
    N. The gap between the two arms' curves is itself a reported finding about the
    value of seasonal coverage, not a discrepancy to explain away.
See this plan's Task 4 "Before You Begin" for the full rationale and the real, verified
year_idx/time_idx chronological-polarity finding this script's ordering relies on.

Step 0 finding (2026-09-03, SSH altay, real archive inspection): the reforecast file's
"time" coordinate carries real datetime64 issue dates ascending from 2017-01-02 through
2018-12-31 (209 dates spanning roughly two calendar years, spaced ~3-4 days apart — this
is the archive's day-of-year template axis, not per-analog-year real dates) — CONFIRMED
real, ascending = later in the annual cycle. The "year" coordinate is a bare int64
positional index 1..20 (analog-year enumeration) with NO confirmed real calendar meaning
— its ascending-is-more-recent ordering is an ASSUMPTION (the standard archive-construction
convention), not a verified fact; sub-year (time_idx) contiguity resolution is real, but
whole-year (year_idx) ordering could in principle be reversed without this script being
able to detect it from the file alone. CHRONOLOGICAL_DESCENDING = True is kept as the
brief specified, given this mixed but not-contradicting finding (time_idx polarity
confirmed consistent with the assumption; year_idx polarity assumed, not verified).

EMOS is fit two ways at each N: "pooled" (one global model across all 49 stations, as
in Phase 2) and "local" (one model per station, fit only on that station's rows in the
subsample). Local EMOS needs >= LOCAL_EMOS_MIN_ROWS rows per station to fit reliably
(4 free parameters); at small N most stations won't have enough contiguous rows, so the
local curve is reported only over the stations it could actually cover, with that
coverage fraction reported alongside it — never silently filled in or extrapolated.

Reforecasts have 11 ensemble members (germany_ensemble_reforecasts_t2m.nc); the test-
period forecasts EMOS is evaluated against have 51. ens_var computed from 11 members is
a noisier (higher-variance) estimator of the true ensemble spread than one from 51 would
be, at every N equally — this is a fixed property of the training archive, not something
that changes with N, and is not corrected here. Documented as a limitation in the task
report, not fixed in code (no clean fix exists without re-simulating the ensemble).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_ids
from zeropp.eval.calibration import empirical_coverage
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles
from zeropp.models.emos import EMOS

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"

ISSUE_DATES_PER_YEAR = 209
DAYS_PER_YEAR = 365
LOCAL_EMOS_MIN_ROWS = 5

# Set from Step 0's verified finding. If year/time carry confirmed real chronological
# values, sort descending by that value (True). If falling back to the ascending-index
# assumption (see Step 0), sort descending by the raw index (also True in this codebase's
# convention where larger year_idx/time_idx already means later — confirm against your
# Step 0 output and flip to False here if your finding says otherwise).
CHRONOLOGICAL_DESCENDING = True


def n_days_to_k(n_days) -> int | str:
    """Convert an 'N days' label to a case count k, or pass through 'full'."""
    if n_days == "full":
        return "full"
    return round(n_days * ISSUE_DATES_PER_YEAR / DAYS_PER_YEAR)


def k_to_calendar_days(k: int) -> int:
    """Inverse of n_days_to_k's ratio, for reporting 'N cases (M calendar days)'."""
    return round(k * DAYS_PER_YEAR / ISSUE_DATES_PER_YEAR)


def sample_contiguous(full_train: pd.DataFrame, n_days) -> pd.DataFrame:
    """The k pairs closest in time to the test period, working backward (deterministic,
    no seed) — see module docstring and this plan's Task 4 Step 0 for the verified
    chronological-ordering rationale behind CHRONOLOGICAL_DESCENDING."""
    if n_days == "full":
        return full_train
    unique_pairs = full_train[["year_idx", "time_idx"]].drop_duplicates()
    k = min(n_days_to_k(n_days), len(unique_pairs))
    ordered = unique_pairs.sort_values(
        ["year_idx", "time_idx"], ascending=not CHRONOLOGICAL_DESCENDING
    )
    sampled_pairs = ordered.iloc[:k]
    return full_train.merge(sampled_pairs, on=["year_idx", "time_idx"], how="inner")


def sample_random(full_train: pd.DataFrame, n_days, seed: int) -> pd.DataFrame:
    """k pairs drawn uniformly at random from the full archive (fixed seed) — the
    prior single-arm design, kept as the secondary arm."""
    if n_days == "full":
        return full_train
    unique_pairs = full_train[["year_idx", "time_idx"]].drop_duplicates()
    k = min(n_days_to_k(n_days), len(unique_pairs))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(unique_pairs), size=k, replace=False)
    sampled_pairs = unique_pairs.iloc[sampled_idx]
    return full_train.merge(sampled_pairs, on=["year_idx", "time_idx"], how="inner")


def compute_metrics(y: np.ndarray, preds: np.ndarray, quantile_levels: list[float]) -> dict:
    """y: (n,1), preds: (n,1,n_quantiles). Returns crps, coverage_80pct (nominal 80%
    band from the q0.1/q0.9 pair, per this plan's Task 1 coverage-label fix), and
    interval_width_k (mean p10-p90 width in Kelvin, i.e. sharpness)."""
    crps = float(crps_from_quantiles(y, preds, quantile_levels).mean())
    coverage = float(empirical_coverage(y, preds, quantile_levels, lower=0.1, upper=0.9))
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)
    width = float(np.mean(preds[:, 0, hi_idx] - preds[:, 0, lo_idx]))
    return {"crps": crps, "coverage_80pct": coverage, "interval_width_k": width}


def fit_predict_pooled_emos(train_subset: pd.DataFrame, quantile_levels: list[float], test_X: dict) -> np.ndarray:
    train_for_emos = train_subset[["station_id", "ens_mean", "ens_var", "t2m_obs"]]
    model = EMOS(quantile_levels=quantile_levels).fit(train_for_emos)
    return model.predict_quantiles({"ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"]})


def fit_predict_local_emos(
    train_subset: pd.DataFrame,
    test_station_ids: np.ndarray,
    test_X: dict,
    quantile_levels: list[float],
    min_rows: int = LOCAL_EMOS_MIN_ROWS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One EMOS per station, fit only on that station's rows in train_subset. A test
    row whose station has < min_rows training rows (or none at all) is EXCLUDED, not
    filled in with a pooled fallback — the covered_mask and coverage_fraction make that
    exclusion explicit and reportable rather than silent."""
    n = len(test_station_ids)
    preds = np.full((n, 1, len(quantile_levels)), np.nan)
    covered_mask = np.zeros(n, dtype=bool)
    counts = train_subset["station_id"].value_counts()
    fittable_stations = counts[counts >= min_rows].index
    for sid in fittable_stations:
        station_train = train_subset[train_subset["station_id"] == sid][
            ["station_id", "ens_mean", "ens_var", "t2m_obs"]
        ]
        model = EMOS(quantile_levels=quantile_levels).fit(station_train)
        rows_mask = test_station_ids == sid
        if not rows_mask.any():
            continue
        station_preds = model.predict_quantiles(
            {"ens_mean": test_X["ens_mean"][rows_mask], "ens_var": test_X["ens_var"][rows_mask]}
        )
        preds[rows_mask] = station_preds
        covered_mask[rows_mask] = True
    coverage_fraction = float(len(fittable_stations)) / float(train_subset["station_id"].nunique() or 1)
    return preds, covered_mask, coverage_fraction


def find_breakpoint(n_cases_axis: list[int], method_values: list[float], reference_value: float, better: str = "lower") -> float | None:
    """Linear-interpolated N (in cases) where method_values crosses reference_value,
    walking n_cases_axis in ascending order. better='lower': method starts worse
    (higher) than reference and crosses to better (lower) — e.g. CRPS. better='closer_to_ref':
    for coverage, 'better' means the gap |value - reference| shrinks past the reference's
    own gap to nominal (not used here since TimesFM-3 IS the reference; instead this mode
    finds where method's coverage crosses the reference's coverage value directly, same as
    'lower'/'higher' but the caller passes the coverage direction it wants). Returns None if
    no crossing occurs within the tested range."""
    pairs = sorted(zip(n_cases_axis, method_values), key=lambda p: p[0])
    for (n0, v0), (n1, v1) in zip(pairs, pairs[1:]):
        if v0 is None or v1 is None or (isinstance(v0, float) and np.isnan(v0)) or (isinstance(v1, float) and np.isnan(v1)):
            continue
        starts_worse = (v0 > reference_value) if better == "lower" else (v0 < reference_value)
        ends_better = (v1 <= reference_value) if better == "lower" else (v1 >= reference_value)
        if starts_worse and ends_better and v1 != v0:
            frac = (reference_value - v0) / (v1 - v0)
            return n0 + frac * (n1 - n0)
    return None


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    data_size_days = config.data_size_days
    sweep_seeds = config.data_size_sweep_seeds

    full_train = build_train_ensemble_stats_with_ids(REFORECAST_PATH, REFORECAST_OBS_PATH)
    test_df = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)

    grouped = test_df.groupby(["station_id", "valid_time", "step_hours"])
    ens_means, ens_vars, obs_values = [], [], []
    for _, group in grouped:
        ens_means.append(group["t2m_forecast"].mean())
        ens_vars.append(group["t2m_forecast"].var())
        obs_values.append(group["t2m_obs"].iloc[0])
    ens_means = np.array(ens_means).reshape(-1, 1)
    ens_vars = np.array(ens_vars).reshape(-1, 1)
    obs_values = np.array(obs_values).reshape(-1, 1)
    test_station_ids = np.array([g[0] for g in grouped.groups.keys()])
    test_X = {"ens_mean": ens_means, "ens_var": ens_vars}

    unique_pairs_full = full_train[["year_idx", "time_idx"]].drop_duplicates()
    n_pairs_full = len(unique_pairs_full)

    rows = []
    for n_days in data_size_days:
        # --- contiguous arm (primary) ---
        train_contig = sample_contiguous(full_train, n_days)
        pooled_metrics = None  # reused below to skip redundant refitting at n_days == "full"
        if len(train_contig) == 0:
            # n_days == "full" always yields the full non-empty archive, so this
            # branch is unreachable for "full" in practice — but k must still be
            # strictly numeric-or-None (never the literal string "full") in case
            # it were ever reached, so n_cases/n_calendar_days_equiv stay a single
            # dtype for write_result's parquet write.
            k = n_days_to_k(n_days) if n_days != "full" else n_pairs_full
            n_calendar = k_to_calendar_days(k) if n_days != "full" else None
            print(f"N={n_days} days, arm=contiguous (0 training rows): EMOS undefined, no data to fit")
            for method in ["emos_pooled", "emos_local"]:
                rows.append({
                    "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                    "sampling_arm": "contiguous", "seed": None, "method": method,
                    "crps": float("nan"), "coverage_80pct": float("nan"), "interval_width_k": float("nan"),
                    "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                    "n_stations_covered": None,
                })
        else:
            k = n_days_to_k(n_days) if n_days != "full" else len(unique_pairs_full)
            n_calendar = k_to_calendar_days(k) if n_days != "full" else None

            pooled_preds = fit_predict_pooled_emos(train_contig, quantile_levels, test_X)
            pooled_metrics = compute_metrics(obs_values, pooled_preds, quantile_levels)
            rows.append({
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "contiguous", "seed": None, "method": "emos_pooled",
                **pooled_metrics,
                "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                "n_stations_covered": None,
            })
            print(f"N={n_days} days, arm=contiguous, method=emos_pooled ({len(train_contig)} rows): {pooled_metrics}")

            local_preds, covered_mask, coverage_fraction = fit_predict_local_emos(
                train_contig, test_station_ids, test_X, quantile_levels
            )
            n_unique_stations = train_contig["station_id"].nunique()
            n_stations_covered = round(coverage_fraction * n_unique_stations)
            if covered_mask.any():
                local_metrics = compute_metrics(
                    obs_values[covered_mask], local_preds[covered_mask], quantile_levels
                )
            else:
                local_metrics = {"crps": float("nan"), "coverage_80pct": float("nan"), "interval_width_k": float("nan")}
            rows.append({
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "contiguous", "seed": None, "method": "emos_local",
                **local_metrics,
                "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                "n_stations_covered": n_stations_covered,
            })
            print(
                f"N={n_days} days, arm=contiguous, method=emos_local "
                f"(coverage {n_stations_covered}/{n_unique_stations} stations): {local_metrics}"
            )

        # --- random arm (secondary, 5 seeds) ---
        k = n_days_to_k(n_days) if n_days != "full" else n_pairs_full
        n_calendar = k_to_calendar_days(k) if n_days != "full" else None

        if n_days == "full":
            # sample_random(full_train, "full", seed) returns full_train unchanged
            # for every seed — identical to the contiguous fit above by
            # construction, since there is nothing left to subsample from. Refitting
            # pooled EMOS 5 more times (plus the contiguous fit = 6x) on the full
            # ~4.3M-row archive was the single most expensive repeated computation
            # in the original run (each full-archive pooled fit dominates the whole
            # sweep's wall-clock time). Reuse the contiguous fit's pooled_metrics
            # instead; std is exactly 0.0 here BY CONSTRUCTION, not an empirical
            # finding (there is no seed-to-seed variation possible at N=full).
            assert pooled_metrics is not None, "full-N contiguous fit must have run before reuse"
            seed_metrics = [pooled_metrics] * len(sweep_seeds)
            for seed in sweep_seeds:
                rows.append({
                    "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                    "sampling_arm": "random", "seed": seed, "method": "emos_pooled",
                    **pooled_metrics,
                    "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                    "n_stations_covered": None,
                })
            print(
                f"N=full days, arm=random (all {len(sweep_seeds)} seeds), method=emos_pooled: "
                f"reused full-archive fit (identical to contiguous by construction, no refit): {pooled_metrics}"
            )
            mean_row = {
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "random_mean", "seed": None, "method": "emos_pooled",
                "crps": pooled_metrics["crps"],
                "coverage_80pct": pooled_metrics["coverage_80pct"],
                "interval_width_k": pooled_metrics["interval_width_k"],
                "crps_std": 0.0, "coverage_80pct_std": 0.0, "interval_width_k_std": 0.0,
                "n_stations_covered": None,
            }
        else:
            seed_metrics = []
            for seed in sweep_seeds:
                train_rand = sample_random(full_train, n_days, seed)
                if len(train_rand) == 0:
                    rand_metrics = {"crps": float("nan"), "coverage_80pct": float("nan"), "interval_width_k": float("nan")}
                    print(f"N={n_days} days, arm=random, seed={seed} (0 training rows): EMOS undefined, no data to fit")
                else:
                    rand_preds = fit_predict_pooled_emos(train_rand, quantile_levels, test_X)
                    rand_metrics = compute_metrics(obs_values, rand_preds, quantile_levels)
                    print(f"N={n_days} days, arm=random, seed={seed}, method=emos_pooled ({len(train_rand)} rows): {rand_metrics}")
                seed_metrics.append(rand_metrics)
                rows.append({
                    "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                    "sampling_arm": "random", "seed": seed, "method": "emos_pooled",
                    **rand_metrics,
                    "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                    "n_stations_covered": None,
                })

            seed_df = pd.DataFrame(seed_metrics)
            mean_row = {
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "random_mean", "seed": None, "method": "emos_pooled",
                "crps": float(seed_df["crps"].mean()),
                "coverage_80pct": float(seed_df["coverage_80pct"].mean()),
                "interval_width_k": float(seed_df["interval_width_k"].mean()),
                "crps_std": float(seed_df["crps"].std()),
                "coverage_80pct_std": float(seed_df["coverage_80pct"].std()),
                "interval_width_k_std": float(seed_df["interval_width_k"].std()),
                "n_stations_covered": None,
            }
        rows.append(mean_row)
        print(f"N={n_days} days, arm=random_mean, method=emos_pooled: {mean_row}")

    # --- N-independent reference lines: raw_ensemble, tsfm3 ---
    raw = pd.read_parquet(RAW_RESULTS_PATH)
    quantile_cols = [f"q{q}" for q in quantile_levels]
    for method in ["raw_ensemble", "tsfm3"]:
        df_method = raw[raw["method"] == method]
        qp = df_method[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
        y = df_method["obs"].to_numpy().reshape(-1, 1)
        ref_metrics = compute_metrics(y, qp, quantile_levels)
        for n_days in data_size_days:
            rows.append({
                "n_days": str(n_days), "n_cases": None, "n_calendar_days_equiv": None,
                "sampling_arm": "n_independent", "seed": None, "method": method,
                **ref_metrics,
                "crps_std": None, "coverage_80pct_std": None, "interval_width_k_std": None,
                "n_stations_covered": None,
            })
        print(f"method={method} (N-independent): {ref_metrics}")

    sweep_df = pd.DataFrame(rows)

    # Dtype audit: n_days is the ONLY string-valued column. n_cases and
    # n_calendar_days_equiv must be strictly numeric-or-None (never the literal
    # string "full") or pyarrow's Table.from_pandas raises ArrowInvalid trying to
    # unify a mixed int/str column — exactly what crashed the first real run here.
    def _numeric_or_none(v):
        return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) or (isinstance(v, float) and np.isnan(v))

    assert sweep_df["n_cases"].map(_numeric_or_none).all(), "n_cases contains a non-numeric value (e.g. literal 'full')"
    assert sweep_df["n_calendar_days_equiv"].map(_numeric_or_none).all(), "n_calendar_days_equiv contains a non-numeric value"

    write_result(
        sweep_df,
        name="phase3_data_size_sweep",
        model_version="phase3-sweep-v2",
        config={
            "data_size_days": data_size_days,
            "data_size_sweep_seeds": sweep_seeds,
            "quantile_levels": quantile_levels,
            "local_emos_min_rows": LOCAL_EMOS_MIN_ROWS,
            "chronological_descending": CHRONOLOGICAL_DESCENDING,
        },
    )

    # --- Breakpoints ---
    tsfm3_row = sweep_df[(sweep_df["method"] == "tsfm3")].iloc[0]
    breakpoint_rows = []
    for metric in ["crps", "coverage_80pct", "interval_width_k"]:
        reference_value = tsfm3_row[metric]
        for emos_variant in ["emos_pooled", "emos_local"]:
            arm_df = sweep_df[
                (sweep_df["sampling_arm"] == "contiguous") & (sweep_df["method"] == emos_variant)
            ].copy()
            if emos_variant == "emos_local":
                arm_df = arm_df[arm_df["n_stations_covered"].fillna(0) > 0]
            n_cases_axis = arm_df["n_cases"].astype(float).tolist()
            method_values = arm_df[metric].tolist()

            # Direction heuristics below inspect the value at the SMALLEST n_cases
            # that has real (non-NaN) data — not index [0] of method_values, which
            # would be the N=0 "undefined, no data to fit" row (NaN). NaN compares
            # False against everything in Python, so using index [0] directly would
            # silently default to the wrong direction whenever N=0's placeholder
            # row sorts first (it always does) — this was caught by comparing the
            # coverage breakpoint result against the figure by eye before finalizing.
            real_pairs = sorted(
                (
                    (n, v) for n, v in zip(n_cases_axis, method_values)
                    if v is not None and not (isinstance(v, float) and np.isnan(v))
                ),
                key=lambda p: p[0],
            )
            smallest_real_value = real_pairs[0][1] if real_pairs else None

            if metric == "crps":
                better = "lower"
            elif metric == "coverage_80pct":
                # Decide direction from the smallest-N real value: if the trained
                # method's coverage starts below the tsfm3 reference and needs to
                # rise toward it, use "higher" (crossing achieved by exceeding the
                # reference from below); if it starts above and falls toward it,
                # use "lower".
                if smallest_real_value is not None and smallest_real_value < reference_value:
                    better = "higher"
                else:
                    better = "lower"
            if metric == "interval_width_k":
                # Purely descriptive: narrower/wider is not "better" on its own
                # (it's a sharpness/calibration tradeoff), so there is no single
                # imposed crossing direction. find_breakpoint's "lower"/"higher"
                # modes are each one-directional (starts on one side, ends on the
                # other); trying only "lower" would silently miss a genuine
                # ascending crossing (which is exactly what emos_pooled does here
                # — its interval width rises from below the tsfm3 reference to
                # above it). Try both directions and report whichever one finds
                # an actual crossing in the tested range.
                bp = find_breakpoint(n_cases_axis, method_values, reference_value, better="lower")
                if bp is None:
                    bp = find_breakpoint(n_cases_axis, method_values, reference_value, better="higher")
            else:
                bp = find_breakpoint(n_cases_axis, method_values, reference_value, better=better)
            bp_calendar = k_to_calendar_days(round(bp)) if bp is not None else None
            breakpoint_rows.append({
                "metric": metric,
                "emos_variant": emos_variant,
                "breakpoint_n_cases": bp,
                "breakpoint_calendar_days": bp_calendar,
            })
            if bp is None:
                print(f"breakpoint: metric={metric}, variant={emos_variant}: no crossing observed in tested range")
            else:
                print(f"breakpoint: metric={metric}, variant={emos_variant}: {bp:.1f} cases (~{bp_calendar} calendar days)")

    breakpoints_df = pd.DataFrame(breakpoint_rows)
    assert breakpoints_df["breakpoint_n_cases"].map(_numeric_or_none).all(), "breakpoint_n_cases contains a non-numeric value"
    assert breakpoints_df["breakpoint_calendar_days"].map(_numeric_or_none).all(), "breakpoint_calendar_days contains a non-numeric value"

    write_result(
        breakpoints_df,
        name="phase3_data_size_sweep_breakpoints",
        model_version="phase3-sweep-v2",
        config={
            "data_size_days": data_size_days,
            "data_size_sweep_seeds": sweep_seeds,
            "quantile_levels": quantile_levels,
            "local_emos_min_rows": LOCAL_EMOS_MIN_ROWS,
            "chronological_descending": CHRONOLOGICAL_DESCENDING,
        },
    )

    # --- Figure ---
    x_order = [str(n) for n in data_size_days]
    x_labels = []
    for n_days in data_size_days:
        if n_days == "full":
            x_labels.append(f"full\n({n_pairs_full} cases)")
        else:
            x_labels.append(f"{n_days}\n({n_days_to_k(n_days)} cases)")

    def _ordered(df_subset):
        d = df_subset.copy()
        d["n_days"] = pd.Categorical(d["n_days"], categories=x_order, ordered=True)
        return d.sort_values("n_days")

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    metrics = ["crps", "coverage_80pct", "interval_width_k"]
    ylabels = ["CRPS", "Coverage @ 80% nominal", "Interval width (K)"]

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        pooled_contig = _ordered(sweep_df[(sweep_df["sampling_arm"] == "contiguous") & (sweep_df["method"] == "emos_pooled")])
        ax.plot(pooled_contig["n_days"].astype(str), pooled_contig[metric], marker="o", linestyle="-", label="EMOS pooled (contiguous)")

        rand_mean = _ordered(sweep_df[(sweep_df["sampling_arm"] == "random_mean") & (sweep_df["method"] == "emos_pooled")])
        ax.plot(rand_mean["n_days"].astype(str), rand_mean[metric], marker=None, linestyle="--", label="EMOS pooled (random, mean)")
        std_col = f"{metric}_std"
        lower = rand_mean[metric] - rand_mean[std_col]
        upper = rand_mean[metric] + rand_mean[std_col]
        ax.fill_between(rand_mean["n_days"].astype(str), lower, upper, alpha=0.2)

        local_contig = sweep_df[
            (sweep_df["sampling_arm"] == "contiguous")
            & (sweep_df["method"] == "emos_local")
            & (sweep_df["n_stations_covered"].fillna(0) > 0)
        ]
        local_contig = _ordered(local_contig)
        ax.plot(local_contig["n_days"].astype(str), local_contig[metric], marker="^", linestyle="-", label="EMOS local (contiguous, covered stations)")

        raw_val = sweep_df[sweep_df["method"] == "raw_ensemble"][metric].iloc[0]
        ax.axhline(raw_val, linestyle=":", label="Raw ensemble (N-independent)")
        tsfm3_val = sweep_df[sweep_df["method"] == "tsfm3"][metric].iloc[0]
        ax.axhline(tsfm3_val, linestyle="-.", label="TimesFM-3 zero-shot (N-independent)")

        ax.set_ylabel(ylabel)

    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Training data size")
    axes[-1].set_xticks(range(len(x_order)))
    axes[-1].set_xticklabels(x_labels)
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/data_size_sweep.png", dpi=150)
    print("Saved figure to figures/data_size_sweep.png")


if __name__ == "__main__":
    main()
