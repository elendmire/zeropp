"""Task 6: lead-time-bucketed analysis, folding in E5a, E5b, E4's optional
lead-time-grouped-EMOS extension, and E1's lead-time-level width-distribution
comparison -- all four need the SAME lead-time-bucketing machinery, so they are
combined here rather than duplicated across separate scripts (per the task
brief's "E2/E5 folded in wherever they're cheapest").

E4 (already resolved by Task 5's review + this repo's own tests/test_build.py
fixture, re-confirmed by tests/test_build.py's new
test_build_train_ensemble_stats_with_lead_step_hours_are_not_averaged_away):
build_train_ensemble_stats_with_ids never averages training rows over the
reforecast archive's 'step' (lead-time) dimension -- it only averages over
'number' (ensemble member). 'step' is merely dropped from the OUTPUT COLUMNS by
that function's final column subselection ([station_id, time_idx, year_idx,
ens_mean, ens_var, t2m_obs]). This is Scenario A from the task brief, not
Scenario B: rows genuinely differ by lead time, they are just unlabeled, and
every EMOS/DRN fit in this project pools across all 21 lead times. Direction of
the resulting bias (not yet magnitude, until the extension below) is derivable
analytically: a single global Gaussian fit minimizing mean CRPS across a mix of
lead times will be too WIDE relative to the true short-lead spread (which is
narrower, more persistence-informed) and too NARROW relative to the true
long-lead spread (which is wider, more purely climatological) -- the pooled fit's
sigma is a rows-weighted average of per-lead sigmas, not a per-lead-optimal one.

E4's optional extension (this script): fits one EMOS PER lead-time bucket (using
the new build_train_ensemble_stats_with_lead, which retains step_hours) and
compares each bucket's own interval width against the SAME pooled-across-all-
leads EMOS model's width, evaluated on that SAME bucket's test rows -- isolating
the effect of training-data lead-time pooling specifically, not a difference in
test population.

E5a: reuses results/phase3_lead_time_breakdown.parquet (already persisted by
scripts/06_lead_time_breakdown.py) directly -- pure aggregation, no new fit.

E5b: re-runs the SAME contiguous-arm EMOS pooled/local fits scripts/
07_data_size_sweep.py already validated (imported verbatim via importlib, the
same mechanism tests/test_data_size_sweep.py uses -- 07_data_size_sweep.py itself
is not modified or even re-executed as __main__ by this import), but computes
metrics and breakpoints SEPARATELY within each lead-time bucket instead of pooled
across all 21 lead times.

E1 (lead-time-level part only; the case-level N-sweep for the variance-inflation
baseline lives in scripts/07_data_size_sweep.py): compares the WIDTH DISTRIBUTION
(mean/std/p10/p50/p90, not just the mean) of TimesFM-3 against both
variance-inflation baseline variants and the full-data pooled EMOS fit, overall
and within each lead-time bucket.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import (
    build_test_long_table,
    build_train_ensemble_stats_with_lead,
)
from zeropp.eval.results import write_result
from zeropp.models.emos import EMOS
from zeropp.models.variance_inflation import VarianceInflationBaseline

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
FORECAST_PATH = "data/raw/germany_ensemble_forecasts_t2m.nc"
FORECAST_OBS_PATH = "data/raw/germany_forecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"
LEAD_TIME_BREAKDOWN_PATH = "results/phase3_lead_time_breakdown.parquet"

# Reuse scripts/07_data_size_sweep.py's already-reviewed helper functions
# VERBATIM (not reimplemented) via the same importlib-from-file mechanism
# tests/test_data_size_sweep.py already uses, since "scripts/07_data_size_sweep.py"
# is not an importable module name (starts with a digit). Only top-level
# definitions execute on load -- main() only runs under `if __name__ ==
# "__main__"`, so this import never re-runs 07's own sweep.
_SWEEP_SCRIPT_PATH = Path(__file__).resolve().parent / "07_data_size_sweep.py"
_SPEC = importlib.util.spec_from_file_location("data_size_sweep_module_08", _SWEEP_SCRIPT_PATH)
_sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_sweep)

sample_contiguous = _sweep.sample_contiguous
fit_predict_pooled_emos = _sweep.fit_predict_pooled_emos
fit_predict_local_emos = _sweep.fit_predict_local_emos
compute_metrics = _sweep.compute_metrics
breakpoint_and_direction = _sweep.breakpoint_and_direction
# Fix-round-1 minor item: E5b's pooled/local EMOS fits previously called
# fit_predict_pooled_emos/fit_predict_local_emos directly, bypassing the
# overflow-warning-counting wrapper Task 4 established for exactly these two
# functions in scripts/07_data_size_sweep.py. Reused verbatim here too.
_catch_overflow_warnings = _sweep._catch_overflow_warnings
k_and_calendar_days = _sweep.k_and_calendar_days
LAMBDA_CLIM = _sweep.LAMBDA_CLIM

# Task 6 E5b's lead-time buckets, per the brief's own example edges (0-24h,
# 24-72h, 72-120h). Half-open except the last (closed at 120h, the archive's max
# lead) -- every one of the archive's 21 step_hours values (0, 6, ..., 120) falls
# in exactly one bucket (see assign_lead_time_bucket's own fail-fast check).
LEAD_TIME_BUCKETS = [(0.0, 24.0, "0-24h"), (24.0, 72.0, "24-72h"), (72.0, 120.0, "72-120h")]


def assign_lead_time_bucket(step_hours) -> np.ndarray:
    step_hours = np.asarray(step_hours, dtype=float)
    labels = np.full(step_hours.shape, None, dtype=object)
    for low, high, label in LEAD_TIME_BUCKETS:
        is_last = label == LEAD_TIME_BUCKETS[-1][2]
        mask = (step_hours >= low) & ((step_hours <= high) if is_last else (step_hours < high))
        labels[mask] = label
    if any(l is None for l in labels):
        raise ValueError(
            "assign_lead_time_bucket: some step_hours values fell outside every "
            f"bucket in {LEAD_TIME_BUCKETS} -- extend the bucket edges before proceeding."
        )
    return labels


def find_first_sign_flip(step_hours_axis: list[float], diff_values: list[float]) -> tuple[float | None, str]:
    """Task 6 E5a: the interpolated lead time (hours) where diff_values
    (crps_tsfm3 - crps_emos) FIRST crosses from negative (tsfm3 better) to
    non-negative (tsfm3 worse or equal) as step_hours_axis ascends. Deliberately
    the first flip only -- see find_last_sign_flip_into_permanent_positive for the
    'durable' crossover, a separate statistic over the same data (the real
    CRPS-vs-lead-time curve is not perfectly monotonic in the sign of this diff)."""
    pairs = sorted(zip(step_hours_axis, diff_values), key=lambda p: p[0])
    for (h0, d0), (h1, d1) in zip(pairs, pairs[1:]):
        if d0 < 0 and d1 >= 0:
            if d1 == d0:
                return h0, f"adjacent points (step_hours={h0},{h1}) have identical diff — flip located at h0, no interpolation possible"
            frac = (0.0 - d0) / (d1 - d0)
            return h0 + frac * (h1 - h0), f"first sign flip between step_hours={h0} and {h1}"
    return None, "no negative-to-positive sign flip found across the tested lead times"


def find_last_sign_flip_into_permanent_positive(
    step_hours_axis: list[float], diff_values: list[float]
) -> tuple[float | None, str]:
    """The 'durable' crossover (Task 6 E5a judgment call -- see task report): the
    last flip after which diff_values (crps_tsfm3 - crps_emos) stays >= 0 for
    EVERY remaining tested lead time, i.e. the point TimesFM-3 never recovers past
    within the tested range. Reported alongside find_first_sign_flip because the
    raw curve flips sign more than once at short-to-medium lead here (real data,
    not synthetic), which would make "the first flip" alone a misleadingly early
    headline number."""
    pairs = sorted(zip(step_hours_axis, diff_values), key=lambda p: p[0])
    for i in range(len(pairs) - 1):
        h0, d0 = pairs[i]
        rest = pairs[i + 1:]
        if d0 < 0 and all(d >= 0 for _, d in rest):
            h1, d1 = rest[0]
            if d1 == d0:
                return h0, f"permanent flip located at step_hours={h0} (identical diff to {h1}, no interpolation)"
            frac = (0.0 - d0) / (d1 - d0)
            return (
                h0 + frac * (h1 - h0),
                f"last/durable sign flip between step_hours={h0} and {h1} — tsfm3 remains worse (or tied) for every longer tested lead time",
            )
    return None, "no durable negative-to-positive flip found (either always non-negative, always negative, or never stabilizes)"


def _instance_width(preds: np.ndarray, quantile_levels: list[float]) -> np.ndarray:
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)
    return preds[:, 0, hi_idx] - preds[:, 0, lo_idx]


def main() -> None:
    config = load_experiment_config()
    quantile_levels = config.quantile_levels
    quantile_cols = [f"q{q}" for q in quantile_levels]
    data_size_days = config.data_size_days

    # ================= E5a: lead-time crossover, from already-persisted data =================
    breakdown = pd.read_parquet(LEAD_TIME_BREAKDOWN_PATH)
    piv = breakdown.pivot(index="step_hours", columns="method", values="crps")
    diff = (piv["tsfm3"] - piv["emos"]).sort_index()
    step_axis = diff.index.astype(float).tolist()
    diff_values = diff.tolist()

    first_flip_h, first_flip_reason = find_first_sign_flip(step_axis, diff_values)
    durable_flip_h, durable_flip_reason = find_last_sign_flip_into_permanent_positive(step_axis, diff_values)
    print(f"E5a: first sign flip at ~{first_flip_h:.2f}h — {first_flip_reason}")
    print(f"E5a: durable sign flip at ~{durable_flip_h:.2f}h — {durable_flip_reason}")

    e5a_df = pd.DataFrame([
        {"crossover_type": "first_sign_flip", "step_hours": first_flip_h, "reason": first_flip_reason},
        {"crossover_type": "durable_sign_flip", "step_hours": durable_flip_h, "reason": durable_flip_reason},
    ])
    write_result(
        e5a_df,
        name="phase3_lead_time_crossover",
        model_version="phase3-lead-analysis-v1",
        config={"source": LEAD_TIME_BREAKDOWN_PATH},
    )

    # ================= Shared data loading: E4 extension / E5b / E1 lead-dist =================
    full_train_lead = build_train_ensemble_stats_with_lead(REFORECAST_PATH, REFORECAST_OBS_PATH)
    test_df = build_test_long_table(FORECAST_PATH, FORECAST_OBS_PATH)
    raw = pd.read_parquet(RAW_RESULTS_PATH)

    grouped = test_df.groupby(["station_id", "valid_time", "step_hours"])
    ens_means, ens_vars, obs_values = [], [], []
    key_station_ids, key_valid_times, key_step_hours = [], [], []
    for key, group in grouped:
        station_id, valid_time, step_hours = key
        ens_means.append(group["t2m_forecast"].mean())
        ens_vars.append(group["t2m_forecast"].var())
        obs_values.append(group["t2m_obs"].iloc[0])
        key_station_ids.append(station_id)
        key_valid_times.append(valid_time)
        key_step_hours.append(step_hours)
    ens_means = np.array(ens_means).reshape(-1, 1)
    ens_vars = np.array(ens_vars).reshape(-1, 1)
    obs_values = np.array(obs_values).reshape(-1, 1)
    test_station_ids = np.array(key_station_ids)
    test_keys_df = pd.DataFrame({
        "station_id": key_station_ids,
        "valid_time": key_valid_times,
        "step_hours": key_step_hours,
        "row_idx": np.arange(len(key_station_ids)),
    })

    # Instance-set join, re-derived independently here (not imported from
    # scripts/07_data_size_sweep.py's main()) -- see module docstring. Checked
    # again with the same assertion 07 uses, since this script re-derives rather
    # than reuses that specific join.
    key_cols = ["station_id", "valid_time", "step_hours"]
    canonical_key_set = raw.loc[raw["method"] == "tsfm3", key_cols].drop_duplicates()
    matched_keys = test_keys_df.merge(canonical_key_set, on=key_cols, how="inner")
    n_matched = len(matched_keys)
    assert n_matched == len(canonical_key_set), (
        f"instance-set join incomplete: matched {n_matched} of {len(canonical_key_set)} "
        "tsfm3 reference instances -- see scripts/07_data_size_sweep.py's identical "
        "assertion for the full rationale."
    )
    matched_row_idx = matched_keys["row_idx"].to_numpy()
    ens_means = ens_means[matched_row_idx]
    ens_vars = ens_vars[matched_row_idx]
    obs_values = obs_values[matched_row_idx]
    test_station_ids = test_station_ids[matched_row_idx]
    step_hours_matched = matched_keys["step_hours"].to_numpy()
    test_X = {"ens_mean": ens_means, "ens_var": ens_vars}
    lead_buckets_matched = assign_lead_time_bucket(step_hours_matched)
    n_unique_test_stations = len(np.unique(test_station_ids))

    tsfm3_ordered = matched_keys[key_cols].merge(raw[raw["method"] == "tsfm3"], on=key_cols, how="left")
    assert tsfm3_ordered["obs"].notna().all(), "tsfm3 left-merge produced an unmatched row"
    tsfm3_qp = tsfm3_ordered[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    tsfm3_y = tsfm3_ordered["obs"].to_numpy().reshape(-1, 1)

    raw_ens_ordered = matched_keys[key_cols].merge(raw[raw["method"] == "raw_ensemble"], on=key_cols, how="left")
    assert raw_ens_ordered["obs"].notna().all(), "raw_ensemble left-merge produced an unmatched row"
    raw_ens_qp = raw_ens_ordered[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    raw_ens_y = raw_ens_ordered["obs"].to_numpy().reshape(-1, 1)

    unique_pairs_full = full_train_lead[["year_idx", "time_idx"]].drop_duplicates()
    n_pairs_full = len(unique_pairs_full)
    full_train_blind = full_train_lead[["station_id", "ens_mean", "ens_var", "t2m_obs"]]

    # ================= E4 extension: pooled-across-leads vs. per-bucket-trained EMOS =================
    pooled_all_leads_model = EMOS(quantile_levels=quantile_levels).fit(full_train_blind)
    pooled_all_leads_preds = pooled_all_leads_model.predict_quantiles(test_X)

    e4_rows = []
    for low, high, label in LEAD_TIME_BUCKETS:
        bucket_mask = lead_buckets_matched == label
        bucket_test_X = {"ens_mean": test_X["ens_mean"][bucket_mask], "ens_var": test_X["ens_var"][bucket_mask]}
        bucket_obs = obs_values[bucket_mask]

        pooled_metrics_bucket = compute_metrics(bucket_obs, pooled_all_leads_preds[bucket_mask], quantile_levels)
        e4_rows.append({
            "lead_time_bucket": label, "training": "pooled_across_all_leads",
            **pooled_metrics_bucket, "n_instances": int(bucket_mask.sum()),
        })

        is_last_bucket = label == LEAD_TIME_BUCKETS[-1][2]
        train_bucket_mask = (full_train_lead["step_hours"] >= low) & (
            (full_train_lead["step_hours"] <= high) if is_last_bucket else (full_train_lead["step_hours"] < high)
        )
        bucket_train = full_train_lead.loc[train_bucket_mask, ["station_id", "ens_mean", "ens_var", "t2m_obs"]]
        bucket_model = EMOS(quantile_levels=quantile_levels).fit(bucket_train)
        bucket_preds = bucket_model.predict_quantiles(bucket_test_X)
        bucket_specific_metrics = compute_metrics(bucket_obs, bucket_preds, quantile_levels)
        e4_rows.append({
            "lead_time_bucket": label, "training": "bucket_specific",
            **bucket_specific_metrics, "n_instances": int(bucket_mask.sum()),
            "n_train_rows": int(train_bucket_mask.sum()),
        })

        width_delta = bucket_specific_metrics["interval_width_k"] - pooled_metrics_bucket["interval_width_k"]
        print(
            f"E4 ext: bucket={label}: pooled-across-leads width={pooled_metrics_bucket['interval_width_k']:.4f}K, "
            f"bucket-specific width={bucket_specific_metrics['interval_width_k']:.4f}K, delta={width_delta:+.4f}K "
            f"(crps pooled={pooled_metrics_bucket['crps']:.4f}, bucket-specific={bucket_specific_metrics['crps']:.4f})"
        )

    e4_df = pd.DataFrame(e4_rows)
    write_result(
        e4_df,
        name="phase3_lead_time_grouped_emos",
        model_version="phase3-lead-analysis-v1",
        config={"lead_time_buckets": LEAD_TIME_BUCKETS, "quantile_levels": quantile_levels},
    )

    # ================= E5b: bucketed breakpoints, over the SAME N axis Task 4 used =================
    e5b_rows = []
    for n_days in data_size_days:
        train_contig = sample_contiguous(full_train_lead, n_days)
        if len(train_contig) == 0:
            print(f"E5b: N={n_days} days (0 training rows) -- skipped, EMOS undefined")
            continue
        k, n_calendar = k_and_calendar_days(n_days, n_pairs_full)

        pooled_preds, n_overflow_pooled = _catch_overflow_warnings(fit_predict_pooled_emos, train_contig, quantile_levels, test_X)
        (local_preds, covered_mask, _), n_overflow_local = _catch_overflow_warnings(
            fit_predict_local_emos, train_contig, test_station_ids, test_X, quantile_levels
        )
        if n_overflow_pooled or n_overflow_local:
            print(
                f"E5b: N={n_days} days: {n_overflow_pooled} pooled + {n_overflow_local} local "
                "'overflow encountered in exp' RuntimeWarning(s) during EMOS optimization "
                "(proxy for potential optimizer instability, not a definitive non-convergence signal)."
            )

        for low, high, label in LEAD_TIME_BUCKETS:
            bucket_mask = lead_buckets_matched == label

            pooled_bucket_metrics = compute_metrics(obs_values[bucket_mask], pooled_preds[bucket_mask], quantile_levels)
            e5b_rows.append({
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "contiguous", "lead_time_bucket": label, "method": "emos_pooled",
                **pooled_bucket_metrics, "n_stations_covered": None,
            })

            local_bucket_mask = bucket_mask & covered_mask
            n_stations_covered_bucket = (
                len(np.unique(test_station_ids[local_bucket_mask])) if local_bucket_mask.any() else 0
            )
            if local_bucket_mask.any():
                local_bucket_metrics = compute_metrics(
                    obs_values[local_bucket_mask], local_preds[local_bucket_mask], quantile_levels
                )
            else:
                local_bucket_metrics = {"crps": float("nan"), "coverage_80pct": float("nan"), "interval_width_k": float("nan")}
            e5b_rows.append({
                "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
                "sampling_arm": "contiguous", "lead_time_bucket": label, "method": "emos_local",
                **local_bucket_metrics, "n_stations_covered": n_stations_covered_bucket,
            })
        print(f"E5b: N={n_days} days (k={k}) bucketed pooled/local EMOS computed for all {len(LEAD_TIME_BUCKETS)} buckets")

    # N-independent bucketed reference rows (raw_ensemble, tsfm3)
    for method, qp, y in [("raw_ensemble", raw_ens_qp, raw_ens_y), ("tsfm3", tsfm3_qp, tsfm3_y)]:
        for low, high, label in LEAD_TIME_BUCKETS:
            bucket_mask = lead_buckets_matched == label
            ref_metrics_bucket = compute_metrics(y[bucket_mask], qp[bucket_mask], quantile_levels)
            for n_days in data_size_days:
                e5b_rows.append({
                    "n_days": str(n_days), "n_cases": None, "n_calendar_days_equiv": None,
                    "sampling_arm": "n_independent", "lead_time_bucket": label, "method": method,
                    **ref_metrics_bucket, "n_stations_covered": None,
                })

    e5b_df = pd.DataFrame(e5b_rows)
    write_result(
        e5b_df,
        name="phase3_lead_time_bucketed_sweep",
        model_version="phase3-lead-analysis-v1",
        config={
            "lead_time_buckets": LEAD_TIME_BUCKETS, "data_size_days": data_size_days,
            "quantile_levels": quantile_levels,
        },
    )

    bp_rows = []
    for low, high, label in LEAD_TIME_BUCKETS:
        tsfm3_bucket_row = e5b_df[(e5b_df["method"] == "tsfm3") & (e5b_df["lead_time_bucket"] == label)].iloc[0]
        for metric in ["crps", "coverage_80pct"]:
            reference_value = tsfm3_bucket_row[metric]
            for variant in ["emos_pooled", "emos_local"]:
                arm = e5b_df[
                    (e5b_df["method"] == variant)
                    & (e5b_df["lead_time_bucket"] == label)
                    & (e5b_df["sampling_arm"] == "contiguous")
                ]
                if variant == "emos_local":
                    arm = arm[arm["n_stations_covered"].fillna(0) > 0]
                n_cases_axis = arm["n_cases"].astype(float).tolist()
                method_values = arm[metric].tolist()

                bp, reason = breakpoint_and_direction(metric, n_cases_axis, method_values, reference_value)
                bp_rows.append({
                    "lead_time_bucket": label, "metric": metric, "emos_variant": variant,
                    "breakpoint_n_cases": bp, "crossing_direction": reason,
                })
                if bp is None:
                    print(f"E5b breakpoint: bucket={label}, metric={metric}, variant={variant}: no crossing — {reason}")
                else:
                    print(f"E5b breakpoint: bucket={label}, metric={metric}, variant={variant}: {bp:.2f} cases — {reason}")

    bp_df = pd.DataFrame(bp_rows)
    write_result(
        bp_df,
        name="phase3_lead_time_bucketed_breakpoints",
        model_version="phase3-lead-analysis-v1",
        config={"lead_time_buckets": LEAD_TIME_BUCKETS, "data_size_days": data_size_days},
    )

    # ================= E1 (lead-time-level part): width DISTRIBUTION comparison =================
    vi_trainfit_full = VarianceInflationBaseline(quantile_levels=quantile_levels).fit(full_train_blind)
    vi_trainfit_preds = vi_trainfit_full.predict_quantiles(test_X)
    vi_fixed_model = VarianceInflationBaseline.from_fixed_multiplier(LAMBDA_CLIM, quantile_levels=quantile_levels)
    vi_fixed_preds = vi_fixed_model.predict_quantiles(test_X)

    width_by_method = {
        "tsfm3": _instance_width(tsfm3_qp, quantile_levels),
        "var_inflation_trainfit_full": _instance_width(vi_trainfit_preds, quantile_levels),
        "var_inflation_fixed": _instance_width(vi_fixed_preds, quantile_levels),
        "emos_pooled_full_all_leads": _instance_width(pooled_all_leads_preds, quantile_levels),
    }

    dist_rows = []
    scopes = [("all_leads", np.ones(len(test_station_ids), dtype=bool))] + [
        (label, lead_buckets_matched == label) for _, _, label in LEAD_TIME_BUCKETS
    ]
    for method_name, widths in width_by_method.items():
        for scope, mask in scopes:
            w = widths[mask]
            dist_rows.append({
                "method": method_name, "scope": scope,
                "width_mean": float(np.mean(w)), "width_std": float(np.std(w)),
                "width_p10": float(np.percentile(w, 10)), "width_p50": float(np.percentile(w, 50)),
                "width_p90": float(np.percentile(w, 90)), "n_instances": int(mask.sum()),
            })
            print(
                f"E1 width distribution: method={method_name}, scope={scope}: "
                f"mean={np.mean(w):.4f}K, std={np.std(w):.4f}K, p10={np.percentile(w, 10):.4f}K, "
                f"p50={np.percentile(w, 50):.4f}K, p90={np.percentile(w, 90):.4f}K"
            )

    dist_df = pd.DataFrame(dist_rows)
    write_result(
        dist_df,
        name="phase3_width_distribution",
        model_version="phase3-lead-analysis-v1",
        config={
            "lead_time_buckets": LEAD_TIME_BUCKETS, "lambda_clim": LAMBDA_CLIM,
            "quantile_levels": quantile_levels,
        },
    )


if __name__ == "__main__":
    main()
