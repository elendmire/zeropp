"""Real TimesFM-3 zero-shot comparison against the raw ensemble and EMOS, on
the real Germany t2m test set, at full scale on GPU.

Design (redesigned after review found a lead-time-collapsing flaw in an
earlier version of this script): forecasts are grouped by
(station_id, issue_time), where issue_time = valid_time - step_hours is the
true NWP issue time (derived here since load_test()'s long table keeps only
the correct valid_time, not the raw issue time). For each such group we build
ONE context (real past station observations strictly before issue_time,
CONTEXT_LENGTH of them) and ONE future covariate array -- the ensemble-mean
t2m_forecast for EACH distinct step_hours (lead time) in that group, averaged
over members, in step_hours order (up to 21 lead times for EUPPBench) -- and
call TimesFM3 ONCE per group with horizon = number of lead times present.
This produces one real quantile forecast per (station, valid_time,
step_hours) instance -- the same unit of analysis scripts/03_run_baselines.py
uses -- instead of collapsing every lead time and every ensemble member into
a single scalar covariate per valid_time, which is what the earlier version
of this script did (and which an independent review + matched-subsample
check showed materially distorted the CRPS comparison).

For every (station, valid_time, step_hours) instance TimesFM-3 is evaluated
on, this script ALSO computes the raw-ensemble CRPS and the EMOS CRPS on that
EXACT SAME instance, so the printed percentages below are a genuine
apples-to-apples comparison computed in one place -- no separate ad hoc
script or report-only arithmetic required to trust the finding.

`past_future_ens_spread` is deliberately NOT passed to
`TimesFM3.predict_quantiles` -- per zeropp.models.tsfm_timesfm.TimesFM3's own
docstring/task-5 finding, the real API only exposes one past-future covariate
slot (filled here by ensemble mean), and passing that key only triggers a
no-op UserWarning.

DEVICE: this script runs TimesFM-3 on GPU ("cuda"). A single predict() call
with horizon=21 was benchmarked on an altay A100 node at ~0.29s (vs ~1.8s/call
on CPU for a single-lead-time call), making the full, non-subsampled test set
tractable (~35,700 (station, issue_time) groups, each one call -> a few
hours). Submit this script as a Slurm GPU batch job; it is not meant to be
run interactively on a CPU-only login node.
"""
import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test, load_train
from zeropp.eval.scores import crps_from_quantiles
from zeropp.models.emos import EMOS
from zeropp.models.tsfm_timesfm import TimesFM3

CONTEXT_LENGTH = 40  # number of past observations fed as context, per station
DEVICE = "cuda"


def main() -> None:
    quantile_levels = load_experiment_config().quantile_levels

    train_df = load_train()
    test_df = load_test()
    test_df["issue_time"] = test_df["valid_time"] - pd.to_timedelta(test_df["step_hours"], unit="h")

    emos = EMOS(quantile_levels=quantile_levels).fit(train_df)
    model = TimesFM3(quantile_levels=quantile_levels, device=DEVICE)

    tsfm_preds, raw_preds, emos_preds, obs_values = [], [], [], []
    n_groups = 0
    n_skipped_short_context = 0

    for station_id, station_df in test_df.groupby("station_id"):
        station_df = station_df.sort_values("valid_time")

        # Real, chronologically sorted, deduplicated-by-valid_time observation
        # series for this station -- shared context source for every issue_time.
        obs_lookup = (
            station_df.drop_duplicates("valid_time")[["valid_time", "t2m_obs"]]
            .sort_values("valid_time")
            .reset_index(drop=True)
        )
        obs_times = obs_lookup["valid_time"].to_numpy()
        obs_vals = obs_lookup["t2m_obs"].to_numpy()

        for issue_time, group in station_df.groupby("issue_time"):
            n_groups += 1
            idx = np.searchsorted(obs_times, np.datetime64(issue_time), side="left")
            if idx < CONTEXT_LENGTH:
                n_skipped_short_context += 1
                continue
            context = obs_vals[idx - CONTEXT_LENGTH: idx]

            by_lead = (
                group.groupby("step_hours")
                .agg(
                    ens_mean=("t2m_forecast", "mean"),
                    ens_var=("t2m_forecast", "var"),
                    obs=("t2m_obs", "first"),
                )
                .sort_index()
            )
            future_ens_means = by_lead["ens_mean"].to_numpy()
            horizon = len(future_ens_means)
            if horizon == 0:
                continue

            covariate = np.concatenate([context, future_ens_means])

            tsfm_pred = model.predict_quantiles({
                "context": [context],
                "past_future_ens_mean": [covariate],
                "horizon": horizon,
            })  # shape (1, horizon, n_quantiles)

            emos_pred = emos.predict_quantiles({
                "ens_mean": by_lead["ens_mean"].to_numpy().reshape(-1, 1),
                "ens_var": by_lead["ens_var"].to_numpy().reshape(-1, 1),
            })  # shape (horizon, 1, n_quantiles)

            for i, step_hours in enumerate(by_lead.index):
                tsfm_preds.append(tsfm_pred[0, i])
                emos_preds.append(emos_pred[i, 0])
                obs_values.append(by_lead["obs"].iloc[i])

                member_vals = group.loc[group["step_hours"] == step_hours, "t2m_forecast"].to_numpy()
                raw_preds.append(np.quantile(member_vals, quantile_levels))

    print(
        f"Processed {n_groups} (station, issue_time) groups "
        f"({n_skipped_short_context} skipped for insufficient prior context)"
    )

    n = len(obs_values)
    tsfm_preds = np.array(tsfm_preds).reshape(n, 1, len(quantile_levels))
    raw_preds = np.array(raw_preds).reshape(n, 1, len(quantile_levels))
    emos_preds = np.array(emos_preds).reshape(n, 1, len(quantile_levels))
    obs_values = np.array(obs_values).reshape(-1, 1)

    tsfm_crps = crps_from_quantiles(obs_values, tsfm_preds, quantile_levels)
    raw_crps = crps_from_quantiles(obs_values, raw_preds, quantile_levels)
    emos_crps = crps_from_quantiles(obs_values, emos_preds, quantile_levels)

    print(f"Evaluated on {n} real (station, valid_time, step_hours) forecast instances")
    print(f"Raw ensemble CRPS (matched instances):  {raw_crps.mean():.4f}")
    print(f"EMOS CRPS (matched instances):          {emos_crps.mean():.4f}")
    print(f"TimesFM-3 zero-shot CRPS (matched):     {tsfm_crps.mean():.4f}")
    print(f"EMOS improvement over raw ensemble (matched instances):      "
          f"{(1 - emos_crps.mean() / raw_crps.mean()) * 100:.1f}%")
    print(f"TimesFM-3 improvement over raw ensemble (matched instances): "
          f"{(1 - tsfm_crps.mean() / raw_crps.mean()) * 100:.1f}%")
    print(f"TimesFM-3 improvement over EMOS (matched instances):         "
          f"{(1 - tsfm_crps.mean() / emos_crps.mean()) * 100:.1f}%")


if __name__ == "__main__":
    main()
