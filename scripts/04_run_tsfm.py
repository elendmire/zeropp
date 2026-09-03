"""Real TimesFM-3 zero-shot comparison against the raw ensemble, on the real Germany t2m test set.

NOTE on subsampling: a full run (every forecast instance at every station) is
~141,000 CPU inference calls at ~1.8s/call (~70 hours) -- benchmarked on the
server before this run. That is not tractable for this slice, so we
subsample: for each station, instead of iterating every index in
range(CONTEXT_LENGTH, len(obs_series) - 1), we take every SUBSAMPLE_STRIDE-th
index. This keeps ~10 evenly-time-spaced forecast instances per station
(~490 total across 49 stations), for a real run of roughly 15 minutes on
CPU, while still covering the full test period and all stations rather than
just an early time window.

Also: `past_future_ens_spread` is deliberately NOT passed to
`TimesFM3.predict_quantiles` -- per zeropp.models.tsfm_timesfm.TimesFM3's own
docstring/task-5 finding, the real API only exposes one past-future covariate
slot (filled here by ensemble mean), and passing that key only triggers a
UserWarning with no effect on the prediction.
"""
import numpy as np

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test
from zeropp.eval.scores import crps_from_quantiles
from zeropp.models.tsfm_timesfm import TimesFM3

CONTEXT_LENGTH = 40  # number of past observations fed as context, per station
SUBSAMPLE_STRIDE = 288  # take every Nth eligible forecast instance per station (see module docstring)


def main() -> None:
    quantile_levels = load_experiment_config().quantile_levels
    test_df = load_test()

    model = TimesFM3(quantile_levels=quantile_levels)

    all_preds, all_obs = [], []
    for station_id, station_df in test_df.groupby("station_id"):
        station_df = station_df.sort_values("valid_time")
        obs_series = station_df.drop_duplicates("valid_time")["t2m_obs"].to_numpy()
        if len(obs_series) <= CONTEXT_LENGTH:
            continue

        for i in range(CONTEXT_LENGTH, len(obs_series) - 1, SUBSAMPLE_STRIDE):
            context = obs_series[i - CONTEXT_LENGTH:i]
            future_row = station_df[station_df["valid_time"] == station_df["valid_time"].unique()[i]]
            if future_row.empty:
                continue
            ens_mean_future = future_row["t2m_forecast"].mean()
            covariate = np.concatenate([context, np.full(1, ens_mean_future)])

            preds = model.predict_quantiles({
                "context": [context],
                "past_future_ens_mean": [covariate],
                "horizon": 1,
            })
            all_preds.append(preds[0])
            all_obs.append(obs_series[i])

    all_preds = np.array(all_preds)
    all_obs = np.array(all_obs).reshape(-1, 1)

    tsfm_crps = crps_from_quantiles(all_obs, all_preds, quantile_levels)
    print(f"TimesFM-3 zero-shot mean CRPS: {tsfm_crps.mean():.4f}")
    print(f"Evaluated on {len(all_obs)} real forecast instances")


if __name__ == "__main__":
    main()
