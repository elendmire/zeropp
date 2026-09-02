import pandas as pd
import xarray as xr


def build_train_ensemble_stats(reforecast_path: str, obs_path: str) -> pd.DataFrame:
    fcs = xr.open_dataset(reforecast_path)
    obs = xr.open_dataset(obs_path)

    ens_mean = fcs["t2m"].mean(dim="number")
    ens_var = fcs["t2m"].var(dim="number")

    merged = xr.Dataset({"ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": obs["t2m"]})
    df = merged.to_dataframe().reset_index()
    df = df[["station_id", "ens_mean", "ens_var", "t2m_obs"]]
    df = df.dropna(subset=["ens_mean", "ens_var", "t2m_obs"])
    return df.reset_index(drop=True)


def build_test_long_table(forecast_path: str, obs_path: str) -> pd.DataFrame:
    fcs = xr.open_dataset(forecast_path)
    try:
        obs = xr.open_dataset(obs_path)
    except OverflowError:
        # Known intermittent bug on the forecasts-observations file: unmasked
        # fill-value sentinels in the time coordinate can overflow the int64
        # conversion during decode_times. Retry once, then fall back to
        # decode_times=False (same proven fix used for the reforecast obs file).
        try:
            obs = xr.open_dataset(obs_path)
        except OverflowError:
            obs = xr.open_dataset(obs_path, decode_times=False)
            if not pd.api.types.is_datetime64_any_dtype(obs["time"].values):
                raise RuntimeError(
                    "obs time coordinate did not decode to datetime64 even after "
                    "the decode_times=False fallback; merging on raw integer time "
                    "offsets against the forecast file's real datetimes would "
                    "silently produce zero matches. Needs manual time decoding "
                    "for this file before build_test_long_table can proceed."
                )

    fcs_df = fcs["t2m"].to_dataframe(name="t2m_forecast").reset_index()
    # Real EUPPBench forecast files carry a genuine (time, step) -> valid_time
    # auxiliary coordinate (init time + lead), which to_dataframe() surfaces as
    # its own column alongside raw "time". Prefer it when present since it is
    # the true verification timestamp; the synthetic test fixtures have no such
    # coordinate, so "time" itself is used as valid_time there.
    has_native_valid_time = "valid_time" in fcs_df.columns
    fcs_keep = ["station_id", "time", "step", "number", "t2m_forecast"]
    if has_native_valid_time:
        fcs_keep.append("valid_time")
    fcs_df = fcs_df[fcs_keep]

    obs_df = obs["t2m"].to_dataframe(name="t2m_obs").reset_index()
    obs_df = obs_df[["station_id", "time", "step", "t2m_obs"]]

    merged = fcs_df.merge(obs_df, on=["station_id", "time", "step"], how="left")
    if not has_native_valid_time:
        merged = merged.rename(columns={"time": "valid_time"})
    else:
        merged = merged.drop(columns=["time"])
    merged = merged.rename(columns={"step": "step_hours", "number": "member"})
    merged = merged.dropna(subset=["t2m_forecast", "t2m_obs"])
    merged = merged[["station_id", "valid_time", "step_hours", "member", "t2m_forecast", "t2m_obs"]]
    merged = merged.sort_values(["station_id", "valid_time"]).reset_index(drop=True)
    return merged
