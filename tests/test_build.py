import numpy as np
import pandas as pd
import pytest
import xarray as xr

from zeropp.data.build import build_test_long_table, build_train_ensemble_stats


@pytest.fixture
def synthetic_reforecast_files(tmp_path):
    # 2 stations, 2 time_idx, 1 year, 2 step, 3 members
    station_id = [100, 200]
    time_idx = [0, 1]
    year_idx = [0]
    step = [0.0, 6.0]
    number = [0, 1, 2]

    rng = np.random.default_rng(0)
    fcs_data = rng.normal(280.0, 2.0, size=(2, 2, 1, 2, 3, 1)).astype("float32")
    fcs = xr.Dataset(
        {"t2m": (("station_id", "time", "year", "step", "number", "surface"), fcs_data)},
        coords={
            "station_id": station_id, "time": time_idx, "year": year_idx,
            "step": step, "number": number, "surface": [0.0],
        },
    )
    fcs_path = tmp_path / "reforecasts.nc"
    fcs.to_netcdf(fcs_path)

    obs_data = np.full((2, 2, 1, 2), 280.0, dtype="float64")
    obs_data[0, 0, 0, 0] = np.nan  # one missing observation to verify it gets dropped
    obs = xr.Dataset(
        {"t2m": (("station_id", "time", "year", "step"), obs_data)},
        coords={"station_id": station_id, "time": time_idx, "year": year_idx, "step": step},
    )
    obs_path = tmp_path / "reforecast_obs.nc"
    obs.to_netcdf(obs_path)

    return str(fcs_path), str(obs_path)


@pytest.fixture
def synthetic_forecast_files(tmp_path):
    station_id = [100, 200]
    time = pd.date_range("2017-01-01", periods=3, freq="12h")
    step = [0.0, 6.0]
    number = [0, 1]

    rng = np.random.default_rng(1)
    fcs_data = rng.normal(280.0, 2.0, size=(2, 3, 2, 2, 1)).astype("float32")
    fcs = xr.Dataset(
        {"t2m": (("station_id", "time", "step", "number", "surface"), fcs_data)},
        coords={"station_id": station_id, "time": time, "step": step, "number": number, "surface": [0.0]},
    )
    fcs_path = tmp_path / "forecasts.nc"
    fcs.to_netcdf(fcs_path)

    obs_data = np.full((2, 3, 2), 280.0, dtype="float64")
    obs_data[1, 2, 1] = np.nan  # one missing observation
    obs = xr.Dataset(
        {"t2m": (("station_id", "time", "step"), obs_data)},
        coords={"station_id": station_id, "time": time, "step": step},
    )
    obs_path = tmp_path / "forecast_obs.nc"
    obs.to_netcdf(obs_path)

    return str(fcs_path), str(obs_path)


def test_build_train_ensemble_stats_shape_and_columns(synthetic_reforecast_files):
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats(fcs_path, obs_path)
    assert list(df.columns) == ["station_id", "ens_mean", "ens_var", "t2m_obs"]
    # 2 stations x 2 time_idx x 1 year x 2 step = 8 combinations, minus 1 dropped NaN obs
    assert len(df) == 7


def test_build_train_ensemble_stats_mean_and_var_are_real_ensemble_stats(synthetic_reforecast_files):
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats(fcs_path, obs_path)
    assert (df["ens_var"] >= 0).all()
    assert df["ens_mean"].between(270, 290).all()  # sanity: within the synthetic generation range


def test_build_test_long_table_shape_and_columns(synthetic_forecast_files):
    fcs_path, obs_path = synthetic_forecast_files
    df = build_test_long_table(fcs_path, obs_path)
    assert list(df.columns) == ["station_id", "valid_time", "step_hours", "member", "t2m_forecast", "t2m_obs"]
    # 2 stations x 3 time x 2 step x 2 members = 24 rows, minus 2 members dropped for the 1 NaN obs row
    assert len(df) == 22


def test_build_test_long_table_sorted_by_station_then_time(synthetic_forecast_files):
    fcs_path, obs_path = synthetic_forecast_files
    df = build_test_long_table(fcs_path, obs_path)
    for station_id, group in df.groupby("station_id"):
        assert group["valid_time"].is_monotonic_increasing


def test_build_test_long_table_valid_time_is_real_timestamp(synthetic_forecast_files):
    fcs_path, obs_path = synthetic_forecast_files
    df = build_test_long_table(fcs_path, obs_path)
    assert pd.api.types.is_datetime64_any_dtype(df["valid_time"])
