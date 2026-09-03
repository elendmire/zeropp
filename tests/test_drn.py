import numpy as np
import pandas as pd
import pytest

from zeropp.models.base import Postprocessor
from zeropp.models.drn import DRN

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def synthetic_train_df():
    rng = np.random.default_rng(0)
    n_per_station = 200
    station_ids = [1, 2, 3]
    station_bias = {1: -2.0, 2: 0.0, 3: 3.0}  # each station has a real, distinct bias

    rows = []
    for sid in station_ids:
        ens_mean = rng.normal(280.0, 5.0, n_per_station)
        ens_var = rng.uniform(0.5, 2.0, n_per_station)
        t2m_obs = ens_mean + station_bias[sid] + rng.normal(0, np.sqrt(ens_var))
        rows.append(pd.DataFrame({"station_id": sid, "ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": t2m_obs}))
    return pd.concat(rows, ignore_index=True)


def test_drn_is_a_postprocessor():
    assert issubclass(DRN, Postprocessor)


def test_drn_fit_returns_self(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5)
    assert model.fit(synthetic_train_df) is model


def test_drn_predict_quantiles_shape(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5).fit(synthetic_train_df)
    X = {
        "ens_mean": np.full((3, 1), 280.0),
        "ens_var": np.full((3, 1), 1.0),
        "station_id": np.array([1, 2, 3]),
    }
    preds = model.predict_quantiles(X)
    assert preds.shape == (3, 1, 9)


def test_drn_predict_before_fit_raises():
    model = DRN(quantile_levels=QUANTILE_LEVELS)
    X = {"ens_mean": np.zeros((1, 1)), "ens_var": np.ones((1, 1)), "station_id": np.array([1])}
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_quantiles(X)


def test_drn_learns_real_per_station_bias(synthetic_train_df):
    # Behavioral test: after training, DRN's median prediction for the same ens_mean
    # should differ meaningfully across stations, tracking the real, distinct biases
    # the synthetic data was generated with (station 3's median should be well above
    # station 1's, since the true generating bias differs by 5.0).
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=200, lr=0.05, seed=0).fit(synthetic_train_df)
    X = {
        "ens_mean": np.full((3, 1), 280.0),
        "ens_var": np.full((3, 1), 1.0),
        "station_id": np.array([1, 2, 3]),
    }
    preds = model.predict_quantiles(X)
    median_idx = QUANTILE_LEVELS.index(0.5)
    medians = preds[:, 0, median_idx]
    assert medians[2] - medians[0] > 2.0  # station 3 vs station 1, true gap is 5.0


def test_drn_predict_quantiles_are_monotonic_increasing(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5).fit(synthetic_train_df)
    X = {"ens_mean": np.full((2, 1), 280.0), "ens_var": np.full((2, 1), 1.0), "station_id": np.array([1, 2])}
    preds = model.predict_quantiles(X)
    assert np.all(np.diff(preds, axis=-1) >= 0)


def test_drn_unseen_station_at_predict_time_does_not_crash(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5).fit(synthetic_train_df)
    X = {"ens_mean": np.full((1, 1), 280.0), "ens_var": np.full((1, 1), 1.0), "station_id": np.array([999])}
    preds = model.predict_quantiles(X)  # station 999 was never in training data
    assert preds.shape == (1, 1, 9)
