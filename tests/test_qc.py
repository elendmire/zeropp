import numpy as np
import pandas as pd
import pytest

from zeropp.data.qc import detect_gaps, interpolate_gaps, to_utc


def test_detect_gaps_finds_missing_hour():
    idx = pd.date_range("2026-01-01", periods=5, freq="h").delete(2)
    series = pd.Series(np.arange(4, dtype=float), index=idx)
    gaps = detect_gaps(series, freq="h")
    assert pd.Timestamp("2026-01-01 02:00") in gaps


def test_detect_gaps_empty_when_continuous():
    idx = pd.date_range("2026-01-01", periods=5, freq="h")
    series = pd.Series(np.arange(5, dtype=float), index=idx)
    gaps = detect_gaps(series, freq="h")
    assert len(gaps) == 0


def test_interpolate_gaps_fills_missing_hour_linearly():
    idx = pd.date_range("2026-01-01", periods=5, freq="h").delete(2)
    series = pd.Series([0.0, 1.0, 3.0, 4.0], index=idx)
    filled = interpolate_gaps(series, freq="h")
    assert len(filled) == 5
    assert filled.loc["2026-01-01 02:00"] == pytest.approx(2.0)


def test_interpolate_gaps_raises_on_edge_nan():
    idx = pd.date_range("2026-01-01", periods=5, freq="h").delete(0)
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    with pytest.raises(ValueError, match="edge"):
        interpolate_gaps(series, freq="h")


def test_to_utc_converts_local_tz_to_utc():
    idx = pd.date_range("2026-03-29 00:00", periods=3, freq="h", tz="Europe/Istanbul")
    series = pd.Series([1.0, 2.0, 3.0], index=idx)
    converted = to_utc(series)
    assert str(converted.index.tz) == "UTC"


def test_to_utc_raises_on_naive_index():
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    series = pd.Series([1.0, 2.0, 3.0], index=idx)
    with pytest.raises(ValueError, match="tz"):
        to_utc(series)
