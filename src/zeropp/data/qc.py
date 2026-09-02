import pandas as pd


def detect_gaps(series: pd.Series, freq: str) -> pd.DatetimeIndex:
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    missing = full_index.difference(series.index)
    return missing


def to_utc(series: pd.Series) -> pd.Series:
    if series.index.tz is None:
        raise ValueError(
            "series index has no tz info; DST/UTC ambiguity cannot be resolved "
            "implicitly — localize the source timezone explicitly before calling to_utc"
        )
    converted = series.copy()
    converted.index = converted.index.tz_convert("UTC")
    return converted


def interpolate_gaps(series: pd.Series, freq: str) -> pd.Series:
    # Create full index from the beginning of the day to the end of the series
    # This ensures we catch leading gaps (e.g., series starting at 01:00 instead of 00:00)
    min_time = series.index.min().normalize()
    max_time = series.index.max()
    full_index = pd.date_range(min_time, max_time, freq=freq)
    reindexed = series.reindex(full_index)
    filled = reindexed.interpolate(method="linear", limit_area="inside")
    if filled.isna().any():
        raise ValueError(
            "edge NaNs remain after linear interpolation — series starts or ends "
            "with missing values, which interpolation cannot fill; trim or extend "
            "the context window instead"
        )
    return filled
