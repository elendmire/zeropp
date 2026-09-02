import numpy as np

from zeropp.eval.scores import _quantile_index


def pit_values(
    y_true: np.ndarray, quantile_preds: np.ndarray, quantile_levels: list[float]
) -> np.ndarray:
    n_samples, n_leads = y_true.shape
    out = np.empty((n_samples, n_leads))
    tau = np.array(quantile_levels)
    for i in range(n_samples):
        for j in range(n_leads):
            q = quantile_preds[i, j, :]
            out[i, j] = np.interp(y_true[i, j], q, tau)
    return out


def pit_histogram(pit: np.ndarray, n_bins: int = 10) -> np.ndarray:
    counts, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    return counts / counts.sum()


def empirical_coverage(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantile_levels: list[float],
    lower: float = 0.1,
    upper: float = 0.9,
) -> float:
    lower_idx, upper_idx = _quantile_index(quantile_levels, (lower, upper))
    q_lower = quantile_preds[..., lower_idx]
    q_upper = quantile_preds[..., upper_idx]
    inside = (y_true >= q_lower) & (y_true <= q_upper)
    return float(np.mean(inside))


def reliability_index(pit: np.ndarray, n_bins: int = 10) -> float:
    observed = pit_histogram(pit, n_bins=n_bins)
    expected = 1.0 / n_bins
    return float(np.sqrt(np.mean((observed - expected) ** 2)))
