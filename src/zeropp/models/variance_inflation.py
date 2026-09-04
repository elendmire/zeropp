import numpy as np
from scipy.stats import norm

from zeropp.models.base import Postprocessor


class VarianceInflationBaseline(Postprocessor):
    """Trivial single-parameter baseline for Task 6 E1: mu = ens_mean (no mean
    adjustment), sigma = multiplier * sqrt(ens_var) -- the same Gaussian parametric
    family EMOS/DRN already use in this project (mu = a + b*ens_mean, sigma =
    sqrt(exp(c) + exp(d)*ens_var)), but with exactly ONE free scalar and the mean
    left untouched, so it is a fair "1 parameter / k cases" contrast against EMOS's
    "4 parameters / k cases".

    The point of this baseline is to check whether TimesFM-3's sharpness/coverage
    can be matched by a trivial rescaling of the raw ensemble's own reported
    spread, with no fitted mean correction and (for the fixed-multiplier variant)
    no fitting step at all.

    Two ways to obtain `multiplier`, per Task 6 E1 -- both are reported, neither is
    picked as "the" baseline:
      - fit(train): the standard second-moment-matching / spread-skill-ratio
        estimator, multiplier = sqrt(mean squared error / mean ensemble variance)
        over the training rows passed in. Callers MUST pass only a training subset
        (e.g. the same k-sized contiguous subset EMOS is fit on at each N in
        scripts/07_data_size_sweep.py) -- fitting this on test data would be
        leakage, exactly as it would be for EMOS.
      - from_fixed_multiplier(value, quantile_levels): a genuinely zero-shot
        constructor -- no fit() call, no training data at all. `value` is meant to
        be a literal constant, e.g. a published climatological ensemble
        under-dispersion inflation factor (see scripts/07_data_size_sweep.py's
        LAMBDA_CLIM constant and comment for the citation and its caveats), matching
        TimesFM-3's own zero-shot status.

    Fix-round-1 finding 4: an instance created via from_fixed_multiplier is
    PERMANENTLY fixed -- calling .fit(train) on it (e.g. by accident, or via generic
    code that calls .fit() on every Postprocessor uniformly) is a no-op that returns
    self unchanged, rather than silently overwriting the fixed multiplier with a
    data-derived one. Without this guard,
    `VarianceInflationBaseline.from_fixed_multiplier(1.5).fit(train)` would quietly
    stop being the zero-shot baseline E1(b) is supposed to be.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels
        self.multiplier = None
        self._fixed = False

    @classmethod
    def from_fixed_multiplier(cls, multiplier: float, quantile_levels: list[float]) -> "VarianceInflationBaseline":
        instance = cls(quantile_levels)
        instance.multiplier = float(multiplier)
        instance._fixed = True
        return instance

    def fit(self, train) -> "VarianceInflationBaseline":
        if self._fixed:
            # Fix-round-1 finding 4: from_fixed_multiplier's whole point is a
            # multiplier that is NOT derived from any training data -- fit() must
            # not be allowed to silently overwrite it.
            return self
        obs = np.asarray(train["t2m_obs"])
        ens_mean = np.asarray(train["ens_mean"])
        ens_var = np.asarray(train["ens_var"])
        mse = np.mean((obs - ens_mean) ** 2)
        mean_var = np.mean(ens_var)
        self.multiplier = float(np.sqrt(mse / mean_var))
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        if self.multiplier is None:
            raise RuntimeError(
                "VarianceInflationBaseline.predict_quantiles called before fit() or "
                "from_fixed_multiplier()"
            )
        ens_mean = X["ens_mean"]
        sigma = self.multiplier * np.sqrt(X["ens_var"])
        quantiles = [ens_mean + sigma * norm.ppf(q) for q in self.quantile_levels]
        return np.stack(quantiles, axis=-1)
