from zeropp.models.base import Postprocessor


class EMOS(Postprocessor):
    """Ensemble Model Output Statistics baseline.

    BLOCKED: needs EUPPBench training data (splits.py) to implement fit().
    Do not implement with synthetic data — the CRPS-minimization fit only
    means something against real ensemble/observation pairs.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "EMOS":
        raise NotImplementedError("blocked: needs EUPPBench training data via splits.py")

    def predict_quantiles(self, X):
        raise NotImplementedError("blocked: needs EUPPBench training data via splits.py")
