from zeropp.models.base import Postprocessor


class DRN(Postprocessor):
    """Distributional Regression Network baseline (Rasp & Lerch 2018).

    BLOCKED: needs EUPPBench training data (splits.py) to implement fit().
    Do not implement with synthetic data — a neural-network fit only means
    something against real ensemble/observation pairs.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "DRN":
        raise NotImplementedError("blocked: needs EUPPBench training data via splits.py")

    def predict_quantiles(self, X):
        raise NotImplementedError("blocked: needs EUPPBench training data via splits.py")
