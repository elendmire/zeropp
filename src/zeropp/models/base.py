from abc import ABC, abstractmethod

import numpy as np


class Postprocessor(ABC):
    """Common contract for every baseline and TSFM model in ZeroPP.

    predict_quantiles must return an array of shape
    (n_samples, n_leads, n_quantiles) at the quantile levels defined in
    configs/experiment.yaml — never a model-specific shape.
    """

    @abstractmethod
    def fit(self, train) -> "Postprocessor":
        raise NotImplementedError

    @abstractmethod
    def predict_quantiles(self, X) -> np.ndarray:
        raise NotImplementedError
