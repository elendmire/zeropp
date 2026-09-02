import warnings

import numpy as np
import timesfm

from zeropp.models.base import Postprocessor

DEFAULT_WEIGHTS_PATH = "/ari/users/oavci/zeropp/model_cache/timesfm-3.0-pytorch"


class TimesFM3(Postprocessor):
    """Frozen TimesFM-3 zero-shot postprocessor with real past-future covariate injection.

    NOTE: `timesfm.TimesFM3Forecaster.predict`/`predict_batch` expose a single
    `past_future_covariates` slot (verified via `inspect.signature` on the
    server against the real installed package) — there is no separate slot
    for a second simultaneous past-future covariate. This implementation
    therefore injects ensemble mean as the covariate and does NOT pass
    ensemble spread (`X["past_future_ens_spread"]`) to the model at all for
    this first slice. See task-5-report.md for the explicit limitation note.
    """

    def __init__(self, quantile_levels: list[float], weights_path: str = DEFAULT_WEIGHTS_PATH):
        self.quantile_levels = quantile_levels
        self.weights_path = weights_path
        self._model = None

    def fit(self, train) -> "TimesFM3":
        return self  # zero-shot: no-op fit, but see predict_quantiles

    def predict_quantiles(self, X: dict) -> np.ndarray:
        if "past_future_ens_spread" in X:
            warnings.warn(
                "X['past_future_ens_spread'] was provided but TimesFM3.predict_quantiles "
                "does not use it — the real TimesFM-3 API only exposes one "
                "past_future_covariates slot, currently filled by ensemble mean. See "
                "TimesFM3's class docstring for details."
            )

        if self._model is None:
            self._model = timesfm.TimesFM3Forecaster.from_pretrained(self.weights_path, device="cpu")

        contexts = X["context"]
        ens_means = X["past_future_ens_mean"]
        horizon = X["horizon"]

        all_quantiles = []
        for ctx, ens_mean in zip(contexts, ens_means):
            out = self._model.predict(
                context=ctx,
                horizon=horizon,
                # TODO: investigate whether past_future_covariates accepts a
                # multi-channel (T, 2) array to combine ens_mean + ens_spread —
                # not verified this session, see task-5-report.md
                past_future_covariates=ens_mean,
                return_quantiles=True,
            )
            all_quantiles.append(np.asarray(out.quantiles))

        return np.stack(all_quantiles, axis=0)
