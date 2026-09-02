from zeropp.models.base import Postprocessor


class TimesFM3(Postprocessor):
    """Frozen TimesFM-3 zero-shot postprocessor.

    BLOCKED: needs the SSH server's `timesfm[torch]` install (scripts/00_setup_env.sh)
    and a verified covariate API (past-future dynamic covariates) before this can be
    implemented for real. Do NOT implement covariate injection approximately —
    verify `python -c "import timesfm; help(timesfm)" | grep -i covariate` on the
    server first, per CLAUDE.md.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "TimesFM3":
        return self  # zero-shot: no-op fit, but see predict_quantiles

    def predict_quantiles(self, X):
        raise NotImplementedError(
            "blocked: needs SSH server timesfm[torch] install and verified "
            "past-future covariate API"
        )
