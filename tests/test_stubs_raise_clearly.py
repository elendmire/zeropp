import pytest

from zeropp.data import download as download_stub
from zeropp.models.emos import EMOS
from zeropp.models.tsfm_timesfm import TimesFM3


def test_download_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="blocked"):
        download_stub.download_euppbench(target_dir="/tmp/whatever")


def test_emos_stub_is_a_postprocessor_but_fit_raises():
    from zeropp.models.base import Postprocessor

    assert issubclass(EMOS, Postprocessor)
    model = EMOS(quantile_levels=[0.1, 0.5, 0.9])
    with pytest.raises(NotImplementedError, match="blocked"):
        model.fit(train=None)


def test_timesfm_stub_predict_raises():
    model = TimesFM3(quantile_levels=[0.1, 0.5, 0.9])
    with pytest.raises(NotImplementedError, match="blocked"):
        model.predict_quantiles(X=None)
