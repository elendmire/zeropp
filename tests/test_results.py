import json
import subprocess

import pandas as pd
import pytest

from zeropp.eval.results import write_result


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


def test_write_result_creates_parquet_and_json(tmp_path, sample_df):
    write_result(sample_df, name="test_result", model_version="emos-v1", config={"k": 1}, out_dir=str(tmp_path))
    assert (tmp_path / "test_result.parquet").exists()
    assert (tmp_path / "test_result.json").exists()


def test_write_result_parquet_content_matches(tmp_path, sample_df):
    write_result(sample_df, name="test_result", model_version="emos-v1", config={"k": 1}, out_dir=str(tmp_path))
    loaded = pd.read_parquet(tmp_path / "test_result.parquet")
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_write_result_json_has_required_provenance_fields(tmp_path, sample_df):
    write_result(sample_df, name="test_result", model_version="emos-v1", config={"k": 1}, out_dir=str(tmp_path))
    meta = json.loads((tmp_path / "test_result.json").read_text())
    assert meta["model_version"] == "emos-v1"
    assert meta["config"] == {"k": 1}
    assert "config_hash" in meta
    assert "git_sha" in meta
    assert "written_at" in meta


def test_write_result_git_sha_matches_real_head(tmp_path, sample_df):
    write_result(sample_df, name="test_result", model_version="emos-v1", config={}, out_dir=str(tmp_path))
    meta = json.loads((tmp_path / "test_result.json").read_text())
    real_head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    assert meta["git_sha"] == real_head


def test_write_result_config_hash_is_deterministic(tmp_path, sample_df):
    write_result(sample_df, name="r1", model_version="v1", config={"a": 1, "b": 2}, out_dir=str(tmp_path))
    write_result(sample_df, name="r2", model_version="v1", config={"b": 2, "a": 1}, out_dir=str(tmp_path))
    meta1 = json.loads((tmp_path / "r1.json").read_text())
    meta2 = json.loads((tmp_path / "r2.json").read_text())
    assert meta1["config_hash"] == meta2["config_hash"]  # key order must not matter


def test_write_result_creates_out_dir_if_missing(tmp_path, sample_df):
    out_dir = tmp_path / "nested" / "results"
    write_result(sample_df, name="r", model_version="v1", config={}, out_dir=str(out_dir))
    assert (out_dir / "r.parquet").exists()
