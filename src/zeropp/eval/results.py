"""Persist result DataFrames with CLAUDE.md's non-negotiable provenance rule 3:
every results/ file carries git SHA, model version, and config hash. Every
later script in this project must call write_result() instead of calling
df.to_parquet() directly."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def _config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def write_result(
    df: pd.DataFrame, *, name: str, model_version: str, config: dict, out_dir: str = "results"
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path / f"{name}.parquet", index=False)

    meta = {
        "model_version": model_version,
        "config": config,
        "config_hash": _config_hash(config),
        "git_sha": _git_sha(),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_path / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
