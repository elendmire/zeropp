"""BLOCKED: needs real model run output (predictions + timing) before this can
write actual results/*.parquet + *.json files. When implemented, every result
file MUST carry: git SHA (of the commit that produced it), model version
string, and a hash of the config used (see zeropp.config) — this is CLAUDE.md's
non-negotiable rule 3. Do not implement partially: no results file may skip
any of the three provenance fields."""

from pathlib import Path


def write_result(predictions, metadata: dict, out_dir: Path) -> None:
    raise NotImplementedError(
        "blocked: needs real model predictions and run metadata (git SHA, "
        "model version, config hash, wall-clock timing) before results can "
        "be written to results/*.parquet + *.json"
    )
