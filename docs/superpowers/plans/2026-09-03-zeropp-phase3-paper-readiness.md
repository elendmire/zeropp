# ZeroPP Phase 3 — Paper-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Phase 2's working-but-not-defensible comparison (raw CRPS numbers on 737k autocorrelated instances, one baseline, no sharpness reporting, no breakpoint curve) into paper-ready evidence: sharpness+correctly-labeled calibration, real significance testing that respects the data's block structure, a lead-time breakdown, the training-data-size breakpoint curve (the project's actual headline contribution), and a second, modern ML baseline (DRN).

**Architecture:** Every task after Task 1 reads from `results/phase2_comparison_raw.parquet` (Phase 2's already-computed, already-persisted per-instance predictions) rather than re-running the ~2.5-hour GPU job — TimesFM-3 is zero-shot and its Phase 2 output is reused verbatim throughout. Only the N-day sweep (Task 4) and DRN (Task 5) require new compute, and both are cheap (EMOS/DRN fits are CPU-seconds each). `src/zeropp/eval/results.py` becomes the one place every task writes through, so every result file satisfies CLAUDE.md's provenance rule (git SHA + model version + config hash) automatically instead of each script reinventing it.

**Tech Stack:** Same as Phase 1/2 (Python 3.11.16, server-side venv on `altay`) plus `matplotlib` (new, for Task 3/4 figures) and `torch` (already present, now used for Task 5's DRN in addition to TimesFM-3).

## Global Constraints

- **Sync-then-test workflow, every task, no exceptions** (identical to Phase 2):
  ```bash
  rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
    -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
  ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest <path> -v'
  ```
  Never run pytest locally on the Mac — no real data, no `torch`/`timesfm`/`climetlab` there.
- `results/phase2_comparison_raw.parquet` already exists on the server (2,213,427 rows — 737,809 matched instances × 3 methods: raw ensemble, EMOS, TimesFM-3). **Task 1's first step is inspecting its real schema** (exact column names/dtypes are not yet known to this plan) before any task writes code against it.
- Quantile levels stay the single global constant from `configs/experiment.yaml`. `DRN` (Task 5) reads them from the constructor argument like every other model, never hardcodes them.
- `Postprocessor.fit(train) -> self` / `predict_quantiles(X) -> ndarray` shaped `(n_samples, n_leads, n_quantiles)` is unchanged — `DRN` must satisfy it exactly.
- No hardcoded seeds in `src/`. The N-day sweep's subsampling seed and DRN's training seed both come from `configs/experiment.yaml`.
- CLAUDE.md rule 3 (every `results/` file carries git SHA, model/method version, config hash) is satisfied by routing every write through `zeropp.eval.results.write_result` (Task 1) — no task writes to `results/` any other way.
- Coverage must be labeled by its true nominal level. `empirical_coverage(lower=0.1, upper=0.9)` measures the **80%**-nominal band, never call it "coverage@90%".
- Statistical claims about method differences must account for the data's block structure (49 stations × ~730 overlapping issue times × 21 correlated lead times) — never report a p-value or CI computed as if the 737,809 instances were independent.
- Figures are produced by scripts, never notebooks (CLAUDE.md: "Notebook'lar sonuc URETMEZ").

---

## File Structure

```
src/zeropp/eval/
├── results.py            # Task 1 — real (replaces the Phase-1 stub)
└── significance.py       # Task 2 — new: block bootstrap CI, station-blocked paired test
src/zeropp/data/
└── build.py               # Task 4 modifies — adds build_train_ensemble_stats_with_ids
src/zeropp/models/
└── drn.py                 # Task 5 — real (replaces the Phase-1 stub)
scripts/
├── 05_summarize_results.py    # Task 1 — richer summary from the persisted parquet
├── 06_lead_time_breakdown.py  # Task 3
└── 07_data_size_sweep.py      # Task 4 (created), Task 5 (extended with a DRN curve)
configs/
└── experiment.yaml        # Task 4 modifies — adds data_size_sweep_seeds
docs/
└── phase2_results_schema.md  # Task 1 — records the real, inspected parquet schema
tests/
├── test_results.py         # Task 1
├── test_significance.py    # Task 2
├── test_build.py           # Task 4 modifies — new tests for the ID-preserving function
└── test_drn.py              # Task 5
```

---

## Task 1: `eval/results.py` (real provenance) + corrected summary metrics

**Files:**
- Modify: `src/zeropp/eval/results.py` (currently a stub raising `NotImplementedError`)
- Create: `scripts/05_summarize_results.py`
- Create: `docs/phase2_results_schema.md`
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: `results/phase2_comparison_raw.parquet` (real file on the server — schema confirmed by this task's Step 1, not assumed).
- Produces (used by every later task): `zeropp.eval.results.write_result(df: pd.DataFrame, *, name: str, model_version: str, config: dict, out_dir: str = "results") -> None`. Writes `{out_dir}/{name}.parquet` (the data, unmodified) and `{out_dir}/{name}.json` (`{"model_version": ..., "config": ..., "config_hash": <sha256 of the canonicalized config, first 16 hex chars>, "git_sha": <git rev-parse HEAD>, "written_at": <UTC ISO timestamp>}`). Every later task's script calls this instead of `df.to_parquet(...)` directly.

## Before You Begin

**Step 1 is a real investigation — the parquet's exact columns are not yet known to this plan.** Do not write `scripts/05_summarize_results.py` until you've inspected the real file and written `docs/phase2_results_schema.md`. If the real schema is structurally different from what's assumed below (e.g., quantile predictions stored as a single array column vs. 9 separate columns), adapt Step 4's code to match reality and note the adaptation in your report — don't force the data to match a guess.

- [ ] **Step 1: Inspect the real results file on the server**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate
python -c "
import pandas as pd
df = pd.read_parquet(\"results/phase2_comparison_raw.parquet\")
print(\"shape:\", df.shape)
print(\"columns:\", list(df.columns))
print(\"dtypes:\")
print(df.dtypes)
print(\"method value counts:\")
print(df[\"method\"].value_counts() if \"method\" in df.columns else \"no method column\")
print(df.head(3).to_string())
"
'
```

- [ ] **Step 2: Write `docs/phase2_results_schema.md`** recording the real, confirmed schema:

```markdown
# Phase 2 results/ schema — confirmed YYYY-MM-DD

`results/phase2_comparison_raw.parquet` (2,213,427 rows, produced by `scripts/04_run_tsfm.py`):

- Columns: <paste the real column list>
- Dtypes: <paste the real dtypes>
- `method` values: <paste the real distinct values, e.g. "raw_ensemble", "emos", "timesfm3">
- How quantile predictions are stored: <one sentence — e.g. "9 columns q0.1..q0.9" or "one array-valued column">

## Consequence for this plan's tasks
<one or two sentences: how Task 1/2/3 must read this file given the real structure>
```

- [ ] **Step 3: Write the failing tests for `results.py`**

```python
# tests/test_results.py
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
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_results.py -v'
```
Expected: FAIL with `NotImplementedError: blocked: ...` (the current stub).

- [ ] **Step 5: Write `src/zeropp/eval/results.py`**

```python
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


def write_result(df: pd.DataFrame, *, name: str, model_version: str, config: dict, out_dir: str = "results") -> None:
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
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_results.py -v'
```
Expected: 6 passed.

- [ ] **Step 7: Write `scripts/05_summarize_results.py`**

Adapt the exact column-reading logic to what Step 1/2 actually found. The logic below assumes per-instance rows with a `method` column and 9 quantile columns matching `configs/experiment.yaml`'s levels — **adjust to the real schema, this is a starting point, not a guarantee**:

```python
"""Recompute rich summary metrics (CRPS, MAE, twCRPS, interval width, coverage,
reliability) from the already-persisted Phase 2 per-instance results — does NOT
re-run the GPU job. See docs/phase2_results_schema.md for the real input schema."""
import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.eval.calibration import empirical_coverage, pit_values, reliability_index
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles, mae_from_quantiles, twcrps_from_quantiles

RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"


def summarize_method(df_method: pd.DataFrame, quantile_levels: list[float]) -> dict:
    quantile_cols = [f"q{q}" for q in quantile_levels]  # adjust if Step 1 found a different column naming
    quantile_preds = df_method[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    obs = df_method["t2m_obs"].to_numpy().reshape(-1, 1)

    interval_width = float(np.mean(quantile_preds[..., -1] - quantile_preds[..., 0]))
    coverage_80 = empirical_coverage(obs, quantile_preds, quantile_levels, lower=0.1, upper=0.9)
    pit = pit_values(obs, quantile_preds, quantile_levels)

    return {
        "crps": float(crps_from_quantiles(obs, quantile_preds, quantile_levels).mean()),
        "mae": float(mae_from_quantiles(obs, quantile_preds, quantile_levels).mean()),
        "twcrps": float(twcrps_from_quantiles(obs, quantile_preds, quantile_levels).mean()),
        "interval_width_p10_p90_kelvin": interval_width,
        "coverage_80pct_nominal": coverage_80,
        "reliability_index": reliability_index(pit.flatten()),
        "n_instances": len(df_method),
    }


def main() -> None:
    quantile_levels = load_experiment_config().quantile_levels
    raw = pd.read_parquet(RAW_RESULTS_PATH)

    rows = []
    for method, df_method in raw.groupby("method"):
        summary = summarize_method(df_method, quantile_levels)
        summary["method"] = method
        rows.append(summary)

    summary_df = pd.DataFrame(rows)

    print("\n===== Phase 3 corrected summary (from persisted Phase 2 results) =====")
    print("NOTE: 'coverage_80pct_nominal' measures the [q0.1, q0.9] interval, which is the")
    print("80%-nominal band, NOT 90% (empirical_coverage(lower=0.1, upper=0.9) cannot compute")
    print("CLAUDE.md's literal 'nominal 90%' ask, which would need q0.05/q0.95 — not available")
    print("on this project's fixed 9-level [0.1,...,0.9] quantile grid).")
    print("LIMITATION: TimesFM-3's quantile output spans only p10-p90 (no tail beyond) — twcrps")
    print("(tail-weighted at q0.8/q0.9) is therefore not a fully fair comparison against EMOS's")
    print("full Gaussian tail or raw ensemble's empirical 51-member tail.")
    print(summary_df.to_string(index=False))

    write_result(
        summary_df,
        name="phase3_summary_metrics",
        model_version="phase3-summary-v1",
        config={"quantile_levels": quantile_levels, "source": RAW_RESULTS_PATH},
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run it for real and record the actual output**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && python scripts/05_summarize_results.py'
```
If this crashes because the real schema differs from Step 7's assumption, fix `summarize_method`'s column access to match reality (per Step 1's findings), then re-run. Record the real printed table in your report — this is the corrected, sharpness-aware, correctly-labeled Phase 2 headline result.

- [ ] **Step 9: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/eval/results.py tests/test_results.py scripts/05_summarize_results.py docs/phase2_results_schema.md
git commit -m "feat: real results.py (CLAUDE.md provenance rule) + corrected sharpness/coverage-labeled summary from persisted Phase 2 results"
```

---

## Task 2: `eval/significance.py` — block bootstrap CI + station-blocked paired test

**Files:**
- Create: `src/zeropp/eval/significance.py`
- Test: `tests/test_significance.py`
- Modify: `scripts/05_summarize_results.py` (add the significance comparison for TimesFM-3 vs raw ensemble and TimesFM-3 vs EMOS)

**Interfaces:**
- Consumes: per-instance loss values (e.g. CRPS per row) and a block-id column (station_id) from `results/phase2_comparison_raw.parquet` via Task 1's schema.
- Produces: `block_bootstrap_skill_score_ci(loss_method: np.ndarray, loss_reference: np.ndarray, block_ids: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]` returning `(point_estimate, ci_low, ci_high)` for the skill score `1 - mean(loss_method)/mean(loss_reference)`. `station_blocked_paired_test(loss_a: np.ndarray, loss_b: np.ndarray, block_ids: np.ndarray) -> dict` returning `{"n_blocks": int, "block_mean_diff": float, "t_statistic": float, "t_pvalue": float, "wilcoxon_statistic": float, "wilcoxon_pvalue": float}`.

## Before You Begin

This is a **station-blocked** test, not a textbook single-series Diebold-Mariano test — the 737,809 instances form a station × time × lead-time panel, not one autocorrelated series, so classical single-series DM/HAC-Newey-West assumptions don't cleanly apply. Aggregating to one mean loss-differential per station (49 independent-ish blocks) and testing THAT is the correct, defensible approach for this data structure. Name things accordingly in code and docs — do not claim this is literally "the Diebold-Mariano test."

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_significance.py
import numpy as np
import pytest

from zeropp.eval.significance import block_bootstrap_skill_score_ci, station_blocked_paired_test


@pytest.fixture
def synthetic_panel():
    # 10 stations, 200 instances each. Method B's loss is systematically 0.5 lower
    # than method A's, with station-correlated noise (each station has its own offset).
    rng = np.random.default_rng(0)
    n_stations = 10
    n_per_station = 200
    block_ids = np.repeat(np.arange(n_stations), n_per_station)
    station_offset = np.repeat(rng.normal(0, 0.3, n_stations), n_per_station)
    loss_a = 2.0 + station_offset + rng.normal(0, 0.2, n_stations * n_per_station)
    loss_b = loss_a - 0.5 + rng.normal(0, 0.05, n_stations * n_per_station)  # B is genuinely better
    return loss_a, loss_b, block_ids


def test_block_bootstrap_skill_score_ci_point_estimate_matches_true_effect(synthetic_panel):
    loss_a, loss_b, block_ids = synthetic_panel
    point, lo, hi = block_bootstrap_skill_score_ci(loss_b, loss_a, block_ids, n_boot=500, seed=1)
    true_skill = 1 - loss_b.mean() / loss_a.mean()
    assert point == pytest.approx(true_skill, abs=1e-9)  # point estimate is exact, not resampled
    assert lo < point < hi


def test_block_bootstrap_skill_score_ci_bounds_are_ordered(synthetic_panel):
    loss_a, loss_b, block_ids = synthetic_panel
    point, lo, hi = block_bootstrap_skill_score_ci(loss_b, loss_a, block_ids, n_boot=500, seed=1)
    assert lo <= point <= hi


def test_block_bootstrap_skill_score_ci_is_reproducible_with_same_seed(synthetic_panel):
    loss_a, loss_b, block_ids = synthetic_panel
    result1 = block_bootstrap_skill_score_ci(loss_b, loss_a, block_ids, n_boot=200, seed=42)
    result2 = block_bootstrap_skill_score_ci(loss_b, loss_a, block_ids, n_boot=200, seed=42)
    assert result1 == result2


def test_station_blocked_paired_test_detects_real_effect(synthetic_panel):
    loss_a, loss_b, block_ids = synthetic_panel
    result = station_blocked_paired_test(loss_a, loss_b, block_ids)
    assert result["n_blocks"] == 10
    assert result["block_mean_diff"] == pytest.approx(0.5, abs=0.1)  # A - B ~= 0.5
    assert result["t_pvalue"] < 0.01  # a real, station-consistent 0.5 effect should be significant
    assert result["wilcoxon_pvalue"] < 0.05


def test_station_blocked_paired_test_null_case_not_significant():
    rng = np.random.default_rng(0)  # verified: this seed gives t_pvalue≈0.375, comfortably non-significant
    n_stations = 20
    n_per_station = 100
    block_ids = np.repeat(np.arange(n_stations), n_per_station)
    loss_a = rng.normal(2.0, 0.3, n_stations * n_per_station)
    loss_b = rng.normal(2.0, 0.3, n_stations * n_per_station)  # no real difference
    result = station_blocked_paired_test(loss_a, loss_b, block_ids)
    assert result["t_pvalue"] > 0.05


def test_station_blocked_paired_test_aggregates_to_one_row_per_block(synthetic_panel):
    loss_a, loss_b, block_ids = synthetic_panel
    result = station_blocked_paired_test(loss_a, loss_b, block_ids)
    # the diff should be far more precisely estimated than raw per-instance noise would suggest,
    # because it's a mean of 10 already-averaged block means, not raw std/sqrt(2000)
    assert result["n_blocks"] == len(np.unique(block_ids))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_significance.py -v'
```
Expected: FAIL with `ModuleNotFoundError: No module named 'zeropp.eval.significance'`.

- [ ] **Step 3: Write `src/zeropp/eval/significance.py`**

```python
import numpy as np
from scipy import stats


def block_bootstrap_skill_score_ci(
    loss_method: np.ndarray,
    loss_reference: np.ndarray,
    block_ids: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    loss_method = np.asarray(loss_method)
    loss_reference = np.asarray(loss_reference)
    block_ids = np.asarray(block_ids)
    unique_blocks = np.unique(block_ids)

    point = 1 - loss_method.mean() / loss_reference.mean()

    rng = np.random.default_rng(seed)
    boot_skills = np.empty(n_boot)
    block_index = {b: np.where(block_ids == b)[0] for b in unique_blocks}

    for i in range(n_boot):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_index[b] for b in sampled_blocks])
        boot_skills[i] = 1 - loss_method[idx].mean() / loss_reference[idx].mean()

    alpha = 1 - ci
    lo, hi = np.quantile(boot_skills, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def station_blocked_paired_test(loss_a: np.ndarray, loss_b: np.ndarray, block_ids: np.ndarray) -> dict:
    loss_a = np.asarray(loss_a)
    loss_b = np.asarray(loss_b)
    block_ids = np.asarray(block_ids)
    unique_blocks = np.unique(block_ids)

    diff = loss_a - loss_b
    block_means = np.array([diff[block_ids == b].mean() for b in unique_blocks])

    t_stat, t_pvalue = stats.ttest_1samp(block_means, popmean=0.0)
    w_stat, w_pvalue = stats.wilcoxon(block_means)

    return {
        "n_blocks": len(unique_blocks),
        "block_mean_diff": float(block_means.mean()),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pvalue),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_significance.py -v'
```
Expected: 6 passed.

- [ ] **Step 5: Add the real significance comparison to `scripts/05_summarize_results.py`**

Append to `main()`, after the summary table is printed (adapt `df_method["crps_per_instance"]` / `df_method["station_id"]` column names to whatever Step 1 of Task 1 actually found — this assumes per-instance CRPS can be computed per row and a `station_id` column exists, both should be true given the schema description, but verify):

```python
from zeropp.eval.significance import block_bootstrap_skill_score_ci, station_blocked_paired_test


def per_instance_crps(df_method: pd.DataFrame, quantile_levels: list[float]) -> np.ndarray:
    quantile_cols = [f"q{q}" for q in quantile_levels]
    quantile_preds = df_method[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
    obs = df_method["t2m_obs"].to_numpy().reshape(-1, 1)
    return crps_from_quantiles(obs, quantile_preds, quantile_levels).flatten()
```

And in `main()`:

```python
    raw_ens = raw[raw["method"] == "raw_ensemble"].reset_index(drop=True)
    emos = raw[raw["method"] == "emos"].reset_index(drop=True)
    tsfm = raw[raw["method"] == "timesfm3"].reset_index(drop=True)
    # adjust the method-name strings above to match Step 1's real confirmed values

    crps_raw = per_instance_crps(raw_ens, quantile_levels)
    crps_emos = per_instance_crps(emos, quantile_levels)
    crps_tsfm = per_instance_crps(tsfm, quantile_levels)
    stations = tsfm["station_id"].to_numpy()

    print("\n===== Significance: TimesFM-3 vs raw ensemble =====")
    point, lo, hi = block_bootstrap_skill_score_ci(crps_tsfm, crps_raw, stations, seed=0)
    print(f"CRPS skill score (block bootstrap, 95% CI): {point:.4f} [{lo:.4f}, {hi:.4f}]")
    test_result = station_blocked_paired_test(crps_raw, crps_tsfm, stations)
    print(f"Station-blocked paired test (n={test_result['n_blocks']} stations): "
          f"mean diff={test_result['block_mean_diff']:.4f}, "
          f"t p-value={test_result['t_pvalue']:.4f}, wilcoxon p-value={test_result['wilcoxon_pvalue']:.4f}")

    print("\n===== Significance: TimesFM-3 vs EMOS =====")
    point, lo, hi = block_bootstrap_skill_score_ci(crps_tsfm, crps_emos, stations, seed=0)
    print(f"CRPS skill score (block bootstrap, 95% CI): {point:.4f} [{lo:.4f}, {hi:.4f}]")
    test_result = station_blocked_paired_test(crps_emos, crps_tsfm, stations)
    print(f"Station-blocked paired test (n={test_result['n_blocks']} stations): "
          f"mean diff={test_result['block_mean_diff']:.4f}, "
          f"t p-value={test_result['t_pvalue']:.4f}, wilcoxon p-value={test_result['wilcoxon_pvalue']:.4f}")
```

- [ ] **Step 6: Run it for real, record the actual significance numbers**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && python scripts/05_summarize_results.py'
```

- [ ] **Step 7: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/eval/significance.py tests/test_significance.py scripts/05_summarize_results.py
git commit -m "feat: station-blocked significance testing (block bootstrap CI + paired test) for the Phase 2 comparison"
```

---

## Task 3: Lead-time breakdown

**Files:**
- Create: `scripts/06_lead_time_breakdown.py`

**Interfaces:**
- Consumes: `results/phase2_comparison_raw.parquet` (same schema as Task 1), `zeropp.eval.results.write_result` (Task 1).
- Produces: `results/phase3_lead_time_breakdown.parquet`+`.json` and `figures/lead_time_breakdown.png` — no new importable functions needed by later tasks.

- [ ] **Step 1: Write `scripts/06_lead_time_breakdown.py`**

Adapt column names per `docs/phase2_results_schema.md` (from Task 1) if they differ from this starting assumption:

```python
"""Per-lead-time CRPS and coverage breakdown, from the already-persisted Phase 2
results — no GPU re-run. Produces a real figure (not a notebook)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.eval.calibration import empirical_coverage
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles

RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"
FIGURE_PATH = "figures/lead_time_breakdown.png"


def main() -> None:
    quantile_levels = load_experiment_config().quantile_levels
    quantile_cols = [f"q{q}" for q in quantile_levels]
    raw = pd.read_parquet(RAW_RESULTS_PATH)

    rows = []
    for (method, step_hours), group in raw.groupby(["method", "step_hours"]):
        quantile_preds = group[quantile_cols].to_numpy().reshape(-1, 1, len(quantile_levels))
        obs = group["t2m_obs"].to_numpy().reshape(-1, 1)
        rows.append({
            "method": method,
            "step_hours": step_hours,
            "crps": float(crps_from_quantiles(obs, quantile_preds, quantile_levels).mean()),
            "coverage_80pct": empirical_coverage(obs, quantile_preds, quantile_levels, lower=0.1, upper=0.9),
            "n_instances": len(group),
        })

    breakdown_df = pd.DataFrame(rows).sort_values(["method", "step_hours"])
    print(breakdown_df.to_string(index=False))

    write_result(
        breakdown_df,
        name="phase3_lead_time_breakdown",
        model_version="phase3-lead-breakdown-v1",
        config={"quantile_levels": quantile_levels, "source": RAW_RESULTS_PATH},
    )

    fig, (ax_crps, ax_cov) = plt.subplots(1, 2, figsize=(12, 5))
    for method, group in breakdown_df.groupby("method"):
        ax_crps.plot(group["step_hours"], group["crps"], marker="o", label=method)
        ax_cov.plot(group["step_hours"], group["coverage_80pct"], marker="o", label=method)

    ax_crps.set_xlabel("Lead time (hours)")
    ax_crps.set_ylabel("CRPS")
    ax_crps.set_title("CRPS vs. lead time")
    ax_crps.legend()

    ax_cov.axhline(0.80, color="gray", linestyle="--", label="nominal 80%")
    ax_cov.set_xlabel("Lead time (hours)")
    ax_cov.set_ylabel("Coverage (p10-p90 band)")
    ax_cov.set_title("Coverage vs. lead time")
    ax_cov.legend()

    fig.tight_layout()
    import os
    os.makedirs("figures", exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `matplotlib` to `pyproject.toml` dependencies and install on the server**

```toml
dependencies = [
    "numpy>=1.26.4",
    "pandas>=2.2",
    "pyyaml>=6.0",
    "scipy>=1.11",
    "pyarrow>=14",
    "matplotlib>=3.8",
]
```

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && uv pip install -e ".[dev]"'
```

- [ ] **Step 3: Run it for real**

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && python scripts/06_lead_time_breakdown.py'
```
Expected: a 21-row-per-method table printed, `results/phase3_lead_time_breakdown.parquet`+`.json` and `figures/lead_time_breakdown.png` created on the server. Copy the figure back to inspect it:
```bash
scp altay:~/zeropp/figures/lead_time_breakdown.png /tmp/lead_time_breakdown.png
```
Look at it — confirm it's a real, sensible plot (not empty, not garbled) before reporting done.

- [ ] **Step 4: Commit**

```bash
cd /Users/farukavci/zeropp
git add scripts/06_lead_time_breakdown.py pyproject.toml
git commit -m "feat: per-lead-time CRPS/coverage breakdown and figure"
```

---

## Task 4: Training-data-size breakpoint curve (N-day sweep)

**Files:**
- Modify: `src/zeropp/data/build.py` (add a new function, do not change the existing `build_train_ensemble_stats`'s signature or output columns)
- Modify: `configs/experiment.yaml` (add seeds for the sweep's random-arm subsampling)
- Modify: `src/zeropp/config.py` (`ExperimentConfig` is a frozen dataclass with an explicit field list, and `load_experiment_config` builds it field-by-field from `raw[...]` — it does NOT pass through arbitrary extra YAML keys. Add a `data_size_sweep_seeds: list[int]` field to `ExperimentConfig` and read `raw["data_size_sweep_seeds"]` into it in `load_experiment_config`, otherwise `config.data_size_sweep_seeds` in the new script raises `AttributeError` even after the YAML key exists.)
- Create: `scripts/07_data_size_sweep.py`
- Test: `tests/test_build.py` (add tests for the new function), `tests/test_config.py` (add a test for the new field)

**Interfaces:**
- Consumes: `zeropp.models.emos.EMOS`, `zeropp.eval.scores.crps_from_quantiles`, `zeropp.eval.calibration.empirical_coverage`, `zeropp.eval.results.write_result` (Task 1), `results/phase2_comparison_raw.parquet` (for TimesFM-3's and raw ensemble's N-independent reference values).
- Produces: `zeropp.data.build.build_train_ensemble_stats_with_ids(reforecast_path: str, obs_path: str) -> pd.DataFrame` — same as `build_train_ensemble_stats` but ALSO includes `time_idx` and `year_idx` columns (needed for N-day subsampling), used by Task 5 too. Also produces, in `scripts/07_data_size_sweep.py` itself (Task 5 imports and reuses these): `sample_contiguous(full_train, n_days) -> pd.DataFrame`, `sample_random(full_train, n_days, seed) -> pd.DataFrame`, `compute_metrics(y, preds, quantile_levels) -> dict`, `fit_predict_local_emos(train_subset, test_station_ids, test_X, quantile_levels, min_rows=5) -> tuple[np.ndarray, np.ndarray, float]` (predictions, covered-row boolean mask, station coverage fraction), `find_breakpoint(n_cases_axis, method_values, reference_value, better="lower") -> float | None`.

## Before You Begin — this task's design changed after a methodological review; read this in full before writing any code

A prior attempt at this task used **pure random subsampling** across the full 20-year archive for every N. That is now known to be the wrong primary design: randomly drawing N cases from 20 years of history gives EMOS full seasonal coverage (a taste of every season) even at very small N, which no real newly-deployed station would actually have (a station with 90 real days of history has seen ONE season, not four). This artificially strengthens EMOS at small N and shifts the CRPS breakpoint in TimesFM-3's favor — biased toward the story we'd like to tell, not the true operational one. Fixed below. If you are resuming uncommitted work from a halted prior attempt (check `git status` and `git diff` before starting — `src/zeropp/data/build.py`, `src/zeropp/config.py`, `configs/experiment.yaml`, `tests/test_build.py`, `tests/test_config.py` may already carry Steps 1-5's changes from that attempt), the build.py refactor and config wiring (Steps 1-5) are STILL CORRECT and reusable as-is — only the single `data_size_sweep_seed` needs upgrading to the plural `data_size_sweep_seeds` list described in Step 5 below, and `scripts/07_data_size_sweep.py` (Step 6 onward) needs to be written fresh per this revised design, not the old single-arm version.

**Operational definition of "N days of training data,"** unchanged from before: the reforecast archive is 20 analog years × ~209 representative issue-dates per year (confirmed structure from Phase 2's `germany_ensemble_reforecasts_t2m.nc`: raw xarray dims `time=209, year=20`, renamed to `time_idx`/`year_idx` columns by `build_train_ensemble_stats_with_ids`). `N` days of training data maps to `k = round(N * 209 / 365)` `(year_idx, time_idx)` pairs — this conversion is unchanged. **What's new is that there are now two distinct ways to pick WHICH `k` pairs, answering two different questions, and both must be run:**

| Arm | How | Answers |
|---|---|---|
| **`sample_contiguous`** (primary) | The `k` pairs closest in time to the test period's start, working backward — i.e. a real cold-start: a station with only `N` days of real history would see only the most recent portion of the archive, likely a single season for small `N`. Deterministic, no seed. | "What happens with a real newly-deployed station's limited, chronologically-contiguous history?" — this is the operationally realistic scenario and the primary story. |
| **`sample_random`** (secondary, kept from before) | `k` pairs drawn uniformly at random (fixed seed) from all pairs in the archive, giving artificial full-season coverage even at small `k`. | "How much does EMOS benefit from artificial seasonal diversity it wouldn't operationally have?" — the *difference* between the two arms' curves is itself a real, reportable finding about the value of seasonal coverage, not something to discard. |

**Step 0 (do this first, before writing any sampling code): verify the chronological polarity of `year_idx`/`time_idx` on the real archive** — do not assume ascending index means more recent. SSH to `altay` and run:

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && python3 -c "
import xarray as xr
fcs = xr.open_dataset(\"data/raw/germany_ensemble_reforecasts_t2m.nc\")
print(\"year coord dtype/values:\", fcs.coords[\"year\"].dtype, fcs.coords[\"year\"].values[:5], \"...\", fcs.coords[\"year\"].values[-5:])
print(\"time coord dtype/values:\", fcs.coords[\"time\"].dtype, fcs.coords[\"time\"].values[:5], \"...\", fcs.coords[\"time\"].values[-5:])
"'
```

Record the printed values in your report. Two possible outcomes:
- **If `year` values are real calendar years** (e.g. `1997...2016`) and/or `time` values are real day-like/date-like values ascending within a year: sort descending by the real value (larger = more recent / later in the year) to build the "closest to test start, working backward" order. This gives sub-year (`k < 209`) resolution — small `N` pulls from a single recent season, exactly the realistic cold-start.
- **If either is a bare positional index with no confirmed chronological meaning**: fall back to treating `year_idx` ascending = chronologically ascending (the standard convention for how such archives are constructed — each successive index is one more reforecast year added) as your best-available ordering, but **state this as an assumption, not a verified fact, in your report** — this is a real, disclosed limitation (contiguity resolution degrades to whole-year granularity rather than single-season granularity for sub-year `N`), not something to silently paper over.

- [ ] **Step 1: Write the failing test for the new build.py function**

```python
# tests/test_build.py — ADD this test to the existing file (do not remove existing tests)
def test_build_train_ensemble_stats_with_ids_includes_time_and_year(synthetic_reforecast_files):
    from zeropp.data.build import build_train_ensemble_stats_with_ids
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats_with_ids(fcs_path, obs_path)
    assert set(["station_id", "time_idx", "year_idx", "ens_mean", "ens_var", "t2m_obs"]) <= set(df.columns)
    assert len(df) == 7  # same row count/NaN-dropping as build_train_ensemble_stats


def test_build_train_ensemble_stats_still_returns_exactly_four_columns(synthetic_reforecast_files):
    # Regression guard: the new with_ids function must not have changed the
    # existing public function's contract.
    from zeropp.data.build import build_train_ensemble_stats
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats(fcs_path, obs_path)
    assert list(df.columns) == ["station_id", "ens_mean", "ens_var", "t2m_obs"]
```

- [ ] **Step 2: Run tests to verify the new one fails**

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_build.py -v'
```
Expected: the new `test_build_train_ensemble_stats_with_ids_includes_time_and_year` FAILS with `ImportError`; the regression-guard test and all pre-existing tests PASS (they test existing, unmodified code).

- [ ] **Step 3: Modify `src/zeropp/data/build.py`** — refactor `build_train_ensemble_stats` to share logic with a new `build_train_ensemble_stats_with_ids`, WITHOUT changing the public function's output columns:

```python
def build_train_ensemble_stats_with_ids(reforecast_path: str, obs_path: str) -> pd.DataFrame:
    fcs = xr.open_dataset(reforecast_path)
    obs = xr.open_dataset(obs_path)

    ens_mean = fcs["t2m"].mean(dim="number")
    ens_var = fcs["t2m"].var(dim="number", ddof=1)

    merged = xr.Dataset({"ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": obs["t2m"]})
    df = merged.to_dataframe().reset_index()
    df = df.rename(columns={"time": "time_idx", "year": "year_idx"})
    df = df[["station_id", "time_idx", "year_idx", "ens_mean", "ens_var", "t2m_obs"]]
    df = df.dropna(subset=["ens_mean", "ens_var", "t2m_obs"])
    return df.reset_index(drop=True)


def build_train_ensemble_stats(reforecast_path: str, obs_path: str) -> pd.DataFrame:
    full = build_train_ensemble_stats_with_ids(reforecast_path, obs_path)
    return full[["station_id", "ens_mean", "ens_var", "t2m_obs"]]
```

(This replaces the existing standalone `build_train_ensemble_stats` function body — the rest of `build.py`, including `build_test_long_table` and its `valid_time` handling, is untouched.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_build.py -v'
```
Expected: all `test_build.py` tests pass (both new ones and every pre-existing one, unmodified — this proves the refactor is behavior-preserving for the public function).

- [ ] **Step 5: Add sweep seeds to `configs/experiment.yaml`, and wire them into `ExperimentConfig`**

```yaml
quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
data_size_days: [0, 30, 90, 365, 1095, "full"]
seeds: [0, 1, 2]
data_size_sweep_seeds: [0, 1, 2, 3, 4]
```

Five seeds, not one — a breakpoint found from a single random-arm seed is noise a reviewer will immediately challenge; the random arm must report mean ± std across these 5. The contiguous arm is deterministic and uses none of these seeds.

`ExperimentConfig` in `src/zeropp/config.py` is a frozen dataclass built field-by-field — adding a YAML key alone does nothing until the dataclass and loader both know about it:

```python
@dataclass(frozen=True)
class ExperimentConfig:
    quantile_levels: list[float]
    data_size_days: list[int | str]
    seeds: list[int]
    data_size_sweep_seeds: list[int]
    source_path: Path


def load_experiment_config(path: Path | None = None) -> ExperimentConfig:
    resolved_path = path or DEFAULT_EXPERIMENT_CONFIG
    raw = yaml.safe_load(resolved_path.read_text())
    return ExperimentConfig(
        quantile_levels=raw["quantile_levels"],
        data_size_days=raw["data_size_days"],
        seeds=raw["seeds"],
        data_size_sweep_seeds=raw["data_size_sweep_seeds"],
        source_path=Path(resolved_path),
    )
```

Add a test to `tests/test_config.py` (create it if it doesn't exist yet — check first; if resuming a halted prior attempt, it may already exist with a `data_size_sweep_seed` singular-field test that needs updating to the plural list) asserting `load_experiment_config().data_size_sweep_seeds == [0, 1, 2, 3, 4]` against the real `configs/experiment.yaml`, run it over SSH, confirm it passes, then commit this step's changes together with Step 8 below.

- [ ] **Step 6: Write `scripts/07_data_size_sweep.py`**

Build it from these pieces. This is longer than earlier scripts in this plan because it now answers five distinct questions (two sampling arms × two EMOS pooling variants × three metrics, plus per-metric breakpoints) instead of one — each piece below is a fully specified function; assemble `main()` to call them in the order shown and you have the complete script.

```python
"""Training-data-size breakpoint curve (RQ2): refit EMOS at increasing amounts of
reforecast training data, under two sampling arms and two pooling variants; TimesFM-3
and raw ensemble are zero-shot / N-independent and reused as flat reference lines from
the already-persisted Phase 2 results.

"N days" is operationalized as k = round(N * 209 / 365) (year_idx, time_idx) issue-date
pairs out of the full 20-year x 209-date reforecast archive. Two arms pick WHICH k pairs:
  - contiguous (primary): the k pairs closest in time to the test period, working
    backward — simulates a real newly-deployed station's limited, chronologically
    contiguous history (small N sees roughly one season, not all four).
  - random (secondary, 5 seeds, mean +/- std reported): k pairs drawn uniformly at
    random from the full archive — gives artificial full-season coverage even at small
    N. The gap between the two arms' curves is itself a reported finding about the
    value of seasonal coverage, not a discrepancy to explain away.
See this plan's Task 4 "Before You Begin" for the full rationale and the real, verified
year_idx/time_idx chronological-polarity finding this script's ordering relies on.

EMOS is fit two ways at each N: "pooled" (one global model across all 49 stations, as
in Phase 2) and "local" (one model per station, fit only on that station's rows in the
subsample). Local EMOS needs >= LOCAL_EMOS_MIN_ROWS rows per station to fit reliably
(4 free parameters); at small N most stations won't have enough contiguous rows, so the
local curve is reported only over the stations it could actually cover, with that
coverage fraction reported alongside it — never silently filled in or extrapolated.

Reforecasts have 11 ensemble members (germany_ensemble_reforecasts_t2m.nc); the test-
period forecasts EMOS is evaluated against have 51. ens_var computed from 11 members is
a noisier (higher-variance) estimator of the true ensemble spread than one from 51 would
be, at every N equally — this is a fixed property of the training archive, not something
that changes with N, and is not corrected here. Documented as a limitation in the task
report, not fixed in code (no clean fix exists without re-simulating the ensemble).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zeropp.config import load_experiment_config
from zeropp.data.build import build_test_long_table, build_train_ensemble_stats_with_ids
from zeropp.eval.calibration import empirical_coverage
from zeropp.eval.results import write_result
from zeropp.eval.scores import crps_from_quantiles
from zeropp.models.emos import EMOS

REFORECAST_PATH = "data/raw/germany_ensemble_reforecasts_t2m.nc"
REFORECAST_OBS_PATH = "data/raw/germany_reforecasts_observations_t2m.nc"
RAW_RESULTS_PATH = "results/phase2_comparison_raw.parquet"

ISSUE_DATES_PER_YEAR = 209
DAYS_PER_YEAR = 365
LOCAL_EMOS_MIN_ROWS = 5

# Set from Step 0's verified finding. If year/time carry confirmed real chronological
# values, sort descending by that value (True). If falling back to the ascending-index
# assumption (see Step 0), sort descending by the raw index (also True in this codebase's
# convention where larger year_idx/time_idx already means later — confirm against your
# Step 0 output and flip to False here if your finding says otherwise).
CHRONOLOGICAL_DESCENDING = True


def n_days_to_k(n_days) -> int | str:
    """Convert an 'N days' label to a case count k, or pass through 'full'."""
    if n_days == "full":
        return "full"
    return round(n_days * ISSUE_DATES_PER_YEAR / DAYS_PER_YEAR)


def k_to_calendar_days(k: int) -> int:
    """Inverse of n_days_to_k's ratio, for reporting 'N cases (M calendar days)'."""
    return round(k * DAYS_PER_YEAR / ISSUE_DATES_PER_YEAR)


def sample_contiguous(full_train: pd.DataFrame, n_days) -> pd.DataFrame:
    """The k pairs closest in time to the test period, working backward (deterministic,
    no seed) — see module docstring and this plan's Task 4 Step 0 for the verified
    chronological-ordering rationale behind CHRONOLOGICAL_DESCENDING."""
    if n_days == "full":
        return full_train
    unique_pairs = full_train[["year_idx", "time_idx"]].drop_duplicates()
    k = min(n_days_to_k(n_days), len(unique_pairs))
    ordered = unique_pairs.sort_values(
        ["year_idx", "time_idx"], ascending=not CHRONOLOGICAL_DESCENDING
    )
    sampled_pairs = ordered.iloc[:k]
    return full_train.merge(sampled_pairs, on=["year_idx", "time_idx"], how="inner")


def sample_random(full_train: pd.DataFrame, n_days, seed: int) -> pd.DataFrame:
    """k pairs drawn uniformly at random from the full archive (fixed seed) — the
    prior single-arm design, kept as the secondary arm."""
    if n_days == "full":
        return full_train
    unique_pairs = full_train[["year_idx", "time_idx"]].drop_duplicates()
    k = min(n_days_to_k(n_days), len(unique_pairs))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(unique_pairs), size=k, replace=False)
    sampled_pairs = unique_pairs.iloc[sampled_idx]
    return full_train.merge(sampled_pairs, on=["year_idx", "time_idx"], how="inner")


def compute_metrics(y: np.ndarray, preds: np.ndarray, quantile_levels: list[float]) -> dict:
    """y: (n,1), preds: (n,1,n_quantiles). Returns crps, coverage_80pct (nominal 80%
    band from the q0.1/q0.9 pair, per this plan's Task 1 coverage-label fix), and
    interval_width_k (mean p10-p90 width in Kelvin, i.e. sharpness)."""
    crps = float(crps_from_quantiles(y, preds, quantile_levels).mean())
    coverage = float(empirical_coverage(y, preds, quantile_levels, lower=0.1, upper=0.9))
    lo_idx, hi_idx = quantile_levels.index(0.1), quantile_levels.index(0.9)
    width = float(np.mean(preds[:, 0, hi_idx] - preds[:, 0, lo_idx]))
    return {"crps": crps, "coverage_80pct": coverage, "interval_width_k": width}


def fit_predict_pooled_emos(train_subset: pd.DataFrame, quantile_levels: list[float], test_X: dict) -> np.ndarray:
    train_for_emos = train_subset[["station_id", "ens_mean", "ens_var", "t2m_obs"]]
    model = EMOS(quantile_levels=quantile_levels).fit(train_for_emos)
    return model.predict_quantiles({"ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"]})


def fit_predict_local_emos(
    train_subset: pd.DataFrame,
    test_station_ids: np.ndarray,
    test_X: dict,
    quantile_levels: list[float],
    min_rows: int = LOCAL_EMOS_MIN_ROWS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One EMOS per station, fit only on that station's rows in train_subset. A test
    row whose station has < min_rows training rows (or none at all) is EXCLUDED, not
    filled in with a pooled fallback — the covered_mask and coverage_fraction make that
    exclusion explicit and reportable rather than silent."""
    n = len(test_station_ids)
    preds = np.full((n, 1, len(quantile_levels)), np.nan)
    covered_mask = np.zeros(n, dtype=bool)
    counts = train_subset["station_id"].value_counts()
    fittable_stations = counts[counts >= min_rows].index
    for sid in fittable_stations:
        station_train = train_subset[train_subset["station_id"] == sid][
            ["station_id", "ens_mean", "ens_var", "t2m_obs"]
        ]
        model = EMOS(quantile_levels=quantile_levels).fit(station_train)
        rows_mask = test_station_ids == sid
        if not rows_mask.any():
            continue
        station_preds = model.predict_quantiles(
            {"ens_mean": test_X["ens_mean"][rows_mask], "ens_var": test_X["ens_var"][rows_mask]}
        )
        preds[rows_mask] = station_preds
        covered_mask[rows_mask] = True
    coverage_fraction = float(len(fittable_stations)) / float(train_subset["station_id"].nunique() or 1)
    return preds, covered_mask, coverage_fraction


def find_breakpoint(n_cases_axis: list[int], method_values: list[float], reference_value: float, better: str = "lower") -> float | None:
    """Linear-interpolated N (in cases) where method_values crosses reference_value,
    walking n_cases_axis in ascending order. better='lower': method starts worse
    (higher) than reference and crosses to better (lower) — e.g. CRPS. better='closer_to_ref':
    for coverage, 'better' means the gap |value - reference| shrinks past the reference's
    own gap to nominal (not used here since TimesFM-3 IS the reference; instead this mode
    finds where method's coverage crosses the reference's coverage value directly, same as
    'lower'/'higher' but the caller passes the coverage direction it wants). Returns None if
    no crossing occurs within the tested range."""
    pairs = sorted(zip(n_cases_axis, method_values), key=lambda p: p[0])
    for (n0, v0), (n1, v1) in zip(pairs, pairs[1:]):
        if v0 is None or v1 is None or (isinstance(v0, float) and np.isnan(v0)) or (isinstance(v1, float) and np.isnan(v1)):
            continue
        starts_worse = (v0 > reference_value) if better == "lower" else (v0 < reference_value)
        ends_better = (v1 <= reference_value) if better == "lower" else (v1 >= reference_value)
        if starts_worse and ends_better and v1 != v0:
            frac = (reference_value - v0) / (v1 - v0)
            return n0 + frac * (n1 - n0)
    return None
```

Now the `main()` that assembles these pieces:

1. Load config, build `full_train = build_train_ensemble_stats_with_ids(...)` and `test_df = build_test_long_table(...)` exactly as the prior single-arm version did (group `test_df` by `(station_id, valid_time, step_hours)`, build `ens_means`/`ens_vars`/`obs_values` arrays shaped `(n,1)`, and `test_station_ids = np.array([g[0] for g in grouped.groups.keys()])` for the local-EMOS station lookup — reuse this exact grouping logic, it is unchanged from before).
2. For each `n_days` in `config.data_size_days`:
   - `train_contig = sample_contiguous(full_train, n_days)`. If empty (N=0), record `crps/coverage_80pct/interval_width_k = nan` for both `emos_pooled` and `emos_local`, `sampling_arm="contiguous"`, `seed=None`, `n_stations_covered=None`; print the same "undefined, no data to fit" note the prior version had. Otherwise: run `fit_predict_pooled_emos` + `compute_metrics` → row with `method="emos_pooled"`; run `fit_predict_local_emos` + (on `covered_mask`, apply `compute_metrics` to only the covered rows of `test_X`/`obs_values`) → row with `method="emos_local"` and `n_stations_covered=coverage_fraction * n_unique_stations_in_train_contig` (round to int). Every row for this arm gets `sampling_arm="contiguous"`, `seed=None`, `n_cases=k` (from `n_days_to_k`), `n_calendar_days_equiv=k_to_calendar_days(k)` (or, for `n_days="full"`, `n_cases=len(unique_pairs)` and skip the calendar-day conversion — just print/store the same string `"full"`).
   - For `seed in config.data_size_sweep_seeds`: `train_rand = sample_random(full_train, n_days, seed)`, pooled EMOS only (no local-EMOS random arm — out of scope for this task, note this scope choice in your report), `compute_metrics` → row with `method="emos_pooled"`, `sampling_arm="random"`, `seed=seed`. After all 5 seeds for this `n_days`, also append one aggregated row per metric: `sampling_arm="random_mean"` with `crps`/`coverage_80pct`/`interval_width_k` = the mean across the 5 seeds' rows, and matching `*_std` columns (add `crps_std`, `coverage_80pct_std`, `interval_width_k_std`, `null` for every other row type) so the figure can plot a mean ± std band.
   - Print one line per `(n_days, arm, method)` combination with its metrics, same style as the prior version's print statements.
3. Reference lines (unchanged logic from before, but now via `compute_metrics` for all three metrics instead of just CRPS): `raw = pd.read_parquet(RAW_RESULTS_PATH)`; for `method in ["raw_ensemble", "tsfm3"]` (real confirmed values, not `"timesfm3"` — see `docs/phase2_results_schema.md`), build `y`/`preds` from `raw[raw["method"]==method]`'s `q0.1..q0.9` columns and `obs` column (real confirmed name, not `t2m_obs`), call `compute_metrics`, and add one row per `n_days` with `sampling_arm="n_independent"`, `seed=None`, `n_stations_covered=None`.
4. `write_result(sweep_df, name="phase3_data_size_sweep", model_version="phase3-sweep-v2", config={"data_size_days": ..., "data_size_sweep_seeds": ..., "quantile_levels": ..., "local_emos_min_rows": LOCAL_EMOS_MIN_ROWS, "chronological_descending": CHRONOLOGICAL_DESCENDING})`.
5. Breakpoints: for each metric in `["crps", "coverage_80pct", "interval_width_k"]`, for each of `emos_pooled` (contiguous) and `emos_local` (contiguous, using only rows where `n_stations_covered > 0`), call `find_breakpoint` against the `tsfm3` reference's value for that metric (`better="lower"` for `crps`; for `coverage_80pct` compare against the reference's own coverage value directly, `better="lower"` if the trained method starts below and needs to rise past it, `better="higher"` — i.e. flip the comparison — if it starts above, decide from the printed numbers which direction actually applies and say so in a comment; for `interval_width_k` there is no single "better" direction since it's a sharpness/calibration tradeoff — report the crossing point without a `better` framing, purely descriptive). Collect these into a small `breakpoints_df` (`columns: metric, emos_variant, breakpoint_n_cases, breakpoint_calendar_days`, with `None` rows where `find_breakpoint` returned `None`, printed as `"no crossing observed in tested range"`), and `write_result(breakpoints_df, name="phase3_data_size_sweep_breakpoints", model_version="phase3-sweep-v2", config={...same config...})`.
6. Figure: one `matplotlib` figure with 3 subplots sharing the x-axis (`plt.subplots(3, 1, figsize=(9, 12), sharex=True)`), one subplot each for `crps`, `coverage_80pct`, `interval_width_k`. In each subplot, plot: `emos_pooled` contiguous (solid line, marker `o`), `emos_pooled` `random_mean` with `fill_between` for ±1 std (dashed line + shaded band), `emos_local` contiguous restricted to `n_stations_covered > 0` (solid line, marker `^`), `raw_ensemble` flat reference (dotted `axhline`), `tsfm3` flat reference (dash-dot `axhline`). X-axis tick labels combine both units, e.g. `f"{n_days}\n({k} cases)"` for numeric `n_days`, `"full\n(4180 cases)"` for `"full"`. Legend once (on the top subplot is fine), shared x-label `"Training data size"`, per-subplot y-labels `"CRPS"`, `"Coverage @ 80% nominal"`, `"Interval width (K)"`. Save to `figures/data_size_sweep.png`, `dpi=150`, `os.makedirs("figures", exist_ok=True)` first.

- [ ] **Step 7: Run it for real**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && python scripts/07_data_size_sweep.py'
```
Expected: one printed line per `(n_days, arm, method)` combination (fast — CPU seconds per EMOS fit, even with the local-EMOS per-station loop), figure saved, two result files written (`results/phase3_data_size_sweep.parquet`/`.json` and `results/phase3_data_size_sweep_breakpoints.parquet`/`.json`). Copy back and inspect: `scp altay:~/zeropp/figures/data_size_sweep.png /tmp/data_size_sweep.png`. In your report, record: the real breakpoint per metric (CRPS, coverage@80%, interval width) for both `emos_pooled` and `emos_local`, whether/how much the contiguous and random arms diverge, `emos_local`'s station-coverage fraction at each N, and the Step 0 finding about `year_idx`/`time_idx` polarity — these are the paper's headline numbers, plural, not a single one.

- [ ] **Step 8: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/data/build.py tests/test_build.py configs/experiment.yaml src/zeropp/config.py tests/test_config.py scripts/07_data_size_sweep.py
git commit -m "feat: training-data-size breakpoint curve (RQ2) — contiguous/random arms x pooled/local EMOS, per-metric breakpoints"
```

---

## Task 5: DRN baseline (Rasp & Lerch 2018)

**Files:**
- Modify: `src/zeropp/models/drn.py` (currently a stub raising `NotImplementedError`)
- Modify: `scripts/07_data_size_sweep.py` (add a DRN curve to the sweep)
- Test: `tests/test_drn.py`

**Interfaces:**
- Consumes: `zeropp.models.base.Postprocessor`. `fit(train: pd.DataFrame)` expects the `load_train()`/`build_train_ensemble_stats` schema (`station_id`, `ens_mean`, `ens_var`, `t2m_obs`). `predict_quantiles(X: dict)` expects `X = {"ens_mean": np.ndarray (n_samples, n_leads), "ens_var": np.ndarray (n_samples, n_leads), "station_id": np.ndarray (n_samples,)}` — note the extra `station_id` key versus `EMOS`, since DRN uses a learned per-station embedding (a real, disclosed architectural difference — document it in the class docstring).
- Produces: `DRN(quantile_levels: list[float], embedding_dim: int = 4, hidden_dim: int = 16, n_epochs: int = 50, lr: float = 1e-2, seed: int = 0, device: str = "cpu")`, a `Postprocessor` subclass.

## Before You Begin

DRN parametrizes a Gaussian (same distributional family as `EMOS`, for a fair architectural comparison — the difference is EMOS's linear `mu = a + b*ens_mean` vs. DRN's nonlinear neural network with a learned per-station embedding) and trains by minimizing the same closed-form Gaussian CRPS Phase 2's `EMOS` already uses (`src/zeropp/models/emos.py`'s `_gaussian_crps`), just implemented in `torch` so it's differentiable for gradient descent instead of `scipy.optimize`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_drn.py
import numpy as np
import pandas as pd
import pytest

from zeropp.models.base import Postprocessor
from zeropp.models.drn import DRN

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def synthetic_train_df():
    rng = np.random.default_rng(0)
    n_per_station = 200
    station_ids = [1, 2, 3]
    station_bias = {1: -2.0, 2: 0.0, 3: 3.0}  # each station has a real, distinct bias

    rows = []
    for sid in station_ids:
        ens_mean = rng.normal(280.0, 5.0, n_per_station)
        ens_var = rng.uniform(0.5, 2.0, n_per_station)
        t2m_obs = ens_mean + station_bias[sid] + rng.normal(0, np.sqrt(ens_var))
        rows.append(pd.DataFrame({"station_id": sid, "ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": t2m_obs}))
    return pd.concat(rows, ignore_index=True)


def test_drn_is_a_postprocessor():
    assert issubclass(DRN, Postprocessor)


def test_drn_fit_returns_self(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5)
    assert model.fit(synthetic_train_df) is model


def test_drn_predict_quantiles_shape(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5).fit(synthetic_train_df)
    X = {
        "ens_mean": np.full((3, 1), 280.0),
        "ens_var": np.full((3, 1), 1.0),
        "station_id": np.array([1, 2, 3]),
    }
    preds = model.predict_quantiles(X)
    assert preds.shape == (3, 1, 9)


def test_drn_predict_before_fit_raises():
    model = DRN(quantile_levels=QUANTILE_LEVELS)
    X = {"ens_mean": np.zeros((1, 1)), "ens_var": np.ones((1, 1)), "station_id": np.array([1])}
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_quantiles(X)


def test_drn_learns_real_per_station_bias(synthetic_train_df):
    # Behavioral test: after training, DRN's median prediction for the same ens_mean
    # should differ meaningfully across stations, tracking the real, distinct biases
    # the synthetic data was generated with (station 3's median should be well above
    # station 1's, since the true generating bias differs by 5.0).
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=200, lr=0.05, seed=0).fit(synthetic_train_df)
    X = {
        "ens_mean": np.full((3, 1), 280.0),
        "ens_var": np.full((3, 1), 1.0),
        "station_id": np.array([1, 2, 3]),
    }
    preds = model.predict_quantiles(X)
    median_idx = QUANTILE_LEVELS.index(0.5)
    medians = preds[:, 0, median_idx]
    assert medians[2] - medians[0] > 2.0  # station 3 vs station 1, true gap is 5.0


def test_drn_predict_quantiles_are_monotonic_increasing(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5).fit(synthetic_train_df)
    X = {"ens_mean": np.full((2, 1), 280.0), "ens_var": np.full((2, 1), 1.0), "station_id": np.array([1, 2])}
    preds = model.predict_quantiles(X)
    assert np.all(np.diff(preds, axis=-1) >= 0)


def test_drn_unseen_station_at_predict_time_does_not_crash(synthetic_train_df):
    model = DRN(quantile_levels=QUANTILE_LEVELS, n_epochs=5).fit(synthetic_train_df)
    X = {"ens_mean": np.full((1, 1), 280.0), "ens_var": np.full((1, 1), 1.0), "station_id": np.array([999])}
    preds = model.predict_quantiles(X)  # station 999 was never in training data
    assert preds.shape == (1, 1, 9)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_drn.py -v'
```
Expected: FAIL with `NotImplementedError: blocked: ...` (the current stub).

- [ ] **Step 3: Write `src/zeropp/models/drn.py`**

```python
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from zeropp.models.base import Postprocessor


def _gaussian_crps_torch(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = (y - mu) / sigma
    normal = torch.distributions.Normal(0.0, 1.0)
    cdf = normal.cdf(z)
    pdf = torch.exp(normal.log_prob(z))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


class _DRNNet(nn.Module):
    def __init__(self, n_stations: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.station_embedding = nn.Embedding(n_stations, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, ens_mean: torch.Tensor, ens_var: torch.Tensor, station_idx: torch.Tensor):
        emb = self.station_embedding(station_idx)
        x = torch.cat([ens_mean.unsqueeze(-1), ens_var.unsqueeze(-1), emb], dim=-1)
        out = self.net(x)
        return out[..., 0], out[..., 1]  # mu, log_sigma2


class DRN(Postprocessor):
    """Distributional Regression Network (Rasp & Lerch 2018): a Gaussian
    postprocessor like EMOS, but with a nonlinear per-station-embedding neural
    network instead of a linear fit, trained by gradient descent on the same
    closed-form Gaussian CRPS loss EMOS minimizes via scipy.optimize.

    predict_quantiles requires an extra "station_id" key in X (unlike EMOS),
    since the learned embedding is indexed by station. A station_id unseen
    during fit() falls back to embedding index 0 (a documented limitation,
    not a silent bug) rather than raising.
    """

    def __init__(
        self,
        quantile_levels: list[float],
        embedding_dim: int = 4,
        hidden_dim: int = 16,
        n_epochs: int = 50,
        lr: float = 1e-2,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.quantile_levels = quantile_levels
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed
        self.device = device
        self._net = None
        self._station_to_idx: dict | None = None

    def fit(self, train) -> "DRN":
        torch.manual_seed(self.seed)

        stations = sorted(train["station_id"].unique())
        self._station_to_idx = {s: i for i, s in enumerate(stations)}

        station_idx = torch.tensor(train["station_id"].map(self._station_to_idx).to_numpy(), dtype=torch.long)
        ens_mean = torch.tensor(train["ens_mean"].to_numpy(), dtype=torch.float32)
        ens_var = torch.tensor(train["ens_var"].to_numpy(), dtype=torch.float32)
        obs = torch.tensor(train["t2m_obs"].to_numpy(), dtype=torch.float32)

        self._net = _DRNNet(len(stations), self.embedding_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            mu, log_sigma2 = self._net(ens_mean, ens_var, station_idx)
            sigma = torch.exp(0.5 * log_sigma2)
            loss = _gaussian_crps_torch(mu, sigma, obs).mean()
            loss.backward()
            optimizer.step()

        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("DRN.predict_quantiles called before fit()")

        ens_mean = X["ens_mean"]
        ens_var = X["ens_var"]
        station_id = X["station_id"]
        n_samples, n_leads = ens_mean.shape

        station_idx_arr = np.array(
            [[self._station_to_idx.get(s, 0)] * n_leads for s in station_id], dtype=np.int64
        )

        self._net.eval()
        with torch.no_grad():
            mu, log_sigma2 = self._net(
                torch.tensor(ens_mean, dtype=torch.float32),
                torch.tensor(ens_var, dtype=torch.float32),
                torch.tensor(station_idx_arr, dtype=torch.long),
            )
            sigma = torch.exp(0.5 * log_sigma2)

        mu_np = mu.numpy()
        sigma_np = sigma.numpy()
        quantiles = [mu_np + sigma_np * norm.ppf(q) for q in self.quantile_levels]
        return np.stack(quantiles, axis=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_drn.py -v'
```
Expected: 7 passed. If `test_drn_learns_real_per_station_bias` doesn't pass with the given hyperparameters, increase `n_epochs`/`lr` in that test's own call (not the class defaults) rather than weakening the assertion — verify by hand that the training loss is actually decreasing before concluding the model can't learn the pattern.

- [ ] **Step 5: Add DRN to the data-size sweep**

**Note: Task 4 went through a post-review fix round (commits 4d80a91..3db6738) that changed several names/helpers in `scripts/07_data_size_sweep.py` from what was originally sketched here — read the actual current file before wiring DRN in, don't rely on the snippet below being a verbatim match.** In particular: the k/calendar-day pair now comes from one helper, `k, n_calendar = k_and_calendar_days(n_days, n_pairs_full)`, not a separate `k_to_calendar_days(k)` call; the EMOS predict calls use a `test_X = {"ens_mean": ens_means, "ens_var": ens_vars}` dict built once after the instance-set join (`main()`, mid-file) — reuse that same dict's contents for DRN's `ens_mean`/`ens_var` inputs so DRN is scored on the identical joined instance set as EMOS (this was exactly Task 4's fix-round-1 finding 1 — don't reintroduce that mismatch for DRN); `breakpoint_and_direction()`/the breakpoints loop now only covers `["crps", "coverage_80pct"]` (`interval_width_k` was deliberately dropped from breakpoints per Task 4's fix round — DRN should follow the same rule, report its width as a curve only, do not add it to the breakpoints loop for that metric).

DRN gets its own curve on the same N-axis as EMOS, using the **contiguous arm only** (DRN's per-station embedding already models station-specific structure the way `emos_local` does, so there is no separate local/pooled DRN split — one `method="drn"` curve is enough; the random arm and multi-seed treatment stay EMOS-only, out of scope here, note this in your report same as Task 4's local-EMOS random-arm scope note). Modify `scripts/07_data_size_sweep.py`'s per-`n_days` contiguous-arm block (the non-empty branch, after the `emos_pooled`/`emos_local` rows are appended) to also fit and evaluate `DRN`, using the same `compute_metrics` helper as EMOS and the same `test_X`/`obs_values`/`test_station_ids` (post-join) that `fit_predict_local_emos` already uses:

```python
from zeropp.models.drn import DRN

# inside the non-empty-train_contig branch, after the emos_pooled/emos_local rows, using
# the same k/n_calendar already computed earlier in this branch via k_and_calendar_days:
        drn_train = train_contig[["station_id", "ens_mean", "ens_var", "t2m_obs"]]
        drn_model = DRN(quantile_levels=quantile_levels, seed=config.data_size_sweep_seeds[0]).fit(drn_train)
        drn_preds = drn_model.predict_quantiles({
            "ens_mean": test_X["ens_mean"], "ens_var": test_X["ens_var"], "station_id": test_station_ids,
        })
        drn_metrics = compute_metrics(obs_values, drn_preds, quantile_levels)
        rows.append({
            "n_days": str(n_days), "n_cases": k, "n_calendar_days_equiv": n_calendar,
            "sampling_arm": "contiguous", "seed": None, "method": "drn", "n_stations_covered": None,
            **drn_metrics,
        })
        print(f"N={n_days} days ({len(train_contig)} training rows): DRN CRPS={drn_metrics['crps']:.4f}")
```

(`config.data_size_sweep_seeds[0]` — i.e. the first configured seed, `0` — is used as `DRN`'s fixed training seed since it's a real value already required to exist by Task 4's config change, not a new hardcoded seed introduced here; CLAUDE.md's "no hardcoded seed in code" rule is satisfied because the value still traces back to `configs/experiment.yaml`.)

Add a `drn` line to each of the 3 subplots in the current figure code (same style as the `emos_local` line, different marker, e.g. `marker="s"`), and add `drn` to the breakpoints loop **for `crps` and `coverage_80pct` only** (same treatment as `emos_pooled`/`emos_local`, against the same `tsfm3` reference) — do not add a breakpoint for `interval_width_k`, matching Task 4's fix-round-1 rationale that sharpness has no inherent "better" direction.

- [ ] **Step 6: Run the extended sweep for real**

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && python scripts/07_data_size_sweep.py'
```
Expected: DRN's metrics printed alongside EMOS's and the reference lines' for each contiguous-arm N, figure's 3 subplots each gain a `drn` series, breakpoints file gains `drn` rows. Copy back and inspect the figure. Record whether DRN beats `emos_pooled`/`emos_local`, beats TimesFM-3, and at what N (per metric, same as Task 4) — this is a real, reportable finding either way, and together with the EMOS curves is intended to be the paper's cover figure (TimesFM-3 flat, EMOS, DRN, all on one N-axis).

- [ ] **Step 7: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/models/drn.py tests/test_drn.py scripts/07_data_size_sweep.py
git commit -m "feat: real DRN baseline (Rasp & Lerch 2018) — station-embedding neural network, added to the data-size sweep"
```

**Addendum for Task 5's review (added after a first implementer attempt stalled and was killed):** the review must confirm the report states the stalled attempt's actual failure cause (traceback if one exists) — if the cause was never identified, the same failure can recur silently on a future run. The review must also sanity-check the reported training-time profile: DRN's k=9 training set is only ~1,029 rows, and a small per-station-embedding MLP should fit in seconds on that, not tens of minutes — if the real run's CPU time was much larger than that, the report must explain why (e.g. `n_epochs` far larger than needed, an accidental full-data-size fit at every N, a stuck optimizer), broken down per N in the results/report, not just a single total.

**Addendum tied to Task 6's E4 (added after Task 4 review):** whether DRN is presentable as a baseline in the paper at all depends on Task 6's E4 finding. If `build_train_ensemble_stats_with_ids` silently drops the lead-time (`step`) dimension before DRN ever sees it, what gets trained here is not really "DRN" — it's a 2-input (`ens_mean`, `ens_var`) MLP with a station embedding, i.e. a nonlinear EMOS variant, missing the auxiliary features and lead-time structure the real Rasp & Lerch (2018) DRN's advantage over EMOS comes from. Do not let Task 5's review conclude "DRN underperforms EMOS, therefore DRN is a weak baseline" without first checking E4 — if the feature set is the constraint, the correct conclusion is "this architecture, under this constrained feature set, was outperformed by EMOS at low N — not evidence about DRN's real capability," and this baseline is not presentable as a fully faithful DRN in the paper's headline comparison without that caveat prominently attached (or without first fixing the feature set once E4 resolves it).

---

## Task 6: Paper-defensibility closeout (E1-E5)

**Do not dispatch this task until Task 4's and Task 5's reviews are both clean.** This task exists because the external technical review of Task 4's real numbers found the breakpoint curve alone is not yet defensible, and because Task 3's lead-time crossover finding — the plan's most interesting result — currently has no task computing the two numbers that would make it citable. **Order matters and is fixed: E4 → E3 → E1, with E2/E5 folded in wherever they're cheapest (all are pure reporting/analysis over already-persisted data, no new GPU/CPU compute beyond what Tasks 3/4/5 already produced, except E1's baseline fit and E3's low-N grid, both CPU-seconds).**

### E4 (do first): resolve what `build_train_ensemble_stats_with_ids` actually does with lead time

Determine definitively whether the reforecast archive's `step` (lead-time) dimension survives into the training dataframe:
- **Scenario A:** rows are duplicated across `step` but the column is silently dropped during column selection (i.e. `k=1` produces ~`49 stations × 21 steps = 1029` rows, not 49) — lead-time information exists in the row count but is unlabeled and gets pooled together in every EMOS/DRN fit.
- **Scenario B:** the training pipeline averages over `step` before this function runs, so `ens_mean`/`ens_var` are already lead-time-averaged — a much deeper problem, since the "sharpness" and "coverage" claims for EMOS would then be comparing a lead-time-blurred quantity against TimesFM-3/raw-ensemble's genuinely per-lead-time metrics.
Check by inspecting the actual row count at `k=1` (or reading `germany_ensemble_reforecasts_t2m.nc`'s real dims for `t2m` directly) — this is a read/count operation on already-downloaded data, not a new model run. State the finding plainly in the task report. If Scenario A: report it as a known limitation for EMOS/DRN's sharpness comparison (a single global fit averaging across lead times will be too wide at short lead, too narrow at long lead — direction of the bias, not magnitude, is derivable analytically and should be stated) and, if time allows, fit one additional EMOS variant that groups training rows by lead time and fits one `(a,b,c,d)` per lead-time group (or per one of the E5b lead-time buckets) to quantify how much the pooled fit's sharpness numbers change — do not just flag it, measure it once cheaply if the grouping columns are available. If Scenario B: this is a blocker for the sharpness story as currently framed and must be escalated back to the controller before Task 6 proceeds to E1/E3, since E1's baseline design and DRN's presentability both depend on which scenario is real.

### E3 (do second, independent of E4, can run in parallel with it): low-N grid

Extend the N-day sweep's tested values to include `k = 1, 2, 3, 5, 7` (in addition to the existing 30/90/365/1095/full days) — using `DAYS_PER_CASE` this is roughly `N ≈ 3, 7, 10, 17, 24` calendar days, but drive the sweep by `k` directly rather than by `n_days` for this grid since the existing conversion rounds coarsely at small N. Re-run `sample_contiguous`/`fit_predict_pooled_emos`/`fit_predict_local_emos`/`compute_metrics` at each of these `k` values (reuse the existing functions verbatim — this grid needs no new code beyond calling them at more `k` values and recomputing `find_breakpoint` over the extended axis). Cost is milliseconds per fit. Report exactly where the CRPS and coverage breakpoints land on this finer grid — the current "k=9" number may not survive as the true breakpoint once k=1..7 are actually measured, and this finer number (e.g. "~7 calendar days" instead of "~31") is a materially more striking headline if it holds up.

### E1 (do third, depends on E4's finding for its sharpness framing): variance-inflation baseline

Add a trivial baseline: scale the raw ensemble's spread by a single constant multiplier, and find the multiplier that makes its interval width match TimesFM-3's (~5.138 K). **Fit this multiplier two ways, report both, do not pick one:**
- **(a) Estimated from training data** (the same `k`-sized contiguous training subset EMOS gets at each N) — this is the fair comparison against EMOS (both get "1 parameter / k cases" vs. "4 parameters / k cases"). Must use only the training subset, never the test set (fitting on test data is leakage and unfairly strengthens this baseline).
- **(b) Estimated from a fixed climatological constant** (a typical published ECMWF/EUPPBench under-dispersion inflation factor, cited) — this is the fair comparison against TimesFM-3, since this variant is genuinely zero-shot with no fitting step at all, matching TimesFM-3's own zero-shot status.
Report CRPS, coverage@80%, and interval width for both variants, at both the case-level and lead-time-level (per E5b's grouping, if E5b is done by this point — a fixed multiplier does not adapt per case or per lead time the way TimesFM-3's output might, and this difference must be shown, not asserted: compare the per-instance/per-lead-time width DISTRIBUTION, not just the mean). If this baseline matches or beats TimesFM-3, that is a real, reportable, paper-shaping finding, not a result to bury.

### E2 (fold in wherever convenient, no new compute): apply Task 2's significance test to Task 4's headline coverage/CRPS gaps

Task 4's headline claims (pooled EMOS's coverage@80% edge over TimesFM-3 at k=9 is +0.0065; its CRPS edge is -0.098) are currently unaccompanied by any significance statement, unlike Task 3's lead-time findings. Apply `zeropp.eval.significance.station_blocked_paired_test` (Task 2) to both differentials at k=9 (and report the same for k=26 if useful for contrast). Depending on the result:
- **coverage not significant:** state "at k=9 (~31 days), EMOS and TimesFM-3 are statistically indistinguishable on coverage; EMOS is significantly better by k=26 (~90 days)" (adjust the second clause to whatever k the test actually turns significant at).
- **coverage significant:** state "EMOS shows a marginal but statistically significant coverage edge even at k=9."
Apply the same test to the CRPS differential (report the p-value even though it's expected to be clearly significant — state it, don't assume it). **The headline sentence for Task 4's result must be built on the CRPS finding, not coverage**: something like "k=9 training cases (~31 calendar days) is sufficient for EMOS to beat TimesFM-3 on CRPS; the two are statistically indistinguishable on coverage at that point, with EMOS pulling significantly ahead by k=26 (~90 days)" — adjust exact numbers to what E2/E3's real results show.

### E5 (fold in wherever convenient, no new compute): quantify Task 3's lead-time crossover

Task 3's lead-time breakdown (`results/phase2_lead_time_breakdown.parquet` or equivalent, already computed and persisted) shows TimesFM-3 best at short lead, worst at long lead — currently reported qualitatively only. Two additions, both pure aggregation over already-persisted data:
- **E5a:** find and report the exact lead time (in hours) where TimesFM-3's CRPS crosses from better-than to worse-than EMOS's (reuse Task 4's `find_breakpoint`-style logic, or simpler: the first `step_hours` value where the sign of `crps_tsfm3 - crps_emos` flips). Report the number plainly (e.g. "the crossover occurs at Xh").
- **E5b:** re-run Task 4's breakpoint calculation (CRPS and coverage, both EMOS variants vs. TimesFM-3) SEPARATELY within lead-time buckets (e.g. 0-24h, 24-72h, 72-120h) instead of pooled across all 21 lead times — this only requires calling the existing breakpoint function once per bucket if lead-time (`step_hours`) is available in Task 4's per-instance data (check this — if Task 4's EMOS predictions were computed without retaining `step_hours` per row, this needs that column added back before bucketing, which may interact with E4's finding about whether lead time is even distinguishable in the training data at all). Report whether the pooled "EMOS beats TimesFM-3 by k=9" headline holds within EVERY bucket, or whether short-lead buckets need much more data (larger k) to catch up — since at short lead, the 40-observation context window carries real persistence information TimesFM-3 can exploit that vanishes at long lead, where the covariate is the only informative signal left and EMOS's structural advantage may be smaller or absent. If the buckets diverge sharply, this reframes the paper's message from "TSFMs aren't ready for postprocessing" to "TSFMs are valuable specifically in the nowcasting regime" — state this framing explicitly in the report if the data supports it, this is potentially the paper's strongest single sentence.

**Files:** primarily `scripts/07_data_size_sweep.py` (E1, E3 additions) and either a new small script or an extension to `scripts/06_lead_time_breakdown.py` (E5a/E5b) plus `scripts/05_summarize_results.py` or wherever Task 4's headline sentence currently lives (E2). Route every new result through `write_result` per the Global Constraints. Write real tests for any new function (the variance-inflation multiplier fit, the lead-time-bucketed breakpoint call) — TDD, not a one-off script with no test coverage, consistent with every other task in this plan.

---

## Self-Review Notes

- **Spec coverage:** B1+O2 → Task 1 (interval width, coverage relabel, tail-truncation limitation note, all in `scripts/05_summarize_results.py`); B2 → Task 2 (`significance.py` + its integration into the same script); O1 → Task 3; S1 → Task 4; O3 → Task 5; CLAUDE.md rule 3 (provenance) → Task 1's `results.py`. The paper-writing placement guidance for the covariate-bug writeup (Code Availability note + Discussion subsection) is explicitly a writing task, not a code task, and is intentionally not a plan task — noted here as a reminder for whoever writes the paper: do NOT narrate the debugging process in the paper's main body, put it in (a) a brief Code/Data Availability note and (b) a "Implementation pitfall: covariate wiring" Discussion subsection.
- **Placeholder scan:** Task 1 Step 1 and Task 4's "Before You Begin" both contain genuine, disclosed investigation/design steps with concrete fallback instructions ("adapt to match reality, note the adaptation") rather than vague hand-waving — consistent with how Phase 2 handled real unknowns (e.g. the forecasts schema investigation). Not a placeholder violation.
- **Type consistency:** `Postprocessor.fit(train) -> self` / `predict_quantiles(X) -> ndarray` shape `(n_samples, n_leads, n_quantiles)` used identically by `DRN` (Task 5) as by `EMOS`/`TimesFM3` (Phase 2). `write_result`'s signature (`df, *, name, model_version, config, out_dir="results"`) defined once in Task 1, reused verbatim by Tasks 3 and 4. `build_train_ensemble_stats_with_ids`'s column names (`station_id, time_idx, year_idx, ens_mean, ens_var, t2m_obs`) defined in Task 4, consumed identically by Task 4's own sweep script and Task 5's DRN training loop.

## Phase 3 Closure Notes (for when this plan's workspace is finished, not a plan task itself)

- **`docs/superpowers/` (plan files, task briefs, review packages) belongs in git, not in the Zenodo archive.** It's process/development-log material with real historical value in the repo's own commit history, but no reproducibility value for someone trying to rerun the analysis — mark it `export-ignore` in `.gitattributes` so `git archive` (the mechanism Zenodo-via-GitHub-release snapshots use) skips it, while it stays fully present in the git repo itself. Do this before cutting the Zenodo release, not as an afterthought.
- **Archive `results/*.parquet`/`*.json`, not just the code.** Every result file already carries git SHA/config hash/model version (CLAUDE.md rule 3, satisfied by `write_result`) — that provenance is only useful to a reader if the actual result files ship with the archive, not just the code that produced them.
- **`git_sha` alone is not sufficient provenance under this project's workflow.** Under this project's rsync-then-run workflow, `git_sha` names the nearest prior commit, not necessarily the exact code that produced this file (proven stale at least once — final-review fix round, item 2); `source_tree_sha256` (added to `write_result`'s sidecar in that same fix round, alongside `git_dirty`) identifies the exact source tree. Prefer `source_tree_sha256` over `git_sha` when checking whether two result files ran identical code.
- **README needs an explicit TimesFM 3.0 Non-Commercial License note** (per `CLAUDE.md`'s "Lisans notu" — academic use is fine, no commercial embedding) before any public release.
- Sharing code is the norm for EUPPBench-based work in this subfield — the project's actual methodological contribution (the covariate-wiring pitfall writeup) only has value to the field if the code proving it is inspectable.
