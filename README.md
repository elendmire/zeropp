# ZeroPP: Zero-shot Postprocessing Benchmark

Can a frozen time-series foundation model (TimesFM-3, Chronos-2, Moirai-2,
CITRAS-FM), given NWP ensemble output as a known future covariate, match
trained station-level postprocessing (EMOS, DRN, QRF, ...) on EUPPBench —
and at how many days of training data does the trained method pull ahead?
See `CLAUDE.md` for the full project brief and non-negotiable rules.

## Status

**Phases 1-3: done.** Phase 1: local architecture only (`Postprocessor` ABC,
QC pipeline, CRPS/pinball/MAE/twCRPS, PIT/coverage/reliability, climatology +
persistence + raw-ensemble baselines). Phase 2: real EUPPBench data + TimesFM-3
zero-shot comparison, run on `altay` over SSH. Phase 3: paper-readiness —
sharpness/calibration-corrected summary, station-blocked significance
testing, lead-time breakdown, the training-data-size breakpoint curve
(headline contribution), DRN baseline, and a paper-defensibility closeout
(low-N grid, variance-inflation baseline, lead-time-bucketed breakpoints).
See `docs/superpowers/plans/2026-09-03-zeropp-phase3-paper-readiness.md` and
`docs/results_index.md` for the full task history and which `results/*.parquet`
file is authoritative for which claim.

## Provenance

Every `results/*.parquet` file has a `.json` sidecar (`zeropp.eval.results.write_result`)
with `model_version`, `config`/`config_hash`, `git_sha`, `git_dirty`, `source_tree_sha256`,
and `written_at`. Under this project's rsync-then-run workflow (sync the working tree to
`altay` over SSH, run there, commit locally afterward), `git_sha` names the nearest prior
commit, not necessarily the exact code that produced this file; `source_tree_sha256` (a
content hash over every file under `src/`, `scripts/`, `configs/`) identifies the exact
source tree that ran, independent of whether it was ever committed — prefer it over
`git_sha` when checking whether two result files ran identical code.

## Running the tests

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

## License note

TimesFM 3.0 weights are distributed under the TimesFM Non-Commercial
License v1.0. Academic use (this project) is fine. Commercial use or
embedding in a product is NOT permitted under that license.
