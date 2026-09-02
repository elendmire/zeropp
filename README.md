# ZeroPP: Zero-shot Postprocessing Benchmark

Can a frozen time-series foundation model (TimesFM-3, Chronos-2, Moirai-2,
CITRAS-FM), given NWP ensemble output as a known future covariate, match
trained station-level postprocessing (EMOS, DRN, QRF, ...) on EUPPBench —
and at how many days of training data does the trained method pull ahead?
See `CLAUDE.md` for the full project brief and non-negotiable rules.

## Status

**Phase 1 (this checkout): done.** Local architecture only — `Postprocessor`
ABC, QC pipeline, CRPS/pinball/MAE/twCRPS, PIT/coverage/reliability,
climatology + persistence + raw-ensemble baselines. No EUPPBench data, no
GPU, no SSH required to run `pytest`.

**Phase 2: blocked on SSH access.** See `docs/PHASE2_BLOCKED.md` for the
exact list of stub modules and what unblocks each one.

## Running the tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

## License note

TimesFM 3.0 weights are distributed under the TimesFM Non-Commercial
License v1.0. Academic use (this project) is fine. Commercial use or
embedding in a product is NOT permitted under that license.
