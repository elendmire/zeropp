# ZeroPP Phase 2 — First Real Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Phase 1's stubs into working code on real EUPPBench data — build a real long-format dataset, a real EMOS baseline, a real covariate-conditioned TimesFM-3 zero-shot forecaster, and a first real CRPS comparison between them, for one variable (t2m) and one country (Germany).

**Architecture:** Code is edited and committed on the Mac (`/Users/farukavci/zeropp`, the single git source of truth). Every test that touches real data or the `torch`/`timesfm`/`climetlab` stack runs over SSH on the `altay` HPC login node, against a synced copy of the repo — there is no local-only test path in this phase, because every task here eventually touches real data or heavy deps. EMOS is trained on the 20-year reforecast archive (ensemble mean/variance → observation pairs, order-independent — no calendar dates needed). TimesFM-3's zero-shot context and covariates are built from the 2017-2018 forecast archive's own real chronological record — the two archives are used for different purposes and are never merged into one table, which is also what makes "never re-split EUPPBench's own train/test boundary" trivial to satisfy: train and test literally come from different source files, never from date-filtering a merged table.

**Tech Stack:** Python 3.11.16 (server-side venv only), pandas, numpy, xarray, scipy (added this phase, for EMOS's CRPS-minimization fit), torch/timesfm (server-side only, already installed), pytest.

## Global Constraints

- **Sync-then-test workflow (every task uses this, no exceptions):** after editing/committing on the Mac, sync to the server, then run tests over SSH:
  ```bash
  rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
    -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
  ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest <path> -v'
  ```
  Never run `pytest` locally on the Mac in this phase — the Mac venv doesn't have `torch`/`timesfm`/`climetlab`, and none of this phase's real-data files live there either.
- Quantile levels stay the single global constant from `configs/experiment.yaml` (`[0.1, ..., 0.9]`) — every model in this phase reads them from `zeropp.config.load_experiment_config()`, never hardcodes them.
- `Postprocessor.fit(train) -> self` / `predict_quantiles(X) -> ndarray` shaped `(n_samples, n_leads, n_quantiles)` is unchanged from Phase 1 — `EMOS` and `TimesFM3` must satisfy it exactly, like `Climatology`/`RawEnsemble` already do.
- EUPPBench's own train/test boundary must never be violated: reforecasts (1997-2016) are train-only, forecasts (2017-2018) are test-only. No function in this phase may combine or re-split across that boundary.
- No hardcoded seeds in `src/` — EMOS's optimizer seed (if any randomness is used — L-BFGS-B here is deterministic, no seed needed) and any future seed usage must come from `configs/experiment.yaml`.
- TimesFM-3 covariate injection must be implemented exactly per the real, verified API (`past_future_covariates` on `TimesFM3Forecaster.predict`/`predict_batch`) or not at all — never approximated. This was already verified working end-to-end this session; Task 5 reuses that exact verified call shape.
- Server paths: repo at `~/zeropp` (`/ari/users/oavci/zeropp`), cached TimesFM-3 weights at `~/zeropp/model_cache/timesfm-3.0-pytorch/`, raw EUPPBench NetCDF at `~/zeropp/data/raw/`. SSH alias `altay` is already configured with key-based auth.
- The server cannot reach `object-store.os-api.cci1.ecmwf.int` (firewalled) — any further EUPPBench download in this phase (Task 1) must run on the Mac, then `rsync` to `altay:~/zeropp/data/raw/`. The Mac already has a working lightweight venv at `/private/tmp/claude-501/-Users-farukavci/59bff915-451d-4e16-956c-72aedec11873/scratchpad/euppbench_dl_venv` (xarray, `zarr<3`, fsspec, aiohttp — deliberately not full `climetlab`, which has native-lib install friction on macOS). Reuse it; don't recreate it.
- The EUPPBench object store disconnects mid-transfer frequently. Every real download in this phase wraps BOTH `xr.open_zarr(..., consolidated=True)` and `.load()` in a retry loop (proven pattern from this session, reused verbatim in Task 1).

---

## File Structure

```
src/zeropp/data/
├── build.py       # Task 2 — raw NetCDF -> modeling-ready tables (real, replaces stub)
└── splits.py      # Task 3 — train/test accessors (real, replaces stub)
src/zeropp/models/
├── emos.py            # Task 4 — real Gaussian CRPS-minimization EMOS (replaces stub)
└── tsfm_timesfm.py    # Task 5 — real predict_quantiles (replaces stub; fit() unchanged)
scripts/
├── 03_run_baselines.py  # Task 6 — real: raw ensemble + EMOS on the real test set
└── 04_run_tsfm.py       # Task 6 — real: TimesFM-3 zero-shot on the real test set
tests/
├── test_build.py          # Task 2
├── test_splits.py         # Task 3
├── test_emos.py           # Task 4
└── test_tsfm_timesfm.py   # Task 5
data/raw/  (on altay only, not in git)
├── germany_ensemble_reforecasts_t2m.nc         # already present (this session)
├── germany_reforecasts_observations_t2m.nc     # already present (this session)
├── germany_ensemble_forecasts_t2m.nc           # Task 1 — new
└── germany_forecasts_observations_t2m.nc       # Task 1 — new
```

---

## Task 1: Download the missing test-side (forecasts) data

**Files:**
- Create (on the Mac, temporary, then transferred): local NetCDF files under the scratchpad download venv's working area
- Create (on altay, via rsync): `~/zeropp/data/raw/germany_ensemble_forecasts_t2m.nc`, `~/zeropp/data/raw/germany_forecasts_observations_t2m.nc`
- Create: `docs/euppbench_forecasts_schema.md` (repo root docs, committed) — records the empirically-confirmed dimension names for the forecasts dataset, since (unlike reforecasts, confirmed this session) the exact dims of the non-reforecast "forecasts" dataset have not yet been inspected

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: two real NetCDF files on the server that Task 2 reads. Task 2 needs to know: the exact dimension names on `germany_ensemble_forecasts_t2m.nc` (predicted: `station_id`, `time`, `number`, `step`, `surface` — NO `year` dimension, since forecasts cover only real 2017-2018 dates directly, unlike the 20-analog-year reforecast structure) and whether `time` decodes to real `datetime64` cleanly (predicted: yes, based on the gridded-forecasts example from EUPPBench's own docs successfully decoding `time: 2017-01-01 ... 2018-12-31` with no overflow — the overflow bug seen this session was specific to the *reforecast* observations file, not any forecasts file). **This task's Step 3 must verify this prediction and update `docs/euppbench_forecasts_schema.md` with what's actually true before Task 2 starts** — if the prediction is wrong, write down what's real, since Task 2 depends on it.

- [ ] **Step 1: Inspect the forecasts dataset's real structure on the Mac**

```bash
VENV=/private/tmp/claude-501/-Users-farukavci/59bff915-451d-4e16-956c-72aedec11873/scratchpad/euppbench_dl_venv
"$VENV/bin/python3" -c "
import fsspec, xarray as xr, time

BASE = 'https://object-store.os-api.cci1.ecmwf.int/eumetnet-postprocessing-benchmark-1st-phase-training-dataset/data/stations_data'

def try_open(url, decode_times, tries=6):
    last_err = None
    for attempt in range(tries):
        try:
            fs = fsspec.filesystem('https', client_kwargs={'timeout': None})
            mapper = fs.get_mapper(url)
            return xr.open_zarr(mapper, consolidated=True, decode_times=decode_times)
        except Exception as e:
            last_err = e
            print(f'  attempt {attempt+1} failed: {type(e).__name__}: {e}')
            time.sleep(3*(attempt+1))
    raise last_err

fcs_url = f'{BASE}/stations_ensemble_forecasts_surface_germany.zarr'
obs_url = f'{BASE}/stations_forecasts_observations_surface_germany.zarr'

print('=== forecasts (decode_times=True) ===')
try:
    ds = try_open(fcs_url, decode_times=True)
    print(ds)
except Exception as e:
    print('decode_times=True FAILED:', e)
    print('=== retrying with decode_times=False ===')
    ds = try_open(fcs_url, decode_times=False)
    print(ds)

print()
print('=== observations (decode_times=True) ===')
try:
    ods = try_open(obs_url, decode_times=True)
    print(ods)
except Exception as e:
    print('decode_times=True FAILED:', e)
    print('=== retrying with decode_times=False ===')
    ods = try_open(obs_url, decode_times=False)
    print(ods)
"
```

Record the exact `Dimensions:` line and whether `time` (and/or `valid_time`) prints as `datetime64[ns]` (success) or required the `decode_times=False` fallback (failure) for BOTH datasets. This determines what Task 2 can rely on.

- [ ] **Step 2: Download t2m for both, with retry-wrapped `.load()`, matching this session's proven pattern**

```bash
VENV=/private/tmp/claude-501/-Users-farukavci/59bff915-451d-4e16-956c-72aedec11873/scratchpad/euppbench_dl_venv
OUT=/private/tmp/claude-501/-Users-farukavci/59bff915-451d-4e16-956c-72aedec11873/scratchpad/euppbench_raw
"$VENV/bin/python3" << 'PYEOF'
import fsspec, xarray as xr, time

BASE = 'https://object-store.os-api.cci1.ecmwf.int/eumetnet-postprocessing-benchmark-1st-phase-training-dataset/data/stations_data'
OUT = '/private/tmp/claude-501/-Users-farukavci/59bff915-451d-4e16-956c-72aedec11873/scratchpad/euppbench_raw'

# Use whichever decode_times value Step 1 determined actually works for each
# dataset — do not assume both use the same value.
DECODE_TIMES_FCS = True   # <-- set to what Step 1 actually found
DECODE_TIMES_OBS = True   # <-- set to what Step 1 actually found

def try_open(url, decode_times, tries=6):
    last_err = None
    for attempt in range(tries):
        try:
            fs = fsspec.filesystem('https', client_kwargs={'timeout': None})
            mapper = fs.get_mapper(url)
            return xr.open_zarr(mapper, consolidated=True, decode_times=decode_times)
        except Exception as e:
            last_err = e
            print(f'  open attempt {attempt+1} failed: {type(e).__name__}: {e}', flush=True)
            time.sleep(3*(attempt+1))
    raise last_err

def try_load(ds, tries=8):
    last_err = None
    for attempt in range(tries):
        try:
            return ds.load()
        except Exception as e:
            last_err = e
            print(f'  load attempt {attempt+1} failed: {type(e).__name__}: {e}', flush=True)
            time.sleep(4*(attempt+1))
    raise last_err

targets = [
    ('germany_ensemble_forecasts_t2m', f'{BASE}/stations_ensemble_forecasts_surface_germany.zarr', DECODE_TIMES_FCS),
    ('germany_forecasts_observations_t2m', f'{BASE}/stations_forecasts_observations_surface_germany.zarr', DECODE_TIMES_OBS),
]

for name, url, decode_times in targets:
    print(f'=== {name} ===', flush=True)
    ds = try_open(url, decode_times)
    ds_t2m = ds[['t2m']]
    print('shape:', ds_t2m.t2m.shape, 'approx MB:', ds_t2m.t2m.nbytes/1e6, flush=True)
    ds_t2m = try_load(ds_t2m)
    outpath = f'{OUT}/{name}.nc'
    ds_t2m.to_netcdf(outpath)
    print('saved to', outpath, flush=True)

print('ALL DONE')
PYEOF
```

- [ ] **Step 3: Write `docs/euppbench_forecasts_schema.md` recording what Step 1 actually found**

```markdown
# EUPPBench forecasts (test-side) dataset schema — confirmed YYYY-MM-DD

Confirmed by direct inspection (see Task 1 of docs/superpowers/plans/2026-09-03-zeropp-phase2-real-slice.md):

## germany_ensemble_forecasts_t2m.nc
- Dimensions: <paste the real `Dimensions:` line from Step 1's output>
- `time` decodes to real datetime64: <yes/no — if no, note what decode_times value was actually needed and why>
- Ensemble member count: <N>

## germany_forecasts_observations_t2m.nc
- Dimensions: <paste the real `Dimensions:` line from Step 1's output>
- `time` decodes to real datetime64: <yes/no>

## Consequence for Task 2 (data/build.py)
<one sentence: e.g. "time decodes cleanly on both files as predicted, build.py uses xr.open_dataset's default decode_times=True and reads `time` directly as the chronological index" — or, if the prediction was wrong, state exactly what build.py must do instead>
```

- [ ] **Step 4: Transfer to altay**

```bash
ssh -o BatchMode=yes altay 'mkdir -p ~/zeropp/data/raw'
rsync -az -e "ssh -o BatchMode=yes" \
  /private/tmp/claude-501/-Users-farukavci/59bff915-451d-4e16-956c-72aedec11873/scratchpad/euppbench_raw/germany_ensemble_forecasts_t2m.nc \
  /private/tmp/claude-501/-Users-farukavci/59bff915-451d-4e16-956c-72aedec11873/scratchpad/euppbench_raw/germany_forecasts_observations_t2m.nc \
  altay:~/zeropp/data/raw/
ssh -o BatchMode=yes altay 'ls -la ~/zeropp/data/raw/ && du -sh ~/zeropp/data/raw/'
```

Expected: both new files listed alongside the two already-present reforecast files, total size a few hundred MB.

- [ ] **Step 5: Sanity-check on the server (values physically consistent, real dates present)**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate
python -c "
import xarray as xr
fcs = xr.open_dataset(\"data/raw/germany_ensemble_forecasts_t2m.nc\")
obs = xr.open_dataset(\"data/raw/germany_forecasts_observations_t2m.nc\")
print(\"fcs dims:\", dict(fcs.sizes))
print(\"obs dims:\", dict(obs.sizes))
print(\"fcs t2m NaN frac:\", float((fcs.t2m.isnull().sum()/fcs.t2m.size).values))
print(\"obs t2m NaN frac:\", float((obs.t2m.isnull().sum()/obs.t2m.size).values))
"
'
```

Expected: dims match what Step 3 recorded, NaN fractions are small (station data — some missing is normal, per CLAUDE.md).

- [ ] **Step 6: Commit** (only the schema doc — raw data files live on the server, not in git, matching Phase 1's `.gitignore` treatment of `results/`)

```bash
cd /Users/farukavci/zeropp
git add docs/euppbench_forecasts_schema.md
git commit -m "docs: confirm EUPPBench forecasts (test-side) dataset schema, download Germany t2m test data"
```

---

## Task 2: `data/build.py` — real long-format tables from raw NetCDF

**Files:**
- Modify: `src/zeropp/data/build.py` (currently a stub raising `NotImplementedError`)
- Test: `tests/test_build.py`

**Interfaces:**
- Consumes: `docs/euppbench_forecasts_schema.md` (Task 1) for the real forecasts-side dimension names; the reforecast files' already-known dims (`station_id`, `time`, `number`, `year`, `step`, `surface` — confirmed this session, no schema doc needed for these, they're stable).
- Produces (read by Task 3, 4, 5, 6 — exact names, keep them):
  - `build_train_ensemble_stats(reforecast_path: str, obs_path: str) -> pd.DataFrame` with columns `station_id` (int), `ens_mean` (float, Kelvin), `ens_var` (float, Kelvin²), `t2m_obs` (float, Kelvin), one row per (station, time_idx, year_idx, step) combination with a non-NaN observation. `ens_mean`/`ens_var` are computed across the `number` (ensemble member) dimension. Rows with NaN observation OR NaN forecast are dropped (this is the QC gate — apply it here, not silently downstream).
  - `build_test_long_table(forecast_path: str, obs_path: str) -> pd.DataFrame` with columns `station_id` (int), `valid_time` (real `pd.Timestamp`, chronologically sortable), `step_hours` (float), `member` (int), `t2m_forecast` (float, Kelvin), `t2m_obs` (float, Kelvin, broadcast across members since obs has no member dimension). Sorted by `station_id`, then `valid_time`. Rows with NaN observation are dropped; rows with NaN forecast are dropped.

## Before You Begin

If Task 1's `docs/euppbench_forecasts_schema.md` shows the forecasts dataset's dimensions differ from `(station_id, time, number, step, surface)`, or `time` did NOT decode cleanly, STOP and ask — the code below assumes it did. Do not guess a workaround; the schema doc from Task 1 is the source of truth.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build.py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from zeropp.data.build import build_test_long_table, build_train_ensemble_stats


@pytest.fixture
def synthetic_reforecast_files(tmp_path):
    # 2 stations, 2 time_idx, 1 year, 2 step, 3 members
    station_id = [100, 200]
    time_idx = [0, 1]
    year_idx = [0]
    step = [0.0, 6.0]
    number = [0, 1, 2]

    rng = np.random.default_rng(0)
    fcs_data = rng.normal(280.0, 2.0, size=(2, 2, 1, 2, 3, 1)).astype("float32")
    fcs = xr.Dataset(
        {"t2m": (("station_id", "time", "year", "step", "number", "surface"), fcs_data)},
        coords={
            "station_id": station_id, "time": time_idx, "year": year_idx,
            "step": step, "number": number, "surface": [0.0],
        },
    )
    fcs_path = tmp_path / "reforecasts.nc"
    fcs.to_netcdf(fcs_path)

    obs_data = np.full((2, 2, 1, 2), 280.0, dtype="float64")
    obs_data[0, 0, 0, 0] = np.nan  # one missing observation to verify it gets dropped
    obs = xr.Dataset(
        {"t2m": (("station_id", "time", "year", "step"), obs_data)},
        coords={"station_id": station_id, "time": time_idx, "year": year_idx, "step": step},
    )
    obs_path = tmp_path / "reforecast_obs.nc"
    obs.to_netcdf(obs_path)

    return str(fcs_path), str(obs_path)


@pytest.fixture
def synthetic_forecast_files(tmp_path):
    station_id = [100, 200]
    time = pd.date_range("2017-01-01", periods=3, freq="12h")
    step = [0.0, 6.0]
    number = [0, 1]

    rng = np.random.default_rng(1)
    fcs_data = rng.normal(280.0, 2.0, size=(2, 3, 2, 2, 1)).astype("float32")
    fcs = xr.Dataset(
        {"t2m": (("station_id", "time", "step", "number", "surface"), fcs_data)},
        coords={"station_id": station_id, "time": time, "step": step, "number": number, "surface": [0.0]},
    )
    fcs_path = tmp_path / "forecasts.nc"
    fcs.to_netcdf(fcs_path)

    obs_data = np.full((2, 3, 2), 280.0, dtype="float64")
    obs_data[1, 2, 1] = np.nan  # one missing observation
    obs = xr.Dataset(
        {"t2m": (("station_id", "time", "step"), obs_data)},
        coords={"station_id": station_id, "time": time, "step": step},
    )
    obs_path = tmp_path / "forecast_obs.nc"
    obs.to_netcdf(obs_path)

    return str(fcs_path), str(obs_path)


def test_build_train_ensemble_stats_shape_and_columns(synthetic_reforecast_files):
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats(fcs_path, obs_path)
    assert list(df.columns) == ["station_id", "ens_mean", "ens_var", "t2m_obs"]
    # 2 stations x 2 time_idx x 1 year x 2 step = 8 combinations, minus 1 dropped NaN obs
    assert len(df) == 7


def test_build_train_ensemble_stats_mean_and_var_are_real_ensemble_stats(synthetic_reforecast_files):
    fcs_path, obs_path = synthetic_reforecast_files
    df = build_train_ensemble_stats(fcs_path, obs_path)
    assert (df["ens_var"] >= 0).all()
    assert df["ens_mean"].between(270, 290).all()  # sanity: within the synthetic generation range


def test_build_test_long_table_shape_and_columns(synthetic_forecast_files):
    fcs_path, obs_path = synthetic_forecast_files
    df = build_test_long_table(fcs_path, obs_path)
    assert list(df.columns) == ["station_id", "valid_time", "step_hours", "member", "t2m_forecast", "t2m_obs"]
    # 2 stations x 3 time x 2 step x 2 members = 24 rows, minus 2 members dropped for the 1 NaN obs row
    assert len(df) == 22


def test_build_test_long_table_sorted_by_station_then_time(synthetic_forecast_files):
    fcs_path, obs_path = synthetic_forecast_files
    df = build_test_long_table(fcs_path, obs_path)
    for station_id, group in df.groupby("station_id"):
        assert group["valid_time"].is_monotonic_increasing


def test_build_test_long_table_valid_time_is_real_timestamp(synthetic_forecast_files):
    fcs_path, obs_path = synthetic_forecast_files
    df = build_test_long_table(fcs_path, obs_path)
    assert pd.api.types.is_datetime64_any_dtype(df["valid_time"])
```

- [ ] **Step 2: Run tests to verify they fail**

Sync and run over SSH (per Global Constraints):
```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_build.py -v'
```
Expected: FAIL with `NotImplementedError: blocked: ...` (the current stub) for every test.

- [ ] **Step 3: Write `src/zeropp/data/build.py`**

```python
import numpy as np
import pandas as pd
import xarray as xr


def build_train_ensemble_stats(reforecast_path: str, obs_path: str) -> pd.DataFrame:
    fcs = xr.open_dataset(reforecast_path)
    obs = xr.open_dataset(obs_path)

    ens_mean = fcs["t2m"].mean(dim="number")
    ens_var = fcs["t2m"].var(dim="number")

    merged = xr.Dataset({"ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": obs["t2m"]})
    df = merged.to_dataframe().reset_index()
    df = df[["station_id", "ens_mean", "ens_var", "t2m_obs"]]
    df = df.dropna(subset=["ens_mean", "ens_var", "t2m_obs"])
    return df.reset_index(drop=True)


def build_test_long_table(forecast_path: str, obs_path: str) -> pd.DataFrame:
    fcs = xr.open_dataset(forecast_path)
    obs = xr.open_dataset(obs_path)

    fcs_df = fcs["t2m"].to_dataframe(name="t2m_forecast").reset_index()
    fcs_df = fcs_df.rename(columns={"time": "valid_time", "step": "step_hours", "number": "member"})

    obs_df = obs["t2m"].to_dataframe(name="t2m_obs").reset_index()
    obs_df = obs_df.rename(columns={"time": "valid_time", "step": "step_hours"})

    merged = fcs_df.merge(obs_df, on=["station_id", "valid_time", "step_hours"], how="left")
    merged = merged.dropna(subset=["t2m_forecast", "t2m_obs"])
    merged = merged[["station_id", "valid_time", "step_hours", "member", "t2m_forecast", "t2m_obs"]]
    merged = merged.sort_values(["station_id", "valid_time"]).reset_index(drop=True)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_build.py -v'
```
Expected: 5 passed.

- [ ] **Step 5: Run against the REAL Germany data as a manual integration check (not a committed test — just verification)**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate
python -c "
from zeropp.data.build import build_train_ensemble_stats, build_test_long_table
train = build_train_ensemble_stats(\"data/raw/germany_ensemble_reforecasts_t2m.nc\", \"data/raw/germany_reforecasts_observations_t2m.nc\")
test = build_test_long_table(\"data/raw/germany_ensemble_forecasts_t2m.nc\", \"data/raw/germany_forecasts_observations_t2m.nc\")
print(\"train rows:\", len(train), train.head())
print(\"test rows:\", len(test), test.head())
print(\"train ens_var range:\", train.ens_var.min(), train.ens_var.max())
"
'
```

Expected: both non-empty, `ens_var` all non-negative and physically plausible (a few K²), no crash. If this fails, the real data's actual column/dimension names differ from what Step 3's code assumes — fix `build.py` to match reality (check with `xr.open_dataset(path)` printed structure) rather than changing the real data.

- [ ] **Step 6: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/data/build.py tests/test_build.py
git commit -m "feat: real data/build.py — long-format tables from raw EUPPBench NetCDF"
```

---

## Task 3: `data/splits.py` — real train/test accessors

**Files:**
- Modify: `src/zeropp/data/splits.py` (currently a stub raising `NotImplementedError`)
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: `build_train_ensemble_stats`, `build_test_long_table` from `zeropp.data.build` (Task 2).
- Produces (read by Task 4, 5, 6): `load_train(data_dir: str = "data/raw") -> pd.DataFrame` (calls `build_train_ensemble_stats` with the two reforecast file paths under `data_dir`), `load_test(data_dir: str = "data/raw") -> pd.DataFrame` (calls `build_test_long_table` with the two forecast file paths under `data_dir`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_splits.py
from unittest.mock import patch

import pandas as pd

from zeropp.data.splits import load_test, load_train


def test_load_train_calls_build_train_ensemble_stats_with_reforecast_paths():
    fake_df = pd.DataFrame({"station_id": [1], "ens_mean": [280.0], "ens_var": [1.0], "t2m_obs": [280.5]})
    with patch("zeropp.data.splits.build_train_ensemble_stats", return_value=fake_df) as mock_build:
        result = load_train(data_dir="/some/dir")
    mock_build.assert_called_once_with(
        "/some/dir/germany_ensemble_reforecasts_t2m.nc",
        "/some/dir/germany_reforecasts_observations_t2m.nc",
    )
    assert result is fake_df


def test_load_test_calls_build_test_long_table_with_forecast_paths():
    fake_df = pd.DataFrame({"station_id": [1], "valid_time": [pd.Timestamp("2017-01-01")],
                             "step_hours": [0.0], "member": [0], "t2m_forecast": [280.0], "t2m_obs": [280.5]})
    with patch("zeropp.data.splits.build_test_long_table", return_value=fake_df) as mock_build:
        result = load_test(data_dir="/some/dir")
    mock_build.assert_called_once_with(
        "/some/dir/germany_ensemble_forecasts_t2m.nc",
        "/some/dir/germany_forecasts_observations_t2m.nc",
    )
    assert result is fake_df


def test_load_train_and_load_test_never_share_a_source_file():
    import inspect
    train_src = inspect.getsource(load_train)
    test_src = inspect.getsource(load_test)
    # The EUPPBench train/test boundary is enforced by construction: these two
    # functions must never reference the same raw filename.
    train_files = {w for w in train_src.split() if w.endswith('.nc"') or w.endswith(".nc'")}
    test_files = {w for w in test_src.split() if w.endswith('.nc"') or w.endswith(".nc'")}
    assert train_files.isdisjoint(test_files)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_splits.py -v'
```
Expected: FAIL with `NotImplementedError: blocked: ...`.

- [ ] **Step 3: Write `src/zeropp/data/splits.py`**

```python
import pandas as pd

from zeropp.data.build import build_test_long_table, build_train_ensemble_stats


def load_train(data_dir: str = "data/raw") -> pd.DataFrame:
    return build_train_ensemble_stats(
        f"{data_dir}/germany_ensemble_reforecasts_t2m.nc",
        f"{data_dir}/germany_reforecasts_observations_t2m.nc",
    )


def load_test(data_dir: str = "data/raw") -> pd.DataFrame:
    return build_test_long_table(
        f"{data_dir}/germany_ensemble_forecasts_t2m.nc",
        f"{data_dir}/germany_forecasts_observations_t2m.nc",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_splits.py -v'
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/data/splits.py tests/test_splits.py
git commit -m "feat: real data/splits.py — reforecasts=train, forecasts=test, enforced disjoint by construction"
```

---

## Task 4: `models/emos.py` — real Gaussian CRPS-minimization EMOS

**Files:**
- Modify: `src/zeropp/models/emos.py` (currently a stub)
- Test: `tests/test_emos.py`

**Interfaces:**
- Consumes: `zeropp.models.base.Postprocessor` (Phase 1). Trains on a DataFrame shaped like `load_train()`'s output (columns `station_id`, `ens_mean`, `ens_var`, `t2m_obs`) — this is what `fit(train)` receives.
- Produces: `EMOS(quantile_levels: list[float])`, a `Postprocessor` subclass. `fit(train: pd.DataFrame) -> self` fits one global (station-pooled) Gaussian EMOS: `μ = a + b·ens_mean`, `σ² = exp(c) + exp(d)·ens_var` (the `exp` reparametrization keeps the variance non-negative without needing a constrained optimizer), minimizing mean training CRPS over `(a, b, c, d)` via `scipy.optimize.minimize` with `method="L-BFGS-B"`. `predict_quantiles(X: dict) -> np.ndarray` expects `X = {"ens_mean": np.ndarray shape (n_samples, n_leads), "ens_var": np.ndarray shape (n_samples, n_leads)}` and returns `(n_samples, n_leads, n_quantiles)` via the inverse normal CDF at each quantile level, using the fitted `(a, b, c, d)`.

## Before You Begin

The closed-form Gaussian CRPS (Gneiting et al. 2005) is:
```
CRPS(N(μ, σ²), y) = σ · [ z·(2·Φ(z) − 1) + 2·φ(z) − 1/√π ],   z = (y − μ) / σ
```
where `Φ` is the standard normal CDF and `φ` the standard normal PDF. This is real, established meteorological statistics — implement it exactly, do not approximate.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_emos.py
import numpy as np
import pandas as pd
import pytest

from zeropp.models.base import Postprocessor
from zeropp.models.emos import EMOS

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def synthetic_train_df():
    rng = np.random.default_rng(42)
    n = 500
    ens_mean = rng.normal(280.0, 5.0, size=n)
    ens_var = rng.uniform(0.5, 3.0, size=n)
    # true relationship: obs ~ N(ens_mean, ens_var) — EMOS should recover a~=0, b~=1
    t2m_obs = ens_mean + rng.normal(0, np.sqrt(ens_var))
    return pd.DataFrame({"station_id": 1, "ens_mean": ens_mean, "ens_var": ens_var, "t2m_obs": t2m_obs})


def test_emos_is_a_postprocessor():
    assert issubclass(EMOS, Postprocessor)


def test_emos_fit_returns_self(synthetic_train_df):
    model = EMOS(quantile_levels=QUANTILE_LEVELS)
    fitted = model.fit(synthetic_train_df)
    assert fitted is model


def test_emos_fit_recovers_near_identity_mean_mapping(synthetic_train_df):
    # Since obs were generated as N(ens_mean, ens_var), a well-fit EMOS should
    # find b close to 1 and a close to 0 — a real check that the optimizer
    # is actually fitting something meaningful, not just returning defaults.
    model = EMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    assert model._params is not None
    a, b, c, d = model._params
    assert abs(b - 1.0) < 0.3
    assert abs(a) < 2.0


def test_emos_predict_quantiles_shape(synthetic_train_df):
    model = EMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    X = {"ens_mean": np.full((3, 2), 280.0), "ens_var": np.full((3, 2), 1.5)}
    preds = model.predict_quantiles(X)
    assert preds.shape == (3, 2, 9)


def test_emos_predict_quantiles_are_monotonic_increasing(synthetic_train_df):
    model = EMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    X = {"ens_mean": np.full((2, 1), 280.0), "ens_var": np.full((2, 1), 1.5)}
    preds = model.predict_quantiles(X)
    assert np.all(np.diff(preds, axis=-1) >= 0)


def test_emos_predict_before_fit_raises():
    model = EMOS(quantile_levels=QUANTILE_LEVELS)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_quantiles({"ens_mean": np.zeros((1, 1)), "ens_var": np.ones((1, 1))})


def test_emos_median_close_to_ensemble_mean_when_b_near_one(synthetic_train_df):
    model = EMOS(quantile_levels=QUANTILE_LEVELS).fit(synthetic_train_df)
    X = {"ens_mean": np.full((1, 1), 280.0), "ens_var": np.full((1, 1), 1.5)}
    preds = model.predict_quantiles(X)
    median_idx = QUANTILE_LEVELS.index(0.5)
    assert abs(preds[0, 0, median_idx] - 280.0) < 3.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_emos.py -v'
```
Expected: FAIL with `NotImplementedError: blocked: ...`.

- [ ] **Step 3: Write `src/zeropp/models/emos.py`**

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from zeropp.models.base import Postprocessor


def _gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


class EMOS(Postprocessor):
    """Ensemble Model Output Statistics: a global Gaussian CRPS-minimization
    fit, mu = a + b*ens_mean, sigma^2 = exp(c) + exp(d)*ens_var (Gneiting et al. 2005)."""

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels
        self._params = None

    def fit(self, train) -> "EMOS":
        ens_mean = train["ens_mean"].to_numpy()
        ens_var = train["ens_var"].to_numpy()
        obs = train["t2m_obs"].to_numpy()

        def negative_mean_crps(params):
            a, b, c, d = params
            mu = a + b * ens_mean
            sigma = np.sqrt(np.exp(c) + np.exp(d) * ens_var)
            return np.mean(_gaussian_crps(mu, sigma, obs))

        result = minimize(negative_mean_crps, x0=[0.0, 1.0, 0.0, 0.0], method="L-BFGS-B")
        self._params = result.x
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("EMOS.predict_quantiles called before fit()")
        a, b, c, d = self._params
        ens_mean = X["ens_mean"]
        ens_var = X["ens_var"]
        mu = a + b * ens_mean
        sigma = np.sqrt(np.exp(c) + np.exp(d) * ens_var)
        quantiles = [mu + sigma * norm.ppf(q) for q in self.quantile_levels]
        return np.stack(quantiles, axis=-1)
```

- [ ] **Step 4: Add `scipy` to `pyproject.toml` dependencies**

```toml
dependencies = [
    "numpy>=1.26.4",
    "pandas>=2.2",
    "pyyaml>=6.0",
    "scipy>=1.11",
]
```

Then reinstall on the server: `ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && uv pip install -e ".[dev]"'`

- [ ] **Step 5: Run tests to verify they pass**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_emos.py -v'
```
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/models/emos.py tests/test_emos.py pyproject.toml
git commit -m "feat: real EMOS — Gaussian CRPS-minimization fit (Gneiting et al. 2005)"
```

---

## Task 5: `models/tsfm_timesfm.py` — real covariate-conditioned predict_quantiles

**Files:**
- Modify: `src/zeropp/models/tsfm_timesfm.py` (currently a stub — `fit()` is already a correct no-op, do not change it)
- Test: `tests/test_tsfm_timesfm.py`

**Interfaces:**
- Consumes: `zeropp.models.base.Postprocessor`. `predict_quantiles(X: dict)` expects `X = {"context": list[np.ndarray]}` where each array in the list is one station's recent real observation history (1-D, chronologically ordered — built by the caller from `load_test()`'s `valid_time`-sorted rows), plus `X["past_future_ens_mean"]` and `X["past_future_ens_spread"]` (each `list[np.ndarray]`, one 1-D array per station spanning the full context+horizon window — ensemble mean and standard deviation computed from `load_test()`'s per-`member` rows, the real covariate this whole project exists to test), and `X["horizon"]` (int).
- Produces: `(n_samples, n_leads, n_quantiles)` where `n_samples = len(X["context"])`.

## Before You Begin

The real, verified call shape (confirmed this session with actual inference against the cached local weights) is:
```python
m = timesfm.TimesFM3Forecaster.from_pretrained("<local weights dir>", device="cpu")
out = m.predict(context=ctx_1d_array, horizon=H, past_future_covariates=covariate_1d_array, return_quantiles=True)
out.forecast   # shape (H,)
out.quantiles  # shape (H, 9)
```
`past_future_covariates` takes ONE array per call spanning the full context+horizon window. This task passes ensemble mean as the covariate array (the model's single `past_future_covariates` slot) — ensemble spread as a SECOND simultaneous covariate requires checking whether `predict`/`predict_batch` accepts multiple named covariates or only one positional array; if the real signature only accepts one, use `ens_mean` alone for this first slice and note the `ens_spread` limitation explicitly in the report rather than silently dropping it without comment. Check the real signature on the server before writing the loop:
```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate
python -c "import timesfm, inspect; print(inspect.signature(timesfm.TimesFM3Forecaster.predict_batch))"
'
```

- [ ] **Step 1: Write the failing tests (these mock the model, they do not need real weights or the server's torch install — but per Global Constraints, still run over SSH since `timesfm` must be importable)**

```python
# tests/test_tsfm_timesfm.py
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zeropp.models.base import Postprocessor
from zeropp.models.tsfm_timesfm import TimesFM3

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_timesfm3_is_a_postprocessor():
    assert issubclass(TimesFM3, Postprocessor)


def test_timesfm3_fit_is_a_noop_returning_self():
    model = TimesFM3(quantile_levels=QUANTILE_LEVELS)
    assert model.fit(train=None) is model


def test_timesfm3_predict_quantiles_calls_predict_with_covariates():
    fake_output = MagicMock()
    fake_output.forecast = np.zeros(4)
    fake_output.quantiles = np.zeros((4, 9))

    fake_model = MagicMock()
    fake_model.predict.return_value = fake_output

    with patch("zeropp.models.tsfm_timesfm.timesfm.TimesFM3Forecaster.from_pretrained", return_value=fake_model):
        model = TimesFM3(quantile_levels=QUANTILE_LEVELS, weights_path="/fake/path")
        X = {
            "context": [np.ones(10)],
            "past_future_ens_mean": [np.ones(14)],
            "past_future_ens_spread": [np.ones(14)],
            "horizon": 4,
        }
        preds = model.predict_quantiles(X)

    assert preds.shape == (1, 4, 9)
    call_kwargs = fake_model.predict.call_args.kwargs
    assert "past_future_covariates" in call_kwargs
    assert call_kwargs["horizon"] == 4
    assert call_kwargs["return_quantiles"] is True


def test_timesfm3_predict_quantiles_handles_multiple_stations():
    fake_output = MagicMock()
    fake_output.forecast = np.zeros(3)
    fake_output.quantiles = np.zeros((3, 9))

    fake_model = MagicMock()
    fake_model.predict.return_value = fake_output

    with patch("zeropp.models.tsfm_timesfm.timesfm.TimesFM3Forecaster.from_pretrained", return_value=fake_model):
        model = TimesFM3(quantile_levels=QUANTILE_LEVELS, weights_path="/fake/path")
        X = {
            "context": [np.ones(10), np.ones(10)],
            "past_future_ens_mean": [np.ones(13), np.ones(13)],
            "past_future_ens_spread": [np.ones(13), np.ones(13)],
            "horizon": 3,
        }
        preds = model.predict_quantiles(X)

    assert preds.shape == (2, 3, 9)
    assert fake_model.predict.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_tsfm_timesfm.py -v'
```
Expected: FAIL — `test_timesfm3_fit_is_a_noop_returning_self` passes already (fit is unchanged from the stub), the rest fail with `NotImplementedError: blocked: ...` from `predict_quantiles`.

- [ ] **Step 3: Write `src/zeropp/models/tsfm_timesfm.py`**

(Adjust the `past_future_covariates=` line if Step 0's signature check showed multiple named covariate parameters exist — if so, pass `past_future_ens_mean` AND `past_future_ens_spread` both; if only one slot exists, pass `past_future_ens_mean` only and note the limitation in this task's report.)

```python
import numpy as np
import timesfm

from zeropp.models.base import Postprocessor

DEFAULT_WEIGHTS_PATH = "/ari/users/oavci/zeropp/model_cache/timesfm-3.0-pytorch"


class TimesFM3(Postprocessor):
    """Frozen TimesFM-3 zero-shot postprocessor with real past-future covariate injection."""

    def __init__(self, quantile_levels: list[float], weights_path: str = DEFAULT_WEIGHTS_PATH):
        self.quantile_levels = quantile_levels
        self.weights_path = weights_path
        self._model = None

    def fit(self, train) -> "TimesFM3":
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
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
                past_future_covariates=ens_mean,
                return_quantiles=True,
            )
            all_quantiles.append(np.asarray(out.quantiles))

        return np.stack(all_quantiles, axis=0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && pytest tests/test_tsfm_timesfm.py -v'
```
Expected: 4 passed.

- [ ] **Step 5: Manual integration check with the REAL cached weights (one real station, not mocked)**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate
python -c "
import numpy as np
from zeropp.models.tsfm_timesfm import TimesFM3
model = TimesFM3(quantile_levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
X = {
    \"context\": [np.random.normal(280, 3, size=40)],
    \"past_future_ens_mean\": [np.random.normal(280, 3, size=64)],
    \"past_future_ens_spread\": [np.random.normal(1, 0.3, size=64)],
    \"horizon\": 24,
}
preds = model.predict_quantiles(X)
print(\"shape:\", preds.shape)
print(\"monotonic:\", bool(np.all(np.diff(preds, axis=-1) >= 0)))
"
'
```
Expected: shape `(1, 24, 9)`, monotonic True, no crash — this is real TimesFM-3 inference with real weights, not a mock.

- [ ] **Step 6: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/models/tsfm_timesfm.py tests/test_tsfm_timesfm.py
git commit -m "feat: real TimesFM-3 predict_quantiles with past_future_covariates wiring"
```

---

## Task 6: First real comparison — raw ensemble vs EMOS vs TimesFM-3 on real Germany t2m

**Files:**
- Modify: `scripts/03_run_baselines.py` (currently a stub)
- Modify: `scripts/04_run_tsfm.py` (currently a stub)

**Interfaces:**
- Consumes: `zeropp.data.splits.load_train`, `load_test` (Task 3); `zeropp.models.emos.EMOS` (Task 4); `zeropp.models.tsfm_timesfm.TimesFM3` (Task 5); `zeropp.eval.scores.crps_from_quantiles` (Phase 1); `zeropp.config.load_experiment_config` (Phase 1, for quantile levels).
- Produces: printed CRPS numbers for three approaches on the real test set — this is the deliverable, a real research finding, not a reusable library function. No new test file for this task (matches how `scripts/00_setup_env.sh` in Phase 1 needed no test — it's an experiment run, verified by executing it and inspecting real output, not by pytest).

- [ ] **Step 1: Write `scripts/03_run_baselines.py`**

```python
"""Real baseline comparison: raw ensemble vs EMOS, on the real Germany t2m test set."""
import numpy as np

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test, load_train
from zeropp.eval.scores import crps_from_quantiles
from zeropp.models.emos import EMOS


def main() -> None:
    quantile_levels = load_experiment_config().quantile_levels

    train_df = load_train()
    test_df = load_test()

    emos = EMOS(quantile_levels=quantile_levels).fit(train_df)

    # Group by (station_id, valid_time, step_hours) to get one ensemble per forecast instance
    grouped = test_df.groupby(["station_id", "valid_time", "step_hours"])
    ens_means, ens_vars, obs_values = [], [], []
    for _, group in grouped:
        ens_means.append(group["t2m_forecast"].mean())
        ens_vars.append(group["t2m_forecast"].var())
        obs_values.append(group["t2m_obs"].iloc[0])

    ens_means = np.array(ens_means).reshape(-1, 1)
    ens_vars = np.array(ens_vars).reshape(-1, 1)
    obs_values = np.array(obs_values).reshape(-1, 1)

    # Raw ensemble: use the empirical quantiles of each ensemble directly
    raw_quantiles = []
    for _, group in grouped:
        raw_quantiles.append(np.quantile(group["t2m_forecast"].to_numpy(), quantile_levels))
    raw_quantiles = np.array(raw_quantiles).reshape(len(obs_values), 1, len(quantile_levels))

    emos_preds = emos.predict_quantiles({"ens_mean": ens_means, "ens_var": ens_vars})

    raw_crps = crps_from_quantiles(obs_values, raw_quantiles, quantile_levels)
    emos_crps = crps_from_quantiles(obs_values, emos_preds, quantile_levels)

    print(f"Raw ensemble mean CRPS: {raw_crps.mean():.4f}")
    print(f"EMOS mean CRPS:         {emos_crps.mean():.4f}")
    print(f"EMOS improvement over raw ensemble: {(1 - emos_crps.mean()/raw_crps.mean())*100:.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real**

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.superpowers' \
  -e "ssh -o BatchMode=yes" /Users/farukavci/zeropp/ altay:~/zeropp/
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && python scripts/03_run_baselines.py'
```
Record the actual printed CRPS numbers in this task's completion report — this is a real finding, write it down verbatim.

- [ ] **Step 3: Write `scripts/04_run_tsfm.py`**

```python
"""Real TimesFM-3 zero-shot comparison against the raw ensemble, on the real Germany t2m test set."""
import numpy as np

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test
from zeropp.eval.scores import crps_from_quantiles
from zeropp.models.tsfm_timesfm import TimesFM3

CONTEXT_LENGTH = 40  # number of past observations fed as context, per station


def main() -> None:
    quantile_levels = load_experiment_config().quantile_levels
    test_df = load_test()

    model = TimesFM3(quantile_levels=quantile_levels)

    all_preds, all_obs = [], []
    for station_id, station_df in test_df.groupby("station_id"):
        station_df = station_df.sort_values("valid_time")
        obs_series = station_df.drop_duplicates("valid_time")["t2m_obs"].to_numpy()
        if len(obs_series) <= CONTEXT_LENGTH:
            continue

        for i in range(CONTEXT_LENGTH, len(obs_series) - 1):
            context = obs_series[i - CONTEXT_LENGTH:i]
            future_row = station_df[station_df["valid_time"] == station_df["valid_time"].unique()[i]]
            if future_row.empty:
                continue
            ens_mean_future = future_row["t2m_forecast"].mean()
            covariate = np.concatenate([context, np.full(1, ens_mean_future)])

            preds = model.predict_quantiles({
                "context": [context],
                "past_future_ens_mean": [covariate],
                "past_future_ens_spread": [np.zeros_like(covariate)],
                "horizon": 1,
            })
            all_preds.append(preds[0])
            all_obs.append(obs_series[i])

    all_preds = np.array(all_preds)
    all_obs = np.array(all_obs).reshape(-1, 1)

    tsfm_crps = crps_from_quantiles(all_obs, all_preds, quantile_levels)
    print(f"TimesFM-3 zero-shot mean CRPS: {tsfm_crps.mean():.4f}")
    print(f"Evaluated on {len(all_obs)} real forecast instances")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it for real**

```bash
ssh -o BatchMode=yes altay 'export PATH="$HOME/.local/bin:$PATH"; cd ~/zeropp && source .venv/bin/activate && python scripts/04_run_tsfm.py'
```

This will be slow (one TimesFM-3 CPU inference call per forecast instance) — if it takes more than ~10 minutes, that's expected on CPU; let it finish. Record the real printed CRPS number.

- [ ] **Step 5: Write the real finding to the task report**

In this task's completion report, state plainly, using the ACTUAL numbers from Steps 2 and 4 (never fabricate or round favorably):
- Raw ensemble CRPS, EMOS CRPS, EMOS's % improvement over raw ensemble
- TimesFM-3 zero-shot CRPS, and its % improvement (or regression) versus raw ensemble
- Whether TimesFM-3 clears CLAUDE.md's week-4 kill-criterion bar (≥5% CRPS improvement over raw ensemble) — if it does not, say so plainly; this is a legitimate, useful negative result per the original project brief's own kill-criteria design, not a failure to hide.

- [ ] **Step 6: Commit**

```bash
cd /Users/farukavci/zeropp
git add scripts/03_run_baselines.py scripts/04_run_tsfm.py
git commit -m "feat: first real comparison — raw ensemble vs EMOS vs TimesFM-3 zero-shot on Germany t2m"
```

---

## Self-Review Notes

- **Spec coverage:** all 6 numbered scope items from the brief map to tasks: (1) missing test data → Task 1, (2) data/build.py → Task 2, (3) data/splits.py → Task 3, (4) models/emos.py → Task 4, (5) models/tsfm_timesfm.py → Task 5, (6) comparison scripts → Task 6. Explicitly-out-of-scope items (other countries, other 9 models, N-day sweep, w10, wrappers, results.py, tables/figures, cli.py) are untouched by this plan — confirmed no task references them. The sync-then-test workflow ambiguity flagged in the brief is resolved: one consistent rule (Global Constraints) applies to every task, no per-task deviation.
- **Placeholder scan:** Task 1's schema doc has bracketed fill-in placeholders (`<paste the real...>`) — these are legitimate, since the values are empirically unknown until Step 1 runs; this is an investigation step with concrete commands and pass/fail criteria, not a vague "TBD," and Task 1 Step 3 explicitly requires filling them from Step 1's real output before the task is done.
- **Type consistency:** `Postprocessor.fit(train) -> self` / `predict_quantiles(X) -> ndarray` shape `(n_samples, n_leads, n_quantiles)` used identically in Task 4 (EMOS) and Task 5 (TimesFM3), matching Phase 1's ABC exactly. `load_train()`/`load_test()` column names (`ens_mean`, `ens_var`, `t2m_obs` / `station_id`, `valid_time`, `step_hours`, `member`, `t2m_forecast`, `t2m_obs`) are defined once in Task 2 and reused verbatim in Task 3 (splits), Task 4 (EMOS's `X` dict keys `ens_mean`/`ens_var` match), Task 6 (both scripts consume these exact column names).
