# Phase 2 blocked work

Everything below raises `NotImplementedError("blocked: ...")` on purpose.
Do not implement any of these with synthetic/fake data — wait for the real
prerequisite, then write it with TDD like every Phase 1 module.

| File | Blocked on |
|---|---|
| scripts/01_download_data.sh | SSH host + Zenodo/GitHub network access |
| src/zeropp/data/download.py | same |
| scripts/02_build_dataset.py | download.py output |
| src/zeropp/data/build.py | download.py output |
| src/zeropp/data/splits.py | build.py output |
| src/zeropp/models/emos.py, qrf.py, drn.py, mos_rf.py | splits.py output |
| scripts/03_run_baselines.py | above baselines |
| src/zeropp/models/tsfm_timesfm.py | SSH `timesfm[torch]` install + verified covariate API |
| src/zeropp/models/tsfm_chronos.py, tsfm_moirai.py | SSH install of chronos/moirai packages |
| scripts/04_run_tsfm.py | tsfm_*.py |
| scripts/05_data_size_sweep.py | baselines + tsfm models both real |
| src/zeropp/models/wrappers/conformal.py, gpd_tail.py, qavg.py | real TSFM quantile output |
| src/zeropp/eval/tables.py, figures.py | real results/*.parquet |
| src/zeropp/cli.py | the scripts it wraps |
| scripts/06_make_report.py | tables.py, figures.py |

Next unblock step: get SSH connection details from the user, run
scripts/00_setup_env.sh on the server, then run the "İlk üç komut" checks
from the project brief (TimesFM import, EUPPBench download, covariate API
grep) before writing any of the above for real.
