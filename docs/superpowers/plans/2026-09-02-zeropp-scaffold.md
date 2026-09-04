# ZeroPP Repo Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the ZeroPP repo locally — directory tree, `CLAUDE.md` brief, configs, and the pure-Python architectural core (model ABC, QC pipeline, CRPS/pinball/calibration metrics, first two concrete baselines) — everything that needs no EUPPBench data, no GPU, and no SSH access, fully TDD'd.

**Architecture:** A `Postprocessor` ABC (`fit`/`predict_quantiles`) is the single contract every model — EMOS, DRN, QRF, TimesFM-3, Chronos-2 — must satisfy, so the eval code never branches on model type. Everything downstream (`eval/scores.py`, `eval/calibration.py`) consumes only the ABC's output shape `(n_samples, n_leads, n_quantiles)`, never a model internals. Data-heavy and SSH-heavy modules (EUPPBench download, TimesFM covariate wiring, DRN/QRF training) are scaffolded as explicit `NotImplementedError` stubs now and implemented in a later phase once SSH credentials and EUPPBench access exist.

**Tech Stack:** Python 3.11, numpy, pandas, PyYAML, pytest. Heavier deps (torch, timesfm, statsmodels, climetlab) are deferred to the SSH-side `uv` environment per `scripts/00_setup_env.sh` — do not install them locally in this plan.

## Global Constraints

- Quantile levels are a single global constant: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]` — defined once in `configs/experiment.yaml`, never hardcoded elsewhere.
- Every `Postprocessor` implements `fit(train) -> self` and `predict_quantiles(X) -> np.ndarray` of shape `(n_samples, n_leads, n_quantiles)`. Zero-shot models make `fit` a no-op but keep the identical signature.
- Notebooks never produce results — this plan creates zero notebook content. Results-producing code lives under `scripts/` and `src/zeropp/`.
- No hardcoded seeds anywhere in `src/`. Seeds live only in `configs/experiment.yaml`.
- EUPPBench's own train/test split must never be altered by any code written here (enforced later once `splits.py` is real; for now `splits.py` is a stub that cannot yet violate this).
- Data-size sweep axis is fixed: `N ∈ {0, 30, 90, 365, 1095, "full"}` days.
- TimesFM 3.0 weights are under the TimesFM Non-Commercial License v1.0 — academic use only, no commercial/product embedding. This must appear in the top-level `README.md`.
- Every module that cannot be implemented without SSH access or EUPPBench data (data download/build/splits, EMOS/DRN/QRF/MOS-RF baselines, all `tsfm_*` models, all `wrappers/*`, `cli.py`, `eval/tables.py`, `eval/figures.py`) is scaffolded as a stub raising `NotImplementedError("blocked: needs <reason>")` — never silently passes, never fakes output.

---

## File Structure

```
zeropp/
├── CLAUDE.md                          # Task 1 — verbatim brief, non-negotiable rules
├── pyproject.toml                     # Task 1
├── README.md                          # Task 9
├── .gitignore                         # Task 1
├── configs/
│   ├── experiment.yaml                # Task 2 — quantile levels, N-day axis, seeds
│   ├── data.yaml                      # Task 2 — EUPPBench version, targets, lead times
│   └── models/
│       ├── emos.yaml                  # Task 8 (stub config)
│       └── timesfm.yaml               # Task 8 (stub config)
├── src/zeropp/
│   ├── __init__.py                    # Task 1
│   ├── config.py                      # Task 2 — typed loader for the two yaml files
│   ├── data/
│   │   ├── __init__.py                # Task 1
│   │   ├── qc.py                      # Task 4 — gap detection, DST/UTC fix, interpolation
│   │   ├── download.py                # Task 8 (stub, blocked on SSH + Zenodo access)
│   │   ├── build.py                   # Task 8 (stub, blocked on downloaded data)
│   │   └── splits.py                  # Task 8 (stub, blocked on downloaded data)
│   ├── models/
│   │   ├── __init__.py                # Task 1
│   │   ├── base.py                    # Task 3 — Postprocessor ABC
│   │   ├── climatology.py             # Task 7
│   │   ├── raw.py                     # Task 7 — raw ensemble + persistence
│   │   ├── emos.py                    # Task 8 (stub, blocked on EUPPBench data)
│   │   ├── qrf.py                     # Task 8 (stub)
│   │   ├── drn.py                     # Task 8 (stub)
│   │   ├── mos_rf.py                  # Task 8 (stub)
│   │   ├── tsfm_timesfm.py            # Task 8 (stub, blocked on SSH + timesfm pkg)
│   │   ├── tsfm_chronos.py            # Task 8 (stub)
│   │   ├── tsfm_moirai.py             # Task 8 (stub)
│   │   └── wrappers/
│   │       ├── __init__.py            # Task 1
│   │       ├── conformal.py           # Task 8 (stub)
│   │       ├── gpd_tail.py            # Task 8 (stub)
│   │       └── qavg.py                # Task 8 (stub)
│   ├── eval/
│   │   ├── __init__.py                # Task 1
│   │   ├── scores.py                  # Task 5 — CRPS, pinball, MAE, twCRPS
│   │   ├── calibration.py             # Task 6 — PIT, coverage, reliability index
│   │   ├── tables.py                  # Task 8 (stub, blocked on real model results)
│   │   └── figures.py                 # Task 8 (stub)
│   └── cli.py                         # Task 8 (stub, blocked on scripts it would wrap)
├── scripts/
│   ├── 00_setup_env.sh                # Task 8 — real, runnable once SSH host is known
│   ├── 01_download_data.sh            # Task 8 (stub, blocked on SSH host + Zenodo)
│   ├── 02_build_dataset.py            # Task 8 (stub)
│   ├── 03_run_baselines.py            # Task 8 (stub)
│   ├── 04_run_tsfm.py                 # Task 8 (stub)
│   ├── 05_data_size_sweep.py          # Task 8 (stub)
│   └── 06_make_report.py              # Task 8 (stub)
├── notebooks/                         # Task 1 — empty, .gitkeep only
├── results/                           # Task 1 — empty, .gitkeep only
├── figures/                           # Task 1 — empty, .gitkeep only
└── tests/
    ├── __init__.py                    # Task 1
    ├── test_config.py                 # Task 2
    ├── test_base.py                   # Task 3
    ├── test_qc.py                     # Task 4
    ├── test_scores.py                 # Task 5
    ├── test_calibration.py            # Task 6
    └── test_baselines.py              # Task 7
```

Stubs (Task 8) are a deliberate architectural decision, not laziness: every stub module raises `NotImplementedError("blocked: <specific reason>")` immediately on call, so an accidental import-and-run in Phase 2 fails loudly instead of returning fabricated numbers. `docs/PHASE2_BLOCKED.md` (Task 8) lists every stub and exactly what unblocks it.

---

## Task 1: Repo scaffold, git init, pyproject, CLAUDE.md

**Files:**
- Create: `/Users/farukavci/zeropp/CLAUDE.md`
- Create: `/Users/farukavci/zeropp/pyproject.toml`
- Create: `/Users/farukavci/zeropp/.gitignore`
- Create: `/Users/farukavci/zeropp/src/zeropp/__init__.py`
- Create: `/Users/farukavci/zeropp/src/zeropp/data/__init__.py`
- Create: `/Users/farukavci/zeropp/src/zeropp/models/__init__.py`
- Create: `/Users/farukavci/zeropp/src/zeropp/models/wrappers/__init__.py`
- Create: `/Users/farukavci/zeropp/src/zeropp/eval/__init__.py`
- Create: `/Users/farukavci/zeropp/tests/__init__.py`
- Create: `/Users/farukavci/zeropp/notebooks/.gitkeep`
- Create: `/Users/farukavci/zeropp/results/.gitkeep`
- Create: `/Users/farukavci/zeropp/figures/.gitkeep`

**Interfaces:**
- Produces: importable package `zeropp` on `sys.path` via editable install (`pip install -e .`), so every later task can `import zeropp.models.base` etc.

- [ ] **Step 1: Create `CLAUDE.md` at repo root**

Content (copy verbatim — this is the user's non-negotiable brief, do not paraphrase or reformat):

```markdown
# ZeroPP: Zero-shot Postprocessing Benchmark

## Ne yapıyoruz
Dondurulmuş zaman serisi temel modellerini (TimesFM-3 basta olmak uzere)
istasyon bazli olasiliksal hava tahmini post-processing'inde, egitimli
EMOS/DRN baseline'larina karsi kiyasliyoruz. Ana soru: kac gunluk egitim
verisinden sonra egitimli yontemler sifir atisli modeli geciyor.

## Veri
EUPPBench (EUMETNET Postprocessing Benchmark) v1.0.
- GitHub: EUPP-benchmark/climetlab-eumetnet-postprocessing-benchmark
- Zenodo DOI: 10.5281/zenodo.7429236
- Hedefler: t2m (birincil), w10 (ikincil)
- EUPPBench'in KENDI train/test bolmesini kullan, yeniden bolme YAPMA.

## Mimari kurallar (pazarlik yok)
1. Her model `src/zeropp/models/base.py` icindeki `Postprocessor` ABC'sini
   implement eder: `fit(train) -> self`, `predict_quantiles(X) -> ndarray`
   sekli (n_samples, n_leads, n_quantiles). Sifir atisli modellerde fit()
   no-op olur ama imza degismez.
2. Kuantil seviyeleri global sabit: [0.1, 0.2, ..., 0.9]. configs/ icinde.
3. Tum sonuclar results/ altina parquet + json. Her dosyaya git SHA,
   model versiyonu ve config hash'i yazilir.
4. Notebook'lar sonuc URETMEZ. Sadece kesif. Sonuc scripts/ ile uretilir.
5. Rastgelelik: her seed configs/experiment.yaml icinde. Kod icinde
   hardcoded seed veya cagri yok.

## TimesFM-3 kovaryat kurgusu
- target: istasyon gozlemi (gecmis penceresi)
- past-future (dynamic) covariates: ensemble ortalamasi, ensemble spread,
  ek NWP alanlari. Bunlar ufuk boyunca BILINEN olarak verilir. Bu projenin
  teknik kalbi burasi, dikkatli implement et ve birim testi yaz.
- Model dondurulmus. Fine-tuning YOK (ayri bir kol olarak eklenene kadar).

## Veri kalitesi kisitlari (TSFM gereksinimleri)
- Context kesintisiz olmali, delik olmamali
- Context ve ufuk ayni frekansta olmali
- NaN'lar model cagrilmadan once lineer interpolasyonla doldurulmali
- DST gecisleri ve UTC/yerel saat karisikligi sahte delik yaratir, qc.py
  bunlari yakalamali ve testleri olmali

## Metrikler (hepsi zorunlu, tek skorla karar verilmez)
CRPS, MAE, pinball loss, PIT histogrami, nominal %90'da ampirik kapsama,
reliability index, threshold-weighted CRPS (kuyruk), wall-clock sure.

## Baseline seti (eksik baseline = reddedilmis makale)
raw ensemble, climatology, persistence, EMOS, AR-EMOS,
time-series EMOS (Jobst 2024), DRN (Rasp & Lerch 2018),
QRF (Taillardat 2016), MOS random forests (Muschinski 2023),
lead-time-continuous (Wessel 2024).
TSFM tarafi: TimesFM-3, Chronos-2, Moirai-2, CITRAS-FM.

## Veri boyutu ekseni
N in {0, 30, 90, 365, 1095, full} gun. Her N icin tum egitimli modeller
yeniden fit edilir, sifir atisli modeller degismez. Cikti: CRPS vs N egrisi.

## Ortam
SSH sunucu, Python 3.11, uv ile yonetilen venv.
GPU olmayabilir, CPU yolunu her zaman calisir tut.
Uzun koşular tmux icinde. SSH kopmasi is oldurmemeli.

## Lisans notu
TimesFM 3.0 agirliklari TimesFM Non-Commercial License v1.0 altinda.
Akademik kullanim uygun. Ticari kullanim veya urune gomme YOK.
README'de bunu belirt.

## Yapma
- EUPPBench bolmesini degistirme
- Sonuclari notebook'ta uretme
- Tek metrikle sonuc iddia etme
- Yeni mimari gelistirme, bu bir uygulama ve benchmark calismasi
- Kovaryat enjeksiyonunu "yaklasik" implement etme, dogru veya hic
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "zeropp"
version = "0.1.0"
description = "Zero-shot Postprocessing Benchmark: frozen TSFMs vs trained EMOS/DRN on EUPPBench"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26.4",
    "pandas>=2.2",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/zeropp"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
results/*.parquet
results/*.json
figures/*.png
!results/.gitkeep
!figures/.gitkeep
.DS_Store
```

- [ ] **Step 4: Create empty package `__init__.py` files and `.gitkeep` placeholders**

```bash
cd /Users/farukavci/zeropp
mkdir -p src/zeropp/data src/zeropp/models/wrappers src/zeropp/eval tests notebooks results figures scripts configs/models
touch src/zeropp/__init__.py src/zeropp/data/__init__.py src/zeropp/models/__init__.py \
      src/zeropp/models/wrappers/__init__.py src/zeropp/eval/__init__.py tests/__init__.py
touch notebooks/.gitkeep results/.gitkeep figures/.gitkeep
```

- [ ] **Step 5: Install the package in editable mode and verify it imports**

```bash
cd /Users/farukavci/zeropp
python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import zeropp; print('zeropp OK')"
```

Expected: `zeropp OK` printed, no import errors.

- [ ] **Step 6: git init and first commit**

```bash
cd /Users/farukavci/zeropp
git init
git add CLAUDE.md pyproject.toml .gitignore src tests notebooks/.gitkeep results/.gitkeep figures/.gitkeep
git commit -m "scaffold: repo skeleton, CLAUDE.md brief, editable package"
```

---

## Task 2: Configs and typed config loader

**Files:**
- Create: `/Users/farukavci/zeropp/configs/experiment.yaml`
- Create: `/Users/farukavci/zeropp/configs/data.yaml`
- Create: `/Users/farukavci/zeropp/src/zeropp/config.py`
- Test: `/Users/farukavci/zeropp/tests/test_config.py`

**Interfaces:**
- Consumes: nothing (reads YAML files directly).
- Produces: `zeropp.config.load_experiment_config(path=None) -> ExperimentConfig` and `zeropp.config.load_data_config(path=None) -> DataConfig`, both dataclasses. `ExperimentConfig.quantile_levels: list[float]`, `ExperimentConfig.data_size_days: list[int | str]`, `ExperimentConfig.seeds: list[int]`. `DataConfig.targets: list[str]`, `DataConfig.max_lead_hours: int`, `DataConfig.split_name: str`. Every later task that needs quantile levels imports `ExperimentConfig.quantile_levels` — never redefines the list.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from zeropp.config import load_experiment_config, load_data_config


def test_experiment_config_quantile_levels():
    cfg = load_experiment_config()
    assert cfg.quantile_levels == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_experiment_config_data_size_axis():
    cfg = load_experiment_config()
    assert cfg.data_size_days == [0, 30, 90, 365, 1095, "full"]


def test_experiment_config_seeds_present():
    cfg = load_experiment_config()
    assert len(cfg.seeds) >= 1
    assert all(isinstance(s, int) for s in cfg.seeds)


def test_data_config_targets():
    cfg = load_data_config()
    assert cfg.targets == ["t2m", "w10"]
    assert cfg.split_name == "euppbench_default"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zeropp.config'` (or `ImportError`).

- [ ] **Step 3: Create `configs/experiment.yaml`**

```yaml
quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
data_size_days: [0, 30, 90, 365, 1095, "full"]
seeds: [0, 1, 2]
```

- [ ] **Step 4: Create `configs/data.yaml`**

```yaml
euppbench_version: "v1.0"
targets: ["t2m", "w10"]
max_lead_hours: 120
split_name: "euppbench_default"
```

- [ ] **Step 5: Write `src/zeropp/config.py`**

```python
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"


@dataclass(frozen=True)
class ExperimentConfig:
    quantile_levels: list[float]
    data_size_days: list[int | str]
    seeds: list[int]


@dataclass(frozen=True)
class DataConfig:
    euppbench_version: str
    targets: list[str]
    max_lead_hours: int
    split_name: str


def load_experiment_config(path: Path | None = None) -> ExperimentConfig:
    path = path or DEFAULT_EXPERIMENT_CONFIG
    raw = yaml.safe_load(path.read_text())
    return ExperimentConfig(
        quantile_levels=raw["quantile_levels"],
        data_size_days=raw["data_size_days"],
        seeds=raw["seeds"],
    )


def load_data_config(path: Path | None = None) -> DataConfig:
    path = path or DEFAULT_DATA_CONFIG
    raw = yaml.safe_load(path.read_text())
    return DataConfig(
        euppbench_version=raw["euppbench_version"],
        targets=raw["targets"],
        max_lead_hours=raw["max_lead_hours"],
        split_name=raw["split_name"],
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
cd /Users/farukavci/zeropp
git add configs/experiment.yaml configs/data.yaml src/zeropp/config.py tests/test_config.py
git commit -m "feat: experiment/data configs and typed loader"
```

---

## Task 3: `Postprocessor` ABC

**Files:**
- Create: `/Users/farukavci/zeropp/src/zeropp/models/base.py`
- Test: `/Users/farukavci/zeropp/tests/test_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `zeropp.models.base.Postprocessor` (ABC) with abstract methods `fit(self, train) -> "Postprocessor"` and `predict_quantiles(self, X) -> np.ndarray` returning shape `(n_samples, n_leads, n_quantiles)`. Every model task from here on (Task 7, and every Task-8 stub) subclasses this exact ABC with these exact method names.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base.py
import numpy as np
import pytest

from zeropp.models.base import Postprocessor


class _DummyPostprocessor(Postprocessor):
    def fit(self, train):
        return self

    def predict_quantiles(self, X):
        n_samples, n_leads, n_quantiles = len(X), 2, 9
        return np.zeros((n_samples, n_leads, n_quantiles))


def test_postprocessor_is_abstract():
    with pytest.raises(TypeError):
        Postprocessor()


def test_concrete_subclass_fit_returns_self():
    model = _DummyPostprocessor()
    fitted = model.fit(train=[1, 2, 3])
    assert fitted is model


def test_concrete_subclass_predict_shape():
    model = _DummyPostprocessor().fit(train=None)
    preds = model.predict_quantiles(X=[0, 1, 2, 3])
    assert preds.shape == (4, 2, 9)


def test_missing_predict_quantiles_raises_typeerror():
    class _Incomplete(Postprocessor):
        def fit(self, train):
            return self

    with pytest.raises(TypeError):
        _Incomplete()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zeropp.models.base'`.

- [ ] **Step 3: Write `src/zeropp/models/base.py`**

```python
from abc import ABC, abstractmethod

import numpy as np


class Postprocessor(ABC):
    """Common contract for every baseline and TSFM model in ZeroPP.

    predict_quantiles must return an array of shape
    (n_samples, n_leads, n_quantiles) at the quantile levels defined in
    configs/experiment.yaml — never a model-specific shape.
    """

    @abstractmethod
    def fit(self, train) -> "Postprocessor":
        raise NotImplementedError

    @abstractmethod
    def predict_quantiles(self, X) -> np.ndarray:
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_base.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/models/base.py tests/test_base.py
git commit -m "feat: Postprocessor ABC — the one interface every model shares"
```

---

## Task 4: QC pipeline — gaps, DST/UTC, interpolation

**Files:**
- Create: `/Users/farukavci/zeropp/src/zeropp/data/qc.py`
- Test: `/Users/farukavci/zeropp/tests/test_qc.py`

**Interfaces:**
- Consumes: nothing beyond pandas/numpy.
- Produces:
  - `zeropp.data.qc.detect_gaps(series: pd.Series, freq: str) -> pd.DatetimeIndex` — timestamps where a gap starts (missing expected step at `freq`).
  - `zeropp.data.qc.to_utc(series: pd.Series) -> pd.Series` — normalizes a tz-aware or tz-naive-local series to UTC, raising `ValueError` if the index has no timezone info and cannot be inferred as already UTC.
  - `zeropp.data.qc.interpolate_gaps(series: pd.Series, freq: str) -> pd.Series` — reindexes onto the full `freq` grid and fills NaNs (including newly introduced ones from reindexing) via linear interpolation. Raises `ValueError` if NaNs remain at the series edges (linear interpolation cannot extrapolate) so callers never silently ship a model input with leading/trailing gaps.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qc.py
import numpy as np
import pandas as pd
import pytest

from zeropp.data.qc import detect_gaps, interpolate_gaps, to_utc


def test_detect_gaps_finds_missing_hour():
    idx = pd.date_range("2026-01-01", periods=5, freq="h").delete(2)
    series = pd.Series(np.arange(4, dtype=float), index=idx)
    gaps = detect_gaps(series, freq="h")
    assert pd.Timestamp("2026-01-01 02:00") in gaps


def test_detect_gaps_empty_when_continuous():
    idx = pd.date_range("2026-01-01", periods=5, freq="h")
    series = pd.Series(np.arange(5, dtype=float), index=idx)
    gaps = detect_gaps(series, freq="h")
    assert len(gaps) == 0


def test_interpolate_gaps_fills_missing_hour_linearly():
    idx = pd.date_range("2026-01-01", periods=5, freq="h").delete(2)
    series = pd.Series([0.0, 1.0, 3.0, 4.0], index=idx)
    filled = interpolate_gaps(series, freq="h")
    assert len(filled) == 5
    assert filled.loc["2026-01-01 02:00"] == pytest.approx(2.0)


def test_interpolate_gaps_raises_on_edge_nan():
    idx = pd.date_range("2026-01-01", periods=5, freq="h")
    series = pd.Series([np.nan, 2.0, 3.0, 4.0, 5.0], index=idx)
    with pytest.raises(ValueError, match="edge"):
        interpolate_gaps(series, freq="h")


def test_to_utc_converts_local_tz_to_utc():
    idx = pd.date_range("2026-03-29 00:00", periods=3, freq="h", tz="Europe/Istanbul")
    series = pd.Series([1.0, 2.0, 3.0], index=idx)
    converted = to_utc(series)
    assert str(converted.index.tz) == "UTC"


def test_to_utc_raises_on_naive_index():
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    series = pd.Series([1.0, 2.0, 3.0], index=idx)
    with pytest.raises(ValueError, match="tz"):
        to_utc(series)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zeropp.data.qc'`.

- [ ] **Step 3: Write `src/zeropp/data/qc.py`**

```python
import pandas as pd


def detect_gaps(series: pd.Series, freq: str) -> pd.DatetimeIndex:
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    missing = full_index.difference(series.index)
    return missing


def to_utc(series: pd.Series) -> pd.Series:
    if series.index.tz is None:
        raise ValueError(
            "series index has no tz info; DST/UTC ambiguity cannot be resolved "
            "implicitly — localize the source timezone explicitly before calling to_utc"
        )
    converted = series.copy()
    converted.index = converted.index.tz_convert("UTC")
    return converted


def interpolate_gaps(series: pd.Series, freq: str) -> pd.Series:
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    reindexed = series.reindex(full_index)
    filled = reindexed.interpolate(method="linear", limit_area="inside")
    if filled.isna().any():
        raise ValueError(
            "edge NaNs remain after linear interpolation — series starts or ends "
            "with missing values, which interpolation cannot fill; trim or extend "
            "the context window instead"
        )
    return filled
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_qc.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/data/qc.py tests/test_qc.py
git commit -m "feat: QC pipeline — gap detection, UTC normalization, edge-safe interpolation"
```

---

## Task 5: Eval scores — CRPS, pinball, MAE, twCRPS

**Files:**
- Create: `/Users/farukavci/zeropp/src/zeropp/eval/scores.py`
- Test: `/Users/farukavci/zeropp/tests/test_scores.py`

**Interfaces:**
- Consumes: quantile levels as a plain `list[float]` argument (caller passes `ExperimentConfig.quantile_levels` from Task 2 — this module does not import config itself, keeping it a pure function library).
- Produces:
  - `pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, tau: float) -> np.ndarray` — elementwise loss, same shape as inputs.
  - `crps_from_quantiles(y_true: np.ndarray, quantile_preds: np.ndarray, quantile_levels: list[float]) -> np.ndarray` — `y_true` shape `(n_samples, n_leads)`, `quantile_preds` shape `(n_samples, n_leads, n_quantiles)`, returns `(n_samples, n_leads)` CRPS approximation via the pinball-loss identity `CRPS ≈ 2 · mean_tau(pinball_tau)`.
  - `mae_from_quantiles(y_true, quantile_preds, quantile_levels) -> np.ndarray` — MAE using the quantile level closest to 0.5 as the point forecast.
  - `twcrps_from_quantiles(y_true, quantile_preds, quantile_levels, tail_levels=(0.8, 0.9)) -> np.ndarray` — same identity as `crps_from_quantiles` but averaged only over `tail_levels`, documented as an upper-tail-weighted approximation (not the full Gneiting–Ranjan chaining functional, which needs a continuous CDF this 9-point quantile forecast doesn't have).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scores.py
import numpy as np
import pytest

from zeropp.eval.scores import (
    crps_from_quantiles,
    mae_from_quantiles,
    pinball_loss,
    twcrps_from_quantiles,
)

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_pinball_loss_zero_at_perfect_forecast():
    y_true = np.array([5.0, 5.0])
    q_pred = np.array([5.0, 5.0])
    loss = pinball_loss(y_true, q_pred, tau=0.5)
    assert np.allclose(loss, 0.0)


def test_pinball_loss_asymmetric_penalty():
    y_true = np.array([10.0])
    q_pred = np.array([8.0])
    low_tau_loss = pinball_loss(y_true, q_pred, tau=0.1)
    high_tau_loss = pinball_loss(y_true, q_pred, tau=0.9)
    assert high_tau_loss > low_tau_loss


def test_crps_zero_for_degenerate_perfect_quantiles():
    y_true = np.full((3, 2), 5.0)
    quantile_preds = np.full((3, 2, len(QUANTILE_LEVELS)), 5.0)
    crps = crps_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS)
    assert crps.shape == (3, 2)
    assert np.allclose(crps, 0.0)


def test_crps_positive_for_wrong_forecast():
    y_true = np.full((2, 1), 10.0)
    quantile_preds = np.tile(
        np.array([QUANTILE_LEVELS]) * 0 + 5.0, (2, 1, 1)
    ).reshape(2, 1, len(QUANTILE_LEVELS))
    crps = crps_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS)
    assert np.all(crps > 0)


def test_mae_uses_median_quantile():
    y_true = np.array([[7.0]])
    quantile_preds = np.arange(1, 10, dtype=float).reshape(1, 1, 9)  # median (tau=0.5) is index 4 -> value 5.0
    mae = mae_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS)
    assert mae.shape == (1, 1)
    assert mae[0, 0] == pytest.approx(2.0)  # |7 - 5|


def test_twcrps_only_uses_tail_levels():
    y_true = np.full((1, 1), 5.0)
    quantile_preds = np.full((1, 1, 9), 5.0)
    twcrps = twcrps_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS, tail_levels=(0.8, 0.9))
    assert twcrps.shape == (1, 1)
    assert np.allclose(twcrps, 0.0)


def test_twcrps_raises_on_level_not_in_quantile_levels():
    y_true = np.full((1, 1), 5.0)
    quantile_preds = np.full((1, 1, 9), 5.0)
    with pytest.raises(ValueError, match="tail_levels"):
        twcrps_from_quantiles(y_true, quantile_preds, QUANTILE_LEVELS, tail_levels=(0.95,))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scores.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zeropp.eval.scores'`.

- [ ] **Step 3: Write `src/zeropp/eval/scores.py`**

```python
import numpy as np


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, tau: float) -> np.ndarray:
    diff = y_true - q_pred
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff)


def _quantile_index(quantile_levels: list[float], levels: tuple[float, ...]) -> list[int]:
    indices = []
    for level in levels:
        if level not in quantile_levels:
            raise ValueError(
                f"tail_levels entry {level} is not in quantile_levels {quantile_levels}"
            )
        indices.append(quantile_levels.index(level))
    return indices


def _mean_pinball_over_levels(
    y_true: np.ndarray, quantile_preds: np.ndarray, quantile_levels: list[float], indices: list[int]
) -> np.ndarray:
    losses = [
        pinball_loss(y_true, quantile_preds[..., i], tau=quantile_levels[i]) for i in indices
    ]
    return np.mean(np.stack(losses, axis=-1), axis=-1)


def crps_from_quantiles(
    y_true: np.ndarray, quantile_preds: np.ndarray, quantile_levels: list[float]
) -> np.ndarray:
    indices = list(range(len(quantile_levels)))
    return 2.0 * _mean_pinball_over_levels(y_true, quantile_preds, quantile_levels, indices)


def mae_from_quantiles(
    y_true: np.ndarray, quantile_preds: np.ndarray, quantile_levels: list[float]
) -> np.ndarray:
    median_index = _quantile_index(quantile_levels, (0.5,))[0]
    return np.abs(y_true - quantile_preds[..., median_index])


def twcrps_from_quantiles(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantile_levels: list[float],
    tail_levels: tuple[float, ...] = (0.8, 0.9),
) -> np.ndarray:
    indices = _quantile_index(quantile_levels, tail_levels)
    return 2.0 * _mean_pinball_over_levels(y_true, quantile_preds, quantile_levels, indices)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scores.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/eval/scores.py tests/test_scores.py
git commit -m "feat: CRPS/pinball/MAE/twCRPS from quantile forecasts"
```

---

## Task 6: Calibration — PIT, coverage, reliability index

**Files:**
- Create: `/Users/farukavci/zeropp/src/zeropp/eval/calibration.py`
- Test: `/Users/farukavci/zeropp/tests/test_calibration.py`

**Interfaces:**
- Consumes: same `(n_samples, n_leads, n_quantiles)` shape convention from Task 5.
- Produces:
  - `pit_values(y_true: np.ndarray, quantile_preds: np.ndarray, quantile_levels: list[float]) -> np.ndarray` — shape `(n_samples, n_leads)`, each entry in `[0, 1]`, computed by linearly interpolating `y_true` into the quantile function per sample/lead (`np.interp`, which clips outside the quantile range to 0/1).
  - `pit_histogram(pit: np.ndarray, n_bins: int = 10) -> np.ndarray` — length-`n_bins` array of observed frequencies (sums to 1.0).
  - `empirical_coverage(y_true, quantile_preds, quantile_levels, lower=0.1, upper=0.9) -> float` — fraction of `y_true` falling within `[q_lower, q_upper]`.
  - `reliability_index(pit: np.ndarray, n_bins: int = 10) -> float` — `sqrt(mean((observed_freq - 1/n_bins)^2))` over the PIT histogram; 0 means perfectly uniform (well-calibrated).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibration.py
import numpy as np
import pytest

from zeropp.eval.calibration import (
    empirical_coverage,
    pit_histogram,
    pit_values,
    reliability_index,
)

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _flat_quantile_preds(n_samples, n_leads):
    base = np.array(QUANTILE_LEVELS) * 10  # quantiles at 1,2,...,9
    return np.tile(base, (n_samples, n_leads, 1))


def test_pit_values_at_median_quantile_is_near_half():
    quantile_preds = _flat_quantile_preds(1, 1)
    y_true = np.array([[5.0]])  # matches the tau=0.5 quantile value exactly
    pit = pit_values(y_true, quantile_preds, QUANTILE_LEVELS)
    assert pit.shape == (1, 1)
    assert pit[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_pit_values_clip_below_min_quantile_to_zero():
    quantile_preds = _flat_quantile_preds(1, 1)
    y_true = np.array([[-100.0]])
    pit = pit_values(y_true, quantile_preds, QUANTILE_LEVELS)
    assert pit[0, 0] == pytest.approx(0.1, abs=1e-6)


def test_pit_histogram_sums_to_one():
    pit = np.array([0.05, 0.15, 0.5, 0.95])
    hist = pit_histogram(pit, n_bins=10)
    assert hist.shape == (10,)
    assert hist.sum() == pytest.approx(1.0)


def test_empirical_coverage_full_when_all_inside_band():
    quantile_preds = _flat_quantile_preds(3, 1)
    y_true = np.array([[2.0], [5.0], [8.0]])
    coverage = empirical_coverage(y_true, quantile_preds, QUANTILE_LEVELS)
    assert coverage == pytest.approx(1.0)


def test_empirical_coverage_partial_when_some_outside_band():
    quantile_preds = _flat_quantile_preds(2, 1)
    y_true = np.array([[-5.0], [5.0]])  # first below q10=1.0, second inside
    coverage = empirical_coverage(y_true, quantile_preds, QUANTILE_LEVELS)
    assert coverage == pytest.approx(0.5)


def test_reliability_index_zero_for_uniform_pit():
    pit = np.linspace(0.0, 1.0, 1000, endpoint=False)
    idx = reliability_index(pit, n_bins=10)
    assert idx == pytest.approx(0.0, abs=1e-3)  # floating-point bin-boundary jitter, not exactly 0


def test_reliability_index_positive_for_skewed_pit():
    pit = np.full(100, 0.5)
    idx = reliability_index(pit, n_bins=10)
    assert idx > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zeropp.eval.calibration'`.

- [ ] **Step 3: Write `src/zeropp/eval/calibration.py`**

```python
import numpy as np

from zeropp.eval.scores import _quantile_index


def pit_values(
    y_true: np.ndarray, quantile_preds: np.ndarray, quantile_levels: list[float]
) -> np.ndarray:
    n_samples, n_leads = y_true.shape
    out = np.empty((n_samples, n_leads))
    tau = np.array(quantile_levels)
    for i in range(n_samples):
        for j in range(n_leads):
            q = quantile_preds[i, j, :]
            out[i, j] = np.interp(y_true[i, j], q, tau)
    return out


def pit_histogram(pit: np.ndarray, n_bins: int = 10) -> np.ndarray:
    counts, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    return counts / counts.sum()


def empirical_coverage(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantile_levels: list[float],
    lower: float = 0.1,
    upper: float = 0.9,
) -> float:
    lower_idx, upper_idx = _quantile_index(quantile_levels, (lower, upper))
    q_lower = quantile_preds[..., lower_idx]
    q_upper = quantile_preds[..., upper_idx]
    inside = (y_true >= q_lower) & (y_true <= q_upper)
    return float(np.mean(inside))


def reliability_index(pit: np.ndarray, n_bins: int = 10) -> float:
    observed = pit_histogram(pit, n_bins=n_bins)
    expected = 1.0 / n_bins
    return float(np.sqrt(np.mean((observed - expected) ** 2)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_calibration.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/eval/calibration.py tests/test_calibration.py
git commit -m "feat: PIT histogram, empirical coverage, reliability index"
```

---

## Task 7: First concrete baselines — climatology and persistence/raw-ensemble

**Files:**
- Create: `/Users/farukavci/zeropp/src/zeropp/models/climatology.py`
- Create: `/Users/farukavci/zeropp/src/zeropp/models/raw.py`
- Test: `/Users/farukavci/zeropp/tests/test_baselines.py`

**Interfaces:**
- Consumes: `zeropp.models.base.Postprocessor` (Task 3).
- Produces: `Climatology(quantile_levels: list[float])` and `RawEnsemble(quantile_levels: list[float])` and `Persistence(quantile_levels: list[float])`, all `Postprocessor` subclasses, all constructed with the quantile levels from config (never hardcoded internally) — this is the pattern every later baseline (EMOS, DRN, etc. in Task 8) will also follow.
  - `Climatology.fit(train: pd.Series)` stores per-quantile empirical values of `train`; `predict_quantiles(X)` broadcasts those same quantiles to every sample/lead in `X` (a `pd.DataFrame`-like with `len(X)` rows and a `n_leads` int attribute — for this task, `X` is a plain `dict` with keys `"n_samples"` and `"n_leads"`, since real feature frames arrive in Task 8 once EUPPBench is loaded).
  - `RawEnsemble.fit` is a no-op (returns self); `predict_quantiles(X)` expects `X` to already carry ensemble-quantile columns and passes them through unchanged — this class is the passthrough baseline for "ham ensemble" quantiles already computed upstream.
  - `Persistence.fit` is a no-op; `predict_quantiles(X)` replicates the last observed value across all quantile levels (a degenerate, zero-spread quantile forecast) — `X` is a `dict` with key `"last_observed"` (shape `(n_samples,)`) and `"n_leads"` (int).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import numpy as np
import pytest

from zeropp.models.base import Postprocessor
from zeropp.models.climatology import Climatology
from zeropp.models.raw import Persistence, RawEnsemble

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_climatology_is_a_postprocessor():
    assert issubclass(Climatology, Postprocessor)


def test_climatology_fit_then_predict_shape():
    model = Climatology(quantile_levels=QUANTILE_LEVELS).fit(train=np.arange(1, 101, dtype=float))
    preds = model.predict_quantiles({"n_samples": 5, "n_leads": 3})
    assert preds.shape == (5, 3, 9)


def test_climatology_predict_matches_empirical_quantiles():
    train = np.arange(1, 101, dtype=float)  # 1..100
    model = Climatology(quantile_levels=QUANTILE_LEVELS).fit(train=train)
    preds = model.predict_quantiles({"n_samples": 1, "n_leads": 1})
    expected_median = np.quantile(train, 0.5)
    median_idx = QUANTILE_LEVELS.index(0.5)
    assert preds[0, 0, median_idx] == pytest.approx(expected_median)


def test_climatology_broadcasts_same_quantiles_to_every_sample_and_lead():
    train = np.arange(1, 101, dtype=float)
    model = Climatology(quantile_levels=QUANTILE_LEVELS).fit(train=train)
    preds = model.predict_quantiles({"n_samples": 4, "n_leads": 2})
    assert np.allclose(preds[0, 0], preds[3, 1])


def test_raw_ensemble_is_passthrough():
    model = RawEnsemble(quantile_levels=QUANTILE_LEVELS).fit(train=None)
    ensemble_quantiles = np.random.rand(3, 2, 9)
    preds = model.predict_quantiles({"ensemble_quantiles": ensemble_quantiles})
    assert np.array_equal(preds, ensemble_quantiles)


def test_persistence_replicates_last_value_across_quantiles():
    model = Persistence(quantile_levels=QUANTILE_LEVELS).fit(train=None)
    last_observed = np.array([5.0, 7.0])
    preds = model.predict_quantiles({"last_observed": last_observed, "n_leads": 3})
    assert preds.shape == (2, 3, 9)
    assert np.all(preds[0] == 5.0)
    assert np.all(preds[1] == 7.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zeropp.models.climatology'`.

- [ ] **Step 3: Write `src/zeropp/models/climatology.py`**

```python
import numpy as np

from zeropp.models.base import Postprocessor


class Climatology(Postprocessor):
    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels
        self._empirical_quantiles: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> "Climatology":
        self._empirical_quantiles = np.quantile(train, self.quantile_levels)
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        n_samples, n_leads = X["n_samples"], X["n_leads"]
        base = self._empirical_quantiles
        return np.tile(base, (n_samples, n_leads, 1))
```

- [ ] **Step 4: Write `src/zeropp/models/raw.py`**

```python
import numpy as np

from zeropp.models.base import Postprocessor


class RawEnsemble(Postprocessor):
    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "RawEnsemble":
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        return X["ensemble_quantiles"]


class Persistence(Postprocessor):
    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "Persistence":
        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        last_observed = X["last_observed"]
        n_leads = X["n_leads"]
        n_quantiles = len(self.quantile_levels)
        return np.tile(last_observed[:, None, None], (1, n_leads, n_quantiles))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_baselines.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/farukavci/zeropp
git add src/zeropp/models/climatology.py src/zeropp/models/raw.py tests/test_baselines.py
git commit -m "feat: climatology, raw-ensemble passthrough, and persistence baselines"
```

---

## Task 8: SSH/data-blocked stubs, real setup script, and blocked-work manifest

**Files:**
- Create: `/Users/farukavci/zeropp/scripts/00_setup_env.sh` (real, runnable)
- Create: `/Users/farukavci/zeropp/scripts/01_download_data.sh` (stub)
- Create: `/Users/farukavci/zeropp/scripts/02_build_dataset.py` (stub)
- Create: `/Users/farukavci/zeropp/scripts/03_run_baselines.py` (stub)
- Create: `/Users/farukavci/zeropp/scripts/04_run_tsfm.py` (stub)
- Create: `/Users/farukavci/zeropp/scripts/05_data_size_sweep.py` (stub)
- Create: `/Users/farukavci/zeropp/scripts/06_make_report.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/data/download.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/data/build.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/data/splits.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/models/emos.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/models/qrf.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/models/drn.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/models/mos_rf.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/models/tsfm_timesfm.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/models/tsfm_chronos.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/models/tsfm_moirai.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/models/wrappers/conformal.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/models/wrappers/gpd_tail.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/models/wrappers/qavg.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/eval/tables.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/eval/figures.py` (stub)
- Create: `/Users/farukavci/zeropp/src/zeropp/cli.py` (stub)
- Create: `/Users/farukavci/zeropp/configs/models/emos.yaml` (stub config)
- Create: `/Users/farukavci/zeropp/configs/models/timesfm.yaml` (stub config)
- Create: `/Users/farukavci/zeropp/docs/PHASE2_BLOCKED.md`
- Test: `/Users/farukavci/zeropp/tests/test_stubs_raise_clearly.py`

**Interfaces:**
- Consumes: `zeropp.models.base.Postprocessor` (Task 3) — every stub model class still subclasses it so `isinstance`/`issubclass` checks in future eval code don't need special-casing "not implemented yet" models.
- Produces: nothing new consumed by other tasks in this plan — this task exists so the full repo tree from CLAUDE.md's spec exists on disk today, with every not-yet-buildable piece failing loudly and specifically instead of silently.

- [ ] **Step 1: Write the failing test for stub behavior**

```python
# tests/test_stubs_raise_clearly.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stubs_raise_clearly.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zeropp.data.download'`.

- [ ] **Step 3: Write the real, runnable `scripts/00_setup_env.sh`**

```bash
#!/usr/bin/env bash
# Run this ON THE SSH SERVER once host/credentials are known. Not runnable locally —
# installs the heavy TSFM/baseline stack this laptop scaffold intentionally excludes.
set -euo pipefail

mkdir -p ~/zeropp && cd ~/zeropp
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate

uv pip install "timesfm[torch]" numpy pandas xarray netcdf4 zarr \
  scikit-learn scipy properscoring pyarrow matplotlib hydra-core
uv pip install statsmodels torch
uv pip install climetlab climetlab-eumetnet-postprocessing-benchmark

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

- [ ] **Step 4: Write the blocked stub scripts (identical pattern, one per file)**

```python
# scripts/01_download_data.sh is a shell stub — see content below
```

```bash
#!/usr/bin/env bash
# BLOCKED: needs SSH host access and outbound network to Zenodo/GitHub for
# EUPP-benchmark/climetlab-eumetnet-postprocessing-benchmark. Ask the user for
# SSH connection details before implementing this.
echo "blocked: SSH host + EUPPBench network access not yet configured" >&2
exit 1
```

For `scripts/02_build_dataset.py` through `scripts/06_make_report.py`, use this identical body (swap the docstring reason per file):

```python
"""BLOCKED: needs EUPPBench data on disk (see 01_download_data.sh) before this
script can be implemented. Do not fill this in with placeholder logic —
implement for real once 02_build_dataset.py output exists."""

import sys


def main() -> None:
    raise NotImplementedError(
        "blocked: needs EUPPBench parquet output from 02_build_dataset.py"
    )


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write the blocked stub library modules (identical pattern)**

`src/zeropp/data/download.py`:

```python
def download_euppbench(target_dir: str) -> None:
    raise NotImplementedError(
        "blocked: needs SSH host access and network access to Zenodo DOI "
        "10.5281/zenodo.7429236 / EUPP-benchmark GitHub — ask the user for "
        "SSH connection details before implementing"
    )
```

`src/zeropp/data/build.py`:

```python
def build_long_format_parquet(raw_dir: str, out_path: str) -> None:
    raise NotImplementedError(
        "blocked: needs downloaded EUPPBench raw files from download.py"
    )
```

`src/zeropp/data/splits.py`:

```python
def load_euppbench_split(data_dir: str):
    raise NotImplementedError(
        "blocked: needs built parquet dataset from build.py; when implemented, "
        "this MUST reuse EUPPBench's own train/test split — never re-split"
    )
```

For each of `src/zeropp/models/{emos,qrf,drn,mos_rf}.py`, follow this pattern (example for `emos.py`, others swap the class name and docstring citation):

```python
from zeropp.models.base import Postprocessor


class EMOS(Postprocessor):
    """Ensemble Model Output Statistics baseline.

    BLOCKED: needs EUPPBench training data (splits.py) to implement fit().
    Do not implement with synthetic data — the CRPS-minimization fit only
    means something against real ensemble/observation pairs.
    """

    def __init__(self, quantile_levels: list[float]):
        self.quantile_levels = quantile_levels

    def fit(self, train) -> "EMOS":
        raise NotImplementedError("blocked: needs EUPPBench training data via splits.py")

    def predict_quantiles(self, X):
        raise NotImplementedError("blocked: needs EUPPBench training data via splits.py")
```

For each of `src/zeropp/models/tsfm_{timesfm,chronos,moirai}.py` (example `tsfm_timesfm.py`):

```python
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
```

For `src/zeropp/models/wrappers/{conformal,gpd_tail,qavg}.py`, `src/zeropp/eval/{tables,figures}.py`, and `src/zeropp/cli.py`, use the same one-function-raises-NotImplementedError-with-a-specific-reason pattern — each citing what upstream output it needs (e.g. `gpd_tail.py` needs real TSFM tail quantile output; `tables.py`/`figures.py` need real `results/*.parquet`; `cli.py` needs the scripts it would wrap to be real).

- [ ] **Step 6: Write `configs/models/emos.yaml` and `configs/models/timesfm.yaml` stub configs**

```yaml
# configs/models/emos.yaml
name: emos
group: classic
fit_method: crps_minimization
# BLOCKED: hyperparameters TBD once real EUPPBench data shape is known.
```

```yaml
# configs/models/timesfm.yaml
name: timesfm-3
group: tsfm_zero_shot
checkpoint: google/timesfm-3.0-pytorch
frozen: true
covariates:
  past_future: [ensemble_mean, ensemble_spread]
# BLOCKED: exact covariate field names TBD once EUPPBench build.py output schema exists.
```

- [ ] **Step 7: Write `docs/PHASE2_BLOCKED.md`**

```markdown
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
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_stubs_raise_clearly.py -v`
Expected: 3 passed.

- [ ] **Step 9: Run the full test suite to confirm nothing broke**

Run: `pytest -v`
Expected: all tests across every test file pass (Tasks 2–8 combined).

- [ ] **Step 10: Commit**

```bash
cd /Users/farukavci/zeropp
chmod +x scripts/00_setup_env.sh scripts/01_download_data.sh
git add scripts src/zeropp/data/download.py src/zeropp/data/build.py src/zeropp/data/splits.py \
        src/zeropp/models/emos.py src/zeropp/models/qrf.py src/zeropp/models/drn.py src/zeropp/models/mos_rf.py \
        src/zeropp/models/tsfm_timesfm.py src/zeropp/models/tsfm_chronos.py src/zeropp/models/tsfm_moirai.py \
        src/zeropp/models/wrappers src/zeropp/eval/tables.py src/zeropp/eval/figures.py src/zeropp/cli.py \
        configs/models docs/PHASE2_BLOCKED.md tests/test_stubs_raise_clearly.py
git commit -m "scaffold: SSH/data-blocked stubs raising explicit NotImplementedError + blocked-work manifest"
```

---

## Task 9: Top-level README and final verification

**Files:**
- Create: `/Users/farukavci/zeropp/README.md`

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing consumed by other code — this is the human-facing entry point.

- [ ] **Step 1: Write `README.md`**

```markdown
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

\`\`\`bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
\`\`\`

## License note

TimesFM 3.0 weights are distributed under the TimesFM Non-Commercial
License v1.0. Academic use (this project) is fine. Commercial use or
embedding in a product is NOT permitted under that license.
```

- [ ] **Step 2: Run the full test suite one final time**

Run: `pytest -v`
Expected: all tests pass, zero errors, zero skips.

- [ ] **Step 3: Verify the tree matches the CLAUDE.md spec**

```bash
cd /Users/farukavci/zeropp
find . -path ./.venv -prune -o -path ./.git -prune -o -type f -print | sort
```

Expected: every path listed in the "File Structure" section above is present.

- [ ] **Step 4: Commit**

```bash
cd /Users/farukavci/zeropp
git add README.md
git commit -m "docs: top-level README with phase status and license note"
```

---

## Self-Review Notes

- **Spec coverage:** `CLAUDE.md`'s five non-negotiable architecture rules map directly to Task 1 (file), Task 3 (rule 1: ABC), Task 2 (rule 2: quantile constant), Task 8's `results/` `.gitkeep` + PHASE2_BLOCKED note (rule 3: parquet+json+SHA — actual writing deferred since it needs real results), Task 1's notebooks/.gitkeep (rule 4), and Task 2's seed field (rule 5). Metrics list → Task 5 + Task 6. Baseline set → Task 7 (2 of 11) + Task 8 (remaining 9, explicitly blocked, not silently dropped). Data-size axis → Task 2's config. Environment/SSH script → Task 8. License note → Task 9. "Yapma" don'ts are enforced structurally: no split logic exists yet to violate EUPPBench's split (Task 8 stub), no notebook content was created, every score function returns multiple metrics rather than one, no new model architecture was invented (only ABC + baselines from the brief's own list), and TimesFM covariate injection is a stub, not an approximation.
- **Placeholder scan:** every Task-8 stub raises `NotImplementedError` with a specific, checkable reason string (asserted by tests in Step 1 of Task 8) — this is a deliberate scaffolding pattern the plan calls out explicitly, not a "TBD" left for later without a test.
- **Type consistency:** `Postprocessor.fit(self, train) -> "Postprocessor"` and `predict_quantiles(self, X) -> np.ndarray` (Task 3) are the exact signatures reused verbatim by `Climatology`/`RawEnsemble`/`Persistence` (Task 7) and every stub model (Task 8). `quantile_levels: list[float]` is the constructor argument name used consistently from Task 2 onward. The `(n_samples, n_leads, n_quantiles)` output shape is used identically in Task 5's docstrings, Task 6's functions, and Task 7's tests.
