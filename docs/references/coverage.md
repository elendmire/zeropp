# ZeroPP reference coverage

Source: 14 Consensus searches run 2026-09-05, 193 unique DOIs harvested,
triaged down to the selection in `selected_dois.tsv`.

**Rule: nothing enters `refs.bib` except via `scripts/build_refs.sh`, which
fetches canonical BibTeX from doi.org. No hand-written entries. No entries
recalled from model memory.**

## Block status

| Block | Topic | Selected | Target | Status |
|---|---|---|---|---|
| B1 | Statistical postprocessing foundations | 8 | ~8 | near-complete, 1 gap |
| B2 | ML-based postprocessing | 13 | ~12 | near-complete, 1 gap |
| B3 | EUPPBench and work on it | 6 | ~6 | complete |
| B4 | Time series foundation models | 10 | ~10 | near-complete, 1 gap |
| B5 | TSFM benchmarking and contamination | 8 | ~6 | complete |
| B6 | TSFMs applied in geoscience | 0 | ~6 | **EMPTY, blocking** |
| B7 | Covariate-aware zero-shot forecasting | 7 | ~6 | complete |
| B8 | Probabilistic evaluation theory | 11 | ~9 | complete, 1 gap |
| B8b | Forecast comparison / significance | 5 | ~4 | near-complete, 1 gap |
| B9 | AI weather prediction models | 9 | ~7 | complete |
| B10 | Downscaling and multivariate alternatives | 8 | ~6 | complete |
| | **Total** | **85** | **~70** | trim LOW priority if over budget |

## Gaps that must be filled before drafting

These are known-essential works that did NOT appear in the harvested searches.
Each needs a targeted search, then its DOI appended to `selected_dois.tsv`.

| Block | Missing work | Why it is mandatory | Search query |
|---|---|---|---|
| B1 | Vannitsem et al. 2020, BAMS | The field's standard review; its absence reads as not knowing the literature | `statistical postprocessing weather forecasts review challenges big data` |
| B2 | Muschinski et al. 2023, NPG | MOS random forests; works with <100 training observations, so it is a direct rival in exactly our small-N regime | `MOS random forests weather adaptive postprocessing` |
| B4 | TimesFM 3.0 technical report / model card | We evaluate this exact model; citing only the 2023 TimesFM paper misattributes the architecture | not on Consensus; cite Google Research release + HF model card |
| B6 | Sun & Sun 2026, Machine Learning: Earth | The closest structural template in geoscience; "are we there yet?" framing | `zero-shot streamflow forecasting time series foundation models CAMELS` |
| B6 | Air Quality Arena 2026 | Large multi-station TSFM benchmark in an environmental domain | `air quality benchmark time series foundation models multi-country stations` |
| B6 | Wildfire PM2.5 generalizability 2026 | Finds trained BiLSTM beats every TSFM on extremes; important counterweight | `foundation model generalizability extreme environmental events wildfire PM2.5` |
| B6 | Rollo et al. 2026, Expert Syst. Appl. | Benchmarks TimesFM specifically on low-cost sensor PM2.5 | `foundation models particulate matter prediction low-cost sensors` |
| B6 | Gao et al. 2026, WWW | Training-free wrapper fixing zero-inflation for frozen TSFMs; the wrapper pattern we discuss | `training-free zero-inflation correction rainfall time series foundation models` |
| B8 | Hamill 2001, MWR | Rank histogram interpretation; our PIT diagnostics rest on it | `rank histogram ensemble verification interpretation` |
| B8b | Diebold & Mariano 1995, JBES | The original test. We cite Diebold 2015 retrospective but not the source | `comparing predictive accuracy Diebold Mariano 1995` |

## DOIs needing manual repair — RESOLVED 2026-09-05

| bibkey | Problem | Resolution |
|---|---|---|
| `hersbach_2000_crps` | DOI recorded as `10.1175/1520-0434(2000)015`, truncated. Legacy AMS DOIs carry an article-ID suffix. | Resolved via Crossref bibliographic search: full DOI is `10.1175/1520-0434(2000)015<0559:dotcrp>2.0.co;2`. `selected_dois.tsv` corrected, re-fetched — title/journal/author confirmed exact match ("Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems", *Weather and Forecasting*, 2000). |
| `coroneo_2024_strong_dependence` | DOI ends `...2024.11.00`, looks cut off. | Resolved via Crossref: full DOI is `10.1016/j.ijforecast.2024.11.003`. `selected_dois.tsv` corrected, re-fetched — confirmed "Testing for equal predictive accuracy with strong dependence", *International Journal of Forecasting*. |
| `ansari_2025_chronos2` | arXiv 2510.15821 assumed to be Chronos-2. | Confirmed: fetched title is exactly "Chronos-2: From Univariate to Universal Forecasting". No DOI change needed. |
| `mulayim_2025_can_tsfm` | Title truncated in harvest. | Fetched title is complete: "Can time-series foundation models perform building energy management tasks?". No DOI change needed. |
| `kreusel_2026_covariates_key` | Title truncated. | Fetched title is complete: "Covariates Are the Key to Accurate Probabilistic Building Energy Forecasting with Time Series Foundation Models". No DOI change needed. |

All 85 DOIs in `selected_dois.tsv` now resolve cleanly via `scripts/build_refs.sh` (`docs/references/failed_dois.txt` is empty).

**Bugs found and fixed in `scripts/build_refs.sh` while resolving the above** (none of these were pre-existing coverage.md items — found during this pass):
1. The publisher-key-to-bibkey normalisation used a GNU-sed-only construct (`0,/regex/` range address + empty-pattern `s//repl/` reuse) that silently no-ops on BSD/macOS sed — every fetched entry kept its raw publisher key instead of our normalised `\cite{}` key. Rewritten in `awk`, portable.
2. Calling the script with zero block-filter arguments (the documented "fetch everything" mode) silently fetched nothing, because the filter-check function's `$# -eq 0` test was checking the function's own argument count, which is never zero once the block name is prepended — not the script's original argument count. Fixed by capturing the script's real argument count in a variable before the loop.
3. **Most serious**: the "did this resolve?" check only verified that some line, anywhere in the response, started with `@` — a doi.org "DOI Not Found" HTML error page happens to contain such a line, so both `hersbach_2000_crps` and `coroneo_2024_strong_dependence`'s truncated DOIs were silently accepted as "successful" fetches, injecting ~230 lines of raw HTML into `refs.bib` under those bibkeys instead of failing loudly. Fixed with a much stricter check: HTTP status must be 200, the body must not contain `<!doctype`/`<html` anywhere, and the first non-blank character of the body must be `@`.

**Bug found and fixed in `scripts/check_references.py` while running it against the real `refs.bib`:** its BibTeX field parser (a single regex anchored per-line) only correctly parsed a multi-line, one-field-per-line layout. doi.org's actual Crossref output is single-line, every-field-comma-separated-on-one-line — a materially different real-world shape the parser had never been run against. Against the real file this silently swallowed every field of 59 of 85 entries into one bogus `title` value, including the `doi` field itself, so `check_references.py` reported 59 false "no DOI or arXiv id" errors on entries that genuinely have DOIs. Replaced with a brace-depth-aware comma splitter (`_split_top_level`/`_parse_fields`) that handles both layouts identically; added a regression test (`test_compact_single_line_crossref_style_entry_parses_correctly` in `tests/test_references.py`) using this project's real Demaeyer-2023 entry verbatim as the fixture. Real run against `refs.bib` after the fix: **0 errors, 1 warning** (the expected "no manuscript files found" — `paper/` doesn't exist yet).

## Findings in the harvest that bear on our own results

Flagged because they affect claims in the paper, not just the bibliography.

**`zhang_2025_context_parroting` — read before finalising the framing.**
"Context parroting: a simple but tough-to-beat baseline for foundation models
in scientific machine learning." This is the same argument as our
variance-inflation baseline: a trivial procedure competitive with a large
pretrained model. It is simultaneously a threat (the general point is already
published) and an asset (it gives our specific instance a home in the
literature). Our contribution is the postprocessing-specific instance with a
calibration axis, not the general observation.

**`kreusel_2026_covariates_key` — supports our covariate-wiring finding.**
Independent evidence that covariate handling dominates TSFM performance,
which is what our misconfigured first run demonstrated the hard way.

**`li_2026_tsfmaudit` and `pan_2026_contamination_electricity` — cite for the
contamination argument.** Our claim that station weather data is unlikely to
sit in TimesFM's pretraining corpus needs external grounding, not just
assertion.

**`coroneo_2020_small_samples` — reviewed against our actual significance test 2026-09-05; NOT a vulnerability, but a naming clarification is needed.**
The paper's abstract (Coroneo & Iacone 2020, *J. Applied Econometrics* 35(4),
391-405): standard Diebold-Mariano's HAC-type long-run-variance estimator is
size-distorted under "increasing-smoothing" asymptotics when there are few
out-of-sample observations in a *single autocorrelated loss-differential
series*; their fix ("fixed-smoothing asymptotics") holds the smoothing
bandwidth fixed as a fraction of sample size, giving a correctly-sized test
statistic even at small T.

Our `zeropp.eval.significance.station_blocked_paired_test` does not use this
class of estimator at all: it collapses the ~737,809 autocorrelated
station×time×lead instances to ONE mean per station block *first*, then runs
a plain paired t-test and Wilcoxon signed-rank test on those ~49
(approximately independent) block means. A paired t-test's null distribution
is exact under normality for any sample size -- it never relies on HAC/
long-run-variance asymptotics, so it is not exposed to the specific failure
mode this paper documents. (This is also exactly why the project's own docs
already refuse to call this test "Diebold-Mariano" -- see `significance.py`'s
docstring.)

One naming correction worth making explicit: "our k=9 comparison" is not the
sample the significance test runs on. k=9 is EMOS's *training*-set size (how
many reforecast cases the model was fit on); the significance test itself
always evaluates the fitted model's test-set performance aggregated to ~49
station means, regardless of k. The already-known, real fragility in our
approach is a different and more mundane one (flagged in Task 3/Task 6's own
findings): a handful of outlier stations can dominate the ~49-point t-test's
variance while the sign-majority still favors Wilcoxon significance,
producing t-vs-Wilcoxon disagreement -- a moderate-N t-test weakness, not
the small-T HAC-asymptotics problem Coroneo & Iacone address. No code change
needed here; reporting both p-values with this caveat (already the project's
practice) remains the right call.

**`gneiting_2007_calibration_sharpness` — the basis of finding 2.**
"Maximise sharpness subject to calibration" is the principle under which
local EMOS dominates TimesFM-3. Currently uncited in our drafts.

**`baran_2026_sharpness_penalty` — closest prior work on our sharpness axis,
and it is on EUPPBench.** Check for overlap before claiming novelty.

**`wessel_2024_leadtime_continuous` and Muschinski 2023 — the small-data
rivals.** Both target the regime our breakpoint curve explores. Neither is
currently a baseline in our code.
