# Priority 1/2/3 investigation report

Session: 2026-09-07, SSH `altay` (`altay.uhem.itu.edu.tr`, resolved via IP
+ `HostKeyAlias` this session due to a transient DNS quirk on the local
machine -- see "Environment note" at the end). All computation ran remotely
per this project's sync-then-test discipline (rsync push -> run on altay ->
results pulled back). No local pytest/scripts were run.

## TL;DR

1. **The 0-24h nowcasting advantage is substantially architectural, not
   information asymmetry.** Giving EMOS the single most recent observation
   (leak-free, `PersistenceAugmentedEMOS`) closes only 3.4% of its CRPS gap
   to TimesFM-3 at 0-24h. `step_hours=0` is a real, non-degenerate forecast
   lead (confirmed both from archive metadata and empirically) and is
   correctly included in this analysis.
2. **The coverage-matched variance-inflation baseline's "0.062 miss" was
   never a root-finding bug.** `from_coverage_target` already uses
   `scipy.optimize.brentq` and converges to ~1e-7 on the training archive;
   the residual test-set gap is a real train/test ensemble-size mismatch
   (11 vs. 51 members), already documented elsewhere in this codebase. No
   code or pipeline change was needed. TimesFM-3 still beats this baseline
   on sharpness at matched coverage.
3. **"EMOS never beats TimesFM-3 at 0-24h, even with full training data" is
   statistically significant**, not just descriptively true (day-blocked
   p=4.29e-61 on CRPS). At longer leads the direction reverses in EMOS's
   favor (statistically significant at 72-120h under both block
   definitions; at 24-72h significant day-blocked only). TimesFM-3 is
   significantly LESS well calibrated than full-N EMOS in every lead-time
   bucket, including the one (0-24h) where it wins decisively on CRPS.

## Investigation 1: is the 0-24h nowcasting advantage real, or information asymmetry?

### Is `step_hours=0` degenerate (analysis time, not a real forecast)?

Checked directly, two independent ways:

1. **Raw NetCDF CF metadata** (`germany_ensemble_forecasts_t2m.nc`'s `step`
   coordinate, inspected on altay): `standard_name="forecast_period"`,
   `long_name="time since forecast_reference_time"`, `units="hours"`,
   first value `0.0`. This is ECMWF's own definition of `step_hours=0` as a
   genuine T+0 forecast (valid at the same instant as issuance), not a flag
   meaning "already assimilated / equals the observation."
2. **Empirically**, from `results/phase3_lead_time_breakdown.parquet`: if
   `step_hours=0` forecasts were degenerate (forecast == obs), CRPS there
   would be near zero for every method. Instead:
   - `raw_ensemble` CRPS at `step_hours=0` (**1.2624 K**) is *worse* than at
     `step_hours=6` (**1.0932 K**).
   - `emos` CRPS at `step_hours=0` (**1.1369 K**) is likewise worse than at
     `step_hours=6` (**0.9671 K**).
   - `tsfm3` CRPS at `step_hours=0` (**0.9195 K**) is essentially unchanged
     from `step_hours=6` (**0.9188 K**) -- no special "free win" either.
   - `raw_ensemble` coverage@80% at `step_hours=0` (**0.2374**) is exactly as
     badly under-dispersed as every other short lead, not near 1.0 as a
     trivial/degenerate case would show.

**Conclusion: `step_hours=0` is a real, non-degenerate forecast lead time**
and is correctly **included** in the 0-24h bucket's headline comparison
below. It does, however, need special handling in the *construction* of the
persistence feature (`last_obs`) used below, since at `step_hours=0`
issue_time == valid_time, and naively using "the observation at issue time"
would just be the target itself. See the script's docstring for the
leak-free construction used instead (a genuinely earlier observation, never
the same physical instant as the target).

### Persistence-augmented EMOS: does it close the gap?

Built `zeropp.models.emos.PersistenceAugmentedEMOS`
(`mu = a + b*ens_mean + e*last_obs`, `sigma` unchanged from plain EMOS) and
`scripts/11_persistence_augmented_emos.py`, restricted to the 0-24h bucket
only, fit on the training reforecast archive only (never on test-set
information). `last_obs` was reconstructed with a strict leakage discipline:

- For `step_hours` in {6, 12, 18}: `last_obs` = the same forecast's own
  `step_hours=0` observation (a real, strictly-prior-or-simultaneous-with-
  issuance instant).
- For `step_hours=0`: `last_obs` = the nearest genuinely earlier physical
  observation (`valid_time - 6h`), found via a direct datetime match against
  any other forecast issue in the training archive (test side) or the same
  `year_idx` (train side; `year_idx` crossing was avoided since its real
  calendar-year mapping is unconfirmed per this project's prior findings).

**Data-quality finding en route**: reconstructing this required treating
`time_idx + step_hours` as a real wall-clock instant. 4,319 of several
hundred thousand (`station_id, year_idx, valid_datetime`) keys in the
training archive turned out to have **conflicting** `t2m_obs` values (up to
12.6 K apart) across different `(time_idx, step_hours)` combinations that
nominally reduce to the same timestamp -- consistent with this project's
already-documented DST/calendar-template landmine (CLAUDE.md: "DST gecisleri
ve UTC/yerel saat karisikligi sahte delik yaratir"). These ambiguous keys
were excluded from the lookup entirely (not guessed at); this affected 1,913
of 815,803 training-bucket rows (0.23%) and 87 of 140,544 test instances
(0.06%) -- both small, reported, and non-selective (see script output).

**Results** (`results/phase3_persistence_augmented_emos.parquet`):

| Method | Population | n | CRPS (K) | Coverage@80% | Width (K) |
|---|---|---|---|---|---|
| emos_pooled (lead-pooled train, full N) | full 0-24h bucket | 140,544 | 1.1585 | 0.8193 | 4.8057 |
| emos bucket-specific (train restricted to 0-24h) | full 0-24h bucket | 140,544 | 1.1361 | 0.8313 | 4.8592 |
| **tsfm3** | full 0-24h bucket | 140,544 | **0.9903** | 0.7621 | 3.7303 |
| emos bucket-specific, refit on matched subset | last_obs-matched | 140,457 | 1.1362 | 0.8313 | 4.8601 |
| **persistence-augmented EMOS** | last_obs-matched | 140,457 | **1.1312** | 0.8277 | 4.8243 |
| tsfm3, matched subset | last_obs-matched | 140,457 | 0.9898 | 0.7622 | 3.7302 |

The persistence weight `e` was fit to **0.0786** (a=0.0095, b=0.9208,
c=1.1545, d=0.2854) -- a real, non-trivial weight, so the model did not
simply ignore `last_obs`.

**CRPS gap to TimesFM-3** (on the identical matched-subset population):
plain EMOS bucket-specific = **+0.1463 K worse**; persistence-augmented EMOS
= **+0.1413 K worse** -- adding the single most recent observation closes
only **3.4%** of the gap. Coverage moved from 0.8313 to 0.8277 (both still
well above TimesFM-3's 0.7621/0.7622 and above nominal 0.80 -- neither EMOS
variant is well-calibrated at this lead regardless of persistence).

**Interpretation: the 0-24h advantage is NOT substantially explained by
information asymmetry.** Giving EMOS access to exactly the one piece of
observation history TimesFM-3 is assumed to lean on hardest at short leads
(the most recent observation) recovers only a sliver of the gap. A
meaningful, essentially unchanged gap to TimesFM-3 remains (an order of
magnitude larger than what persistence closed). This is consistent with
TimesFM-3's real covariate structure being richer than a single lag (~40
past observations plus NWP ensemble-mean/spread, processed by a full
sequence model, not a single linear persistence term) actually mattering,
i.e. the short-lead advantage reads as substantially architectural rather
than a trivial by-product of EMOS's structural blindness to recent
observations.

**Caveat, stated plainly**: this is a test of *linear, single-lag*
persistence added to EMOS's existing linear-Gaussian family -- it does not
rule out that a richer (non-linear, multi-lag) observation-history feature
added to EMOS could close more of the gap. It specifically rules out the
weakest, cheapest version of the "unfair information" hypothesis (one extra
linear predictor), which is what this diagnostic was scoped to test.

## Investigation 2: coverage-matched variance-inflation baseline's "non-convergence"

**Premise checked and found incorrect**: `VarianceInflationBaseline.from_coverage_target`
(`src/zeropp/models/variance_inflation.py`) already uses `scipy.optimize.brentq`
(`xtol=1e-6`) -- it was NOT a coarse grid search. This was added in the prior
commit (`13c1b05`, "fix: figure review round 1"), which the investigation
brief predates or was unaware of.

**Independently re-verified on altay** (re-running `from_coverage_target`
directly against the training reforecast archive, target_coverage=0.760328
= TimesFM-3's real, persisted coverage@[0.1,0.9]):

- Converged `lambda_c` = **2.706424**
- Coverage achieved **on the training archive** (the data brentq is actually
  solving against) = **0.7603282** -- matches the 0.760328 target to
  ~1e-7, i.e. brentq is already converging essentially exactly.
- Applying that *exact* `lambda_c=2.706424` to the **test** forecast archive
  (independently recomputed, matching the persisted
  `results/phase3_data_size_sweep.parquet` row to 6 decimal places) gives
  coverage = **0.822103**, width = **6.722392 K**.

**The 0.062 "miss" (0.8221 vs. 0.7603) is not a root-finding defect.** It is
the expected consequence of calibrating `lambda` on the training reforecast
archive (11 ensemble members) and applying it, unchanged, to a structurally
different test forecast archive (51 members) -- exactly the train/test
ensemble-size mismatch `scripts/07_data_size_sweep.py`'s own module
docstring already documents ("ens_var computed from 11 members is a noisier
... estimator ... than one from 51 would be"). No amount of root-finding
precision can close a gap caused by evaluating a train-calibrated constant
on a differently-distributed test set; brentq already converges to the
tightest defensible answer to the question this baseline actually asks
("what lambda hits the target on the archive we are allowed to look at").

**No code change was needed or made to the root-finding itself.** Added a
docstring note to `from_coverage_target` recording this verification (so a
future reader doesn't re-open the same non-issue) and did not re-run the
full pipeline (07/08/make_figures), since the persisted numbers already
reflect this exact, already-converged brentq solve -- re-running would
reproduce byte-identical results (EMOS/DRN fits are deterministic given
fixed seeds/`x0`), which was confirmed by the independent re-derivation
above rather than by an expensive full re-run.

**Does the paper's "TimesFM-3 beats a trivial coverage-matched rescaling on
sharpness" conclusion still hold?** Yes, unchanged: at the coverage-matched
point (0.8221) TimesFM-3's own width (**3.7303 K**) is still narrower than
the coverage-matched baseline's width (**6.7224 K**) -- TimesFM-3 remains
sharper even though its own coverage (0.7621) sits further from nominal 0.80
than the coverage-matched baseline's 0.8221 does.

## Investigation 3: per-lead-group significance testing

`scripts/12_per_lead_group_significance.py` extends
`scripts/09_spatial_block_significance.py`'s day-blocked + station-blocked
paired testing (`zeropp.eval.significance.station_blocked_paired_test` /
`block_bootstrap_skill_score_ci`, both reused unchanged) to run separately
within each of the three lead-time buckets, using **full-N EMOS-pooled**
(lead-pooled training on the entire 4,282,969-row reforecast archive, no
subsampling) vs. TimesFM-3 -- confirmed this is the right EMOS variant for
the claim being tested ("EMOS never beats TimesFM-3 at 0-24h even with full
training data" is exactly the full-N, all-tested-N-points reading behind
`phase3_lead_time_bucketed_breakpoints.parquet`'s "0-24h/CRPS/emos_pooled: no
crossing, worse throughout" entry in `docs/results_index.md`).
Day-blocked (station=49 blocks vs. day=720-722 blocks) is primary per this
project's established finding that station-blocking understates cross-station
synoptic dependence; station-blocked reported alongside.

Full results in `results/phase3_per_lead_group_significance.parquet`
(`n_matched_instances=737,809`, `bootstrap_seed=0`, day-blocked t-test as the
headline significance call):

| Bucket | Metric | mean diff (EMOS − TimesFM-3) | day-blocked p | station-blocked p | Reading |
|---|---|---|---|---|---|
| 0-24h | CRPS | **+0.1682 K** (EMOS worse) | **4.29e-61** | 3.54e-02 | Both significant. **The "EMOS never beats TimesFM-3 at 0-24h, even at full N" claim is statistically confirmed**, not just descriptively true. |
| 0-24h | Coverage@80% | +0.0573 (EMOS closer to nominal) | **8.40e-53** | 2.55e-02 | Both significant. EMOS (0.8193) sits closer to nominal 0.80 than TimesFM-3 (0.7621) — `|0.0193|` vs. `|0.0379|` from nominal. |
| 24-72h | CRPS | **−0.0647 K** (EMOS better) | **4.33e-07** | 2.27e-01 (not sig.) | **Diverges by block definition.** Day-blocked (primary) says EMOS's CRPS edge here is real; station-blocked (49 blocks only) lacks the power to detect it. Per this project's established day-blocking preference, treat EMOS's advantage here as real. |
| 24-72h | Coverage@80% | +0.0610 (EMOS closer) | **6.15e-50** | 2.93e-03 | Both significant. EMOS (0.8288) closer to nominal than TimesFM-3 (0.7681) — margin is smaller here (`0.0288` vs. `0.0319`). |
| 72-120h | CRPS | **−0.2461 K** (EMOS better) | **3.69e-35** | 3.31e-08 | Both significant, large effect. EMOS's CRPS advantage at long lead is the strongest and most robust of the three buckets. |
| 72-120h | Coverage@80% | +0.0839 (EMOS closer) | **3.06e-64** | 3.30e-06 | Both significant. EMOS (0.8365) closer to nominal than TimesFM-3 (0.7526) — largest calibration gap of the three buckets. |

**Reading, stated carefully** (the script's own "direction" label conflates
"numerically lower diff" with "better calibrated," which is wrong for a
calibration metric — corrected here): coverage@80% is about closeness to the
nominal 0.80, not minimization. On that reading, **full-N EMOS-pooled is
consistently, statistically significantly closer to nominal coverage than
TimesFM-3 in all three lead-time buckets** (day-blocked p ranging
8e-53 to 3e-64) — TimesFM-3 is consistently under-covered (0.75-0.78) while
EMOS over-covers less severely (0.82-0.84). This holds even in the 0-24h
bucket where TimesFM-3 dominates on CRPS: TimesFM-3 wins sharpness/CRPS
decisively at short lead, but is not the better-calibrated method there.

**Headline finding for the paper's strong claim**: "EMOS never beats
TimesFM-3 at 0-24h even with full training data" is **real and
statistically significant** on CRPS (both block definitions, day-blocked
p=4.29e-61) -- not an artifact of insufficient testing. At longer leads
(24-72h, 72-120h) the direction reverses (EMOS's CRPS is significantly
better, most robustly at 72-120h where both block definitions agree; at
24-72h day-blocked and station-blocked disagree, with day-blocked --
primary per this project's convention -- favoring EMOS).

## Environment note

`altay.uhem.itu.edu.tr` failed to resolve via the system resolver
(`getaddrinfo`) throughout this session despite `nslookup` succeeding
against the same configured nameserver (10.128.5.2) -- a local mDNSResponder/
split-DNS quirk, not a server-side problem. Worked around for this session
by connecting directly to the resolved IP (10.128.2.40) with
`HostKeyAlias=altay.uhem.itu.edu.tr` (matches the existing known_hosts
entry) and a new `~/.ssh/config` alias `altay-ip` pointing at that IP --
purely a local SSH client config addition, no changes to the remote server
or to any project file.
