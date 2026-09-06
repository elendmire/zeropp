# `scripts/make_figures.py` — implementation report

**Status: DONE.**

Two prior attempts at this exact dispatch made zero progress (one
rate-limited, one blocked on `altay` being unreachable over VPN). This
attempt started clean with no code to recover. `altay` connectivity was
verified working (`ssh altay 'echo ok'`) before any implementation work
began, and stayed working for the whole session — no blocking issues.

## Workflow followed

Sync-then-test throughout, one background job at a time, as mandated:

1. `rsync` the working tree (excluding `.venv/`, `data/`, `model_cache/`,
   `logs/`, `__pycache__`, `.git`, `*.sbatch`, `*.out`, `*.err`) from this
   Mac to `altay:~/zeropp/`. `altay`'s copy of `results/*.parquet` and
   `data/raw/*.nc` was already present and byte-identical to this Mac's
   (verified by file size); only source/doc/script files needed syncing.
2. All schema inspection, the PIT computation, and every figure render ran
   over SSH on `altay` inside its `.venv` (Python 3.11, matplotlib 3.11.1) —
   nothing ran locally.
3. Rendered figures were `rsync`'d back to this Mac and viewed with the
   Read tool (actual pixel content, not just "no error") before being
   accepted; several rounds of layout fixes were needed (see below).

## What was built

- **`scripts/make_figures.py`** — the deliverable. One `STYLE` dict; five
  independently callable `make_f1()`..`make_f5()` functions; a shared set of
  small helpers (`_load`, `_save_figure`, `_update_caption`, `_style_grid`,
  `_annotate_stack`, `_add_top_headroom`, `_days_per_case_from_data`, ...).
  `__main__`/`main()` just calls all five for convenience — no figure
  depends on another having run first.
- **`scripts/10_compute_pit_histograms.py`** — new. No PIT histogram data
  existed anywhere in `results/` before this task. `raw_ensemble`, `tsfm3`
  and `emos` (pooled, full-N) quantile predictions already existed in
  `results/phase2_comparison_raw.parquet` and are reused directly; `emos_local`
  at full N had never been fit or persisted anywhere in this project (the
  N-sweep in `07_data_size_sweep.py` fits it at every N but only ever
  persists aggregate metrics), so it is fit fresh here — once, at full N,
  following the same per-station / `LOCAL_EMOS_MIN_ROWS=5` / coverage-fraction
  pattern as `07_data_size_sweep.py`'s `fit_predict_local_emos` (duplicated,
  not imported, since `07_data_size_sweep.py` is a numerically-prefixed
  script module, not a package). All four methods are scored on the
  identical (station_id, valid_time, step_hours) instance set (tsfm3's own
  key set, matched via inner join, with fail-fast asserts). Ran once on
  `altay` (~15 min, dominated by loading the two ~150-200MB NetCDF archives
  and 49 per-station L-BFGS-B fits); persisted via `write_result` to
  `results/phase3_pit_histograms.parquet` (+ `.json` sidecar). Station
  coverage came back 100% (49/49 test stations had ≥5 training rows at full
  N), so `emos_local`'s PIT histogram uses the full 737,809-instance test set,
  identical in size to the other three methods.
- **`figures/f1_breakpoint_curve.{pdf,svg,png}`** through
  **`f5_breakpoint_by_lead_group.{pdf,svg,png}`**, plus **`figures/captions.md`**
  (auto-generated captions with every number read from `results/*.parquet`
  at render time — see `_update_caption`). PNGs are gitignored (existing repo
  convention, preview-only); PDF/SVG are tracked.

## Data sources and the breakpoint-file ambiguity

Per `docs/results_index.md`'s "rule of thumb": F1's and F3's `emos_pooled`/
`emos_local` breakpoints/trajectories combine `results/phase3_data_size_sweep.parquet`
(k=9..4180, contiguous arm) with `results/phase3_low_n_grid.parquet` (k=1,2,3,5,7);
F1's in-panel breakpoint annotations for `emos_pooled`/`emos_local` read from
`phase3_low_n_grid_breakpoints.parquet` (authoritative — the main-sweep file's
own breakpoints for these two methods are superseded, per the index), while
`drn` (not tested at low N) reads from `phase3_data_size_sweep_breakpoints.parquet`,
the only source for it. F5 reads `phase3_lead_time_bucketed_breakpoints.parquet`
directly (no low-N re-test exists for this file; F5's caption explicitly
flags that the 0-24h row's "no crossing" claims have not been re-tested below
k=9, per the index's caveat). F2's durable crossover (step_hours ≈ 43.5h) and
first sign flip (≈ 22.4h, caption-only, not annotated on the panel) both come
from `results/phase3_lead_time_crossover.parquet`.

## Deliberate interpretation calls (documented, not silent)

- **All five figures render at double-column width (190mm)**, not a mix of
  90mm/190mm. A first-draft render of F1/F3/F5 at 90mm (single-column)
  produced illegible, overlapping annotation text, trajectory labels, and
  clipped legends — the style guide's own sizing rule ("render at final size
  and read the tick labels before accepting a figure") ruled that draft out.
  190mm gives every figure's in-panel R2/R5 annotations and F1/F3's secondary
  axis / trajectory arrows room to be legible without overlap.
- **`var_inflation_trainfit` is not plotted in F1 or F3.** The style guide's
  palette table has exactly 6 data-series colors (raw/inflated/tsfm3/pooled/
  local/drn) plus one reference-line grey — matching F1's spec exactly (3
  horizontal + 3 curves = 6). `var_inflation_fixed` (genuinely zero-shot, no
  fit, N-independent — matches the palette's "Zero-shot methods... render as
  flat horizontal lines" description) is used as "the" variance-inflated
  baseline series. `var_inflation_trainfit` (fit per-N from the training
  subsample, technically trained though only 1 parameter) has no distinct
  palette entry and is not part of either figure's 6-series spec, so it is
  omitted from F1/F3 rather than inventing a 7th color/style not in the fixed
  table. It remains fully available in its own results file for anyone citing
  it directly.
- **F3's zero-shot points use a diamond marker.** The palette table's
  "marker: none" for raw/inflated/tsfm3 describes their flat-line rendering
  in N-axis figures; F3 needs a visible marker for a single point, and
  diamond is unclaimed elsewhere in the table.
- **F1's calendar-day secondary axis** is not driven by a hardcoded
  days-per-case constant. It's recovered at render time as the mean of
  `n_calendar_days_equiv / n_cases` across every real row of
  `phase3_data_size_sweep.parquet` + `phase3_low_n_grid.parquet`
  (`_days_per_case_from_data`) — this keeps the "numbers come from results
  files" rule true for the axis mapping too, not just the annotations.

## Rendering iterations (what "render and view" caught)

First full render (single-column F1/F3/F5, `tight_layout`) had real
problems: F1 had a large blank region at the top (an interaction between
`tight_layout` and `ax.secondary_xaxis` matplotlib itself warns about),
fig-level legends clipped off the bottom of the canvas entirely (negative
`bbox_to_anchor` y with no `bbox_inches='tight'`), F3's three "N=4180"
endpoint labels collided into unreadable overlapping text and its caption
sentence overflowed the right edge, F5's breakpoint/"no crossing" annotations
overlapped each other across rows. Fixed by: dropping `tight_layout` in favor
of explicit `fig.subplots_adjust` margins everywhere; anchoring every
fig-level legend at a small **positive** figure-fraction y (guaranteed inside
the fixed canvas, since `bbox_inches='tight'` is deliberately never used —
the point of fixing figure size in code); adding top headroom
(`_add_top_headroom`) before stacking R2 annotations so they never sit on top
of plotted lines; per-method offset points for F3's N-labels; switching F5's
annotations to `offset points` anchoring; and widening F5's left margin
(0.22 → 0.27) after the y-axis category labels were clipped. Each round was
re-rendered on `altay` and re-viewed as PNG before moving on. Final F1 also
needed the secondary-axis-vs-legend vertical spacing tuned twice (the
secondary axis and the legend collided, then a subsequent fix left an
oversized gap when the legend's column count was temporarily too wide and
clipped the canvas edges — settled on `ncol=2`, 4 legend rows, which fits
190mm without clipping).

## R1–R7 compliance, per figure

- **R1** (small multiples, shared axes labelled once): F1's 3 stacked panels
  share x (only the bottom panel carries the x-label + secondary calendar-day
  axis); F4's 1×4 row uses `fig.supxlabel`/`fig.supylabel` for the one shared
  x/y label instead of repeating per panel; F5's 1×3 row shares y (category
  labels only on the leftmost panel) and `fig.supxlabel` for the one x-label.
- **R2** (in-panel, colour-matched key numbers): F1's CRPS/coverage panels
  stack each trained method's breakpoint in that method's own color; F2
  annotates the durable crossover in the reference grey (the number belongs
  to the crossing itself, not one series); F5 prints every breakpoint (or
  "no crossing (k min–max)") directly at its marker in that method's color.
- **R3** (one shared horizontal legend, bottom): every figure uses a single
  `fig.legend(loc="lower center")`; no per-panel legends anywhere.
- **R4** (reference lines grey dashed): nominal 0.80 (F1 coverage panel, F2
  coverage panel, F3), the durable crossover (F2), and F4's uniform
  expectation line all use `STYLE["color"]["reference"]` + the dashed style —
  never a palette color.
- **R5** (values printed on the geometry): F5 prints every breakpoint value
  (or the "no crossing" disclosure with its tested k-range) directly next to
  its marker.
- **R6** (minimal chrome): top/right spines disabled globally; light grey
  (`#DDDDDD`) gridlines at 0.5pt; white background; F4's PIT bars are filled
  at 0.7 alpha with a same-hue full-opacity edge (reads darker against the
  blended fill).
- **R7** (one fixed palette everywhere): the 7-row Okabe-Ito table is used
  verbatim in every figure via `STYLE`; zero-shot methods (raw, var-inflated,
  TimesFM-3) render as flat lines/single points in F1/F3, trained methods
  (EMOS pooled/local, DRN) as curves/trajectories with markers — the visual
  distinction the style guide calls "itself the paper's argument" holds in
  every figure that has an N axis.

## Mechanical verification performed (not just visual)

- `grep -oE '#[0-9A-Fa-f]{6}' scripts/make_figures.py` outside the `STYLE`
  block: **0 matches** (all 8 hex colors live inside `STYLE`, verified via a
  brace-matching script, not just line-range grep).
- Scanned every numeric literal in the file: everything outside `STYLE` is a
  layout constant (`subplots_adjust` margins, alpha values, offset points,
  header fractions) or the legitimate fixed "nominal 0.80" design constant
  (same convention as the style guide's own "nominal 80% coverage" reference
  lines) — no breakpoint/crossover/coverage value is retyped; every such
  number traces to a `_load(...)` call.
- PDF metadata verified directly on the rendered files (`strings f1_*.pdf |
  grep Subject/Keywords`): `/Subject` carries the exact source parquet
  path(s), `/Keywords` carries `git_sha:<sha>`, for every one of the 5 PDFs.
  `MediaBox` confirmed at exactly 538.58×{...}pt = 190.0mm wide (72pt/in ×
  25.4mm/in), matching the figsize set in code — no post-hoc scaling.
- `python3 -m py_compile` clean on both scripts.
- Every rendered PNG was opened and visually inspected (not just "ran without
  error") at each iteration; the final round confirmed no clipped legends, no
  overlapping annotation text, legible tick labels at the rendered size, and
  correct in-panel numbers against the source parquet values printed during
  schema inspection.

## Known, disclosed limitations

- **Font weight**: panel titles are requested at "medium" weight per the
  typography table; `altay`'s installed font set (DejaVu Sans / whatever
  Arial/Helvetica substitute is resolved) has no medium-weight variant, so
  matplotlib silently substitutes regular (400) weight
  (`findfont: Failed to find font weight medium, now using 400`). Purely a
  rendering-host font-availability issue, not a code defect; not fixable
  without installing a specific font family on `altay`.
- **`git_sha` in PDF metadata**: follows the same "nearest known commit, not
  a bootstrap-perfect self-hash" convention this project's own
  `write_result()` already documents and accepts (a commit's hash cannot
  self-referentially describe an artifact of that same commit). Since this
  project's mandatory workflow is sync-then-run with git commits happening
  locally only, `altay`'s independent local git history was several commits
  behind this Mac's before this task — its `.git` was refreshed to match
  this session's final local commit before the last production render, so
  the embedded SHA is this commit, not a stale one.
- **F4 bin 0 (PIT ∈ [0.0, 0.1)) is always exactly 0** for every method. This
  is a real, inherent property of `pit_values` as implemented (untouched,
  per instructions): it interpolates against `quantile_levels = [0.1..0.9]`
  and clamps out-of-range observations to the nearest end (`tau=0.1` or
  `tau=0.9`), so no PIT value can fall strictly below 0.1 — this is why bins
  1 and 9 (adjacent to the clamped edges) carry elevated mass, most visibly
  for `raw_ensemble`. Not a bug in the new PIT script; a real consequence of
  scoring PIT against a 9-point quantile grid instead of the full [0,1]
  range, and part of why the caption's "assessed descriptively, no formal
  test" language matters.
- **Var_inflation_trainfit** is omitted from F1/F3 (see above) — available
  directly in `results/phase3_data_size_sweep.parquet` /
  `_breakpoints.parquet` for anyone citing it, just not plotted here.

## Files touched

- `scripts/make_figures.py` (new)
- `scripts/10_compute_pit_histograms.py` (new)
- `results/phase3_pit_histograms.parquet` + `.json` (new, produced by the
  above)
- `figures/f1_breakpoint_curve.{pdf,svg,png}` .. `f5_breakpoint_by_lead_group.{pdf,svg,png}` (new)
- `figures/captions.md` (new)
- `docs/figure_style_guide.md` (pre-existing, untracked before this task;
  committed alongside this work since `make_figures.py` implements it
  directly)
- `docs/figures_implementation_report.md` (this file)
