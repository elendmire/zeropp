"""SUPERSEDED: this was the Phase 1 scaffold's stub for the CRPS-vs-N training-size
sweep. That work is now implemented for real in scripts/07_data_size_sweep.py (Task 4
of the Phase 3 paper-readiness plan) — a two-arm (chronologically-contiguous cold-start
+ random subsampling), two-pooling-variant (pooled + per-station local EMOS) design
that superseded a single-arm design this stub number was originally reserved for.

Kept as a thin pointer (rather than deleted) because docs/PHASE2_BLOCKED.md and
docs/superpowers/plans/2026-09-02-zeropp-scaffold.md still reference
"scripts/05_data_size_sweep.py" by name as historical record of the Phase 1 scaffold's
blocked-work manifest; deleting the file outright would turn those into dead links for
no benefit. A future runner who naively runs this file (per those older docs) gets
pointed at the real script instead of a NotImplementedError with no forwarding
information."""
import sys


def main() -> None:
    sys.exit(
        "scripts/05_data_size_sweep.py is superseded — run "
        "scripts/07_data_size_sweep.py instead (see this file's module docstring)."
    )


if __name__ == "__main__":
    main()
