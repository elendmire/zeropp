"""BLOCKED: needs both the trained baseline models (via 03_run_baselines.py)
and the zero-shot TSFM models (via 04_run_tsfm.py) to be real implementations
before the CRPS-vs-N training-size sweep can be run. Do not fill this in with
placeholder logic — implement for real once both are ready."""

import sys


def main() -> None:
    raise NotImplementedError(
        "blocked: needs real (non-stub) baseline models and real (non-stub) "
        "TSFM models to sweep N in {0, 30, 90, 365, 1095, full} days"
    )


if __name__ == "__main__":
    sys.exit(main())
