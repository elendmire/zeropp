"""BLOCKED: needs the built parquet dataset from 02_build_dataset.py, and needs
the trained baseline models (emos.py, qrf.py, drn.py, mos_rf.py) to be real
implementations, before this script can be implemented. Do not fill this in
with placeholder logic — implement for real once both are ready."""

import sys


def main() -> None:
    raise NotImplementedError(
        "blocked: needs EUPPBench parquet output from 02_build_dataset.py and "
        "real (non-stub) baseline models in zeropp.models"
    )


if __name__ == "__main__":
    sys.exit(main())
