"""BLOCKED: needs zeropp.eval.tables and zeropp.eval.figures to be real
implementations, which themselves need real results/*.parquet output from
model runs. Do not fill this in with placeholder logic — implement for real
once tables.py and figures.py are ready."""

import sys


def main() -> None:
    raise NotImplementedError(
        "blocked: needs real (non-stub) zeropp.eval.tables and "
        "zeropp.eval.figures modules"
    )


if __name__ == "__main__":
    sys.exit(main())
