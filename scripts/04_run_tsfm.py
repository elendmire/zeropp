"""BLOCKED: needs the TSFM model stubs (tsfm_timesfm.py, tsfm_chronos.py,
tsfm_moirai.py) to be real implementations, which in turn need the SSH
server's heavy packages installed via 00_setup_env.sh. Do not fill this in
with placeholder logic — implement for real once the TSFM models are ready."""

import sys


def main() -> None:
    raise NotImplementedError(
        "blocked: needs real (non-stub) TSFM models in zeropp.models "
        "(tsfm_timesfm.py, tsfm_chronos.py, tsfm_moirai.py)"
    )


if __name__ == "__main__":
    sys.exit(main())
