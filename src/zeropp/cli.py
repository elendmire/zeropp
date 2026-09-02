import sys


def main() -> None:
    """Command-line entry point wrapping scripts/02-06.

    BLOCKED: needs the scripts it would wrap (02_build_dataset.py through
    06_make_report.py) to be real implementations before this CLI can do
    anything but fail — wrapping stub scripts would hide their real status
    behind a working-looking `zeropp` command.
    """
    raise NotImplementedError(
        "blocked: needs scripts/02_build_dataset.py through "
        "scripts/06_make_report.py to be real (non-stub) implementations"
    )


if __name__ == "__main__":
    sys.exit(main())
