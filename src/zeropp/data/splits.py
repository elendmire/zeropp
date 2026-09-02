def load_euppbench_split(data_dir: str):
    """Load EUPPBench's own train/test split.

    BLOCKED: needs built parquet dataset from build.py. When implemented,
    this MUST reuse EUPPBench's own train/test split — never re-split the
    data ourselves, per CLAUDE.md's non-negotiable rule.
    """
    raise NotImplementedError(
        "blocked: needs built parquet dataset from build.py; when implemented, "
        "this MUST reuse EUPPBench's own train/test split — never re-split"
    )
