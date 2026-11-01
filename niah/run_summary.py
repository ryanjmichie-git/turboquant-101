"""Hand each quickstart step's headline numbers to the end-of-run summary.

quickstart.sh exports TQ101_RUN_DIR (a scratch folder inside .venv). Each
script it runs -- cpu_demo.py, verify.py, demo.py -- drops one small JSON
file there with the numbers it just printed, and scripts/summary.py reads
them back into one screen. Nothing is re-derived or hard-coded: the summary
can only show what this run actually produced.

Run a script on its own (no TQ101_RUN_DIR) and record() does nothing.
"""

from __future__ import annotations

import json
import os
import pathlib

ENV_VAR = "TQ101_RUN_DIR"


def run_dir() -> pathlib.Path | None:
    d = os.environ.get(ENV_VAR)
    return pathlib.Path(d) if d else None


def under_quickstart() -> bool:
    """True when quickstart.sh is the caller (it prints its own next steps)."""
    return run_dir() is not None


def record(step: str, data: dict) -> None:
    """Write <TQ101_RUN_DIR>/<step>.json. Never raises: a summary is a
    convenience, and must not turn a successful step into a failure."""
    d = run_dir()
    if d is None:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{step}.json").write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def load(step: str, directory: pathlib.Path) -> dict | None:
    try:
        return json.loads((directory / f"{step}.json").read_text())
    except (OSError, ValueError):
        return None
