"""Shared test plumbing: make the repo root importable and load
scripts/verify.py (which is a script, not a package module)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_verify():
    spec = importlib.util.spec_from_file_location(
        "verify", ROOT / "scripts" / "verify.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
