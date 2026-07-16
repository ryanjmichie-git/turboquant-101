"""quickstart.sh must refuse to run on Python < 3.10 (macOS system
python3 is 3.9) and must not reuse a stale .venv built by an older
interpreter."""

import os
import subprocess
import sys

import pytest

from conftest import ROOT


@pytest.mark.skipif(sys.platform == "win32", reason="bash required")
def test_rejects_old_system_python(tmp_path):
    # Shim python3 whose version gate fails (simulates macOS 3.9): the
    # gate runs `python3 -c '...sys.exit(0 if >= 3.10 else 1)'`.
    shim = tmp_path / "python3"
    shim.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = "-V" ]; then echo "Python 3.9.6"; exit 0; fi\n'
        'exit 1\n'
    )
    shim.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    r = subprocess.run(
        ["bash", str(ROOT / "quickstart.sh")],
        capture_output=True, text=True, env=env, timeout=60,
    )
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "3.10" in out, f"expected a Python-floor message, got: {out[-400:]}"


def test_quickstart_recreates_stale_venv_structurally():
    """The stale-venv branch can't be executed safely against the real
    repo dir (it would rebuild the working .venv), so pin its presence
    and shape instead: the interpreter check must target .venv's own
    python and lead to removal."""
    text = (ROOT / "quickstart.sh").read_text(encoding="utf-8")
    assert ".venv/bin/python -c" in text
    assert "rm -rf .venv" in text
    # the gate and the recreation check must both enforce the same floor
    assert text.count("(3, 10)") >= 2
