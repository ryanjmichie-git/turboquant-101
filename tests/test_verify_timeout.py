"""--startup-timeout must fire even when the server produces no output.
The old readline() loop only checked the deadline between newline-
terminated lines, so a silently stalling llama-server hung the probe
indefinitely (worst during first-run model downloads)."""

import sys
import time
import types

import pytest

from conftest import load_verify

verify = load_verify()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell stub")
def test_silent_server_respects_startup_timeout(tmp_path):
    stub = tmp_path / "silent-server"
    stub.write_text("#!/bin/sh\nsleep 30\n")
    stub.chmod(0o755)
    args = types.SimpleNamespace(
        server_bin=str(stub), gguf="unused", ctx=512,
        extra_server_args=None, startup_timeout=2,
    )
    t0 = time.time()
    total, log = verify._run_llamacpp_probe(args, "f16", "f16")
    elapsed = time.time() - t0
    assert total is None
    assert log == ""  # nothing was printed; nothing should be captured
    assert elapsed < 10, (
        f"probe took {elapsed:.1f}s for a 2s timeout -- the deadline is "
        "not being enforced against a silent process"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell stub")
def test_exiting_server_returns_promptly_with_log(tmp_path):
    stub = tmp_path / "crashing-server"
    stub.write_text('#!/bin/sh\necho "error: model not found"\nexit 1\n')
    stub.chmod(0o755)
    args = types.SimpleNamespace(
        server_bin=str(stub), gguf="unused", ctx=512,
        extra_server_args=None, startup_timeout=30,
    )
    t0 = time.time()
    total, log = verify._run_llamacpp_probe(args, "f16", "f16")
    assert total is None
    assert "model not found" in log
    assert time.time() - t0 < 10
