"""A vLLM probe that crashes at import must be reported as a crash with
its fix -- not as "your vLLM version logs differently". Regression for a
fresh WSL2 run (2026-09-27) where torchcodec raised OSError inside the
probe after quickstart's `import vllm` check had passed."""

import types

from conftest import ROOT, load_verify

verify = load_verify()

TORCHCODEC_CRASH = """\
Traceback (most recent call last):
  File ".venv/lib/python3.10/site-packages/vllm/multimodal/video.py", line 35, in <module>
    from torchcodec.decoders import VideoDecoder
  File ".venv/lib/python3.10/site-packages/torch/_ops.py", line 1505, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: .venv/lib/python3.10/site-packages/torchcodec/libtorchcodec_image.so
"""


def test_torchcodec_crash_names_the_fix():
    hint = verify.diagnose_probe_crash(TORCHCODEC_CRASH)
    assert hint and "pip uninstall -y torchcodec" in hint


def test_unknown_traceback_is_still_called_a_crash():
    log = "Traceback (most recent call last):\n  ...\nValueError: boom\n"
    hint = verify.diagnose_probe_crash(log)
    assert hint and "crashed" in hint


def test_healthy_log_is_not_a_crash():
    log = "INFO GPU KV cache size: 44,336 tokens\nPROBE_RESULT {}\n"
    assert verify.diagnose_probe_crash(log) is None


def test_verify_vllm_reports_crash_not_log_format(monkeypatch, capsys):
    monkeypatch.setattr(verify, "_run_vllm_probe",
                        lambda *a, **k: (None, TORCHCODEC_CRASH))
    args = types.SimpleNamespace(cache_dtype=None, max_model_len=8192,
                                 gpu_mem_util=0.85, hf_overrides=None)
    rc = verify.verify_vllm(args)
    out = capsys.readouterr().out
    assert rc == 2
    assert "pip uninstall -y torchcodec" in out
    assert "logs differently" not in out


def test_setup_tests_torchcodec_itself():
    """`import vllm` alone passes while torchcodec is broken (lazy import),
    so the guard must import torchcodec directly and then the engine path."""
    text = (ROOT / "setup.sh").read_text(encoding="utf-8")
    assert "python -c 'import torchcodec'" in text
    assert "from vllm import SamplingParams" in text
    assert "if ! python -c 'import vllm'" not in text


def _pass_run(monkeypatch, capsys, util):
    counts = iter([(38544, "log"), (121984, "log")])
    monkeypatch.setattr(verify, "_run_vllm_probe", lambda *a, **k: next(counts))
    args = types.SimpleNamespace(cache_dtype=None, max_model_len=8192,
                                 gpu_mem_util=util, hf_overrides=None)
    assert verify.verify_vllm(args) == 0
    return capsys.readouterr().out


def test_default_util_explains_readme_mismatch(monkeypatch, capsys):
    out = _pass_run(monkeypatch, capsys, 0.85)
    assert "same GPU memory budget (85%)" in out
    assert "44,336 -> 140,320" in out and "90%" in out
    assert "compare ratios" in out


def test_reference_util_has_no_mismatch_note(monkeypatch, capsys):
    out = _pass_run(monkeypatch, capsys, 0.90)
    assert "same GPU memory budget (90%)" in out
    assert "44,336" not in out
