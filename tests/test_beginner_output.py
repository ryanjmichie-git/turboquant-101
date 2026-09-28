"""Beginner-facing quickstart output (2026-09-28 review): short CPU step,
plain-English verify readout, quiet needle rows, and an end-of-run summary
built only from what this run recorded. Nothing that proves compression
may disappear -- detail moves behind --verbose."""

import json
import os
import subprocess
import sys
import types

import pytest

from conftest import ROOT, load_verify

sys.path.insert(0, str(ROOT / "scripts"))
import demo  # noqa: E402
import summary  # noqa: E402

verify = load_verify()


# ---------------------------------------------------------------- cpu_demo
def _cpu_demo(*args, run_dir=None):
    env = dict(os.environ)
    env.pop("TQ101_RUN_DIR", None)
    if run_dir:
        env["TQ101_RUN_DIR"] = str(run_dir)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "cpu_demo.py"), *args],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_brief_is_short_and_keeps_the_nc_row(tmp_path):
    out = _cpu_demo("--brief", run_dir=tmp_path)
    assert len(out.strip().splitlines()) <= 25
    for label in ("uncompressed (FP16)", "3-bit, naive",
                  "3-bit + rotation + better grid", "norm correction ('_nc')"):
        assert label in out
    # the '_nc' row is what Steps 2-3 run -- it must not be dropped for
    # scoring a hair lower than plain rotation
    assert "what Steps 2-3 use" in out
    assert "python scripts/cpu_demo.py" in out  # pointer to the full version
    rec = json.loads((tmp_path / "cpu.json").read_text())
    assert rec["naive"] < rec["rotated"] <= rec["fp16"]
    assert 0 < rec["rotated_nc"] <= 1


def test_full_demo_keeps_every_section():
    out = _cpu_demo()
    for heading in ("1. WHY THE KV CACHE", "2. THE QUANTIZER",
                    "3. CACHE RATIO IS NOT CONTEXT EXPANSION"):
        assert heading in out
    assert "RMSE" in out and "norm err" in out
    # the old toy table printed the same text on every row
    assert "realized < 3.5x" not in out


def test_full_demo_next_step_pointer_only_when_standalone(tmp_path):
    assert "Or just run ./quickstart.sh" in _cpu_demo()
    assert "Or just run ./quickstart.sh" not in _cpu_demo(run_dir=tmp_path)


# ------------------------------------------------------------------ verify
PROBE_LOGS = ["INFO GPU KV cache size: 38,544 tokens\n",
              "INFO GPU KV cache size: 121,984 tokens\n"]


def _verify_out(monkeypatch, capsys, logs=PROBE_LOGS, verbose=False):
    it = iter(logs)

    def fake_probe(*a, **k):
        log = next(it)
        return verify._parse_capacity(log)[0], log

    monkeypatch.setattr(verify, "_run_vllm_probe", fake_probe)
    args = types.SimpleNamespace(cache_dtype=None, max_model_len=8192,
                                 gpu_mem_util=0.85, hf_overrides=None,
                                 verbose=verbose)
    rc = verify.verify_vllm(args)
    return rc, capsys.readouterr().out


def test_verify_default_readout_is_plain_english(monkeypatch, capsys):
    rc, out = _verify_out(monkeypatch, capsys)
    assert rc == 0
    assert "38,544" in out and "121,984" in out and "3.16x" in out
    assert 'vLLM\'s own "GPU KV cache size" startup line' in out
    assert "log pattern" not in out  # regex is a maintainer detail
    assert "[PASS]" in out and "Not proof quality held" in out
    assert max(len(line) for line in out.splitlines()) <= 80


def test_verify_verbose_shows_the_regex(monkeypatch, capsys):
    _, out = _verify_out(monkeypatch, capsys, verbose=True)
    assert "baseline capacity via log pattern" in out
    assert "compressed capacity via log pattern" in out


def test_attr_fallback_caveat_is_never_hidden(monkeypatch, capsys):
    logs = ['PROBE_RESULT {"attr_tokens": 38544, "attr_chain": "a.b"}\n',
            'PROBE_RESULT {"attr_tokens": 121984, "attr_chain": "a.b"}\n']
    rc, out = _verify_out(monkeypatch, capsys, logs=logs)
    assert rc == 0
    assert "UNDER-report" in out


def test_verify_records_for_summary(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("TQ101_RUN_DIR", str(tmp_path))
    _verify_out(monkeypatch, capsys)
    rec = json.loads((tmp_path / "verify.json").read_text())
    assert rec["base_tokens"] == 38544 and rec["comp_tokens"] == 121984
    assert rec["passed"] is True and rec["gpu_mem_util"] == 0.85


# -------------------------------------------------------------------- demo
def test_exact_hit_is_one_clean_line():
    row = demo.format_row(True, "DYNAMO-1572", "DYNAMO-1572\n", 10, 2.7)
    assert row.strip() == "FOUND   DYNAMO-1572"
    assert "tok/s" not in row


def test_answer_shown_when_not_exactly_the_code():
    row = demo.format_row(True, "FATHOM-4879", "The code is FATHOM-4879.", 12, 1.0)
    assert "model answered: 'The code is FATHOM-4879.'" in row
    miss = demo.format_row(False, "LANTERN-2612", "<think>", 64, 3.0)
    assert miss.lstrip().startswith("MISSED") and "<think>" in miss


def test_verbose_timing_is_labelled_not_speed():
    row = demo.format_row(True, "DYNAMO-1572", "DYNAMO-1572", 10, 2.7, verbose=True)
    assert "tok/s" in row and "not decode speed" in row


# ----------------------------------------------------------------- summary
def _write(d, name, data):
    (d / f"{name}.json").write_text(json.dumps(data))


@pytest.fixture
def full_run(tmp_path):
    _write(tmp_path, "cpu", {"fp16": .986, "naive": .828, "rotated": .982,
                             "rotated_nc": .979, "n_queries": 512})
    _write(tmp_path, "verify", {"backend": "vllm", "base_tokens": 38544,
                                "comp_tokens": 121984, "ratio": 121984 / 38544,
                                "gpu_mem_util": .85, "passed": True,
                                "attr_fallback": False})
    _write(tmp_path, "demo", {"hits": 5, "total": 5, "context": 8192})
    return tmp_path


def test_summary_labels_every_number(full_run):
    text = "\n".join(summary.build(full_run))
    assert "82.8%" in text and "98.2%" in text and "97.9%" in text
    assert "synthetic vectors, your CPU" in text
    assert "38,544 -> 121,984 tokens at 85%" in text
    assert text.count("[measured, your GPU") == 2
    assert "5/5 codes found" in text
    # the honesty caveats travel with the numbers
    assert "not \"3.16x longer context\"" in text
    assert "not proof of zero" in text
    # README / upstream figures never appear in a this-run summary
    assert "44,336" not in text and "3.05x" not in text
    assert max(len(line) for line in text.splitlines()) <= 80


def test_summary_cpu_only_has_no_gpu_claims(tmp_path):
    _write(tmp_path, "cpu", {"fp16": .986, "naive": .828, "rotated": .982,
                             "rotated_nc": .979, "n_queries": 512})
    text = "\n".join(summary.build(tmp_path))
    assert "CPU idea" in text
    assert "measured" not in text and "What this does NOT show" not in text


def test_summary_empty_dir_prints_nothing(tmp_path):
    assert summary.build(tmp_path) == []


def test_summary_llamacpp_says_mac(tmp_path):
    _write(tmp_path, "verify", {"backend": "llamacpp", "cache_dtype": "q8_0/q8_0",
                                "base_mib": 1152.0, "comp_mib": 612.0,
                                "ratio": 1.88, "passed": True})
    text = "\n".join(summary.build(tmp_path))
    assert "measured, your Mac" in text and "1.88x smaller" in text


# -------------------------------------------------------------- quickstart
@pytest.mark.skipif(sys.platform == "win32", reason="bash required")
def test_quickstart_rejects_unknown_option_before_setup():
    r = subprocess.run(["bash", str(ROOT / "quickstart.sh"), "--verbos"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 1
    assert "usage: ./quickstart.sh [--verbose]" in r.stdout
    assert "Creating virtualenv" not in r.stdout


def test_quickstart_passes_verbose_and_prints_summary():
    text = (ROOT / "quickstart.sh").read_text(encoding="utf-8")
    assert "cpu_demo.py --brief" in text
    assert "verify.py --backend vllm $VERBOSE" in text
    assert "--cache-dtype turboquant_k3v4_nc $VERBOSE" in text
    assert "TQ101_RUN_DIR" in text and "scripts/summary.py" in text


# ------------------------------------------- round 2 (2026-09-28 verbose run)
class _FakeBackend:
    """Echoes each needle back, so demo.main runs with no GPU."""

    def __init__(self):
        self.code = None

    def count_tokens(self, text):
        return len(text) // 4

    def generate(self, prompt):
        import re
        code = re.search(r"The secret code is ([A-Z]+-\d+)", prompt).group(1)
        from niah import GenResult
        return GenResult(text=code, gen_tokens=6, seconds=1.2)

    def describe(self):
        return "fake / kv_cache_dtype=turboquant_k3v4_nc"

    def close(self):
        pass


def _run_demo(monkeypatch, capsys, *extra):
    monkeypatch.setattr(demo, "build_backend", lambda args: _FakeBackend())
    monkeypatch.setattr(sys, "argv", ["demo.py", "--backend", "llamacpp", *extra])
    demo.main()
    return capsys.readouterr().out


def test_demo_output_fits_80_columns(monkeypatch, capsys):
    for extra in ((), ("--verbose",)):
        out = _run_demo(monkeypatch, capsys, *extra)
        assert "Retrieved 5/5." in out
        assert max(len(line) for line in out.splitlines()) <= 80, extra


def test_demo_verbose_puts_each_trial_on_its_own_line(monkeypatch, capsys):
    """vLLM logs mid-run under --verbose; a shared progress line got split
    ("trial 1/5INFO ... [hf.py:548] ...") on the 2026-09-28 run."""
    quiet = _run_demo(monkeypatch, capsys)
    assert "  trial 1/5 2/5 3/5 4/5 5/5 done" in quiet
    loud = _run_demo(monkeypatch, capsys, "--verbose")
    assert all(f"  trial {i}/5..." in loud.splitlines() for i in range(1, 6))


def test_demo_answer_truncated_to_fit():
    row = demo.format_row(True, "FATHOM-4879",
                          "Sure! The secret code in the document is FATHOM-4879.",
                          20, 1.0)
    assert len(row) <= 80 and "..." in row


BASE_LOG = ("INFO Using FLASH_ATTN attention backend out of potential "
            "backends\nINFO GPU KV cache size: 38,544 tokens\n")
COMP_LOG = ("INFO Using FLASH_ATTN attention backend out of x\nINFO Using "
            "TURBOQUANT attention backend out of potential backends\n"
            "INFO GPU KV cache size: 121,984 tokens\n")


def test_verbose_verify_quotes_the_engines(monkeypatch, capsys):
    _, out = _verify_out(monkeypatch, capsys, logs=[BASE_LOG, COMP_LOG],
                         verbose=True)
    assert "What each engine logged (verbatim):" in out
    assert '"GPU KV cache size: 38,544 tokens"' in out
    assert '"Using TURBOQUANT attention backend"' in out
    quotes = out.split("What each engine logged")[1].split("=" * 72)[0]
    assert max(len(line) for line in quotes.splitlines()) <= 80


def test_default_verify_has_no_engine_quotes_and_times_each_load(monkeypatch, capsys):
    _, out = _verify_out(monkeypatch, capsys, logs=[BASE_LOG, COMP_LOG])
    assert "What each engine logged" not in out
    assert "loading with the normal cache ... done (" in out
    assert out.rstrip().endswith("=" * 72)  # banner ends the step, no blank


def test_summary_reports_elapsed(full_run):
    text = "\n".join(summary.build(full_run, elapsed=85))
    assert "Whole run took 1 min 25 s" in text
    assert summary.format_elapsed(42) == "42 s"


def test_quickstart_delegates_setup_and_prints_one_ready_line():
    """Installs/downloads moved to setup.sh (2026-09-28); quickstart runs it
    with --brief, which prints a single "Setup ready" header when done."""
    text = (ROOT / "quickstart.sh").read_text(encoding="utf-8")
    assert "bash ./setup.sh --brief" in text
    assert "pip install" not in text and "requirements-cuda.txt" not in text
    assert '--elapsed "$(( $(date +%s) - START_TIME ))"' in text
    setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
    assert 'say "Setup ready: $PLATFORM"' in setup


def test_cpu_brief_has_no_trailing_blank_line(tmp_path):
    out = _cpu_demo("--brief", run_dir=tmp_path)
    assert out.endswith("python scripts/cpu_demo.py\n")
