"""setup.sh + scripts/download_model.py (2026-09-28): the one-time
installs and the ~8 GB model download happen in setup, visibly, with a
disk-space check first -- never hidden inside verify.py's first probe."""

import importlib.util
import json
import subprocess
import sys
import types
from collections import namedtuple

import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location(
    "download_model", ROOT / "scripts" / "download_model.py")
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)


@pytest.mark.skipif(sys.platform == "win32", reason="bash required")
def test_setup_rejects_unknown_option_before_doing_anything():
    r = subprocess.run(["bash", str(ROOT / "setup.sh"), "--skip-disk"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 1
    assert "usage: ./setup.sh" in r.stdout
    assert "Creating the Python environment" not in r.stdout


def test_setup_checks_disk_before_downloading():
    text = (ROOT / "setup.sh").read_text(encoding="utf-8")
    disk = text.index("--disk-check")
    assert disk < text.index("pip install -q -r requirements-cuda.txt")
    assert disk < text.index("python scripts/download_model.py\n")
    assert "Nothing is measured here" in text


def test_verify_and_quickstart_never_download_explicitly():
    for name in ("quickstart.sh", "scripts/verify.py", "scripts/demo.py"):
        assert "download_model" not in (ROOT / name).read_text(encoding="utf-8")


def test_model_id_comes_from_backends():
    from niah.backends import DEFAULT_MODEL
    assert dm.DEFAULT_MODEL == DEFAULT_MODEL


def test_hf_cache_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert dm.hf_cache_dir() == tmp_path / "hf" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "direct"))
    assert dm.hf_cache_dir() == tmp_path / "direct"


Usage = namedtuple("Usage", "total used free")


def test_disk_needs_on_one_filesystem_are_summed(monkeypatch, tmp_path):
    monkeypatch.setattr(dm.shutil, "disk_usage",
                        lambda p: Usage(0, 0, 20 * 2**30))
    needs = {tmp_path / ".venv": 15, tmp_path / "cache": 9}  # same disk
    problems = dm.free_space_problems(needs)
    assert len(problems) == 1 and "about 24 GB" in problems[0]
    assert dm.free_space_problems({tmp_path: 15}) == []
    assert dm.free_space_problems({tmp_path: 0}) == []


def _fake_hub(monkeypatch, snapshot):
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo, allow_patterns=None, local_files_only=False):
        if snapshot is None:
            raise FileNotFoundError(repo)
        return str(snapshot)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_is_cached_requires_every_shard(monkeypatch, tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    (snap / "model-00001-of-00002.safetensors").write_text("x")
    _fake_hub(monkeypatch, snap)
    assert dm.is_cached() is False  # interrupted download: shard 2 missing
    (snap / "model-00002-of-00002.safetensors").write_text("x")
    assert dm.is_cached() is True


def test_is_cached_false_when_nothing_downloaded(monkeypatch):
    _fake_hub(monkeypatch, None)
    assert dm.is_cached() is False
