#!/usr/bin/env python3
"""Download Qwen/Qwen3-4B for the vLLM path -- once, with a progress bar.

setup.sh runs this so the ~8 GB download happens in setup, visibly, and
never inside verify.py (which hides its engines' output: a first-run
download there looked frozen for many minutes). Later engine starts
read the same Hugging Face cache and skip the download.

  python scripts/download_model.py            download (no-op if cached)
  python scripts/download_model.py --check    exit 0 if fully cached, else 1
  python scripts/download_model.py --disk-check --need-venv-gb 15
                                             exit 1 if there isn't room

The model id comes from niah.backends.DEFAULT_MODEL, so it can't drift
from what demo.py / benchmark.py / verify.py load.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from niah.backends import DEFAULT_MODEL  # noqa: E402

# What vLLM reads for a safetensors model: config/tokenizer JSON, the
# tokenizer's merges/vocab text, and the weight shards + their index.
ALLOW = ["*.json", "*.safetensors", "*.txt"]
MODEL_GB = 9  # Qwen3-4B BF16 checkpoint is ~7.5 GiB; margin for the rest


def hf_cache_dir() -> pathlib.Path:
    """Where huggingface_hub stores models (honours HF_HUB_CACHE/HF_HOME)
    -- computed without importing huggingface_hub, so --disk-check works
    before vLLM (which brings it) is installed."""
    if os.environ.get("HF_HUB_CACHE"):
        return pathlib.Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "huggingface")
    return pathlib.Path(home) / "hub"


def is_cached(model: str = DEFAULT_MODEL) -> bool:
    """True only if every weight shard is on disk. A snapshot folder alone
    isn't enough: an interrupted download leaves one behind."""
    try:
        from huggingface_hub import snapshot_download
        path = pathlib.Path(snapshot_download(
            model, allow_patterns=ALLOW, local_files_only=True))
    except Exception:
        return False
    if not (path / "config.json").is_file():
        return False
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    return all((path / s).is_file() for s in shards)


def _existing(p: pathlib.Path) -> pathlib.Path:
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def free_space_problems(needs: dict[pathlib.Path, float]) -> list[str]:
    """needs = {path: GB it will need}. Sums needs that share a filesystem
    (WSL2: the repo and ~/.cache usually do) and reports any shortfall."""
    by_dev: dict[int, list] = {}
    for path, gb in needs.items():
        if gb <= 0:
            continue
        p = _existing(path)
        by_dev.setdefault(os.stat(p).st_dev, [p, 0.0])[1] += gb
    problems = []
    for p, need in by_dev.values():
        free = shutil.disk_usage(p).free / 2**30
        if free < need:
            problems.append(f"{free:.0f} GB free on the disk holding {p}, "
                            f"setup needs about {need:.0f} GB there")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 0 if the model is fully downloaded, else 1")
    ap.add_argument("--disk-check", action="store_true",
                    help="exit 1 if there isn't room for what's still missing")
    ap.add_argument("--need-venv-gb", type=float, default=0,
                    help="GB the Python install still needs (disk check)")
    args = ap.parse_args()

    if args.check:
        return 0 if is_cached() else 1

    if args.disk_check:
        # is_cached() needs huggingface_hub; before vLLM is installed it
        # isn't there, which correctly means "model still to download".
        needs = {pathlib.Path.cwd() / ".venv": args.need_venv_gb,
                 hf_cache_dir(): 0 if is_cached() else MODEL_GB}
        problems = free_space_problems(needs)
        for msg in problems:
            print(f"  Not enough disk space: {msg}.")
        return 1 if problems else 0

    if is_cached():
        print(f"  {DEFAULT_MODEL} is already downloaded.")
        return 0
    from huggingface_hub import snapshot_download
    path = snapshot_download(DEFAULT_MODEL, allow_patterns=ALLOW)
    if not is_cached():
        print(f"  Download finished but some weight files are missing "
              f"under {path}. Re-run ./setup.sh to resume.")
        return 1
    print(f"  Saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
