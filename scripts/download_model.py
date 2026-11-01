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
import fnmatch
import json
import os
import pathlib
import shutil
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from niah.backends import DEFAULT_MODEL  # noqa: E402

# Set before huggingface_hub is first imported (it reads them at import):
# setup.sh prints its own, friendlier HF_TOKEN tip, and the Hub's version
# would land in the middle of the progress line; our single progress line
# replaces the Hub's bars. Anything you export yourself wins.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

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


def expected_bytes(model: str = DEFAULT_MODEL) -> int | None:
    """Total size of the files we'll fetch, from the Hub's file list.
    None if the Hub can't be asked (the ticker then shows GB so far)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model, files_metadata=True)
        return sum(s.size or 0 for s in info.siblings
                   if any(fnmatch.fnmatch(s.rfilename, pat) for pat in ALLOW))
    except Exception:
        return None


def _bytes_on_disk(folder: pathlib.Path) -> int:
    """Size of the model's blobs so far, partial (.incomplete) files too."""
    total = 0
    if folder.is_dir():
        for f in folder.iterdir():
            try:
                total += f.stat().st_size
            except OSError:
                pass  # renamed from .incomplete mid-scan
    return total


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def progress_line(done: int, total: int | None, elapsed: float) -> str:
    gb = done / 2**30
    rate = done / 2**20 / elapsed if elapsed > 0 else 0
    if total:
        pct = min(100, int(100 * done / total))
        head = f"{gb:.1f} of {total / 2**30:.1f} GB ({pct}%)"
    else:
        head = f"{gb:.1f} GB so far"
    return f"  {head}  {rate:.0f} MB/s  {_fmt_time(elapsed)}"


def download(model: str = DEFAULT_MODEL) -> str:
    """snapshot_download with ONE progress line instead of the Hub's
    three (on the 2026-09-28 WSL2 run: "Downloading bytes", a
    "Reconstructing" bar that read 3.89 kB/s -- looks stalled -- and
    "Fetching 10 files", with the Hub's HF_TOKEN warning printed into the
    middle of them). Progress = bytes written to the model's blob folder,
    so it can lag the network briefly while files are reassembled, but it
    only moves forward and ends at 100%. Elapsed time always ticks."""
    from huggingface_hub import constants, snapshot_download

    total = expected_bytes(model)
    blobs = (pathlib.Path(constants.HF_HUB_CACHE)
             / f"models--{model.replace('/', '--')}" / "blobs")
    result: dict = {}

    def run():
        try:
            result["path"] = snapshot_download(model, allow_patterns=ALLOW)
        except BaseException as e:  # re-raised in the main thread
            result["error"] = e

    worker = threading.Thread(target=run, daemon=True)
    t0 = time.time()
    worker.start()
    tty = sys.stdout.isatty()
    last_print = 0.0
    while worker.is_alive():
        worker.join(timeout=1.0)
        now = time.time()
        line = progress_line(_bytes_on_disk(blobs), total, now - t0)
        if tty:
            print("\r" + line.ljust(60), end="", flush=True)
        elif now - last_print >= 15:  # logs/CI: a line every 15 s
            print(line, flush=True)
            last_print = now
    if tty:
        print()
    if "error" in result:
        raise result["error"]
    print(f"  Done in {_fmt_time(time.time() - t0)}. Saved under "
          f"{constants.HF_HUB_CACHE}")
    return result["path"]


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
    try:
        path = download()
    except KeyboardInterrupt:
        print("\n  Stopped. Re-run ./setup.sh to resume where it left off.")
        return 1
    except Exception as e:
        print(f"\n  Download failed: {e}\n  Check your connection and re-run "
              f"./setup.sh -- it resumes where it left off.")
        return 1
    if not is_cached():
        print(f"  Download finished but some weight files are missing "
              f"under {path}. Re-run ./setup.sh to resume.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
