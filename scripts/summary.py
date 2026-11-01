#!/usr/bin/env python3
"""One-screen recap of what ./quickstart.sh just showed.

quickstart.sh runs this last. It reads the small JSON files that
cpu_demo.py, verify.py and demo.py wrote during THIS run (see
niah/run_summary.py) -- nothing here is typed in by hand, so the summary
can't drift from what the steps printed. Each line says where its number
came from: synthetic vectors on the CPU, or a measurement on this machine.
Upstream / README figures never appear here.

  python scripts/summary.py <run-dir> [--elapsed SECONDS]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from niah.run_summary import ENV_VAR, load, run_dir  # noqa: E402

RULE = "=" * 72
LABEL_W = 12  # "  1. CPU idea" column


def _item(n: int, label: str, lines: list[str]) -> list[str]:
    head = f"  {n}. {label:<{LABEL_W}}"
    pad = " " * len(head)
    return [head + lines[0]] + [pad + ln for ln in lines[1:]]


def format_elapsed(seconds: int) -> str:
    minutes, secs = divmod(max(0, int(seconds)), 60)
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"


def build(directory: pathlib.Path, elapsed: int | None = None) -> list[str]:
    cpu = load("cpu", directory)
    ver = load("verify", directory)
    demo = load("demo", directory)
    if not (cpu or ver or demo):
        return []

    device = "your GPU"
    if ver and ver.get("backend") == "llamacpp":
        device = "your Mac"

    out = [RULE, "  What you just showed on this machine", RULE]
    caveats = []

    if cpu:
        out += _item(1, "CPU idea", [
            f"3-bit keys: {cpu['naive']:.1%} naive -> {cpu['rotated']:.1%} "
            f"with rotation,",
            f"{cpu['rotated_nc']:.1%} with norm correction; "
            f"{cpu['fp16']:.1%} uncompressed",
            "[toy needle test, synthetic vectors, your CPU]",
        ])

    if ver:
        verdict = "PASS" if ver.get("passed") else "FAIL"
        if ver.get("backend") == "llamacpp":
            lines = [
                f"{verdict}: KV cache {ver['ratio']:.2f}x smaller "
                f"({ver['cache_dtype']} vs f16)",
                f"({ver['base_mib']:,.0f} -> {ver['comp_mib']:,.0f} MiB)"
                f"   [measured, {device}]",
            ]
        else:
            lines = [
                f"{verdict}: {ver['ratio']:.2f}x more KV cache fits in the "
                f"same memory",
                f"({ver['base_tokens']:,} -> {ver['comp_tokens']:,} tokens at "
                f"{ver['gpu_mem_util']:.0%})   [measured, {device}]",
            ]
            if ver.get("attr_fallback"):
                lines.append("(read from engine attributes -- may under-report)")
        out += _item(2, "Verify", lines)
        caveats.append(
            f"{ver['ratio']:.2f}x is a cache ratio, not \"{ver['ratio']:.2f}x "
            f"longer context\":\n    weights and workspace don't shrink "
            f"(python scripts/cpu_demo.py, section 3)."
        )

    if demo:
        out += _item(3, "Needles", [
            f"{demo['hits']}/{demo['total']} codes found in "
            f"{demo['context']:,}-token documents",
            f"[measured, {device}, compressed cache]",
        ])
        caveats.append(
            f"{demo['hits']}/{demo['total']} on one easy test (one length, "
            f"one depth) is not proof of zero\n    quality loss -- the A/B "
            f"benchmark compares against the uncompressed cache."
        )

    if caveats:
        out += ["", "  What this does NOT show:"]
        out += [f"  - {c}" for c in caveats]
    if elapsed is not None:
        out += ["", f"  Whole run took {format_elapsed(elapsed)} "
                    f"(including setup)."]
    out.append(RULE)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--elapsed", type=int, default=None,
                    help="seconds since quickstart started")
    args = ap.parse_args()
    directory = pathlib.Path(args.run_dir) if args.run_dir else run_dir()
    if directory is None:
        print(f"usage: summary.py <run-dir>   (or set {ENV_VAR})", file=sys.stderr)
        return 2
    lines = build(directory, args.elapsed)
    if lines:
        print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
