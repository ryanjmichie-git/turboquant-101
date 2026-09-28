#!/usr/bin/env python3
"""Five-trial needle-in-a-haystack quick win.

Hides five secret codes at 50% depth of an 8K-token document and asks the
model to retrieve each one. Takes a couple of minutes; the full A/B
protocol lives in scripts/benchmark.py.

  CUDA:   python scripts/demo.py --backend vllm --cache-dtype turboquant_k3v4_nc
  Mac:    python scripts/demo.py --backend llamacpp    (server already running)
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from niah import build_backend, add_backend_args, build_prompt, make_needles, make_trials, score  # noqa: E402
from niah.run_summary import record, under_quickstart  # noqa: E402

CORPUS = pathlib.Path(__file__).resolve().parents[1] / "data" / "corpus.txt"


def format_row(ok: bool, needle: str, text: str, gen_tokens: int,
               seconds: float, verbose: bool = False) -> str:
    """One result line. The model's answer is shown whenever it isn't
    exactly the code (a MISS, or a hit wrapped in extra words), so a
    FOUND can always be checked. Timing is --verbose only: at ~10 answer
    tokens per request, tokens/second is dominated by reading the 8K
    document (prefill), so it reads as a speed figure when it isn't one.
    The benchmark's decode probe measures actual decode rate."""
    answer = text.strip()
    row = f"  {'FOUND ' if ok else 'MISSED'}  {needle}"
    if answer != needle or verbose:
        row = f"{row:<30}  model answered: {answer[:40]!r}"
    if verbose:
        rate = gen_tokens / seconds if seconds > 0 else 0
        row += (f"\n{'':<10}{seconds:.1f} s, {rate:.1f} tok/s -- includes "
                f"reading the document; not decode speed")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_backend_args(ap)
    ap.add_argument("--context", type=int, default=8192)
    args = ap.parse_args()
    if not args.backend:
        ap.error("--backend is required")

    corpus = CORPUS.read_text()
    if args.backend == "vllm" and not args.verbose:
        print("Loading the model (vLLM engine log hidden; add --verbose to see it)...",
              flush=True)
    t_load = time.perf_counter()
    backend = build_backend(args)
    if args.backend == "vllm":
        print(f"Model loaded in {time.perf_counter() - t_load:.0f} s.")
    print(f"\nBackend: {backend.describe()}")
    print(f"5 trials: hide a secret code halfway into a ~{args.context:,}-token "
          f"document,\nthen ask the model to find it.\n")

    trials = make_trials(
        needles=make_needles(5), depths=[50], contexts=[args.context], reps=1
    )
    # Collect results, shut the engine down, THEN print: vLLM's teardown
    # writes its own output, and the answer should be the last thing on
    # screen, not buried above it.
    rows, hits = [], 0
    try:
        print("  trial", end="", flush=True)
        for i, t in enumerate(trials, 1):
            print(f" {i}/{len(trials)}", end="", flush=True)
            prompt = build_prompt(corpus, t, backend.count_tokens)
            res = backend.generate(prompt)
            ok = score(res.text, t.needle)
            hits += ok
            rows.append(format_row(ok, t.needle, res.text, res.gen_tokens,
                                   res.seconds, args.verbose))
        print(" done", flush=True)
    finally:
        backend.close()

    total = len(trials)
    record("demo", {"hits": hits, "total": total, "context": args.context,
                    "depth_pct": 50, "backend": backend.describe()})
    print("\nResults:")
    print("\n".join(rows))
    print(f"\n  Retrieved {hits}/{total}.")
    if hits == total:
        print(f"  No damage this easy test can detect ({args.context:,}-token "
              f"documents, codes at 50% depth).")
        if not under_quickstart():
            print("  That is not proof quality held: the A/B benchmark "
                  "(scripts/benchmark.py)\n  compares against the "
                  "uncompressed cache at more depths and lengths.")
        print()
    else:
        print("  Misses at 8K/50% depth usually mean thinking-mode leaked "
              "(<think> in output above?),\n  the wrong model loaded, or the "
              "context didn't fit. Run scripts/verify.py first.\n")


if __name__ == "__main__":
    main()
