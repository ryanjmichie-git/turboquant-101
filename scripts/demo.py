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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from niah import build_backend, add_backend_args, build_prompt, make_needles, make_trials, score  # noqa: E402

CORPUS = pathlib.Path(__file__).resolve().parents[1] / "data" / "corpus.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_backend_args(ap)
    ap.add_argument("--context", type=int, default=8192)
    args = ap.parse_args()
    if not args.backend:
        ap.error("--backend is required")

    corpus = CORPUS.read_text()
    if args.backend == "vllm" and not args.verbose:
        print("Loading the model (vLLM engine log hidden; add --verbose to see it)...")
    backend = build_backend(args)
    print(f"\nBackend: {backend.describe()}")
    print(f"Hiding 5 needles at 50% depth of ~{args.context:,}-token documents...\n")

    trials = make_trials(
        needles=make_needles(5), depths=[50], contexts=[args.context], reps=1
    )
    # Collect results, shut the engine down, THEN print: vLLM's teardown
    # writes its own output, and the answer should be the last thing on
    # screen, not buried above it.
    rows, hits = [], 0
    try:
        for i, t in enumerate(trials, 1):
            print(f"  trial {i}/{len(trials)}...", flush=True)
            prompt = build_prompt(corpus, t, backend.count_tokens)
            res = backend.generate(prompt)
            ok = score(res.text, t.needle)
            hits += ok
            # Whole-request rate (prefill included) -- a sanity number for
            # the ~10-token answers here, NOT a decode-speed figure. The
            # benchmark's decode probe measures actual decode rate.
            speed = res.gen_tokens / res.seconds if res.seconds > 0 else 0
            rows.append(f"  {'FOUND ' if ok else 'MISSED'}  expected {t.needle:<18} "
                        f"got: {res.text.strip()[:40]!r}  ({speed:.1f} tok/s end-to-end)")
    finally:
        backend.close()

    print(f"\nResults ({backend.describe()}):")
    print("\n".join(rows))
    print(f"\n  Retrieved {hits}/5.")
    if hits == 5:
        print("  The model's memory of the document survived. Now do it "
              "properly:\n  run the A/B benchmark (baseline vs compressed) "
              "-- see scripts/benchmark.py.\n")
    else:
        print("  Misses at 8K/50% depth usually mean thinking-mode leaked "
              "(<think> in output above?),\n  the wrong model loaded, or the "
              "context didn't fit. Run scripts/verify.py first.\n")


if __name__ == "__main__":
    main()
