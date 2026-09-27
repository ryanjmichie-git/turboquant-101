#!/usr/bin/env python3
"""The full NIAH A/B benchmark.

Reference protocol: 5 needles x 3 depths (25/50/75%) x 2 contexts (8K/16K)
x 2 reps = 60 trials per condition. You run it once per condition and give
each run a label; --report merges everything in results/ into one table.

  CUDA example (two runs, then the comparison):
    python scripts/benchmark.py --backend vllm --label baseline
    python scripts/benchmark.py --backend vllm --label turboquant \\
        --cache-dtype turboquant_k3v4_nc
    python scripts/benchmark.py --report

  llama.cpp example: start llama-server with the cache types you want
  (docs/apple-silicon.md), then:
    python scripts/benchmark.py --backend llamacpp --label q8_0-kv

One condition per invocation is deliberate: engine-level cache settings
are fixed at startup, and keeping each run a separate process makes it
impossible to accidentally benchmark a stale engine.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from niah import build_backend, add_backend_args, build_prompt, make_needles, make_trials, score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus.txt"
RESULTS = ROOT / "results"


def run(args) -> int:
    if not args.backend:
        print("error: --backend is required (or use --report)", file=sys.stderr)
        return 2
    corpus = CORPUS.read_text()
    backend = build_backend(args)
    label = args.label or ("compressed" if args.cache_dtype else "baseline")
    print(f"\nBackend: {backend.describe()}")

    trials = make_trials(
        needles=make_needles(args.needles),
        depths=args.depths,
        contexts=args.contexts,
        reps=args.reps,
    )
    print(f"Label '{label}': {len(trials)} trials "
          f"({args.needles} needles x {len(args.depths)} depths x "
          f"{len(args.contexts)} contexts x {args.reps} reps)\n")

    records, t_start = [], time.time()
    for i, t in enumerate(trials, 1):
        prompt = build_prompt(corpus, t, backend.count_tokens)
        res = backend.generate(prompt)
        ok = score(res.text, t.needle)
        records.append({
            "needle": t.needle, "depth_pct": t.depth_pct,
            "context_tokens": t.context_tokens, "rep": t.rep,
            "hit": bool(ok), "response": res.text.strip()[:120],
            "gen_tokens": res.gen_tokens, "seconds": res.seconds,
        })
        mark = "+" if ok else "MISS"
        print(f"  [{i:>3}/{len(trials)}] ctx={t.context_tokens:>6} "
              f"depth={t.depth_pct:>2}% {t.needle:<18} {mark}")

    # Decode-speed probe. The per-trial "tok/s" divides ~10 generated
    # tokens by the WHOLE request time, prefill included -- fine as a
    # sanity number, useless as a decode-speed A/B. The backends measure
    # decode-only rate separately (see decode_speed in niah/backends.py).
    # Distinct needles (own seed) give two distinct same-length documents
    # that also cannot collide with any trial prompt via prefix caching.
    probe_needles = make_needles(2, seed=4321)
    decode_probe = {}
    for ctx in args.contexts:
        prompts = [
            build_prompt(
                corpus,
                make_trials([n], depths=[50], contexts=[ctx], reps=1)[0],
                backend.count_tokens,
            )
            for n in probe_needles
        ]
        speed = backend.decode_speed(prompts[0], prompts[1])
        if speed:
            decode_probe[str(ctx)] = round(speed, 1)
            print(f"  decode-speed probe ctx={ctx:>6}: {speed:.0f} tok/s")

    out = {
        "label": label,
        "backend": backend.describe(),
        "cache_dtype": args.cache_dtype,
        "protocol": {
            "needles": args.needles, "depths": args.depths,
            "contexts": args.contexts, "reps": args.reps,
        },
        "decode_probe": decode_probe,
        "wall_seconds": time.time() - t_start,
        "trials": records,
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{label}.json"
    path.write_text(json.dumps(out, indent=2))
    # Results are on disk; shut the engine down before the summary so
    # vLLM's teardown output can't bury it (see VLLMBackend.close).
    backend.close()

    hits = sum(r["hit"] for r in records)
    speeds = [r["gen_tokens"] / r["seconds"] for r in records if r["seconds"] > 0]
    print(f"\n  {label}: {hits}/{len(records)} retrieved | "
          f"median e2e {statistics.median(speeds):.0f} tok/s (not decode; see probe) | "
          f"saved to {path.relative_to(ROOT)}")
    print("  Run the other condition, then: python scripts/benchmark.py --report\n")
    return 0


def report() -> int:
    files = sorted(RESULTS.glob("*.json"))
    if not files:
        print("No results in results/ yet.")
        return 1

    print(f"\n{'label':<16} {'context':>8} {'retrieved':>10} "
          f"{'decode tok/s':>13} {'e2e tok/s':>10}")
    print(f"{'-'*16} {'-'*8} {'-'*10} {'-'*13} {'-'*10}")
    for f in files:
        data = json.loads(f.read_text())
        by_ctx: dict[int, list[dict]] = {}
        for r in data["trials"]:
            by_ctx.setdefault(r["context_tokens"], []).append(r)
        for ctx, rs in sorted(by_ctx.items()):
            hits = sum(r["hit"] for r in rs)
            speeds = [r["gen_tokens"] / r["seconds"] for r in rs if r["seconds"] > 0]
            med = statistics.median(speeds) if speeds else 0
            dec = data.get("decode_probe", {}).get(str(ctx))
            dec_s = f"{dec:>13.0f}" if dec else f"{'n/a':>13}"
            print(f"{data['label']:<16} {ctx:>8,} {hits:>6}/{len(rs):<3} "
                  f"{dec_s} {med:>10.0f}")
    print(
        "\nRead this table with the honesty rules from LEARN.md: matched "
        "retrieval means\n'no damage detectable by THIS task', the decode "
        "tok/s difference is the real cost,\nand context expansion must be "
        "measured (scripts/verify.py), not inferred.\n('e2e tok/s' is "
        "answer-tokens / whole-request time INCLUDING prefill -- a\nsanity "
        "number for ~10-token NIAH answers, not a throughput figure.)\n"
    )
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_backend_args(ap)
    ap.add_argument("--needles", type=int, default=5)
    ap.add_argument("--depths", type=int, nargs="+", default=[25, 50, 75])
    ap.add_argument("--contexts", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--report", action="store_true",
                    help="merge results/*.json into a comparison table")
    args = ap.parse_args()
    if args.report:
        sys.exit(report())
    sys.exit(run(args))


if __name__ == "__main__":
    main()
