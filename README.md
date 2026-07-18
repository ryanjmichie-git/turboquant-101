# turboquant-101

**Run 2-3x more context on the GPU you already own, and understand exactly
what you paid for it.**

Every token a language model reads leaves behind a memory footprint (the
KV cache). Think of the model as a hiker whose backpack fills up as the
trail gets longer: eventually the pack is full and the hike ends --
that's your out-of-memory error. TurboQuant-style KV cache compression
vacuum-packs what goes into the backpack. The hiker walks a bit slower,
but the trail can be almost three times longer.

This repo is a hands-on, honest introduction: you'll run the core math on
your CPU in one minute, verify on your own GPU that compression is
genuinely active (not silently disabled -- this happens more than you'd
think), and reproduce a real retrieval benchmark.

## Quickstart

```bash
git clone <this-repo> && cd turboquant-101
./quickstart.sh
```

That's it. The script runs the CPU demo everywhere, then detects your
platform: NVIDIA GPUs get the vLLM path with the real
`turboquant_k3v4_nc` cache (docs/cuda.md), Apple Silicon gets the
llama.cpp quantized-KV path (docs/apple-silicon.md).

## The one distinction that prevents most confusion

* **Cache compression ratio** (~3.2-5x here): how much smaller each
  cached token gets. A property of the algorithm (and of which layers
  the backend actually compresses -- see docs/cuda.md).
* **Context expansion** (3.05x on the validated run, and only with a
  rope-scaling override -- the model's own position limit caps you
  first): how much more text actually fits on your GPU. A property of
  your whole system, because weights and activations don't shrink.

Never quote one as the other. The CPU demo walks through why they differ.

## Reference results

Validated runs, 2026-07-12/18: RTX 5080 16 GB (WSL2 Ubuntu 22.04, vLLM
0.25.0, torch 2.11.0+cu130, Qwen3-4B in BF16), baseline BF16 cache vs
`turboquant_k3v4_nc` (3-bit keys, 4-bit values, norm correction; the
backend keeps the first/last 2 layers uncompressed -- docs/cuda.md).
The full evidence trail — including a 180-trial-per-condition run on
both this card and an M4 Mac, with confidence intervals and a failure
autopsy — is in VALIDATION_REPORT.md:

| metric | baseline | compressed |
|---|---|---|
| Needle retrieval (paired trials) | 180/180 | 180/180 |
| KV capacity @ 0.9 util | 44,336 tok | 140,320 tok (**3.16x**) |
| Max context, as shipped | 40,960 tok* | 40,960 tok* |
| Max context, YaRN x4 override | 43,008 tok | 131,072 tok (**3.05x**) |
| Decode speed @16K (probe) | ~77 tok/s | ~45 tok/s |

*Qwen3-4B's own `max_position_embeddings` is 40,960: as shipped, BOTH
conditions hit the model's position limit before memory. There the
compression buys concurrency (3.4x more parallel 40K requests), not
length; the context-expansion row requires Qwen's documented YaRN
rope-scaling override to mean anything. Cache ratio is still not context
expansion -- and neither survives an unstated model ceiling.

(The original pre-validation reference run reported "~45K -> ~128K
(2.86x)" and ~79 -> ~46 tok/s decode; retrieval and decode match what we
measure, but its context numbers are only consistent with a YaRN-style
override that its notes never recorded. Kept for history; trust the
table above.)

These are one machine's numbers from one run -- treat them as a reference
point to reproduce, not a spec sheet. Your numbers will differ; that's
the point of running it yourself.

## What we will NOT tell you

Honesty section, because this field has a hype problem:

* **Perfect NIAH is not "no quality loss."** Single-needle retrieval at
  8-16K is an easy task. vLLM's own benchmarks show `k3v4_nc` costs
  ~+10% perplexity and measurably hurts reasoning and very-long-context
  tasks. If quality is the priority, `turboquant_4bit_nc` (~3.8x,
  +2.7% PPL) is the smarter preset. Details in LEARN.md.
* **No open-source implementation is the full paper.** Community
  consensus dropped the paper's QJL residual stage (it amplified error
  through softmax); what everyone ships is rotation + optimized
  low-bit grids. Still very useful -- just not the headline algorithm.
* **The paper's "8x" figure is attention-compute speedup on H100s**, not
  a memory compression ratio. Different claim entirely.
* **Compression costs speed.** ~40% slower decode on the reference run.
  You're trading compute for memory; whether that's a good trade depends
  on whether memory is your bottleneck.

## Map

```
quickstart.sh          platform-detecting entry point
scripts/cpu_demo.py    the quantizer math, no GPU (start here)
scripts/verify.py      proves compression is actually engaged  <- run this
scripts/demo.py        5-needle retrieval quick win
scripts/benchmark.py   full A/B protocol + report table
niah/                  the benchmark protocol + backend adapters
docs/cuda.md           NVIDIA/WSL2 setup and gotchas
docs/apple-silicon.md  Mac setup (and the honest llama.cpp story)
LEARN.md               the deep explainer -- what, why, when NOT to
```

## Credits

TurboQuant is from Zandieh, Daliri, Hadian & Mirrokni (Google), ICLR 2026
(arXiv:2504.19874). The vLLM backend used here was contributed upstream
by @vibhavagarwal5 (vllm-project/vllm PR #38479). The llama.cpp
exploration credits @elusznik's PR #21089, @ggerganov's rotation work
(PR #21038), and @TheTom's fork for the Metal path. Filler corpus:
Darwin's *Voyage of the Beagle* via Project Gutenberg (public domain).

MIT licensed. Built as a learning project; corrections welcome --
especially measured numbers that disagree with ours.
