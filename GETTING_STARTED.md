# Getting started (total beginner edition)

New to running language models on your own computer? This guide walks you
through this repo from zero, in plain language. No prior LLM experience
needed.

## What this repo does, in one paragraph

When an AI model reads text, it keeps notes about every word in a memory
scratchpad called the **KV cache**. The longer the text, the bigger the
scratchpad — until your graphics card runs out of memory and the model
stops. This repo demonstrates **compression** for that scratchpad: the
same memory holds roughly 3x more text, at the cost of somewhat slower
responses. You'll run the math on your own machine, *prove* the
compression is really on (it's surprisingly easy to think it's on when it
isn't), and measure whether the model still finds facts in long
documents.

## Five words you'll see

- **Token** — a chunk of text, roughly ¾ of a word. Models read and
  write in tokens.
- **Context** — how much text the model can hold in mind at once,
  measured in tokens.
- **KV cache** — the scratchpad above. Grows with every token read.
- **Quantization** — storing numbers with fewer bits to save memory.
  It's the "compression" here.
- **Needle test** — hide a code word (the needle) in a long document
  (the haystack) and ask the model to find it. If it can, its memory of
  the document survived compression.

## What you need

Pick the row that matches your machine — the first steps are the same
for everyone:

| Your machine | What you can run |
|---|---|
| Any computer | The CPU demo — the core math, no GPU needed. Start here regardless. |
| NVIDIA GPU (~16 GB VRAM), Windows or Linux | The full experience: real compressed cache in vLLM. |
| Apple Silicon Mac (M1–M4, 16 GB+) | The Mac path: quantized cache in llama.cpp. |

You also need about 10 GB of free disk space for the GPU paths (the
model itself is a few GB), and Python 3.10 or newer.

## Step 1 — one-time setup for your platform

**Windows (NVIDIA):** the GPU path runs inside WSL2 (a Linux environment
Windows provides). In PowerShell run `wsl --install`, reboot, and open
the "Ubuntu" app. Do everything below inside that Ubuntu window. Keep
the repo in your Linux home folder (e.g. `~/`), **not** on `/mnt/c/...`
— model loading from the Windows drive is painfully slow (the script
will warn you).

**Mac (Apple Silicon):** install [Homebrew](https://brew.sh) if you
don't have it, then:

```bash
brew install python@3.12 llama.cpp
```

(macOS's built-in Python is too old; the script checks and will tell
you.)

**Linux (NVIDIA):** you just need `git`, `python3` (3.10+), and a
working NVIDIA driver (`nvidia-smi` should print your GPU).

## Step 2 — get the repo

```bash
git clone https://github.com/ryanjmichie-git/turboquant-101
cd turboquant-101
```

## Step 3 — run one script

```bash
./quickstart.sh
```

That's the whole interface. It sets up an isolated Python environment
(`.venv` — it won't touch anything else on your system), then does three
things:

1. **CPU demo (~1 minute, every machine).** Runs the compression math on
   fake attention data. Watch the table it prints: naive 3-bit
   compression tanks retrieval to ~83%, but the same 3 bits with a
   rotation trick recovers ~98% — that one contrast is the entire idea
   of TurboQuant.
2. **Verify (GPU machines).** Starts the model twice — cache compression
   off, then on — and reads the engine's own memory report. You want to
   see `PASS` and a ratio around **3.16x** (NVIDIA) or a cache size
   roughly **half** of f16 (Mac). This step exists because compression
   can silently not engage while everything *looks* normal.
3. **Five-needle demo (GPU machines).** Hides 5 code words in long
   documents and asks the model to find them, with the compressed cache
   on. Expect 5/5.

**Heads up on downloads:** the first GPU run downloads the model —
~7.6 GB on the NVIDIA path, ~2.5 GB on the Mac — plus vLLM itself on
NVIDIA (also large). First run is a coffee break; later runs are fast.

## Step 4 — read your results

- CPU demo: the story is the *gap between rows* (naive vs rotated), not
  the exact percentages.
- Verify: `PASS 3.16x` means "the same memory now holds 3.16x more
  tokens." If it FAILs, compression isn't actually on — see
  `docs/cuda.md` / `docs/apple-silicon.md` before trusting anything
  else.
- Demo: 5/5 means compression didn't break memory *at this difficulty*.
  A perfect score here does NOT mean compression is harmless everywhere
  — see `LEARN.md` §4 for what this test can't detect.

## Step 5 (optional) — the full benchmark

When you're curious, run the real A/B experiment (60 trials per side,
identical documents with compression off vs on):

```bash
# NVIDIA — one run per condition:
python scripts/benchmark.py --backend vllm --label baseline
python scripts/benchmark.py --backend vllm --label turboquant \
    --cache-dtype turboquant_k3v4_nc
python scripts/benchmark.py --report
```

On our 16 GB RTX 5080, both sides scored 60/60 on retrieval, and the
compressed side generated ~40% slower at 16K context. That's the honest
trade: memory for speed. (Mac commands are in `docs/apple-silicon.md`.)

One nuance: `turboquant_k3v4_nc` is the *maximum-capacity experiment*
config — chosen to make the memory win big enough to measure, not the
preset you'd serve with every day. The day-to-day ladder is in
`docs/cuda.md`.

## If something goes wrong

The script already auto-fixes the common traps (missing `python3-venv`,
a broken `torchcodec` dependency, two WSL2-specific engine errors). If
it still fails:

- **"python3 >= 3.10 is required"** — install a newer Python (Step 1)
  and re-run; the script rebuilds `.venv` itself.
- **"llama-server not found" (Mac)** — `brew install llama.cpp`, re-run.
- **NVIDIA engine won't start** — read `docs/cuda.md`; every failure we
  hit on a fresh machine is documented there with its fix.
- **Anything else** — open an issue with the error text. Corrections and
  disagreeing measurements are explicitly welcome.

## Where to go next

- `LEARN.md` — the friendly deep-dive: how the compression works, what
  the honest limits are, and when you *shouldn't* use it.
- `README.md` — the measured reference numbers and the "what we will
  NOT tell you" section.
- `VALIDATION_REPORT.md` — the full evidence trail, if you want to see
  how every number was checked.

One idea to take with you even if you never run another script: **don't
trust a compression setting until something independent proves it's
actually on.** That habit is the most useful thing in this repo.
