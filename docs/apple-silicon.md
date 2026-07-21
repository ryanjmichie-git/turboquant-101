# Apple Silicon path (llama.cpp)

Reference target: Mac mini M4, 16 GB unified memory.

## First, the honest situation (July 2026)

Mainline llama.cpp **does not ship TurboQuant cache types**. The community
PR that added them (`tbq3_0` / `tbq4_0`, PR #21089) was **closed by the
maintainers in June 2026** -- their position: nobody demonstrated a clear
quality-per-bit win over the existing cache quants, especially after the
Hadamard-rotation idea (TurboQuant's core trick) was absorbed into
mainline's existing quants via a separate change (PR #21038). Even on the
PR branch, Metal KV offload crashed and needed a keep-KV-on-CPU
workaround. The full saga is a genuinely useful case study -- see
"A short history" in LEARN.md.

So this repo's Mac path teaches KV cache compression with what mainline
actually supports, and points you at a fork if you want the real
TurboQuant types.

## Path A (default): mainline llama.cpp, quantized KV cache

```bash
brew install llama.cpp
./quickstart.sh        # or by hand:

llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M -c 16384 -fa on \
  -ctk q8_0 -ctv q8_0 --port 8080
```

`-ctk` / `-ctv` set the Key and Value cache types. `f16` is the baseline;
`q8_0` shrinks the cache ~1.9x with negligible quality cost; `q4_0` gives
~3.6x with a measurable but usually acceptable hit. **Flash attention
(`-fa on`) is required for a quantized V cache** -- without it llama.cpp
either refuses or silently dequantizes per step and gets slower.

Then, in another terminal:

```bash
python scripts/verify.py --backend llamacpp --ctk q4_0 --ctv q4_0
python scripts/demo.py   --backend llamacpp
python scripts/benchmark.py --backend llamacpp --label q4_0-kv
```

Same concept as the CUDA path, same benchmark, same honesty rules. Note
what you are and aren't testing: q8_0/q4_0 KV are simple per-block
quantizers, not the rotate-then-quantize TurboQuant recipe. The
*phenomenon* (trade a little fidelity for a lot of cache memory) is
identical; the *algorithm* is not. Do not present Path A numbers as
TurboQuant numbers.

### Chat with the server you already have

The `llama-server` command above stays up until you stop it (the one
quickstart.sh starts is temporary -- it's killed when the script exits).
It speaks the OpenAI API, so once it's healthy you can just talk to it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say hello. /no_think"}],
       "max_tokens": 64}'
```

For a chat window instead of curl, point any OpenAI-compatible UI --
[Open WebUI](https://docs.openwebui.com/), for example -- at
`http://127.0.0.1:8080/v1`. That's everyday local serving, with the
quantized cache you just verified underneath it.

## Measured results (reference M4 16 GB, llama.cpp b9960, Qwen3-4B Q4_K_M)

180 paired trials per condition (identical documents), 2026-07-18:

| condition | KV @ 8K ctx | retrieval | think artifacts | decode 8K/16K | prefill vs f16 |
|---|---|---|---|---|---|
| f16 | 1,152 MiB | 180/180 | 0 | 26.3 / 19.8 tok/s | — |
| q8_0 | 612 MiB (1.88x) | 180/180 | 0 | 25.7 / 20.1 | −20% / −29% |
| q4_0 | 324 MiB (3.56x) | 165/180 | 22 trials | 25.2 / 19.5 | −17% / −25% |

Three lessons the table compresses:

* **On Metal, quantized KV costs prefill, not decode** — the opposite
  of TurboQuant on CUDA, where decode pays and prefill doesn't. Where
  the bill lands depends on the implementation, not the concept.
* **q8_0 is the sweet spot on Apple Silicon**: indistinguishable from
  f16 in 180 paired trials (and in decode speed) at half the cache.
* **q4_0's failures were NOT lost memory.** Replaying every failing
  prompt showed the model retrieved the exact needle 52/52 times — but
  emitted it inside a thinking block, which llama-server routes to
  `reasoning_content`, leaving the answer channel empty. What breaks
  under q4_0 is the `/no_think` soft switch's reliability (plus rare
  token-level noise: a stray fragment, one case-flipped character).
  Strict scoring counts those as task failures — quote them as
  "instruction discipline degrades", never as "forgets the document".

## Path B (advanced): a fork with real TurboQuant-family types

The most maintained option is the `TheTom/llama-cpp-turboquant` fork,
which ships Metal kernels for WHT-rotated polar-codebook cache types
(selected through the same `--cache-type-k/v` flags) plus prebuilt
binaries. Practical notes as of mid-2026:

* Use their prebuilt releases if one matches your machine; an Apple
  Silicon Metal startup-crash fix landed in their PR #200, so avoid
  older builds.
* Their own guidance: keep K conservative (f16 or q8_0) and compress V
  with the turbo types first; symmetric aggressive K+V settings break
  some model families.
* It is a fork. It drifts from mainline, its benchmarks are
  self-reported, and anything you measure on it should be labeled as
  fork results.

## 16 GB unified-memory reality check

Weights (Q4_K_M Qwen3-4B ~= 2.5 GB) + macOS + your apps all share the same
16 GB. The KV cache is still the thing that scales with context, so cache
quantization still buys you real context headroom -- but expect smaller
absolute numbers than the 16 GB *dedicated-VRAM* CUDA card, and measure
rather than extrapolate.

## Gotchas

* **KV size lines are hidden at the default log level.** Current
  llama.cpp builds (verified on build 9960) only print the KV buffer
  allocation lines at raised verbosity, which used to make verify.py
  report "could not find KV buffer size lines". verify.py now passes
  `-lv 4` to the server automatically; add it yourself if you run
  llama-server by hand and want to see the sizes.
* **First run downloads the model.** `-hf Qwen/Qwen3-4B-GGUF:Q4_K_M`
  pulls ~2.5 GB into your HF cache (confirmed resolving on brew
  llama.cpp build 9960; 2.32 GiB download). If that repo/tag doesn't
  resolve on your llama.cpp version, download a Qwen3-4B GGUF manually
  and pass `-m /path/to/model.gguf` (and `--gguf` to verify.py).
* **Qwen3 thinking mode.** The scripts append `/no_think`; if you drive
  the server yourself and answers start with `<think>`, that's why.
  Measured caveat: under q4_0 KV the soft switch itself gets flaky
  (~12% of trials on the reference M4) — see the results table above.
* **Cache type names drift.** `llama-server --help | grep cache-type`
  is the source of truth for your build, not any document -- including
  this one.
