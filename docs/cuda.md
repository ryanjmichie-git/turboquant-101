# NVIDIA / CUDA path (vLLM)

The reference configuration this repo was built around: RTX 5080 16 GB,
vLLM under WSL2 Ubuntu, model `Qwen/Qwen3-4B` (chosen because it is
ungated and was used to validate the upstream vLLM TurboQuant PR).

## The one-line version

```bash
pip install -r requirements-cuda.txt
python scripts/verify.py --backend vllm            # ALWAYS first
python scripts/demo.py   --backend vllm --cache-dtype turboquant_k3v4_nc
```

The knob doing all the work is vLLM's `kv_cache_dtype`. The upstream
TurboQuant backend (merged April 2026 via PR #38479, first shipped in the
v0.20.0 release -- v0.19.x predates it) ships four presets, and the same
knob also accepts vLLM's older built-in `fp8` cache:

| preset | keys / values | cache ratio* | PPL delta* |
|---|---|---|---|
| `fp8` (not TurboQuant) | FP8 K / FP8 V | 2.00x (measured) | negligible (per vLLM) |
| `turboquant_k8v4` | FP8 K / 4-bit V | ~2.6x | +1.2% |
| `turboquant_4bit_nc` | 4-bit K / 4-bit V + NC | ~3.8x | +2.7% |
| `turboquant_k3v4_nc` | 3-bit K / 4-bit V + NC | ~3.5x | +10.6% |
| `turboquant_3bit_nc` | 3-bit K / 3-bit V + NC | ~4.9x | +20.6% |

*TurboQuant rows are from vLLM's own documentation. Yes, `k3v4_nc` — this
repo's reference config — costs perplexity even though simple NIAH
retrieval stays perfect. That is not a contradiction; it is the gap
between an easy retrieval task and general language modeling. See
LEARN.md §4.

The `fp8` row is vLLM's plain FP8 KV cache, which predates TurboQuant:
no rotation, no norm correction, and no boundary-protected layers -- so
its ratio is the clean 16->8-bit arithmetic. The 2.00x is not a docs
number; it is what `verify.py --backend vllm --cache-dtype fp8` measures
on the reference RTX 5080. We haven't found a vLLM-published perplexity
figure for it; vLLM's own guidance simply treats it as the safe first
step, which matches the ladder below.

### Which one should you actually run?

The table is a menu, not a recommendation. Day to day, climb the ladder
and stop at the first rung that fits in your memory budget:

1. **BF16 baseline (`auto`)** -- measure first. If memory isn't your
   bottleneck, compression buys you nothing and costs decode speed
   (LEARN.md §6).
2. **`fp8`** -- the safe default: 2x cache for effectively free, per
   vLLM's own guidance.
3. **`turboquant_4bit_nc`** -- the practical TurboQuant config: ~3.8x
   for +2.7% PPL.
4. **`turboquant_k3v4_nc`** -- this repo's reference config,
   *deliberately*: it is the maximum-capacity experiment, picked because
   it makes the memory win big enough to measure convincingly. Reproduce
   it; think twice before serving real work at +10.6% PPL.
5. **`turboquant_3bit_nc`** -- demo/warning territory: +20.6% PPL.

## Which layers actually get compressed (read before quoting ratios)

The shipped backend applies **boundary protection**: on dense models it
always leaves the first 2 and last 2 attention layers uncompressed
(`TurboQuantConfig.get_boundary_skip_layers`, n=2 — vLLM's own testing
found GSM8K drops ~30 points on Qwen3-4B without it for the aggressive
presets). On Qwen3-4B that means layers 0, 1, 34, 35 stay at the model
dtype and 32/36 layers are compressed.

Two consequences:

* The *measured* capacity ratio is a blend. For `turboquant_k3v4_nc` on
  Qwen3-4B: compressed layers store 118 bytes/token/head vs 512 at BF16
  (4.34x), but the four protected layers dilute that to **~3.16x** --
  which is exactly what `verify.py` reports on this config.
* There is **no supported way to compress every layer** (the
  "skip_layers=0" clean A/B the reference protocol called for). The
  engine takes the union of your `kv_cache_dtype_skip_layers` with the
  boundary set, so user config can only *add* skipped layers, never
  remove the boundary four. You can skip *more* layers via
  `--extra-engine-kwargs '{"kv_cache_dtype_skip_layers": ["2", "3"]}'`
  (accepts layer indices or "sliding_window"). Published numbers from
  this repo therefore use the backend's default coverage (32/36).

## Gotchas, in the order you will hit them

**WSL2 filesystem.** Keep the repo and the HF cache on the Linux side
(`~/`), not under `/mnt/c/...`. Model loading across the Windows mount is
several times slower.

**WSL2: "RuntimeError: UVA is not available".** vLLM >= 0.25 requires
pinned host memory (UVA) in its GPU worker, but disables pinned memory by
default under WSL2. The scripts in this repo set
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` for you; if you drive vLLM yourself under
WSL2, export it or the engine dies at startup. (Requires a WSL2 kernel
>= 4.19.121 -- run `wsl --update` if yours is older.)

**"Could not load libtorchcodec" / "Could not load this library:
.../libtorchcodec_image.so".** vLLM's `torchcodec` dependency hard-fails
on machines without FFmpeg shared libraries (RuntimeError or OSError, both
of which escape vLLM's ImportError guard). It is loaded lazily, so a bare
`import vllm` can succeed and the crash only appears when the engine
starts -- in `verify.py` this shows up as a `[????]` "probe crashed"
banner whose log ends in the torchcodec traceback. This repo is
text-only: `pip uninstall -y torchcodec` and re-run. quickstart.sh now
tests `import torchcodec` directly and removes it automatically.

**"Could not find nvcc" / FlashInfer JIT errors during engine warm-up.**
vLLM's FlashInfer sampler JIT-compiles CUDA on first use, which needs a
full CUDA toolkit + host compiler; a fresh WSL2 install has neither, and
(as of flashinfer 0.6.13) the pip-shipped cu13 toolkit's headers clash
with FlashInfer's bundled CCCL anyway. The scripts in this repo therefore
default to vLLM's native sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`) --
identical results for the greedy decoding used here. If you have a real
CUDA toolkit installed and want the FlashInfer sampler back, export
`VLLM_USE_FLASHINFER_SAMPLER=1`.

**Blackwell (sm_120) support.** Verified working out of the box with the
pinned vLLM (0.25.0 pulls torch 2.11.0+cu130, whose wheels include
sm_120). Older vLLM/PyTorch combos may need a newer PyTorch first -- if
you see "no kernel image" or sm_120 errors, install the current PyTorch
for your CUDA version, then reinstall vLLM.

**Attention backend.** On the pinned vLLM (0.25.0) no workaround is
needed: turboquant dtypes select the dedicated TURBOQUANT attention
backend for compressed layers, plus FLASH_ATTN for the boundary-protected
layers (with FlashAttention force-downgraded to v2; the startup warning
"TurboQuant is not yet compatible with FlashAttention >= 3" is expected
and harmless). Early releases of the backend had a known issue (vllm
#40094) where FLASH_ATTN/TRITON_ATTN rejected turboquant dtypes with
`kv_cache_dtype not supported`; if you run one of those versions, try
`VLLM_ATTENTION_BACKEND=TRITON_ATTN` -- or better, upgrade.

**Qwen3 thinking mode.** Qwen3 emits a `<think>` block by default, which
eats the 64-token answer budget. The backends in this repo disable it
(`enable_thinking=False`); if you write your own harness and retrieval
mysteriously scores zero, check for `<think>` in the raw outputs before
blaming the compression.

**Memory headroom.** `gpu_memory_utilization=0.9` is the reference
setting. If the second verify probe OOMs, close whatever else is using
VRAM (browsers count) or drop to 0.85.

**Version drift.** `requirements-cuda.txt` pins the known-good vLLM.
Newer versions will usually work, but preset names and internals move --
if `verify.py` can't read the KV capacity after an upgrade, see the
pattern lists at the top of that script.

## What a healthy run looks like

`verify.py` should report a capacity ratio around **3-4.5x** and print
PASS (measured: **3.16x** on the validated RTX 5080 run -- see the
layer-coverage section above for why it isn't the headline ~3.5x). On
that card the KV pool holds 44,336 baseline vs 140,320 compressed tokens
at `gpu_memory_utilization=0.9`, and decode drops from ~77 to ~45 tok/s
at 16K. Note the context ceiling: Qwen3-4B's config caps `max_model_len`
at 40,960, so *as shipped* both conditions max out there and compression
buys concurrency instead of length. With Qwen's documented YaRN override
(`--hf-overrides` on verify.py, factor 4) the measured max context is
43,008 baseline vs 131,072 compressed -- a **3.05x context expansion**.
Compression buys memory with compute; the benchmark makes you look at
both numbers side by side.

## Keep it running (optional)

Everything above spins the model up, measures, and exits. To leave a
server running and actually chat with it, use vLLM's OpenAI-compatible
server -- with the daily-driver preset from the ladder, not the
`k3v4_nc` experiment:

```bash
vllm serve Qwen/Qwen3-4B --kv-cache-dtype turboquant_4bit_nc \
    --max-model-len 16384
```

Driving vLLM yourself means the scripts' automatic env vars don't
apply: under WSL2, `export VLLM_WSL2_ENABLE_PIN_MEMORY=1` first, and
add `VLLM_USE_FLASHINFER_SAMPLER=0` if warm-up hits the FlashInfer
gotcha above. The server listens on port 8000:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-4B",
       "messages": [{"role": "user", "content": "Say hello. /no_think"}],
       "max_tokens": 64}'
```

For a chat window instead of curl, point any OpenAI-compatible UI --
[Open WebUI](https://docs.openwebui.com/), for example -- at
`http://localhost:8000/v1`. That's the whole bridge: serving is vLLM's
job; this repo's job was proving the compression underneath it.
