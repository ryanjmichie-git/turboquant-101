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
v0.20.0 release -- v0.19.x predates it) ships four presets:

| preset | keys / values | cache ratio* | PPL delta* |
|---|---|---|---|
| `turboquant_k8v4` | FP8 K / 4-bit V | ~2.6x | +1.2% |
| `turboquant_4bit_nc` | 4-bit K / 4-bit V + NC | ~3.8x | +2.7% |
| `turboquant_k3v4_nc` | 3-bit K / 4-bit V + NC | ~3.5x | +10.6% |
| `turboquant_3bit_nc` | 3-bit K / 3-bit V + NC | ~4.9x | +20.6% |

*From vLLM's own documentation. Yes, `k3v4_nc` — this repo's reference
config — costs perplexity even though simple NIAH retrieval stays perfect.
That is not a contradiction; it is the gap between an easy retrieval task
and general language modeling. If quality matters more than the last bit
of memory, `turboquant_4bit_nc` is the better trade. See LEARN.md.

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

**"Could not load libtorchcodec" at `import vllm`.** vLLM's `torchcodec`
dependency hard-fails on machines without FFmpeg shared libraries, and
the error (RuntimeError) escapes vLLM's ImportError guard. This repo is
text-only: `pip uninstall -y torchcodec` and move on. quickstart.sh does
this automatically.

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
