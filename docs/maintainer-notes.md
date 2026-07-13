# Maintainer notes: hardware validation checklist

This repo was assembled in a CPU-only environment. The CPU demo is tested;
everything below the GPU line is written against documented behavior and
needs one pass on real hardware before the repo is shared publicly. Work
through this list on the RTX 5080 and the M4.

> **Status 2026-07-13:** the CUDA/vLLM section (items 1-5) and the
> both-platforms section (items 10-11) were validated on the RTX 5080
> (2026-07-12), and the Apple Silicon section (items 6-8) got a first
> pass on the Mac mini M4 (2026-07-13, via an independent agent run) --
> see `VALIDATION_REPORT.md` in the repo root for the full evidence.
> Inline `RESOLVED` notes below summarize each outcome. Item 9 (fork
> path) and the full Mac A/B benchmark remain open.

## CUDA / vLLM (RTX 5080, WSL2)

1. **Pin check.** `requirements-cuda.txt` pins `vllm==0.19.1` (first
   release with the merged PR #38479 backend). Confirm the exact version
   on the machine where the 60/60 run happened and update the pin +
   README table footnote to match.

   > RESOLVED: the premise was wrong -- vllm 0.19.1 (2026-04-18) was cut
   > from the pre-merge 0.19 branch and contains NO turboquant code; the
   > first release with the backend is 0.20.0 (2026-04-27). Repo now pins
   > 0.25.0, verified end-to-end on the RTX 5080 (torch 2.11.0+cu130,
   > native sm_120 -- no nightly needed). Three environment fixes were
   > required and are now in the scripts/docs: `VLLM_WSL2_ENABLE_PIN_MEMORY=1`
   > (WSL2), removing `torchcodec` (no-FFmpeg systems), and
   > `VLLM_USE_FLASHINFER_SAMPLER=0` (no CUDA toolkit on host).
2. **verify.py detection path.** Run `python scripts/verify.py --backend
   vllm` and note whether capacity came from engine attributes or from a
   log regex. If neither fires, paste the actual startup line that
   reports KV capacity into `VLLM_LOG_PATTERNS` / `VLLM_ATTR_CHAINS`.

   > RESOLVED: on vLLM 0.25.0 the log regex `GPU KV cache size: N tokens`
   > fires. The attr chain `llm_engine.vllm_config.cache_config.num_gpu_blocks`
   > sometimes resolves too, but blocks x block_size UNDER-REPORTS
   > turboquant capacity by 2x (packed slots hold more tokens per block),
   > so verify.py now prefers the log line and prints which source it
   > used. Measured: 38,544 -> 121,984 tokens = 3.16x PASS
   > (8192/util 0.85).
3. **skip_layers.** The reference protocol calls for `skip_layers=0`
   (compress every layer) for a clean A/B. The exact flag/config name for
   this in the shipped backend wasn't verifiable offline. Find it
   (`vllm serve --help`, the PR #38479 docs page) and wire it in via
   `--extra-engine-kwargs '{"...": 0}'`, then document it in
   docs/cuda.md. Until then, runs use the backend's default layer
   coverage -- note this next to any published numbers.

   > RESOLVED (negatively): the mechanism is
   > `kv_cache_dtype_skip_layers` (CLI `--kv-cache-dtype-skip-layers`,
   > wireable via `--extra-engine-kwargs`), but the engine UNIONS it with
   > a hard-coded boundary-protection set (first/last 2 layers,
   > `TurboQuantConfig.get_boundary_skip_layers`, n=2) whenever a
   > turboquant dtype is active. `skip_layers=0` is therefore NOT
   > achievable in this vLLM version; all published numbers use the
   > default 32/36-layer coverage (documented in docs/cuda.md).
4. **Attention backend.** If verify hits `kv_cache_dtype not supported`
   (issue #40094), confirm the `VLLM_ATTENTION_BACKEND=TRITON_ATTN`
   workaround and promote it from "gotcha" to default guidance if needed.

   > RESOLVED: not needed on 0.25.0. Turboquant dtypes select a dedicated
   > TURBOQUANT attention backend (compressed layers) + FLASH_ATTN
   > auto-downgraded to v2 (boundary layers); the "not yet compatible
   > with FlashAttention >= 3" startup warning is expected. docs/cuda.md
   > updated; #40094 note kept for older releases.
5. **Reproduce the reference numbers.** Full benchmark, both conditions.
   Update the README results table with fresh measured values and the
   exact versions used.

   > RESOLVED: 60/60 retrieval in both conditions; decode probe 87/77
   > tok/s baseline vs 61/45 tok/s turboquant (8K/16K). Max context as
   > shipped is 40,960 for BOTH conditions (Qwen3-4B's own position
   > limit); with the YaRN factor-4 override, 43,008 vs 131,072 (3.05x).
   > The original "~45K -> ~128K" row was only reproducible with YaRN --
   > README table updated with measured values + versions, original run
   > kept as history.

## Apple Silicon / llama.cpp (M4, 16 GB)

6. **Model resolution.** Confirm `-hf Qwen/Qwen3-4B-GGUF:Q4_K_M`
   resolves and downloads on current brew llama.cpp. Fallbacks if not:
   the unsloth or bartowski Qwen3-4B GGUF repos; update quickstart.sh
   and verify.py's `HF_GGUF` default.

   > RESOLVED: resolves and downloads (2.32 GiB) on brew llama.cpp
   > build 9960, Mac mini M4 / macOS 15.7.4. No fallback needed.
7. **KV log patterns.** Run verify.py --backend llamacpp and check the
   KV MiB lines match `LLAMACPP_KV_PATTERNS`; add your build's format if
   not.

   > RESOLVED (two bugs found): (a) current builds hide the KV lines at
   > default log level entirely -- verify.py now passes `-lv 4` to the
   > server; (b) the old flat sum matched both the "KV self size" total
   > AND its same-line K/V components, reporting exactly 2x the true MiB
   > (ratios survived only by symmetric luck) -- patterns are now
   > categorized and the first matching category wins. Measured on the
   > M4 at 8K ctx: f16 1,152 MiB; q8_0 612 MiB (1.88x); q4_0 324 MiB
   > (3.56x) -- f16 confirms 144 KiB/token exactly, and the ratios match
   > this doc's ~1.9x / ~3.6x claims.
8. **Thinking mode.** Spot-check demo outputs for `<think>` leakage with
   the `/no_think` suffix on the current server build.

   > MOSTLY RESOLVED: the q8_0 demo scored 5/5 exact-match on the M4
   > (build 9960), which implies `/no_think` worked -- but the raw
   > outputs were not explicitly grepped for `<think>`; do that check
   > when running the full Mac A/B benchmark.
9. **(Optional) Fork path.** If you want real TurboQuant numbers on
   Metal, test the TheTom fork prebuilds (>= the PR #200 Metal fix) and
   add measured results to docs/apple-silicon.md, clearly labeled as
   fork results.

## Both platforms

10. **Memory-math sanity.** cpu_demo derives 288 KiB/token for Qwen3-4B
    from (2 x 36 layers x 8 KV heads x 128 dim x 2 bytes). That figure
    doesn't obviously reconcile with the measured 45K-token baseline on
    a 16 GB card -- possibly different weight precision, engine
    accounting, or a config detail worth pinning down. The docs
    deliberately teach "measure, don't derive" partly for this reason;
    if you find the reconciliation, add a worked example to LEARN.md.

    > RESOLVED: it was an arithmetic slip, not an accounting mystery.
    > 2 x 36 x 8 x 128 x 2 bytes = 147,456 B = **144 KiB**, not 288; the
    > code's table always used 144 KiB, only the prose was wrong (fixed).
    > With 144 KiB everything reconciles to within block rounding:
    > 5.29 GiB measured KV pool / 144 KiB = 38.5K tokens (engine reports
    > 38,544). Worked example added to LEARN.md §1.
11. **Timing.** Reported decode tok/s in benchmark.py is generation
    tokens / wall time per request (includes a small HTTP/engine
    overhead on short generations). Fine for A/B comparison; don't quote
    it as an absolute throughput figure without noting that.

    > RESOLVED (the note understated the problem): for ~10-token NIAH
    > answers the per-request figure is dominated by PREFILL -- it is not
    > a decode rate at all and cannot show the compression's decode cost.
    > benchmark.py now runs a dedicated decode-speed probe per context
    > (paired short/long generations on two distinct same-length
    > documents so prefix caching can't contaminate the delta; llama.cpp
    > path uses the server's own `timings`). The report table now shows
    > "decode tok/s" (the probe) and "e2e tok/s" (the old number,
    > relabeled honestly).
