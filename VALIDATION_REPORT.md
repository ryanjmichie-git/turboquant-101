# Hardware validation report — turboquant-101

**Date:** 2026-07-12 · **Scope:** Phase 0 (census), Phase 1 (CPU demo),
Phase 2 (CUDA/vLLM path). Phase 3 (Apple Silicon / llama.cpp) was **not
run** — this is the CUDA machine; Q7/Q8 remain open for the M4 pass.

**Verdict:** the repo's core claims survive on real hardware — verify.py
PASSES (3.16x), retrieval is 60/60 in both conditions, and the decode
cost is real (~-42% at 16K). But the shipped repo could not run at all
as written: the vLLM pin predates the TurboQuant backend entirely, and
three independent environment bugs each prevented engine startup on a
fresh WSL2 box. Several documented numbers (max context, decode
methodology, the 288 KiB/token arithmetic, paper author names) did not
survive contact with measurement and have been corrected.

---

## 1. Environment census

| item | value |
|---|---|
| Host OS | Windows 11 Pro 10.0.26200 |
| WSL | 2.6.3.0, kernel 6.6.87.2-microsoft-standard-WSL2 |
| Distro | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB (15.92 GiB), Blackwell sm_120 (compute 12.0) |
| Driver / CUDA | 591.74 (WDDM) / CUDA 13.1 (per nvidia-smi) |
| CPU / RAM | AMD Ryzen 9 9950X3D / 64 GB (31 GiB visible in WSL) |
| Python | 3.10.12 (WSL), venv `.venv` in-repo, pip 26.1.2 |
| Repo location | `~/turboquant-101` on ext4 (copied off `/mnt/c` per docs) |
| Ambient VRAM | ~1,004 MiB used by Windows desktop processes at start (dwm, Edge, VS Code, Claude, ChatGPT, Teams, Ollama — `ollama ps` showed **no** model resident). Nothing was closed. |
| Model | Qwen/Qwen3-4B, BF16 safetensors, 7.6 GB snapshot `1cfa9a72` (HF cache already contained blobs dated 2026-05-04 — this machine had run Qwen3-4B before) |

Validated software stack: **vllm 0.25.0, torch 2.11.0+cu130, transformers
5.13.1, flashinfer 0.6.13 (sampler disabled), Python 3.10.12**.

## 2. Phase 1 — CPU demo (control)

PASS. Naive 3-bit retrieval 82.8% (well below FP16's 98.6%); rotated+MSE
3-bit 98.2% ≈ FP16; norm correction takes norm error to 0.0%. Matches the
expected story exactly.

One friction point fixed: stock Ubuntu/WSL2 lacks `python3.10-venv`, so
`python3 -m venv` fails (and there is no passwordless sudo). quickstart.sh
now falls back to `venv --without-pip` + a get-pip bootstrap.

## 3. Answers to the maintainer-notes questions

### Q1 — which exact vLLM version works? Does vllm==0.19.1 install and run?

`vllm==0.19.1` **installs and runs** on this machine (its bundled torch
2.10.0+cu128 supports sm_120 natively — no PyTorch nightly needed), **but
it does not contain the TurboQuant backend at all**: its `CacheDType`
literal has no `turboquant_*` values and the package has zero turboquant
code. The pin's comment ("first release with the merged PR #38479
backend") was wrong: 0.19.1 was released 2026-04-18 but cut from the
pre-merge 0.19 branch; PR #38479 merged into main 2026-04-15 and **first
shipped in v0.20.0 (2026-04-27)**.

Verified working pin: **vllm==0.25.0** (pulls torch 2.11.0+cu130; sm_120
in the wheel's arch list; kernels verified on the RTX 5080).
`requirements-cuda.txt` updated accordingly. Three environment fixes were
required to actually run it (each fatal on a fresh WSL2 install):

1. **torchcodec import crash.** vLLM 0.25.0's `torchcodec` dependency
   raises `RuntimeError` at `import vllm` on machines without FFmpeg
   shared libraries, escaping vLLM's `except ImportError` guard. Fix:
   uninstall torchcodec (repo is text-only); quickstart.sh now does this
   automatically if `import vllm` fails.
2. **"RuntimeError: UVA is not available".** vLLM ≥0.25's GPU worker
   requires pinned host memory, which vLLM disables by default under
   WSL2. Fix: `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, now set (setdefault) by
   `niah/backends.py` and `scripts/verify.py`.
3. **FlashInfer sampler JIT.** Engine warm-up JIT-compiles FlashInfer's
   sampling kernels, needing nvcc + a host toolchain; no CUDA toolkit
   exists on a fresh WSL2, and pointing `CUDA_HOME` at the pip-shipped
   cu13 toolkit fails anyway ("CUDA compiler and CUDA toolkit headers are
   incompatible" against flashinfer 0.6.13's bundled CCCL). Fix: scripts
   default to `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampler; identical
   for greedy decoding; overridable).

Cosmetic: a stale `nvidia-cutlass-dsl-libs-cu12` from the 0.19.1 install
triggers a pip resolver warning next to 0.25.0's cu13 set; harmless in
practice (fresh installs never see it).

### Q2 — verify.py detection path

Capacity came from the **log regex** `GPU KV cache size:\s*([\d,]+)\s*tokens`
(first pattern in `VLLM_LOG_PATTERNS`) for both probes. The attribute
chains cannot be relied on in vLLM 0.25.0: the V1 engine core runs in a
separate process, and although `llm_engine.vllm_config.cache_config.num_gpu_blocks`
did resolve in some runs, `num_gpu_blocks × block_size` **under-reports
turboquant capacity by exactly 2×** (TQ's packed slots store more tokens
per block than `cache_config.block_size` implies; measured 70,160 via
attrs vs 140,320 via the engine log for the same engine). Had the attr
chain fired for both probes, verify would have reported ~1.6x — a false
FAIL. verify.py was patched to (a) prefer the engine log, (b) print which
source produced each number, and (c) compute the "Maximum concurrency"
fallback correctly (it previously captured `max_model_len`, which would
have returned identical numbers for both probes → false FAIL).

Measured (defaults: max_model_len 8192, util 0.85): baseline **38,544**
tokens, compressed **121,984** tokens, ratio **3.16x** → PASS, exit 0.
Re-verified on the final script state: same numbers, PASS, exit 0.

### Q3 — skip_layers

The real mechanism in this vLLM version is
`kv_cache_dtype_skip_layers: list[str]` (CLI
`--kv-cache-dtype-skip-layers`; accepts layer indices or
`"sliding_window"`). It **is** wireable via
`--extra-engine-kwargs '{"kv_cache_dtype_skip_layers": [...]}'` — but it
can only *add* skipped layers. Whenever a turboquant dtype is set,
`create_engine_config` unconditionally **unions** the user list with
`TurboQuantConfig.get_boundary_skip_layers(model_config, n=2)`: on dense
models the first 2 and last 2 attention layers are always left
uncompressed (vLLM's rationale, in-source: GSM8K drops ~30 points on
Qwen3-4B without it on aggressive presets). There is no flag, env var, or
kwarg to disable it. **`skip_layers=0` — the reference protocol's clean
A/B — is not achievable in this vLLM version.** All published numbers use
the default coverage: 32/36 layers compressed (layers 0, 1, 34, 35 stay
BF16). Documented in docs/cuda.md ("Which layers actually get
compressed"), including the arithmetic consequence: per-compressed-layer
ratio 4.34x (512 B → 118 B) blends to the measured 3.16x.

### Q4 — `<think>` leakage

None. demo.py with `turboquant_k3v4_nc` retrieved **5/5** with bare-code
answers; `grep -ci think` over the full run log = 0. The
`chat_template_kwargs={"enable_thinking": False}` path works directly on
vLLM 0.25.0 — the TypeError fallback never fired (0 occurrences).

### Q5 — max context at gpu_memory_utilization=0.9

Engine-init bisection (verify.py's probe subprocess; init success/failure
as the signal):

| condition | max_model_len that initializes | limiter |
|---|---|---|
| baseline, as shipped | **40,960** | model config (`max_position_embeddings`) |
| turboquant_k3v4_nc, as shipped | **40,960** | model config |
| baseline, YaRN ×4 override | **43,008** (44,032 fails) | memory (KV capacity 43,951 tok) |
| turboquant_k3v4_nc, YaRN ×4 | **131,072** | YaRN config ceiling (memory would allow ~139K) |

The headline discovery: **Qwen3-4B's own config caps context at 40,960
tokens**, so as shipped the reference-style "max context that fits"
comparison is 40,960 vs 40,960 — compression buys *concurrency* (KV
capacity 44,336 vs 140,320 tokens; "maximum concurrency" 1.07x vs 3.43x),
not length. Only with Qwen's documented YaRN rope-scaling override
(factor 4) does context expansion become measurable: **43,008 → 131,072 =
3.05x**, vs the reference claim of ~45K → ~128K (2.86x). The original
reference numbers are only consistent with a YaRN-style override that was
never recorded. verify.py gained `--hf-overrides` so this is now
reproducible first-class.

### Q6 — memory-math reconciliation

Resolved: it was an **arithmetic slip in the docs, not an accounting
mystery**. `2 × 36 × 8 × 128 × 2 bytes = 147,456 B = 144 KiB/token` — the
prose in cpu_demo.py and LEARN.md stated "288 KiB" for its own formula
(the code's table always used the correct value). With 144 KiB everything
closes to within block rounding, from the live engine logs (util 0.85):

- BF16 weights: "Model loading took 7.56 GiB"
- "Available KV cache memory: 5.29 GiB" → 5.29 GiB ÷ 144 KiB = 38.5K
  tokens; engine reports 38,544 ✓
- Budget: 15.92 GiB × 0.85 = 13.53 GiB − 7.56 (weights) − ~0.7
  (activation/workspace profiling) ≈ 5.29 GiB ✓
- Compressed: 32×8×118 B + 4×8×512 B = 46,592 B/token → 5.29 GiB ÷
  46,592 B = 121.9K; engine reports 121,984 ✓ (ratio 3.165x = measured)

A worked example with these numbers was added to LEARN.md §1. The naive
"288 KiB → can't fit 45K on 16 GB" tension that motivated note #10
disappears once the factor-2 slip is fixed.

### Q7 / Q8 — Apple Silicon (GGUF resolution, KV log patterns)

**Not validated** — this pass ran on the CUDA machine; llama.cpp/M4 items
(maintainer notes 6–9, quickstart mac path, `/no_think` on llama-server)
remain open. No guesses recorded.

## 4. Benchmark results (validated run)

Protocol: 5 needles × 3 depths (25/50/75%) × 2 contexts (8K/16K) × 2
jittered reps = 60 trials/condition, greedy, seed 1234, max_tokens 64.
Stack: vllm 0.25.0 / torch 2.11.0+cu130 / Qwen3-4B BF16 / RTX 5080 16 GB
/ WSL2. `max_model_len=20000`, `gpu_memory_utilization=0.90`.

```
label             context  retrieved  decode tok/s  e2e tok/s
---------------- -------- ---------- ------------- ----------
baseline            8,192     30/30             87          9
baseline           16,384     30/30             77          4
turboquant          8,192     30/30             61          9
turboquant         16,384     30/30             45          4
```

- Retrieval: **60/60 both conditions** — matches the original reference.
- Decode cost at 16K: 77 → 45 tok/s (**−42%**), consistent with the
  original reference's ~79 → ~46 and the "compression costs speed" story.
- "decode tok/s" is a new dedicated probe (see §5 item 9) — the
  previously shipped per-trial number ("e2e tok/s" column) divides
  ~10-token answers by whole-request time including prefill and is
  useless as a decode A/B (it reads 9 vs 9 and 4 vs 4 above).

Max-context and capacity numbers: §3 Q5.

## 5. Every change made to the repo

1. **requirements-cuda.txt** — repinned `vllm==0.19.1` → `vllm==0.25.0`;
   comment now records the 0.19.1/0.20.0 release history and the
   torchcodec gotcha. *Why:* 0.19.1 has no TurboQuant backend; 0.25.0 is
   the version actually validated on this hardware.
2. **niah/backends.py** — `VLLMBackend` sets
   `VLLM_WSL2_ENABLE_PIN_MEMORY=1` and `VLLM_USE_FLASHINFER_SAMPLER=0`
   (both `setdefault`, overridable) before importing vLLM. *Why:* engine
   cannot start on WSL2 / without a CUDA toolkit otherwise (Q1).
3. **niah/backends.py** — added `decode_speed()` to both backends: vLLM
   uses paired short/long generations on two *distinct* same-length
   documents (`ignore_eos=True`), so prefill cancels and prefix caching
   cannot contaminate the delta; llama.cpp uses the server's own
   `timings.predicted_per_second`. *Why:* the shipped per-trial "tok/s"
   is prefill-dominated for 10-token answers (maintainer note #11
   understated this); the first probe design without distinct documents
   measured an impossible 182 tok/s at 16K vs 115 at 8K due to vLLM's
   default prefix caching.
4. **scripts/benchmark.py** — runs the decode probe per context (distinct
   needle seed so probe prompts can't prefix-collide with trial prompts),
   stores `decode_probe` in results JSON; report table now shows
   `decode tok/s` and relabels the old column `e2e tok/s` with an
   explanatory footnote. *Why:* same as (3).
5. **scripts/verify.py** — probe env fixes (same as 2); prefers the
   engine startup log over attribute chains and prints which source
   produced each capacity number; fixed the "Maximum concurrency"
   fallback pattern to multiply `max_model_len × multiplier` (it
   previously captured max_model_len itself → identical values for both
   probes → guaranteed false FAIL if it ever fired); records the matched
   attr chain in the probe JSON; optional `TQ101_PROBE_LOG_DIR` dump of
   full probe logs (tags include an `-ovr` suffix for override runs);
   new `--hf-overrides` flag for probing beyond the model's native
   context. *Why:* Q2's attr-chain 2× under-report for TQ configs; Q5
   needs rope overrides; the probe logs are the ground truth for the
   memory math.
6. **quickstart.sh** — venv creation falls back to
   `--without-pip` + get-pip.py when `python3 -m venv` fails (stock
   Ubuntu without python3-venv, no sudo); uninstalls torchcodec if
   `import vllm` fails after the CUDA install. *Why:* both were hit on
   this fresh WSL2 box.
7. **scripts/cpu_demo.py** — "288 KiB PER TOKEN" → "144 KiB PER TOKEN"
   (prose now matches its own formula and the code); context-expansion
   blurb updated to measured numbers + model-ceiling caveat.
8. **LEARN.md** — 144 KiB fix (§1) plus a fully-measured worked example
   reconciling arithmetic with the live engine (§1); §5 updated with
   measured 3.16x → 3.05x numbers and a new paragraph on the model's
   position-limit ceiling; paper author names corrected in Further
   reading.
9. **README.md** — reference-results table replaced with the validated
   run (hardware + versions labeled), original reference run kept and
   marked as pre-validation history with its YaRN inconsistency noted;
   "one distinction" section updated (3.05x, layer-coverage and
   model-ceiling caveats); credits author names corrected ("Hanov &
   Zadeh" → "Hadian & Mirrokni", per arXiv:2504.19874).
10. **docs/cuda.md** — new section "Which layers actually get compressed"
    (boundary protection, union semantics, 4.34x-vs-3.16x arithmetic);
    gotchas updated: WSL2 UVA/pin-memory, torchcodec, FlashInfer-sampler
    JIT, Blackwell marked verified-working with the pin, attention
    backend section rewritten for 0.25.0 (dedicated TURBOQUANT backend +
    expected FA2 downgrade warning; #40094 workaround kept for old
    versions only); "healthy run" numbers replaced with measured values;
    PR-shipping note (v0.20.0).
11. **docs/maintainer-notes.md** — validation-status preamble; inline
    `RESOLVED` annotations under items 1–5, 10, 11 with the measured
    outcomes; items 6–9 explicitly marked not-yet-validated.
12. **VALIDATION_REPORT.md** — this file.

## 6. Docs claims contradicted by measured reality

Fixed (unambiguous):

- "0.19.1 is the first release with the backend" — false; 0.20.0 is
  (requirements-cuda.txt, docs/cuda.md).
- "288 KiB per token" — arithmetic slip; 144 KiB (cpu_demo.py, LEARN.md).
- "Max context ~45K → ~128K (2.86x)" presented unconditionally — as
  shipped, both conditions cap at the model's 40,960 limit; the expansion
  requires a YaRN override (README, docs/cuda.md, LEARN.md).
- Issue #40094 TRITON_ATTN workaround presented as the expected path —
  obsolete on the pinned version (docs/cuda.md).
- Paper credited to "Zandieh, Daliri, Hanov & Zadeh" — actual authors
  Zandieh, Daliri, Hadian, Mirrokni (README, LEARN.md).
- Maintainer note #11's "small HTTP/engine overhead" — the overhead is
  the entire prefill; the number was never a decode rate (benchmark.py,
  maintainer-notes annotation).

Verified accurate (no change needed): the four-preset table and its
ratio/PPL numbers (match vLLM's docs); the QJL-omitted story (verbatim in
vLLM's own source docstring); the llama.cpp history in LEARN.md §7
(PR #21089 closed unmerged 2026-06-02; PR #21038 rotation merged
2026-04-01); PR #38479 author and merge date; the vLLM blog's
FP8-first/avoid-k3v4_nc guidance.

## 7. Remaining risks for publishing

1. **Apple Silicon path is entirely unvalidated** (Q7/Q8, maintainer
   notes 6–9, quickstart mac branch, `LLAMACPP_KV_PATTERNS`, the
   `-hf Qwen/Qwen3-4B-GGUF:Q4_K_M` resolution, the TheTom-fork story).
   Run the M4 pass before publishing, or mark the mac path as untested.
2. **The vLLM environment is fragile across versions.** Three of the
   four startup blockers found here are vLLM/dependency packaging issues
   (torchcodec, UVA-on-WSL2, FlashInfer JIT) that may appear, disappear,
   or mutate in other releases; the pin comment says "see
   maintainer-notes before bumping" and that advice is now load-bearing.
3. **Boundary protection is upstream policy.** If a future vLLM makes
   n configurable (or changes n=2), the measured 3.16x and the
   layer-coverage doc section shift; the numbers are correct only for
   the pinned version.
4. **Decode-speed probe granularity.** The probe is one paired
   measurement per context per run (no reps); run-to-run variance of a
   few tok/s is expected. Fine for the A/B story; don't quote single
   tok/s digits as precise.
5. **Windows desktop VRAM pressure.** ~1 GB of ambient VRAM use by the
   host desktop was present during all measurements; a cleaner headless
   box would shift capacities slightly upward. (Nothing was closed
   during this pass.)
6. **The original reference run's provenance** (whose numbers seeded the
   README) remains unknown — its context row implies an unrecorded YaRN
   override. The README now says this explicitly; if the original run's
   notes surface, reconcile.
