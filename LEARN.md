# LEARN.md — KV cache compression, properly explained

You can run everything in this repo without reading this file. Read it
anyway: the difference between using a flag and understanding a flag is
the difference between reproducing a benchmark and being fooled by one.

## 1. What the KV cache is

A transformer generates one token at a time, and each new token attends
to every token before it. Recomputing all of that history for every step
would be quadratic misery, so the model caches, per layer, the Key and
Value vectors of every token it has seen. That store is the KV cache: the
model's working memory of your document.

Its size is pure arithmetic:

```
bytes/token = 2 (K and V) x layers x kv_heads x head_dim x bytes/element
Qwen3-4B @ FP16: 2 x 36 x 8 x 128 x 2 = 147,456 bytes = 144 KiB per token
```

Weights are a fixed cost; the KV cache is the cost that *grows with
context*. At long context on consumer hardware it becomes THE bottleneck:
the difference between a 45K and a 128K context on the same 16 GB card is
not a bigger model or a better GPU -- it's what you do to this cache.

(How many tokens actually fit is an engine-and-system question -- weight
precision, activation workspace, allocator behavior. Derive per-token
cost from architecture; **measure** capacity with `scripts/verify.py`.)

### Worked example: reconciling the arithmetic with a measurement

From a real vLLM startup log (RTX 5080 16 GB, WSL2, vLLM 0.25.0,
`gpu_memory_utilization=0.85`, Qwen3-4B loaded in BF16):

```
Model loading took 7.56 GiB
Available KV cache memory: 5.29 GiB
GPU KV cache size: 38,544 tokens
```

Check it: 5.29 GiB / 144 KiB-per-token = 5.29 x 2^30 / 147,456 = ~38.5K
tokens -- the engine's own number, to within block rounding. The budget
side also closes: a 16 GB card is 15.92 GiB; times 0.85 utilization is
13.53 GiB for vLLM; minus 7.56 GiB of BF16 weights and ~0.7 GiB of
activation/workspace reservation (vLLM profiles this at startup) leaves
the 5.29 GiB KV pool above. Everything reconciles once you (a) use the
correct 144 KiB/token -- an earlier draft of this repo stated 288 KiB, a
factor-2 slip that exactly this measurement caught -- and (b) remember
that weights and workspace come off the top.

The compressed condition is the same arithmetic with a different
per-token cost. `turboquant_k3v4_nc` packs each token into 118
bytes/head/layer (3-bit keys + an fp16 norm = 50 B; 4-bit values +
fp16 scale and zero-point = 68 B) on the 32 compressed layers, while 4
boundary-protected layers (see docs/cuda.md) stay BF16 at 512 B:
32x8x118 + 4x8x512 = 46,592 bytes/token. The same 5.29 GiB pool then
holds ~121.9K tokens -- the engine reports 121,984. Ratio: 3.16x. That
is why verify.py prints ~3.16x on this config, not the per-layer 4.34x
and not the headline "~3.5x": measured blends always include metadata
and whatever the backend chose not to compress.

## 2. Quantization in three paragraphs

Quantization stores numbers with fewer bits. FP16 gives you ~65,000
representable magnitudes per number; 3-bit gives you eight. The art is
choosing WHICH eight values (the "grid") so that the vectors you actually
have lose as little as possible.

The naive approach -- scale by the max absolute value, round -- dies on
**outliers**. Attention keys have a few channels with huge values;
covering them stretches the grid so far that every normal value rounds to
the same one or two levels. Run `scripts/cpu_demo.py` and watch retrieval
fall apart in the "3-bit naive" row for exactly this reason.

Two classic fixes, both used here: pick the grid to minimize
mean-squared error rather than to merely cover the max (Lloyd-Max
codebooks in the real kernels; a scale search in our demo), and
**rotate** the vectors first.

## 3. What TurboQuant actually does

The pipeline behind vLLM's `turboquant_k3v4_nc`:

1. **Randomized Walsh-Hadamard rotation.** An orthogonal transform that
   smears outlier energy evenly across all dimensions, making every
   channel roughly Gaussian -- the distribution low-bit grids handle
   best. Orthogonal means dot products (what attention computes) are
   preserved; you can rotate, quantize, and un-rotate without changing
   the geometry the model cares about.
2. **MSE-optimal low-bit grids** on the rotated vectors -- 3-bit for
   Keys, 4-bit uniform for Values in the k3v4 preset.
3. **Norm correction** (`_nc`): quantization systematically shrinks
   vector norms; each vector is rescaled back to its original length.
   Attention scores are dot products, so norms matter, not just
   directions.
4. **Asymmetric K/V budgets.** Keys route attention through a softmax --
   errors there get exponentially amplified and corrupt WHICH tokens the
   model looks at. Values are averaged linearly -- errors there just blur
   WHAT it retrieves, which is far more forgiving. Hence unequal
   treatment of K and V, and hence why community testing consistently
   finds V-side compression nearly free while K-side is where quality
   lives. The paper additionally splits outlier channels into a
   higher-precision side path; most shipping implementations skip that.

## 4. The honesty section

**QJL is not in your build -- anyone's build.** The paper's full recipe
adds a second stage (a quantized Johnson-Lindenstrauss residual
correction, "TurboQuant-prod"). Five-plus independent community groups
found that at 3+ bits it *hurt* end quality -- residual error injected
before a softmax gets amplified, not averaged away -- and every shipping
implementation (vLLM's included, per its own docs) omits it. So "the
TurboQuant everyone runs" is really: randomized rotation + Lloyd-Max
grids + norm tricks. Closer to what the literature calls PolarQuant. The
paper's most aggressive compression claims (4.5-6x at quality parity)
lean on machinery you are not running.

**Perfect NIAH ≠ no quality loss.** Our reference run scored 60/60. vLLM's
own benchmark blog reports `k3v4_nc` at roughly **+10.6% perplexity**,
with meaningful degradation on reasoning and very-long-context tasks --
they flag it as a poor production choice, recommending `4bit_nc` (+2.7%
PPL at 3.8x) when quality matters. Both things are true at once:
single-needle retrieval at 8-16K is simply too easy to detect that
damage. Multi-needle, aggregation tasks, and 64K+ contexts are where
cracks show. A clean NIAH score means "no damage detectable by this
task" -- quote it that way.

**The "8x" in the paper is not memory.** It's attention-computation
speedup on H100-class hardware from operating directly on quantized
data. The memory story is the ~3.5-5x cache ratio, and the end-to-end
story on our 16 GB card was 2.86x more context at ~40% slower decode.
Three different numbers, three different claims. Precision about which
one you're citing is most of what separates a trustworthy writeup from
hype.

## 5. Cache ratio vs context expansion, with numbers

Compression shrinks only the KV slice of memory. On a 16 GB card with
~8-9 GB of weights and 1-2 GB of activations/workspace, the cache pool is
maybe 5-6 GB. Compress the cache 3.2x and *the pool* holds 3.2x more
tokens -- but serving a longer context also grows activation and
workspace needs, which eats part of the win. Measured on the validated
RTX 5080 run (vLLM 0.25.0): 3.16x capacity ratio -> 3.05x realized
context expansion with YaRN rope scaling (43,008 -> 131,072 tokens at
`gpu_memory_utilization=0.9`). The larger the model and the longer the
context, the more the KV cache dominates total memory and the closer
these two numbers converge.

And one ceiling the arithmetic never shows: **the model config caps you
first**. Qwen3-4B ships with `max_position_embeddings=40960`, so without
a YaRN `rope_scaling` override *both* conditions max out at 40,960
tokens -- compression then buys concurrency (3.4x more parallel 40K
requests), not longer context. "How much more context fits" is a
three-way minimum: memory, engine accounting, and the model's own
position limit.

## 6. When you should NOT use this

KV cache compression is a targeted tool, not a default:

* **Short contexts.** At 2-4K tokens the cache is a rounding error; you
  pay the decode slowdown and metadata overhead for nothing.
* **When memory isn't the bottleneck.** If you're compute-bound or
  weight-bound, compress *weights* (GGUF Q4_K_M, AWQ, ...) -- that's a
  different technique solving a different problem, and it usually comes
  first.
* **Vision / short-prompt batch workloads.** A part-photo classifier
  with 500-token prompts gains nothing from a 3.5x smaller cache; batch
  size and weight precision are the levers there. (Ask us how we know.)
* **Quality-critical reasoning at aggressive presets.** See the honesty
  section; use `4bit_nc` or FP8 KV, or don't compress.

Match the technique to the bottleneck, not to the headline number.

## 7. A short history you should know

Two mature inference stacks looked at the same paper in early 2026 and
went opposite ways -- instructive to watch:

* **vLLM merged it** (PR #38479, April 2026) as first-class
  `kv_cache_dtype` presets, after the contributor shipped NIAH and
  GSM8K evidence, then published its own follow-up benchmarks -- which
  is where the unflattering perplexity numbers above come from. Merging
  and honest re-measurement, both.
* **llama.cpp declined it.** The community PR (#21089, tbq3_0/tbq4_0)
  drew real interest but the maintainers closed it in June 2026: nobody
  had shown a clear KLD-per-bit win over the cache quants llama.cpp
  already ships (q4_0/q8_0), *especially* after the core maintainer
  extracted TurboQuant's best idea -- the Hadamard rotation -- into a
  separate change (#21038) that improved the existing quants and
  shrank the gap the new types were supposed to justify. Metal support
  never existed outside forks; even on the PR branch, Apple-GPU KV
  offload crashed and needed a keep-KV-on-CPU workaround.

Neither project was wrong. One valued the capability, one valued the
maintenance budget, and both demanded evidence. The lesson for you as a
learner: "the technique works" and "this project should ship it" are
separate questions -- and always check whether the thing a blog post says
is merged is actually merged, in the actual repo, this month. (This repo
had to correct itself on exactly that point while being written.)

## 8. How our numbers were made (and their limits)

Protocol: 5 unique WORD-NNNN needles x 3 depths (25/50/75%) x 2 contexts
(8K/16K) x 2 jittered reps = 60 trials per condition, greedy decoding,
fixed seed, 64-token budget, exact-match scoring, filler drawn from
public-domain prose. Everything is in `niah/` -- about 200 lines, no
magic, deliberately boring.

Known limits: single needle per document (multi-needle is harder);
contexts stop at 16K (degradation typically shows at 64K+); exact-match
scoring can't detect partial corruption; one model family. It answers
"did compression destroy retrieval at these lengths?" -- nothing more.
When you extend it, extend the claims and the caveats together.

## Further reading

* TurboQuant paper: Zandieh, Daliri, Hadian, Mirrokni (ICLR 2026),
  arXiv:2504.19874
* vLLM: PR #38479 (the backend), the vLLM TurboQuant docs page (preset
  table, QJL note), and the May 2026 vLLM blog benchmark study
* llama.cpp: PR #21089 (closed; read the maintainer discussion), PR
  #21038 (the rotation change), Discussion #20969
* Related work worth knowing: PolarQuant (the rotation+codebook family
  these implementations actually resemble), KVTC (NVIDIA's
  PCA-calibrated alternative), and llama.cpp's own q4_0/q8_0 cache
  types as the baseline any new method must beat
