#!/usr/bin/env python3
"""TurboQuant on your CPU, in one minute, no GPU required.

This is a simplified, educational reimplementation of the core idea behind
TurboQuant-style KV cache compression: rotate vectors with a (randomized)
Walsh-Hadamard transform so outliers get spread across dimensions, then
quantize to very few bits with an MSE-minimizing grid, then correct the
vector norm. It is NOT the production kernel (real implementations use
Lloyd-Max codebooks and fused GPU/Metal/CUDA ops), but the numbers you see
here are the same phenomenon those kernels exploit.

Run it:  python scripts/cpu_demo.py            (full walk-through)
         python scripts/cpu_demo.py --brief    (the one table that matters;
                                               what ./quickstart.sh shows)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from niah.run_summary import record, under_quickstart  # noqa: E402

HEAD_DIM = 128  # typical attention head dimension (must be a power of 2)


# --------------------------------------------------------------------------
# The rotation: fast Walsh-Hadamard transform (orthonormal, self-inverse)
# --------------------------------------------------------------------------
def fwht(x: np.ndarray) -> np.ndarray:
    """Orthonormal FWHT along the last axis. O(d log d)."""
    y = np.array(x, dtype=np.float64, copy=True).reshape(-1, x.shape[-1])
    n = y.shape[-1]
    assert n & (n - 1) == 0, "FWHT needs a power-of-2 dimension"
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = y[:, i : i + h].copy()
            b = y[:, i + h : i + 2 * h].copy()
            y[:, i : i + h] = a + b
            y[:, i + h : i + 2 * h] = a - b
        h *= 2
    return (y / np.sqrt(n)).reshape(x.shape).astype(np.float32)


def make_rotation(dim: int, seed: int):
    """Randomized Hadamard rotation: flip random signs, then FWHT.
    Returns (rotate, unrotate) closures. Orthonormal, so unrotate(rotate(x))
    == x up to float error."""
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=dim).astype(np.float32)
    return (lambda x: fwht(x * signs)), (lambda x: fwht(x) * signs)


# --------------------------------------------------------------------------
# The quantizers
# --------------------------------------------------------------------------
def quant_sym(x: np.ndarray, bits: int) -> np.ndarray:
    """Naive symmetric uniform quantizer: scale = max|x| per vector."""
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(x).max(axis=-1, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    return np.clip(np.round(x / s), -qmax, qmax) * s


def quant_mse(x: np.ndarray, bits: int, n_grid: int = 48) -> np.ndarray:
    """Symmetric uniform quantizer with an MSE-optimal scale, found by a
    small per-vector search. (Real TurboQuant uses Lloyd-Max codebooks
    tuned to the post-rotation distribution; a scale search is our simple
    stand-in for 'pick the grid that minimizes error, not the grid that
    merely covers the max value'.)"""
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(x).max(axis=-1, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    best, best_err = None, None
    for f in np.linspace(0.25, 1.0, n_grid):
        s = amax * f / qmax
        deq = np.clip(np.round(x / s), -qmax, qmax) * s
        err = ((x - deq) ** 2).sum(axis=-1, keepdims=True)
        if best is None:
            best, best_err = deq, err
        else:
            m = err < best_err
            best = np.where(m, deq, best)
            best_err = np.where(m, err, best_err)
    return best


def quant_minmax(x: np.ndarray, bits: int) -> np.ndarray:
    """Asymmetric min/max uniform quantizer (the classic choice for
    Values, which tolerate noise better than Keys)."""
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    step = np.where(hi > lo, (hi - lo) / (2**bits - 1), 1.0)
    return np.round((x - lo) / step) * step + lo


def norm_correct(deq: np.ndarray, orig: np.ndarray) -> np.ndarray:
    """Rescale each dequantized vector to its original L2 norm (the '_nc'
    in turboquant_k3v4_nc). Attention cares about directions AND norms;
    quantization tends to shrink norms, and this puts them back."""
    on = np.linalg.norm(orig, axis=-1, keepdims=True)
    dn = np.linalg.norm(deq, axis=-1, keepdims=True)
    return deq * (on / np.where(dn == 0, 1.0, dn))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def cos_sim(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return float((num / np.where(den == 0, 1.0, den)).mean())


def norm_error(orig, deq):
    on = np.linalg.norm(orig, axis=-1)
    dn = np.linalg.norm(deq, axis=-1)
    return float((np.abs(dn - on) / np.where(on == 0, 1.0, on)).mean())


def retrieval_top1(queries, keys_deq, targets):
    """Miniature needle-in-a-haystack: each query is a noisy copy of one
    specific key (its 'needle'). Does argmax(q . K) over the QUANTIZED
    keys still find it? This is the property attention actually needs:
    the model must still 'look at' the right token."""
    return float((queries @ keys_deq.T).argmax(-1).__eq__(targets).mean())


# --------------------------------------------------------------------------
# Demo sections
# --------------------------------------------------------------------------
# Qwen3-4B: 2 (K+V) x 36 layers x 8 KV heads x 128 dims x 2 bytes (FP16)
PER_TOKEN_BYTES = 2 * 36 * 8 * 128 * 2


def section_memory_math():
    print("=" * 72)
    print("1. WHY THE KV CACHE IS THE PROBLEM")
    print("=" * 72)
    print(
        """
Every token the model reads leaves behind a Key vector and a Value vector
in every layer -- that's the KV cache, the model's working memory. It grows
linearly with context. For Qwen3-4B (36 layers, 8 KV heads, head_dim 128)
at FP16, that is:

    2 (K+V) x 36 layers x 8 heads x 128 dims x 2 bytes = 144 KiB PER TOKEN
"""
    )
    per_tok = PER_TOKEN_BYTES
    print(f"    {'context':>10} | {'FP16 cache':>12} | {'~3.5x compressed':>16}")
    print(f"    {'-'*10} | {'-'*12} | {'-'*16}")
    for tokens in (4096, 16384, 65536, 131072):
        gb = per_tok * tokens / 2**30
        print(f"    {tokens:>10,} | {gb:>8.1f} GiB | {gb/3.5:>12.1f} GiB")
    print(
        """
One thing this table does NOT tell you: how many tokens fit on YOUR GPU.
That depends on what else lives in VRAM (weights, activations, workspace)
and on the engine's memory accounting. Derive per-token cost from the
architecture; MEASURE capacity on real hardware (that's what scripts/
verify.py and the benchmark report). Never trust arithmetic where a
measurement is available.
"""
    )


def quantizer_results(seed: int) -> list[dict]:
    """Quantize one synthetic Key tensor several ways; return one row of
    metrics per scheme (name, rmse, cos, norm_err, retrieval, n_queries)."""
    rng = np.random.default_rng(seed)

    # Synthetic "Keys": mostly Gaussian, with a few outlier channels --
    # the pattern that makes naive low-bit quantization fall apart.
    n_keys = 2048
    keys = rng.standard_normal((n_keys, HEAD_DIM)).astype(np.float32)
    outlier_channels = rng.choice(HEAD_DIM, size=4, replace=False)
    keys[:, outlier_channels] *= 4.0

    # Miniature NIAH: each query is a noisy copy of one specific key, and
    # for each query we also plant a DISTRACTOR key -- cosine ~0.92 with
    # the true target, same norm. With FP16 keys, argmax(q . K) still finds
    # the true key essentially every time; the margin over the distractor
    # is real but thin. The question: does that thin margin survive 3-bit
    # quantization? (This is what long-context retrieval boils down to.)
    n_queries = 512
    targets = rng.integers(0, n_keys, size=n_queries)
    t_keys = keys[targets]
    queries = t_keys + 0.22 * rng.standard_normal(
        (n_queries, HEAD_DIM)
    ).astype(np.float32)

    nd = rng.standard_normal(t_keys.shape).astype(np.float32)
    t_norm = np.linalg.norm(t_keys, axis=-1, keepdims=True)
    nd_norm = np.linalg.norm(nd, axis=-1, keepdims=True)
    cos_target = 0.92
    distractors = cos_target * t_keys + nd * (
        t_norm * np.sqrt(1 - cos_target**2) / nd_norm
    )
    keys = np.vstack([keys, distractors])  # distractors join the haystack

    rotate, unrotate = make_rotation(HEAD_DIM, seed)
    rot_mse = unrotate(quant_mse(rotate(keys), 3))

    candidates = {
        "FP16 (no quantization)": keys,
        "3-bit naive (no rotation)": quant_sym(keys, 3),
        "3-bit + WHT rotation + MSE grid": rot_mse,
        "  ... + norm correction ('_nc')": norm_correct(rot_mse, keys),
        "Values-style 4-bit min/max ('v4')": quant_minmax(keys, 4),
    }

    return [
        dict(name=name, rmse=rmse(keys, deq), cos=cos_sim(keys, deq),
             norm_err=norm_error(keys, deq),
             retrieval=retrieval_top1(queries, deq, targets),
             n_queries=n_queries)
        for name, deq in candidates.items()
    ]


def section_quantizer(rows: list[dict]):
    print("=" * 72)
    print("2. THE QUANTIZER, ON ONE TENSOR")
    print("=" * 72)
    print(
        f"\n{'scheme':<35} {'RMSE':>7} {'cos sim':>8} {'norm err':>9} {'retrieval':>10}"
    )
    print(f"{'-'*35} {'-'*7} {'-'*8} {'-'*9} {'-'*10}")
    for r in rows:
        print(
            f"{r['name']:<35} {r['rmse']:>7.4f} {r['cos']:>8.4f} "
            f"{r['norm_err']:>8.1%} {r['retrieval']:>9.1%}"
        )
    print(
        """
What just happened: a handful of outlier channels forced the naive 3-bit
grid to spend its 8 levels covering huge values, crushing everything else
toward zero. The Hadamard rotation smeared those outliers evenly across
all 128 dimensions (making every dimension roughly Gaussian), so the same
3 bits suddenly go much further. Norm correction then repairs the
systematic norm shrinkage ('norm err' column) that dot-product attention
is sensitive to. 'retrieval' is the number to care about: it is a
miniature needle-in-a-haystack asking whether attention still finds the
right token after compression.
"""
    )


def section_ratio_vs_expansion():
    print("=" * 72)
    print("3. CACHE RATIO IS NOT CONTEXT EXPANSION (the #1 confusion)")
    print("=" * 72)
    print(
        """
Two different numbers, constantly conflated:

  * CACHE COMPRESSION RATIO -- how much smaller each cached token gets.
    For k3v4-style settings this lands around 3.5-5x depending on
    metadata overhead (scales, norms) and which layers are covered.

  * CONTEXT EXPANSION -- how much more text actually fits on your GPU.
    On the validated 16 GB RTX 5080 run this was 3.05x (43K -> 131K,
    with YaRN rope scaling; without it, the MODEL's own 40,960-token
    position limit caps both conditions first).

Why is expansion smaller? Compression only shrinks the KV slice of memory.
Weights don't shrink. Activation and attention workspace don't shrink --
and some of that grows WITH context length, eating into the headroom the
compression just created.

Example: a 16 GB card holding 8 GB of weights and ~2 GB of other overhead
leaves a ~6 GB pool for the KV cache. A 3.5x cache ratio lets that pool
hold 3.5x more tokens -- but running a longer context also needs more
workspace, which comes out of the same card, so the context you can
actually run grows by LESS than 3.5x.

Rule for the README of your life: quote the compression ratio as a cache
property, quote context expansion only as a measured, hardware-specific
result -- and never present one as the other. The bigger the model and the
longer the context, the more KV dominates and the closer the two numbers
get."""
    )


# Row names from quantizer_results() that the brief view shows, with the
# beginner-facing label and an optional pointer. The '_nc' row stays in
# even though it scores a hair lower on this toy test: it is the setting
# Steps 2-3 actually run, and showing only the best row would be spin.
BRIEF_ROWS = [
    ("FP16 (no quantization)", "uncompressed (FP16)", ""),
    ("3-bit naive (no rotation)", "3-bit, naive", "<- a few outlier values wreck it"),
    ("3-bit + WHT rotation + MSE grid", "3-bit + rotation + better grid",
     "<- the TurboQuant idea"),
    ("  ... + norm correction ('_nc')", "... + norm correction ('_nc')",
     "<- what Steps 2-3 use"),
]


def section_brief(rows: list[dict]):
    by_name = {r["name"]: r for r in rows}
    per_tok_kib = PER_TOKEN_BYTES / 1024
    gib_131k = PER_TOKEN_BYTES * 131072 / 2**30
    print(
        f"""The KV cache is the model's working memory: every token it reads leaves
a Key and a Value vector in every layer. For Qwen3-4B that costs
{per_tok_kib:.0f} KiB per token at FP16 -- {gib_131k:.1f} GiB at 131,072 tokens.
TurboQuant stores Keys in 3 bits and Values in 4, instead of 16 bits each.
Does attention still find the right token afterwards?

  Toy needle test on synthetic vectors (retrieval, higher is better):
"""
    )
    for key, label, note in BRIEF_ROWS:
        r = by_name[key]
        print(f"    {label:<34} {r['retrieval']:>6.1%}   {note}".rstrip())
    rot = by_name["3-bit + WHT rotation + MSE grid"]
    nc = by_name["  ... + norm correction ('_nc')"]
    gap = round(abs(rot["retrieval"] - nc["retrieval"]) * rot["n_queries"])
    print(
        f"""
Rotation spreads those outliers across all 128 dimensions, so the same
3 bits go much further. Norm correction puts each vector's length back
(attention is sensitive to it); on this small toy test it differs from
plain rotation by {gap} of {rot['n_queries']} queries.
Full walk-through (memory table, error metrics, ratio vs. context):
  python scripts/cpu_demo.py"""
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--brief", action="store_true",
                    help="print only the retrieval table and the idea "
                         "behind it (what ./quickstart.sh shows)")
    args = ap.parse_args()

    rows = quantizer_results(args.seed)
    by_name = {r["name"]: r for r in rows}
    record("cpu", {
        "fp16": by_name["FP16 (no quantization)"]["retrieval"],
        "naive": by_name["3-bit naive (no rotation)"]["retrieval"],
        "rotated": by_name["3-bit + WHT rotation + MSE grid"]["retrieval"],
        "rotated_nc": by_name["  ... + norm correction ('_nc')"]["retrieval"],
        "n_queries": rows[0]["n_queries"],
        "seed": args.seed,
    })

    if args.brief:
        section_brief(rows)
        return

    print("\nTurboQuant-101 CPU demo -- no GPU, just the ideas.\n")
    section_memory_math()
    section_quantizer(rows)
    section_ratio_vs_expansion()
    if under_quickstart():
        return  # quickstart.sh prints its own next steps
    print(
        "\nNext step: run the real thing on a model.\n"
        "  NVIDIA GPU:     see docs/cuda.md   (vLLM, turboquant_k3v4_nc)\n"
        "  Apple Silicon:  see docs/apple-silicon.md (llama.cpp KV cache types)\n"
        "Or just run ./quickstart.sh and let it detect your platform.\n"
    )


if __name__ == "__main__":
    main()
