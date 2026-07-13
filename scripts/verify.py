#!/usr/bin/env python3
"""VERIFY THAT COMPRESSION IS ACTUALLY ENGAGED.

This is the most important script in the repo. Multiple public TurboQuant
"benchmarks" silently measured the plain FP16 baseline because the
compression hook never fired (torch.compile bypassing monkey-patches,
prefix caching evading capture, etc.) -- and nothing errored. The numbers
looked plausible and were completely wrong.

The defense is simple: measure KV capacity twice, baseline vs compressed,
and demand a big ratio. Memory doesn't lie.

Usage:
  CUDA / vLLM:
      python scripts/verify.py --backend vllm
      (spawns two short-lived engines: baseline, then turboquant_k3v4_nc)

  llama.cpp:
      python scripts/verify.py --backend llamacpp [--ctk q8_0 --ctv q8_0]
      (spawns llama-server twice: f16 KV, then your chosen cache types,
       and reads the KV buffer sizes it reports at startup)

Exit code 0 = PASS, 1 = FAIL, 2 = couldn't measure (see output).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

DEFAULT_MODEL = "Qwen/Qwen3-4B"
HF_GGUF = "Qwen/Qwen3-4B-GGUF:Q4_K_M"  # llama-server auto-downloads via -hf

PASS_RATIO = 2.0  # compressed capacity must be at least this x baseline
EXPECT_HINT = "~3-4.5x is typical for k3v4-style settings"


def banner(ok: bool | None, msg: str):
    tag = {True: "PASS", False: "FAIL", None: "????"}[ok]
    line = "=" * 72
    print(f"\n{line}\n  [{tag}]  {msg}\n{line}\n")


# ==========================================================================
# vLLM path
# ==========================================================================
# The probe reads the engine's own startup log first (authoritative --
# it accounts for the real per-layer cache layout), then falls back to
# known engine attribute chains. The attr fallback can under-report
# capacity for turboquant dtypes (num_gpu_blocks x block_size assumes the
# baseline layout), which is why it is no longer preferred. If BOTH fail
# on your vLLM version, paste your startup log into a new pattern below --
# and please open an issue so we can add it.
VLLM_LOG_PATTERNS = [
    # Primary (vLLM V1, verified on 0.25.0): "GPU KV cache size: 121,984 tokens"
    (re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"), "tokens"),
    # Fallback: "Maximum concurrency for 8,192 tokens per request: 14.89x"
    # -- capacity is max_model_len x multiplier. (Capturing only the first
    # number would return max_model_len for BOTH probes -> false FAIL.)
    (re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:"
                r"\s*([\d.]+)x"), "concurrency"),
    # Legacy V0 engines: "# GPU blocks: 2409"
    (re.compile(r"#\s*GPU blocks:\s*([\d,]+)"), "blocks"),
]
VLLM_ATTR_CHAINS = [
    "llm_engine.cache_config.num_gpu_blocks",
    "llm_engine.vllm_config.cache_config.num_gpu_blocks",
    "llm_engine.model_executor.cache_config.num_gpu_blocks",
]


def _vllm_probe(cache_dtype: str | None, max_model_len: int, gpu_mem_util: float,
                hf_overrides: str | None = None):
    """Child-process mode: start an engine, report KV capacity as JSON."""
    from vllm import LLM

    kwargs = dict(
        model=DEFAULT_MODEL,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True,  # skip CUDA graph capture; startup only
    )
    if cache_dtype:
        kwargs["kv_cache_dtype"] = cache_dtype
    if hf_overrides:
        # e.g. '{"rope_scaling": {"rope_type": "yarn", "factor": 4.0,
        #        "original_max_position_embeddings": 32768}}' to probe
        # context lengths beyond the model's native limit.
        kwargs["hf_overrides"] = json.loads(hf_overrides)
    llm = LLM(**kwargs)

    result = {"attr_tokens": None, "attr_chain": None, "block_size": 16}
    for chain in VLLM_ATTR_CHAINS:
        obj = llm
        try:
            for part in chain.split("."):
                obj = getattr(obj, part)
            if obj:
                bs = 16
                try:
                    bs = llm.llm_engine.cache_config.block_size or 16
                except Exception:
                    pass
                result["attr_tokens"] = int(obj) * int(bs)
                result["attr_chain"] = chain
                break
        except Exception:
            continue
    print("PROBE_RESULT " + json.dumps(result), flush=True)


def _run_vllm_probe(cache_dtype, max_model_len, gpu_mem_util,
                    hf_overrides=None) -> tuple[int | None, str]:
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--_vllm-probe", cache_dtype or "",
        "--max-model-len", str(max_model_len),
        "--gpu-mem-util", str(gpu_mem_util),
    ]
    if hf_overrides:
        cmd += ["--hf-overrides", hf_overrides]
    env = dict(os.environ, VLLM_LOGGING_LEVEL=os.environ.get("VLLM_LOGGING_LEVEL", "INFO"))
    # vLLM >= 0.25 requires UVA (pinned host memory) in its GPU worker, but
    # disables pinned memory by default under WSL2; without this the engine
    # dies at startup with "RuntimeError: UVA is not available". Only
    # consulted when vLLM detects WSL, so it is harmless elsewhere.
    env.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    # vLLM's FlashInfer sampler JIT-compiles CUDA at first use (during
    # engine warm-up), which requires a full CUDA toolkit + compiler on the
    # host -- absent on a fresh WSL2 install, and the pip-shipped toolkit's
    # headers clash with FlashInfer's bundled CCCL. This repo only does
    # greedy decoding, so default to vLLM's native sampler. Export
    # VLLM_USE_FLASHINFER_SAMPLER=1 to override.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    log = proc.stdout + "\n" + proc.stderr

    # Optional debugging aid: dump each probe's full engine log to a file
    # (set TQ101_PROBE_LOG_DIR=/some/dir). The startup log is the ground
    # truth for memory accounting; see docs/maintainer-notes.md #10.
    dump_dir = os.environ.get("TQ101_PROBE_LOG_DIR")
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        tag = f"{cache_dtype or 'baseline'}-len{max_model_len}-util{gpu_mem_util}"
        if hf_overrides:
            tag += "-ovr"
        with open(os.path.join(dump_dir, f"probe-{tag}.log"), "w") as f:
            f.write(log)

    # The engine's own startup log is authoritative: "GPU KV cache size"
    # accounts for the actual per-layer cache layout. The attribute chains
    # are only a fallback -- num_gpu_blocks x block_size UNDER-REPORTS
    # capacity by 2x for turboquant dtypes (packed slots store more tokens
    # per block than cache_config.block_size implies; measured on 0.25.0).
    tokens, source = None, None
    for pat, unit in VLLM_LOG_PATTERNS:
        hit = pat.search(log)
        if hit:
            val = int(hit.group(1).replace(",", ""))
            if unit == "blocks":
                tokens = val * 16
            elif unit == "concurrency":
                tokens = int(val * float(hit.group(2)))
            else:
                tokens = val
            source = f"log pattern: {pat.pattern!r}"
            break
    if tokens is None:  # fall back to engine attributes
        m = re.search(r"PROBE_RESULT (\{.*\})", log)
        if m:
            data = json.loads(m.group(1))
            tokens = data.get("attr_tokens")
            if tokens is not None:
                source = f"attr chain: {data.get('attr_chain')} (NOTE: may " \
                         f"under-report turboquant capacity; see comment above)"
    if source:
        print(f"  (capacity via {source})")
    return tokens, log


def verify_vllm(args) -> int:
    dtype = args.cache_dtype or "turboquant_k3v4_nc"
    print(f"Probing baseline engine (auto dtype), then {dtype} ...")
    print("(each probe loads the model; expect a few minutes total)\n")

    base_tokens, base_log = _run_vllm_probe(
        None, args.max_model_len, args.gpu_mem_util, args.hf_overrides)
    comp_tokens, comp_log = _run_vllm_probe(
        dtype, args.max_model_len, args.gpu_mem_util, args.hf_overrides)

    # Sanity check: did the engine actually accept the dtype?
    if re.search(r"(?i)(invalid|unsupported).{0,40}kv[_ ]cache", comp_log):
        banner(False, f"Engine rejected kv_cache_dtype={dtype}. Check vLLM version "
                      f"(needs the PR #38479 backend, vllm>=0.19.1).")
        return 1

    if base_tokens is None or comp_tokens is None:
        banner(None, "Could not extract KV capacity from engine internals OR "
                     "startup logs. Your vLLM version logs differently -- see "
                     "the patterns at the top of this file and docs/"
                     "maintainer-notes.md.")
        print("--- last 30 lines of the compressed probe log ---")
        print("\n".join(comp_log.strip().splitlines()[-30:]))
        return 2

    ratio = comp_tokens / max(base_tokens, 1)
    print(f"  baseline KV capacity:   {base_tokens:>10,} tokens")
    print(f"  compressed KV capacity: {comp_tokens:>10,} tokens")
    print(f"  ratio:                  {ratio:>10.2f}x   ({EXPECT_HINT})")

    if ratio >= PASS_RATIO:
        banner(True, f"Compression is ACTIVE ({ratio:.2f}x more KV capacity). "
                     f"Your benchmark numbers can be trusted.")
        return 0
    banner(False, f"Ratio {ratio:.2f}x is below {PASS_RATIO}x -- compression did "
                  f"NOT meaningfully engage. Do NOT trust any benchmark run in "
                  f"this configuration. Common causes: wrong vLLM version, dtype "
                  f"silently ignored, or both probes ran the same config.")
    return 1


# ==========================================================================
# llama.cpp path
# ==========================================================================
# llama.cpp prints its KV buffer sizes at startup (only at raised log
# verbosity on current builds -- the probe passes -lv 4 for this). Formats
# drift between versions, so we know several patterns, grouped into
# priority-ordered categories; the FIRST category with any matches is used
# ALONE. Never sum across categories: the "KV self size" summary line
# repeats the K/V component sizes on the same line, so a flat sum
# double-counts (reported MiB came out exactly 2x on real M4 logs).
# If nothing matches on your build, run llama-server by hand with -lv 4,
# find the KV size lines, and add a pattern here.
LLAMACPP_KV_PATTERNS = [
    # per-backend allocation lines: "Metal KV buffer size = 1152.00 MiB"
    ("buffer", re.compile(r"KV buffer size\s*=\s*([\d.]+)\s*MiB")),
    # summary total: "KV self size  = 1152.00 MiB, K (f16): ..., V (f16): ..."
    ("self", re.compile(r"KV self size\s*=\s*([\d.]+)\s*MiB")),
    # component fallback: "K (f16):  576.00 MiB" / "V (f16):  576.00 MiB"
    ("component", re.compile(r"\b[KV] \(\w+\):\s*([\d.]+)\s*MiB")),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_llamacpp_probe(args, ctk: str, ctv: str) -> tuple[float | None, str]:
    port = _free_port()
    cmd = [
        args.server_bin, "-hf", args.gguf,
        "-c", str(args.ctx), "-fa", "on",
        "-ctk", ctk, "-ctv", ctv,
        "--port", str(port), "--no-webui",
        # Current llama.cpp builds only print the KV allocation lines at a
        # raised log verbosity; without this the probe has nothing to read.
        "-lv", "4",
    ]
    if args.extra_server_args:
        cmd += args.extra_server_args.split()
    print(f"  starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    log_lines, deadline = [], time.time() + args.startup_timeout
    kv_mib = {cat: [] for cat, _ in LLAMACPP_KV_PATTERNS}
    try:
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            log_lines.append(line.rstrip())
            for cat, pat in LLAMACPP_KV_PATTERNS:
                for m in pat.finditer(line):
                    kv_mib[cat].append(float(m.group(1)))
            # server ready => all init logs have been printed
            if "listening" in line.lower() or "server is listening" in line.lower():
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    total = None
    for cat, _ in LLAMACPP_KV_PATTERNS:  # first matching category wins
        if kv_mib[cat]:
            total = sum(kv_mib[cat])
            break
    return total, "\n".join(log_lines)


def verify_llamacpp(args) -> int:
    if shutil.which(args.server_bin) is None:
        banner(None, f"'{args.server_bin}' not found on PATH. "
                     f"Install llama.cpp first (see docs/apple-silicon.md).")
        return 2

    print(f"Probing f16 KV baseline, then -ctk {args.ctk} -ctv {args.ctv} ...")
    print("(first run downloads the model; be patient)\n")
    base_mib, base_log = _run_llamacpp_probe(args, "f16", "f16")
    comp_mib, comp_log = _run_llamacpp_probe(args, args.ctk, args.ctv)

    if base_mib is None or comp_mib is None:
        banner(None, "Could not find KV buffer size lines in the server log. "
                     "Your llama.cpp build logs differently -- see the "
                     "patterns at the top of this file.")
        print("--- last 30 lines of the compressed probe log ---")
        print("\n".join(comp_log.strip().splitlines()[-30:]))
        return 2

    ratio = base_mib / max(comp_mib, 0.001)
    print(f"  f16 KV size:        {base_mib:>10.1f} MiB")
    print(f"  compressed KV size: {comp_mib:>10.1f} MiB")
    print(f"  ratio:              {ratio:>10.2f}x smaller")

    if ratio >= PASS_RATIO:
        banner(True, f"Compression is ACTIVE (KV cache {ratio:.2f}x smaller).")
        return 0
    if ratio >= 1.5:
        banner(True, f"Compression is active but modest ({ratio:.2f}x). q8_0 KV "
                     f"lands here by design (~1.9x); use q4_0 for ~3.6x.")
        return 0
    banner(False, f"KV size barely changed ({ratio:.2f}x). The cache type was "
                  f"likely ignored or fell back to f16 -- check the startup log "
                  f"for the cache type it actually used, and confirm your build "
                  f"supports '{args.ctk}' (llama-server --help | grep cache-type).")
    return 1


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["vllm", "llamacpp"])
    # vLLM options
    ap.add_argument("--cache-dtype", default=None,
                    help="vLLM dtype to verify (default turboquant_k3v4_nc)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--hf-overrides", default=None,
                    help="JSON dict of HF config overrides for the probe "
                         "engines (e.g. a rope_scaling block to test context "
                         "lengths beyond the model's native maximum)")
    ap.add_argument("--_vllm-probe", default=None, help=argparse.SUPPRESS)
    # llama.cpp options
    ap.add_argument("--server-bin", default="llama-server")
    ap.add_argument("--gguf", default=HF_GGUF)
    ap.add_argument("--ctk", default="q8_0")
    ap.add_argument("--ctv", default="q8_0")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--extra-server-args", default=None)
    args = ap.parse_args()

    if args._vllm_probe is not None:
        _vllm_probe(args._vllm_probe or None, args.max_model_len,
                    args.gpu_mem_util, args.hf_overrides)
        return 0
    if not args.backend:
        print("error: --backend is required", file=sys.stderr)
        return 2
    if args.backend == "vllm":
        return verify_vllm(args)
    return verify_llamacpp(args)


if __name__ == "__main__":
    sys.exit(main())
