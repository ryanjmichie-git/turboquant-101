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
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
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
    # Put pip's CUDA 13 libraries (libnvrtc.so.13) on the loader path for
    # the probe process -- see niah/cuda_env.py.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from niah.cuda_env import with_cu13_libs
    with_cu13_libs(env)
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


# Known ways a vLLM probe dies BEFORE the engine reports anything. When a
# probe crashes, the "couldn't find the capacity line" message is true but
# misleading -- there was never a log to parse. Match the real cause and
# name its fix. (First entry: fresh WSL2 run, 2026-09-27 -- OSError from
# torchcodec surfaced here, not at quickstart's import check.)
PROBE_CRASH_HINTS = [
    (re.compile(r"(?s)torchcodec.*(Could not load|libtorchcodec)"),
     "vLLM crashed importing torchcodec (a video library that needs FFmpeg). "
     "This repo is text-only: run `pip uninstall -y torchcodec`, then re-run "
     "this script."),
    (re.compile(r"UVA is not available"),
     "vLLM needs pinned host memory (UVA). On WSL2 run `wsl --update` in "
     "PowerShell, reopen Ubuntu, and re-run."),
    (re.compile(r"(?i)CUDA out of memory|OutOfMemoryError"),
     "The GPU ran out of memory while loading. Close browsers and other GPU "
     "apps, or re-run with --gpu-mem-util 0.8."),
    (re.compile(r"(?i)could not find nvcc|flashinfer.*(jit|compile)"),
     "FlashInfer tried to JIT-compile CUDA. Export "
     "VLLM_USE_FLASHINFER_SAMPLER=0 and re-run (docs/cuda.md)."),
]


def diagnose_probe_crash(log: str) -> str | None:
    """Return a one-line fix if the probe log shows a crash, else None.

    A log with a Python traceback and no PROBE_RESULT line means the engine
    never came up; that's a crash, not a log-format mismatch."""
    for pat, hint in PROBE_CRASH_HINTS:
        if pat.search(log):
            return hint
    if "Traceback (most recent call last)" in log and "PROBE_RESULT" not in log:
        return ("The probe crashed before the engine started -- the traceback "
                "below is the real error (this is not a log-format problem).")
    return None


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
                      f"(needs the PR #38479 backend, first shipped in vllm 0.20.0; "
                      f"note 0.19.x does NOT contain it).")
        return 1

    if base_tokens is None or comp_tokens is None:
        # Crash first: an engine that never started has no log to parse.
        for label, tokens, log in (("baseline", base_tokens, base_log),
                                   ("compressed", comp_tokens, comp_log)):
            hint = diagnose_probe_crash(log) if tokens is None else None
            if hint:
                banner(None, f"The {label} probe crashed before measuring "
                             f"anything. {hint}")
                print(f"--- last 30 lines of the {label} probe log ---")
                print("\n".join(log.strip().splitlines()[-30:]))
                return 2
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
                     f"This proves it is ON -- not that quality held; the "
                     f"needle demo and benchmark test that.")
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


def _extract_kv_mib(lines) -> float | None:
    """Total KV cache MiB from server log lines.

    Categories in LLAMACPP_KV_PATTERNS are priority-ordered and the first
    category with any matches is summed ALONE -- summing across categories
    double-counts (the "KV self size" summary repeats the K/V component
    figures on the same line)."""
    by_cat = {cat: [] for cat, _ in LLAMACPP_KV_PATTERNS}
    for line in lines:
        for cat, pat in LLAMACPP_KV_PATTERNS:
            for m in pat.finditer(line):
                by_cat[cat].append(float(m.group(1)))
    for cat, _ in LLAMACPP_KV_PATTERNS:
        if by_cat[cat]:
            return sum(by_cat[cat])
    return None


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

    # Read stdout on a background thread. A plain readline() loop blocks
    # until a NEWLINE arrives, so a server that stalls silently -- or one
    # emitting only carriage-return progress updates during a model
    # download -- made --startup-timeout ineffective exactly when it was
    # most needed. The queue lets the deadline fire regardless.
    lines_q = queue.Queue()

    def _pump():
        for raw in proc.stdout:
            lines_q.put(raw)
        lines_q.put(None)  # EOF marker

    threading.Thread(target=_pump, daemon=True).start()

    log_lines, deadline = [], time.time() + args.startup_timeout
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                line = lines_q.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if line is None:  # EOF: server exited or closed stdout
                break
            log_lines.append(line.rstrip())
            # server ready => all init logs have been printed
            if "listening" in line.lower():
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    return _extract_kv_mib(log_lines), "\n".join(log_lines)


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
