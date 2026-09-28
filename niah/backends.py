"""Backend adapters.

One NIAH protocol, two very different runtimes:

  * VLLMBackend      -- in-process vLLM engine (NVIDIA CUDA path)
  * LlamaServerBackend -- HTTP client for a running llama-server
                          (Apple Silicon / CPU path)

Both expose the same three methods so scripts/demo.py and
scripts/benchmark.py don't care which one they're driving:

    count_tokens(text) -> int
    generate(prompt)   -> GenResult(text, gen_tokens, seconds)
    describe()         -> str

A note on Qwen3 "thinking mode": Qwen3 chat templates emit a <think>...
block by default, which would eat the 64-token generation budget before
the model ever states the code. We disable it (enable_thinking=False on
vLLM; the "/no_think" soft switch on llama.cpp). If your outputs start
with "<think>", this is the first thing to check.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

DEFAULT_MODEL = "Qwen/Qwen3-4B"
MAX_NEW_TOKENS = 64
SEED = 1234


@dataclass
class GenResult:
    text: str
    gen_tokens: int
    seconds: float


class VLLMBackend:
    """In-process vLLM engine.

    cache_dtype=None (or "auto") gives the FP16/BF16 baseline;
    cache_dtype="turboquant_k3v4_nc" enables the compressed cache
    (3-bit MSE keys + 4-bit uniform values + norm correction).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dtype: str | None = None,
        max_model_len: int = 20000,
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = False,
        extra_engine_kwargs: dict | None = None,
        verbose: bool = False,
    ):
        # Quiet by default. Run in-process, vLLM prints ~100 lines of INFO
        # (a full config dump, compile/graph-capture progress) plus a
        # harmless 15-line libnvrtc WARNING traceback on 0.25.0 -- enough to
        # bury the five result lines. ERROR keeps real failures visible;
        # --verbose (or exporting VLLM_LOGGING_LEVEL yourself) restores the
        # full log, which is where the engine's own evidence lines live
        # ("Using TURBOQUANT attention backend", "GPU KV cache size").
        # Must be set before vllm is imported; the engine's worker process
        # inherits it.
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO" if verbose else "ERROR")
        # vLLM >= 0.25 requires UVA (pinned host memory) in its GPU worker,
        # but disables pinned memory by default under WSL2; without this the
        # engine dies at startup with "RuntimeError: UVA is not available".
        # Only consulted when vLLM detects WSL, so it is harmless elsewhere.
        os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
        # vLLM's FlashInfer sampler JIT-compiles CUDA at first use (during
        # engine warm-up), which requires a full CUDA toolkit + compiler on
        # the host -- absent on a fresh WSL2 install, and the pip-shipped
        # toolkit's headers clash with FlashInfer's bundled CCCL. This repo
        # only does greedy decoding, so default to vLLM's native sampler.
        # Export VLLM_USE_FLASHINFER_SAMPLER=1 to override.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # Let vLLM's engine process find pip's libnvrtc.so.13 (see
        # niah/cuda_env.py); without it, a traceback at startup and another
        # at shutdown. Inherited by the engine process spawned below.
        from .cuda_env import with_cu13_libs
        with_cu13_libs(os.environ)

        from vllm import LLM, SamplingParams  # lazy: only the CUDA path needs it

        kwargs = dict(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )
        if cache_dtype and cache_dtype != "auto":
            kwargs["kv_cache_dtype"] = cache_dtype
        # Escape hatch for version-specific knobs (e.g. skip_layers) without
        # editing this file -- pass them as a JSON dict from the CLI.
        kwargs.update(extra_engine_kwargs or {})

        self.cache_dtype = cache_dtype or "auto (fp16/bf16 baseline)"
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(
            temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=SEED
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate(self, prompt: str) -> GenResult:
        messages = [{"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        try:
            outs = self.llm.chat(
                messages,
                self.sampling,
                chat_template_kwargs={"enable_thinking": False},
                use_tqdm=False,
            )
        except TypeError:
            # Older vLLM without chat_template_kwargs: fall back to the
            # Qwen3 "/no_think" soft switch inside the message itself.
            messages[0]["content"] = prompt + " /no_think"
            outs = self.llm.chat(messages, self.sampling, use_tqdm=False)
        dt = time.perf_counter() - t0
        out = outs[0].outputs[0]
        return GenResult(text=out.text, gen_tokens=len(out.token_ids), seconds=dt)

    def decode_speed(self, prompt_lo: str, prompt_hi: str | None = None,
                     lo_tokens: int = 16, hi_tokens: int = 256) -> float | None:
        """Decode-only tok/s. NIAH answers are ~10 tokens, so tokens/wall
        time per request is dominated by PREFILL, not decode. Instead:
        generate a short and a long completion (EOS ignored) and divide the
        token delta by the time delta -- the near-equal prefills cancel out.

        The two prompts must be DIFFERENT documents of the same token
        length: with vLLM's default prefix caching, reusing one prompt lets
        the second call skip its prefill entirely, which silently inflates
        the measured decode rate (observed: 182 "tok/s" at 16K vs 115 at 8K
        on identical hardware)."""
        from vllm import SamplingParams

        timings = []
        for n, prompt in ((lo_tokens, prompt_lo),
                          (hi_tokens, prompt_hi or prompt_lo)):
            sp = SamplingParams(
                temperature=0.0, max_tokens=n, seed=SEED, ignore_eos=True
            )
            messages = [{"role": "user", "content": prompt}]
            t0 = time.perf_counter()
            try:
                outs = self.llm.chat(
                    messages, sp,
                    chat_template_kwargs={"enable_thinking": False},
                    use_tqdm=False,
                )
            except TypeError:
                messages[0]["content"] = prompt + " /no_think"
                outs = self.llm.chat(messages, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            timings.append((len(outs[0].outputs[0].token_ids), dt))
        (n_lo, t_lo), (n_hi, t_hi) = timings
        if n_hi > n_lo and t_hi > t_lo:
            return (n_hi - n_lo) / (t_hi - t_lo)
        return None

    def describe(self) -> str:
        return f"vLLM / {DEFAULT_MODEL} / kv_cache_dtype={self.cache_dtype}"

    def close(self) -> None:
        """Shut the engine down now instead of at interpreter exit.

        vLLM's engine teardown prints its own noise (on 0.25.0 the worker
        hits the libnvrtc ImportError while shutting down -- docs/cuda.md).
        Closing explicitly makes that happen BEFORE the caller prints its
        summary, so the result is the last thing on screen rather than
        buried above a traceback. Safe to call twice; never raises."""
        llm, self.llm = getattr(self, "llm", None), None
        if llm is None:
            return
        try:
            engine = getattr(llm, "llm_engine", None)
            for target in (engine, getattr(engine, "engine_core", None)):
                shutdown = getattr(target, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                    break
        except Exception:
            pass  # teardown noise is vLLM's; our results are already in hand
        finally:
            del llm
            import gc
            gc.collect()


class LlamaServerBackend:
    """HTTP client for llama-server (llama.cpp).

    You start the server yourself (see docs/apple-silicon.md), choosing the
    KV cache types with -ctk / -ctv. This script only measures what the
    server does -- keeping the compression configuration visible in YOUR
    terminal, not buried in Python.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", label: str = ""):
        self.base_url = base_url.rstrip("/")
        self.label = label or base_url

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())

    def count_tokens(self, text: str) -> int:
        return len(self._post("/tokenize", {"content": text}).get("tokens", []))

    def generate(self, prompt: str) -> GenResult:
        payload = {
            # "/no_think" is Qwen3's soft switch to skip the <think> block.
            "messages": [{"role": "user", "content": prompt + " /no_think"}],
            "temperature": 0.0,
            "max_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
        }
        t0 = time.perf_counter()
        resp = self._post("/v1/chat/completions", payload)
        dt = time.perf_counter() - t0
        text = resp["choices"][0]["message"]["content"] or ""
        usage = resp.get("usage", {})
        gen_tokens = usage.get("completion_tokens", max(1, len(text) // 4))
        return GenResult(text=text, gen_tokens=gen_tokens, seconds=dt)

    def decode_speed(self, prompt_lo: str, prompt_hi: str | None = None,
                     lo_tokens: int = 16, hi_tokens: int = 256) -> float | None:
        """Decode-only tok/s. llama-server reports decode timings itself
        (`timings.predicted_per_second`), so one long generation suffices
        (prompt_hi is accepted for interface parity and ignored)."""
        payload = {
            "messages": [{"role": "user", "content": prompt_lo + " /no_think"}],
            "temperature": 0.0,
            "max_tokens": hi_tokens,
            "seed": SEED,
        }
        resp = self._post("/v1/chat/completions", payload)
        t = resp.get("timings") or {}
        pps = t.get("predicted_per_second")
        return float(pps) if pps else None

    def describe(self) -> str:
        return f"llama-server @ {self.base_url} ({self.label})"

    def close(self) -> None:
        """Nothing to release: you own the server process."""


def build_backend(args) -> "VLLMBackend | LlamaServerBackend":
    """Construct a backend from argparse args shared by demo/benchmark."""
    if args.backend == "vllm":
        extra = json.loads(args.extra_engine_kwargs) if args.extra_engine_kwargs else None
        return VLLMBackend(
            model=args.model,
            cache_dtype=args.cache_dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            extra_engine_kwargs=extra,
            verbose=getattr(args, "verbose", False),
        )
    return LlamaServerBackend(base_url=args.url, label=args.label or "")


def add_backend_args(parser) -> None:
    parser.add_argument("--backend", choices=["vllm", "llamacpp"])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="vLLM model id")
    parser.add_argument(
        "--cache-dtype",
        default=None,
        help="vLLM kv_cache_dtype; omit for baseline, "
        "'turboquant_k3v4_nc' for the compressed reference config",
    )
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument(
        "--extra-engine-kwargs",
        default=None,
        help='JSON dict of extra vLLM engine kwargs, e.g. \'{"foo": 1}\'',
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:8080", help="llama-server base URL"
    )
    parser.add_argument(
        "--label", default=None, help="name for this run (used in results files)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="show vLLM's full engine log (hidden by default; errors always show)",
    )
