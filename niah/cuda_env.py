"""Put pip-installed CUDA 13 libraries on the dynamic loader path.

vLLM 0.25.0's cu130 wheels pull torch 2.11.0+cu130, whose CUDA runtime
libraries (including libnvrtc.so.13) are pip-installed under
site-packages/nvidia/cu13/lib. PyTorch preloads what it needs from there
itself, but two vLLM extensions -- vllm.third_party.deep_gemm and
vllm.cumem_allocator -- rely on the system loader, which doesn't search
that folder. Result on a fresh WSL2 install (2026-09-27): a WARNING
traceback at engine startup, and an ImportError during engine shutdown
that also skips cleanup (hence NCCL's "destroy_process_group() was not
called"). Adding the folder to LD_LIBRARY_PATH fixed both, measured.
(deep_gemm then fails a later, unrelated check -- no CUDA toolkit, so
`assert cuda_home is not None` -- still a harmless WARNING, visible only
with --verbose; see docs/cuda.md.)

The loader reads LD_LIBRARY_PATH only when a process starts, so this must
run before vLLM spawns its engine process (backends) or before verify.py
launches its probe subprocess. It changes nothing when the folder doesn't
exist (Mac, CPU-only, or a different CUDA layout).
"""

from __future__ import annotations

import os
import pathlib
import sysconfig


def cu13_lib_dir() -> str | None:
    d = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib"
    return str(d) if d.is_dir() else None


def with_cu13_libs(env) -> "dict | os._Environ":
    """Prepend the cu13 lib folder to env['LD_LIBRARY_PATH'] (in place)."""
    d = cu13_lib_dir()
    if d:
        parts = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
        if d not in parts:
            env["LD_LIBRARY_PATH"] = os.pathsep.join([d, *parts])
    return env
