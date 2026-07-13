#!/usr/bin/env bash
# turboquant-101 quickstart: detects your platform and runs the shortest
# path to a working demo. Safe to re-run; everything lives in ./.venv.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "turboquant-101"
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }

# ---- 1. venv + base deps (numpy for the CPU demo) -------------------------
if [ ! -d .venv ]; then
  say "Creating virtualenv (.venv)"
  # Stock Ubuntu (incl. fresh WSL2) ships python3 without the venv module's
  # ensurepip (needs the python3-venv apt package). Fall back to a pip
  # bootstrap so quickstart works without sudo.
  if ! python3 -m venv .venv 2>/dev/null; then
    echo "python3 -m venv failed (python3-venv not installed?); bootstrapping pip manually"
    rm -rf .venv
    python3 -m venv --without-pip .venv
    curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python -
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements-base.txt

# ---- 2. the CPU demo: everyone gets this win, no GPU needed ---------------
say "Step 1/3: the quantizer math, on your CPU"
python scripts/cpu_demo.py

# ---- 3. platform detection ------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  say "NVIDIA GPU detected -> vLLM path"
  case "$(pwd)" in /mnt/*)
    echo "WARNING: you are under /mnt/ (Windows filesystem). On WSL2, move"
    echo "this repo to your Linux home (e.g. ~/turboquant-101) first --"
    echo "model loading from /mnt/ is painfully slow. Continuing anyway."
  ;; esac

  say "Installing vLLM (this is a large download; grab a coffee)"
  pip install -q -r requirements-cuda.txt

  # vllm's torchcodec dependency hard-fails at `import vllm` on systems
  # without FFmpeg shared libraries (it raises RuntimeError, which escapes
  # vLLM's ImportError guard). This repo is text-only and never decodes
  # video, so drop torchcodec if it breaks the import.
  if ! python -c 'import vllm' >/dev/null 2>&1; then
    echo "import vllm failed -- removing optional torchcodec and retrying"
    pip uninstall -y -q torchcodec || true
  fi

  say "Step 2/3: verifying compression is ACTUALLY engaged"
  python scripts/verify.py --backend vllm

  say "Step 3/3: five-needle retrieval demo (compressed cache)"
  python scripts/demo.py --backend vllm --cache-dtype turboquant_k3v4_nc
  echo
  echo "Next: the full A/B benchmark -- see the commands at the top of"
  echo "scripts/benchmark.py, and docs/cuda.md for RTX/WSL2 gotchas."

elif [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  say "Apple Silicon detected -> llama.cpp path"
  if ! command -v llama-server >/dev/null 2>&1; then
    echo "llama-server not found. Install it first:"
    echo "    brew install llama.cpp"
    echo "then re-run ./quickstart.sh"
    exit 1
  fi

  say "Step 2/3: verifying the KV cache actually shrinks (f16 vs q8_0)"
  echo "(first run downloads the ~2.5 GB Qwen3-4B GGUF; be patient)"
  python scripts/verify.py --backend llamacpp --ctk q8_0 --ctv q8_0

  say "Step 3/3: five-needle retrieval demo against a live server"
  # Mainline llama.cpp rejected the TurboQuant cache types (see LEARN.md
  # for that story), so the zero-friction path uses the built-in q8_0
  # quantized KV cache -- same concept, ~1.9x smaller cache. q4_0 gives
  # ~3.6x; real TurboQuant types need a community fork (docs/apple-silicon.md).
  llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M -c 16384 -fa on \
    -ctk q8_0 -ctv q8_0 --port 8080 --no-webui >/tmp/tq101-server.log 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
  echo "waiting for llama-server to come up..."
  for _ in $(seq 1 120); do
    curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break
    sleep 2
  done
  python scripts/demo.py --backend llamacpp --url http://127.0.0.1:8080 \
    --label q8_0-kv
  echo
  echo "Next: docs/apple-silicon.md for q4_0 and the fork with real"
  echo "TurboQuant types, then scripts/benchmark.py for the full A/B."

else
  say "No NVIDIA GPU or Apple Silicon detected"
  echo "The CPU demo above is the full experience on this machine. To run"
  echo "the model benchmarks you need an NVIDIA GPU (docs/cuda.md) or an"
  echo "Apple Silicon Mac (docs/apple-silicon.md)."
fi
