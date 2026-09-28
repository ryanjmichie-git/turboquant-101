#!/usr/bin/env bash
# turboquant-101 quickstart: detects your platform and runs the shortest
# path to a working demo. Safe to re-run; everything lives in ./.venv.
#
#   ./quickstart.sh             short, beginner-friendly output
#   ./quickstart.sh --verbose   full CPU walk-through, where each capacity
#                               number was parsed from, vLLM's engine log,
#                               and per-request timings
set -euo pipefail
cd "$(dirname "$0")"

# Parse options before doing anything, so a typo can't half-run the setup.
# VERBOSE is passed UNQUOTED to the scripts below: empty = no argument.
# (A plain string, not an array: macOS /bin/bash 3.2 + `set -u` rejects
# "${empty_array[@]}".)
VERBOSE=""
for arg in "$@"; do
  case "$arg" in
    -v|--verbose) VERBOSE="--verbose" ;;
    -h|--help) echo "usage: ./quickstart.sh [--verbose]"; exit 0 ;;
    *) echo "unknown option: $arg"; echo "usage: ./quickstart.sh [--verbose]"; exit 1 ;;
  esac
done

START_TIME=$(date +%s)  # for the "finished in" line in the summary

# pip's "new release available" notice tempts beginners into upgrading pip
# mid-setup; it has nothing to do with this repo.
export PIP_DISABLE_PIP_VERSION_CHECK=1

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "turboquant-101"
echo "Three steps: (1) the idea, on your CPU; (2) prove compression is really"
echo "on; (3) a needle-in-a-haystack test. A one-screen summary prints at the end."
if [ -z "$VERBOSE" ]; then
  echo "(Add --verbose for the full detail at every step.)"
fi
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
# Python >= 3.10 required: the CUDA path (vLLM) needs it, and macOS's
# system /usr/bin/python3 (3.9) pairs with NumPy 2.x in ways that spew
# spurious warnings through the CPU demo. On macOS: brew install python@3.12
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "python3 >= 3.10 is required (found: $(python3 -V 2>&1))."
  echo "macOS: brew install python@3.12   Ubuntu: apt install python3.10-venv"
  exit 1
fi

# ---- 1. venv + base deps (numpy for the CPU demo) -------------------------
# A pre-existing .venv may date from before the version check above (e.g.
# created by macOS's system Python 3.9). Recreate it rather than silently
# reusing a stale interpreter.
if [ -d .venv ] && ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "Existing .venv uses Python < 3.10 (or is broken); recreating it."
  rm -rf .venv
fi

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

# Each step drops its headline numbers here; scripts/summary.py reads them
# back at the end (niah/run_summary.py). Cleared every run, so the summary
# can only show THIS run's results.
export TQ101_RUN_DIR="$PWD/.venv/tq101-last-run"
rm -rf "$TQ101_RUN_DIR"
mkdir -p "$TQ101_RUN_DIR"

summary() {
  python scripts/summary.py "$TQ101_RUN_DIR" \
    --elapsed "$(( $(date +%s) - START_TIME ))"
}
verbose_hint() {
  if [ -z "$VERBOSE" ]; then
    echo "More detail on any step: ./quickstart.sh --verbose"
  fi
}

# ---- 2. the CPU demo: everyone gets this win, no GPU needed ---------------
say "Step 1/3: the quantizer math, on your CPU"
# Short version here (one table + the idea); the full walk-through --
# memory table, error metrics, ratio vs. context -- with --verbose, or any
# time via `python scripts/cpu_demo.py`.
if [ -n "$VERBOSE" ]; then
  python scripts/cpu_demo.py
else
  python scripts/cpu_demo.py --brief
fi

# ---- 3. platform detection ------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  # Skip the install when the pinned vLLM is already present. Re-running
  # `pip install -r` is not a no-op: pip re-resolves vLLM's dependencies and
  # reinstalls the torchcodec the guard below just removed, so every rerun
  # would download it and print the removal message again (seen on the
  # 2026-09-27 WSL2 rerun). One header either way (reruns used to print two).
  VLLM_PIN="$(sed -n 's/^vllm==\([^[:space:]]*\).*/\1/p' requirements-cuda.txt)"
  VLLM_READY=0
  if [ -n "$VLLM_PIN" ] && python -c "import importlib.metadata as m, sys; sys.exit(0 if m.version('vllm') == sys.argv[1] else 1)" "$VLLM_PIN" 2>/dev/null; then
    VLLM_READY=1
    say "NVIDIA GPU detected -> vLLM path (vLLM $VLLM_PIN already installed)"
  else
    say "NVIDIA GPU detected -> vLLM path"
  fi
  case "$(pwd)" in /mnt/*)
    echo "WARNING: you are under /mnt/ (Windows filesystem). On WSL2, move"
    echo "this repo to your Linux home (e.g. ~/turboquant-101) first --"
    echo "model loading from /mnt/ is painfully slow. Continuing anyway."
  ;; esac
  if [ "$VLLM_READY" = 0 ]; then
    say "Installing vLLM (this is a large download; grab a coffee)"
    pip install -q -r requirements-cuda.txt
  fi

  # vllm's torchcodec dependency hard-fails on systems without FFmpeg
  # shared libraries (RuntimeError or OSError "Could not load this
  # library: .../libtorchcodec_image.so"), which escapes vLLM's
  # ImportError guard. A bare `import vllm` does NOT catch it: vLLM loads
  # torchcodec lazily, and on 0.25.0 the crash only fires once the engine
  # imports vllm.sampling_params -- i.e. inside verify.py, after this
  # check has already passed (hit on a fresh WSL2 run, 2026-09-27). So
  # test torchcodec itself. This repo is text-only and never decodes
  # video, so a torchcodec that can't load is safe to remove.
  if python -m pip show -q torchcodec >/dev/null 2>&1 \
      && ! python -c 'import torchcodec' >/dev/null 2>&1; then
    echo "torchcodec is installed but cannot load (no FFmpeg libraries?)"
    echo "-- removing it; this repo is text-only and never needs it"
    pip uninstall -y -q torchcodec || true
  fi
  # Then import the same path the engine uses, so any other import-time
  # failure surfaces here with its real traceback, not later as a vague
  # "couldn't measure" from verify.py.
  if ! python -c 'from vllm import SamplingParams' 2>/tmp/tq101-vllm-import.log; then
    echo "vLLM still fails to import; last lines of the error:"
    tail -n 15 /tmp/tq101-vllm-import.log || true
    echo "See docs/cuda.md (Gotchas) for the known fixes."
    exit 1
  fi

  say "Step 2/3: verifying compression is ACTUALLY engaged"
  # shellcheck disable=SC2086  # $VERBOSE: empty = no argument
  python scripts/verify.py --backend vllm $VERBOSE

  say "Step 3/3: five-needle retrieval demo (compressed cache)"
  # shellcheck disable=SC2086
  python scripts/demo.py --backend vllm --cache-dtype turboquant_k3v4_nc $VERBOSE
  summary
  echo
  echo "Next: the full A/B benchmark -- see the commands at the top of"
  echo "scripts/benchmark.py, and docs/cuda.md for RTX/WSL2 gotchas."
  verbose_hint

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
  # shellcheck disable=SC2086
  python scripts/verify.py --backend llamacpp --ctk q8_0 --ctv q8_0 $VERBOSE

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
  # The loop above falls through after 120 tries -- don't run the demo
  # against a server that never came up.
  if ! curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "llama-server never became healthy; last lines of its log:"
    tail -n 20 /tmp/tq101-server.log || true
    exit 1
  fi
  # shellcheck disable=SC2086
  python scripts/demo.py --backend llamacpp --url http://127.0.0.1:8080 \
    --label q8_0-kv $VERBOSE
  summary
  echo
  echo "Next: docs/apple-silicon.md for q4_0 and the fork with real"
  echo "TurboQuant types, then scripts/benchmark.py for the full A/B."
  verbose_hint

else
  say "No NVIDIA GPU or Apple Silicon detected"
  summary
  echo
  echo "The CPU demo is the full experience on this machine -- see all of it"
  echo "with: python scripts/cpu_demo.py. To run the model steps you need an"
  echo "NVIDIA GPU (docs/cuda.md) or an Apple Silicon Mac (docs/apple-silicon.md)."
fi
