#!/usr/bin/env bash
# turboquant-101 setup: install everything the chapters use, once.
#
#   ./setup.sh                  first run: Python env, vLLM (NVIDIA) or
#                               llama.cpp check (Mac), and the model download
#   ./setup.sh --skip-disk-check
#
# Slow, large, and machine-specific work lives here, so the chapters (and
# ./quickstart.sh) never stall on a download or fail for machine reasons.
# Nothing is MEASURED here -- proving compression is Chapter 3's job.
# Safe to re-run: finished parts are skipped. quickstart.sh runs this
# first with --brief, so running quickstart alone still works.
set -euo pipefail
cd "$(dirname "$0")"

BRIEF=0
DISK_CHECK=1
for arg in "$@"; do
  case "$arg" in
    --brief) BRIEF=1 ;;             # quickstart's call: one line when ready
    --skip-disk-check) DISK_CHECK=0 ;;
    -h|--help) echo "usage: ./setup.sh [--skip-disk-check]"; exit 0 ;;
    *) echo "unknown option: $arg"; echo "usage: ./setup.sh [--skip-disk-check]"; exit 1 ;;
  esac
done

START_TIME=$(date +%s)
# pip's "new release available" notice tempts beginners into upgrading pip
# mid-setup; it has nothing to do with this repo.
export PIP_DISABLE_PIP_VERSION_CHECK=1

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
elapsed() {
  local s=$(( $(date +%s) - START_TIME ))
  if [ "$s" -ge 60 ]; then echo "$((s / 60)) min $((s % 60)) s"; else echo "$s s"; fi
}

if [ "$BRIEF" = 0 ]; then
  say "turboquant-101 setup"
  echo "Installs what the chapters use, once: a Python environment (.venv),"
  echo "the inference engine, and the Qwen3-4B model. Nothing is measured here."
  echo "Safe to re-run: finished parts are skipped."
fi

# ---- 1. Python >= 3.10 + venv + base deps ---------------------------------
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
# Python >= 3.10 required: the CUDA path (vLLM) needs it, and macOS's
# system /usr/bin/python3 (3.9) pairs with NumPy 2.x in ways that spew
# spurious warnings through the CPU demo. On macOS: brew install python@3.12
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "python3 >= 3.10 is required (found: $(python3 -V 2>&1))."
  echo "macOS: brew install python@3.12   Ubuntu: apt install python3.10-venv"
  exit 1
fi

# A pre-existing .venv may date from before the version check above (e.g.
# created by macOS's system Python 3.9). Recreate it rather than silently
# reusing a stale interpreter.
if [ -d .venv ] && ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "Existing .venv uses Python < 3.10 (or is broken); recreating it."
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  say "Creating the Python environment (.venv)"
  # Stock Ubuntu (incl. fresh WSL2) ships python3 without the venv module's
  # ensurepip (needs the python3-venv apt package). Fall back to a pip
  # bootstrap so setup works without sudo.
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
PY_VER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

# ---- 2. platform-specific engine + model ----------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  PLATFORM="NVIDIA GPU -> vLLM path"
  if [ "$BRIEF" = 0 ]; then say "NVIDIA GPU detected -> vLLM path"; fi
  case "$(pwd)" in /mnt/*)
    echo "WARNING: you are under /mnt/ (Windows filesystem). On WSL2, move"
    echo "this repo to your Linux home (e.g. ~/turboquant-101) first --"
    echo "model loading from /mnt/ is painfully slow. Continuing anyway."
  ;; esac

  # Skip the install when the pinned vLLM is already present. Re-running
  # `pip install -r` is not a no-op: pip re-resolves vLLM's dependencies and
  # reinstalls the torchcodec the guard below just removed, so every rerun
  # would download it and print the removal message again (seen on the
  # 2026-09-27 WSL2 rerun).
  VLLM_PIN="$(sed -n 's/^vllm==\([^[:space:]]*\).*/\1/p' requirements-cuda.txt)"
  VLLM_READY=0
  if [ -n "$VLLM_PIN" ] && python -c "import importlib.metadata as m, sys; sys.exit(0 if m.version('vllm') == sys.argv[1] else 1)" "$VLLM_PIN" 2>/dev/null; then
    VLLM_READY=1
  fi
  MODEL_READY=0
  if [ "$VLLM_READY" = 1 ] && python scripts/download_model.py --check; then
    MODEL_READY=1
  fi

  # Check for room BEFORE the big downloads, not halfway through them.
  if [ "$DISK_CHECK" = 1 ] && { [ "$VLLM_READY" = 0 ] || [ "$MODEL_READY" = 0 ]; }; then
    NEED_VENV_GB=0
    if [ "$VLLM_READY" = 0 ]; then NEED_VENV_GB=15; fi
    if ! python scripts/download_model.py --disk-check --need-venv-gb "$NEED_VENV_GB"; then
      echo "Free up space and re-run ./setup.sh (or add --skip-disk-check if"
      echo "you're sure the estimate is wrong for your machine)."
      exit 1
    fi
  fi

  if [ "$VLLM_READY" = 0 ]; then
    say "Installing vLLM $VLLM_PIN (a large download; grab a coffee)"
    pip install -q -r requirements-cuda.txt
  elif [ "$BRIEF" = 0 ]; then
    echo "  vLLM $VLLM_PIN: already installed"
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

  MODEL="$(python -c 'from niah.backends import DEFAULT_MODEL; print(DEFAULT_MODEL)')"
  if [ "$MODEL_READY" = 0 ] && ! python scripts/download_model.py --check; then
    say "Downloading $MODEL (~8 GB, one time)"
    if [ -z "${HF_TOKEN:-}" ]; then
      echo "Tip: a free Hugging Face token makes this faster (export HF_TOKEN=...)."
      echo "Not required -- the download works without one."
    fi
    python scripts/download_model.py
  elif [ "$BRIEF" = 0 ]; then
    echo "  $MODEL: already downloaded"
  fi
  READY_LINE="Python $PY_VER, vLLM $VLLM_PIN, $MODEL downloaded"

elif [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  PLATFORM="Apple Silicon -> llama.cpp path"
  if [ "$BRIEF" = 0 ]; then say "Apple Silicon detected -> llama.cpp path"; fi
  if ! command -v llama-server >/dev/null 2>&1; then
    echo "llama-server not found. Install it first:"
    echo "    brew install llama.cpp"
    echo "then re-run ./setup.sh"
    exit 1
  fi
  # Pre-fetch the GGUF by letting llama-server itself download it (so it
  # lands exactly where `llama-server -hf` looks later), then stop it.
  # Output is shown: this is the download you'd otherwise wait on blind
  # inside verify.py. A marker in .venv skips this on reruns.
  GGUF="Qwen/Qwen3-4B-GGUF:Q4_K_M"
  MARKER=".venv/.tq101-gguf-ready"
  if [ ! -f "$MARKER" ]; then
    say "Downloading $GGUF (~2.5 GB, one time)"
    PORT=8089
    llama-server -hf "$GGUF" -c 512 --port "$PORT" --no-webui &
    PREFETCH_PID=$!
    trap 'kill $PREFETCH_PID 2>/dev/null || true' EXIT
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
      if ! kill -0 "$PREFETCH_PID" 2>/dev/null; then
        echo "llama-server exited before the model was ready -- see above."
        exit 1
      fi
      sleep 2
    done
    kill "$PREFETCH_PID" 2>/dev/null || true
    wait "$PREFETCH_PID" 2>/dev/null || true
    trap - EXIT
    touch "$MARKER"
  elif [ "$BRIEF" = 0 ]; then
    echo "  $GGUF: already downloaded"
  fi
  READY_LINE="Python $PY_VER, llama.cpp, $GGUF downloaded"

else
  PLATFORM="no NVIDIA GPU or Apple Silicon -> CPU demo only"
  READY_LINE="Python $PY_VER"
fi

# ---- 3. done ----------------------------------------------------------------
if [ "$BRIEF" = 1 ]; then
  say "Setup ready: $PLATFORM"
  echo "($READY_LINE)"
else
  say "Setup ready ($(elapsed))"
  echo "  $PLATFORM: $READY_LINE."
  echo "  Nothing has been measured yet. Next: ./quickstart.sh runs the checks"
  echo "  (about 1-2 min once set up)."
fi
