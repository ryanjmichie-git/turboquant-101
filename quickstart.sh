#!/usr/bin/env bash
# turboquant-101 quickstart: the three-step first win (CPU idea, verify,
# needle demo) plus a one-screen summary. Installs and downloads live in
# ./setup.sh, which this runs first (a one-line check once it's done), so
# `./quickstart.sh` alone still works on a fresh machine.
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

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "turboquant-101"
echo "Three steps: (1) the idea, on your CPU; (2) prove compression is really"
echo "on; (3) a needle-in-a-haystack test. A one-screen summary prints at the end."
if [ -z "$VERBOSE" ]; then
  echo "(Add --verbose for the full detail at every step.)"
fi

# Installs, downloads, machine checks: all in setup.sh. Already set up =
# one "Setup ready" line; otherwise its full progress shows here.
bash ./setup.sh --brief
# shellcheck disable=SC1091
source .venv/bin/activate

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

# ---- 1. the CPU demo: everyone gets this win, no GPU needed ---------------
say "Step 1/3: the quantizer math, on your CPU"
# Short version here (one table + the idea); the full walk-through --
# memory table, error metrics, ratio vs. context -- with --verbose, or any
# time via `python scripts/cpu_demo.py`.
if [ -n "$VERBOSE" ]; then
  python scripts/cpu_demo.py
else
  python scripts/cpu_demo.py --brief
fi

# ---- 2. the model steps (setup.sh already installed the engine + model) ---
if command -v nvidia-smi >/dev/null 2>&1; then
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
  say "Step 2/3: verifying the KV cache actually shrinks (f16 vs q8_0)"
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
  summary
  echo
  echo "The CPU demo is the full experience on this machine -- see all of it"
  echo "with: python scripts/cpu_demo.py. To run the model steps you need an"
  echo "NVIDIA GPU (docs/cuda.md) or an Apple Silicon Mac (docs/apple-silicon.md)."
fi
