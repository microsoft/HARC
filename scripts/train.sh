#!/usr/bin/env bash
# Train the HARC LoRA on Llama-3.1-8B and Qwen-2.5-7B in parallel (one GPU each).
# Hyperparameters live in the per-model configs.
#
# All paths are env-overridable; defaults are repo-relative.
#   PYTHON     python interpreter            (default: python)
#   PROJ       repo root (with PYTHONPATH=.) (default: repo containing this script)
#   HF_HOME    HF cache dir                  (default: unset -> HF default)
#   HF_TOKEN   HF token                      (default: unset)
#   LOG        log dir                       (default: $PROJ/runs/logs)
#   GPU_LLAMA  GPU index for the Llama run   (default: 0)
#   GPU_QWEN   GPU index for the Qwen run    (default: 1)
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=${PROJ:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-python}
LOG=${LOG:-$PROJ/runs/logs}
GPU_LLAMA=${GPU_LLAMA:-0}
GPU_QWEN=${GPU_QWEN:-1}

mkdir -p "$LOG"
cd "$PROJ"
export PYTHONPATH=${PYTHONPATH:-$PROJ}

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG/train.log"; }

run_train() {
  local name="$1" cfg="$2" gpu="$3"
  log "[$name] -> GPU $gpu  config=$cfg"
  CUDA_VISIBLE_DEVICES=$gpu ${HF_HOME:+HF_HOME=$HF_HOME} ${HF_TOKEN:+HF_TOKEN=$HF_TOKEN} PYTHONUNBUFFERED=1 \
    nohup $PYTHON -m main.train --config "$cfg" \
    > "$LOG/train_${name}.log" 2>&1 &
  echo "  $name PID $!"
}

log "=== Train HARC LoRA on Llama-3.1-8B + Qwen-2.5-7B ==="

run_train llama3.1_8b "main/configs/llama3.1_8b.yaml" "$GPU_LLAMA"
run_train qwen2_5_7b  "main/configs/qwen2_5_7b.yaml"  "$GPU_QWEN"

log "=== 2 PIDs launched. Logs: $LOG/train_{llama3.1_8b,qwen2_5_7b}.log ==="
wait
log "=== both runs finished ==="
