#!/usr/bin/env bash
# GPTQ baseline on Llama-2-7B (3-bit, per-channel symmetric).
# Override values from the command line, e.g.: WBITS=4 GROUPSIZE=128 ./scripts/l2-7b/gptq.sh
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-2-7b-hf"}
DATASET=${DATASET:-c4}
WBITS=${WBITS:-3}
GROUPSIZE=${GROUPSIZE:--1}
NSAMPLES=${NSAMPLES:-128}
SEED=${SEED:-0}

python -u llama_step.py \
    "$MODEL_PATH" "$DATASET" \
    --method gptq \
    --wbits "$WBITS" \
    --groupsize "$GROUPSIZE" \
    --nsamples "$NSAMPLES" \
    --seed "$SEED" \
    --sym \
    --true-sequential \
    --act-order \
    --eval
