#!/usr/bin/env bash
# LDLQ baseline on Llama-2-7B (3-bit, per-channel symmetric).
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-2-7b-hf"}
DATASET=${DATASET:-c4}
WBITS=${WBITS:-3}
GROUPSIZE=${GROUPSIZE:--1}
NSAMPLES=${NSAMPLES:-128}
SEED=${SEED:-0}

python -u llama_step.py \
    "$MODEL_PATH" "$DATASET" \
    --method ldlq \
    --wbits "$WBITS" \
    --groupsize "$GROUPSIZE" \
    --nsamples "$NSAMPLES" \
    --seed "$SEED" \
    --sym \
    --true-sequential \
    --eval
