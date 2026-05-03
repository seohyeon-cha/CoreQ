#!/usr/bin/env bash
# CoreQ (no beam search) on Llama-2-13B.
# ALPHA_METHOD=corr selects the data-driven, per-row α from the paper.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-2-13b-hf"}
DATASET=${DATASET:-c4}
WBITS=${WBITS:-3}
GROUPSIZE=${GROUPSIZE:--1}
NSAMPLES=${NSAMPLES:-128}
SEED=${SEED:-0}
ALPHA_METHOD=${ALPHA_METHOD:-corr}
ALPHA=${ALPHA:-0.5}

python -u llama_step.py \
    "$MODEL_PATH" "$DATASET" \
    --method coreq \
    --wbits "$WBITS" \
    --groupsize "$GROUPSIZE" \
    --nsamples "$NSAMPLES" \
    --seed "$SEED" \
    --alpha-method "$ALPHA_METHOD" \
    --alpha "$ALPHA" \
    --beam-size 1 \
    --sym \
    --true-sequential \
    --eval
