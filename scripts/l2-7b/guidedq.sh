#!/usr/bin/env bash
# GuidedQuant baseline on Llama-2-7B.
# Requires precomputed saliency tensors in $SALIENCY_PATH (one l{i}.pt
# per transformer block), produced by an external precomputation step.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-2-7b-hf"}
DATASET=${DATASET:-c4}
WBITS=${WBITS:-3}
GROUPSIZE=${GROUPSIZE:--1}
NSAMPLES=${NSAMPLES:-128}
SEED=${SEED:-0}
SALIENCY_PATH=${SALIENCY_PATH:-cache/saliency}
GUIDED_NUM_GROUPS=${GUIDED_NUM_GROUPS:-4}

python -u llama_step.py \
    "$MODEL_PATH" "$DATASET" \
    --method guidedq \
    --wbits "$WBITS" \
    --groupsize "$GROUPSIZE" \
    --nsamples "$NSAMPLES" \
    --seed "$SEED" \
    --saliency-path "$SALIENCY_PATH" \
    --guided-num-groups "$GUIDED_NUM_GROUPS" \
    --sym \
    --true-sequential \
    --eval
