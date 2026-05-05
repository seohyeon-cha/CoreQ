#!/usr/bin/env bash
# CoreQ with successive (beam) rounding on Llama-2-13B.
# Set BEAM_SIZE > 1 to enable multi-beam search per block.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-2-13b-hf"}
DATASET=${DATASET:-c4}
WBITS=${WBITS:-3}
GROUPSIZE=${GROUPSIZE:--1}
NSAMPLES=${NSAMPLES:-128}
SEED=${SEED:-0}
ALPHA_METHOD=${ALPHA_METHOD:-corr}
ALPHA=${ALPHA:-0.5}

BEAM_SIZE=${BEAM_SIZE:-4}
CD_PASSES=${CD_PASSES:-0}

python -u llama_step.py \
    "$MODEL_PATH" "$DATASET" \
    --method coreq_beam \
    --wbits "$WBITS" \
    --groupsize "$GROUPSIZE" \
    --nsamples "$NSAMPLES" \
    --seed "$SEED" \
    --alpha-method "$ALPHA_METHOD" \
    --alpha "$ALPHA" \
    --beam-size "$BEAM_SIZE" \
    --cd_passes "$CD_PASSES" \
    --sym \
    --true-sequential \
    --eval
