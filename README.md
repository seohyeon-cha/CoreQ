# [NeurIPS 2026 Spotlight] CoreQ: Learning-Free Mismatch Correction and Successive Rounding for Post-Training Quantization

This repository contains the reference implementation of **CoreQ**, a
post-training weight-only quantization method for large language models.
The code reproduces the LLaMA-family results reported in the paper.

CoreQ has two ingredients:

1. **Learning-free mismatch correction** — a closed-form, per-layer
   coefficient α derived from a local correlation estimate. In this
   implementation α is a single scalar per linear sub-module, obtained by
   summing the correlation statistics over all rows and columns of that
   module. Implemented in `algorithms/coreq.py`.
2. **Successive (beam) rounding** — a row-wise beam search over the
   discrete codewords that refines the CoreQ solution. Implemented in
   `algorithms/coreq_beam.py`.

The repository also includes the reference baselines GPTQ, LDLQ, GPTAQ,
and GuidedQuant for comparison.

---

## Repository layout

```
.
├── llama_step.py             # Main entry point (per-block layer-wise quant.)
├── algorithms/
│   ├── gptq.py               # Baseline rounding (per-layer GPTQ)
│   ├── ldlq.py               # Baseline rounding (LDL-based)
│   ├── gptaq.py              # Baseline: GPTQ + closed-form α correction
│   ├── guidedquant.py        # Baseline: row-grouped saliency-weighted Hessian
│   ├── coreq.py              # CoreQ rounding (no beam)
│   └── coreq_beam.py         # CoreQ + successive (beam) rounding
├── quant/                    # Quantizer / packed-linear kernels
├── utils/                    # Data loaders, model utilities, plotting
├── scripts/                  # Ready-to-run shell scripts (see below)
│   ├── l2-7b/                #   Llama-2-7B
│   ├── l2-13b/               #   Llama-2-13B
│   ├── l3-8b/                #   Llama-3-8B
│   └── l2-70b/               #   Llama-2-70B
├── requirements.txt
└── README.md
```

Six algorithms are supported via `--method`: `gptq`, `ldlq`, `gptaq`,
`guidedq`, `coreq`, `coreq_beam`.

---

## Installation

```bash
# Create a fresh conda environment (Python 3.10+)
conda create -n coreq python=3.10 -y
conda activate coreq

# Install dependencies (CUDA 12.4 wheels assumed for torch/triton)
pip install --upgrade pip
pip install -r requirements.txt
```

The `fast_hadamard_transform` package is installed from source.

---

## Quick start

All shell scripts are simple Python launchers; they do **not** assume any
particular cluster or scheduler. Run them from the repository root:

```bash
# 3-bit per-channel CoreQ on Llama-2-7B (uses HuggingFace Hub by default)
bash scripts/l2-7b/coreq.sh

# CoreQ + beam rounding (beam size 7 by default)
bash scripts/l2-7b/coreq_beam.sh

# Baselines
bash scripts/l2-7b/gptq.sh
bash scripts/l2-7b/ldlq.sh
bash scripts/l2-7b/gptaq.sh
bash scripts/l2-7b/guidedq.sh   # requires precomputed saliency tensors
```

Each script exposes a small set of environment variables for overrides:

| Variable     | Default                         | Meaning                                          |
|--------------|---------------------------------|--------------------------------------------------|
| `MODEL_PATH` | e.g. `meta-llama/Llama-2-7b-hf` | HuggingFace model id or local path               |
| `DATASET`    | `c4`                            | Calibration set (`c4`, `wikitext2`, `ptb`)       |
| `WBITS`      | `3`                             | Weight bit-width (`2`, `3`, `4`, …)              |
| `GROUPSIZE`  | `-1`                            | Quant. groupsize (`-1` = per channel)            |
| `NSAMPLES`   | `128`                           | # calibration samples                            |
| `SEED`       | `0`                             | Calibration sampling seed                        |
| `ALPHA`      | `0.5`                           | Mismatch-correction coefficient (CoreQ / GPTAQ)  |
| `BEAM_SIZE`  | `7` (in `coreq_beam.sh`)        | Beam width for successive rounding               |

Example: 4-bit, groupsize-128 CoreQ on Llama-3-8B, seed 1:

```bash
WBITS=4 GROUPSIZE=128 SEED=1 bash scripts/l3-8b/coreq.sh
```

---

## Running directly with `llama_step.py`

The shell scripts wrap a single command:

```bash
python llama_step.py <MODEL> <DATASET> \
    --method {gptq|ldlq|gptaq|guidedq|coreq|coreq_beam} \
    --wbits 3 --groupsize -1 --nsamples 128 --seed 0 \
    --sym --true-sequential --eval
```

Useful additional flags:

- `--act-order` — column reordering heuristic (used by GPTQ/GPTAQ).
- `--beam-size K` — beam width for `coreq_beam` (`K=1` falls back to CoreQ).
- `--saliency-path DIR` and `--guided-num-groups G` — saliency directory
  and group count for `--method guidedq` (saliency tensors must be
  precomputed externally).
- `--lm-eval` — evaluate downstream tasks (PIQA / ARC / HellaSwag /
  WinoGrande / BoolQ) via `lm-evaluation-harness`.

Run `python llama_step.py --help` for the full list.

### Choosing α for CoreQ (`--alpha-method`)

CoreQ supports two modes for the mismatch-correction coefficient α:

- **`--alpha-method corr` (default).** CoreQ automatically computes the
  per-layer α_corr coefficient introduced in the paper from the
  calibration statistics (the closed-form data-driven α derived from a
  local correlation / SNR estimate). Whatever value is passed to
  `--alpha` is overwritten layer-by-layer by α_corr. 
- **`--alpha-method fixed`.** Use the user-supplied `--alpha` value
  unchanged for every layer (no calibration-driven update). This is the
  ablation setting; pass e.g. `--alpha 0.5` together with
  `--alpha-method fixed` to reproduce a single-α baseline. 

---

## Code references and acknowledgments

Our code is implemented based on the following open-source repositories,
and we thank the authors for releasing their work:

- [GPTQ](https://github.com/ist-daslab/gptq)
- [QuIP](https://github.com/Cornell-RelaxML/QuIP)
- [GPTAQ](https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ)

