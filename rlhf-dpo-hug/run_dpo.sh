#!/bin/bash -e
# Optional: uncomment to debug GPU visibility on LUMI
# echo "Rank $SLURM_PROCID - $(taskset -p $$) $ROCR_VISIBLE_DEVICES"
# if [ "${SLURM_LOCALID:-0}" -eq 0 ]; then rocm-smi; fi
# sleep 2

# Hugging Face cache (set WORKDIR or default to /workdir for Singularity bind)
export HF_HOME="${WORKDIR:-/workdir}"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN must be set (Hugging Face access token)" >&2
    exit 1
fi

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/rlhf_dpo_multi_gpu.py"
