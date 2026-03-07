#!/bin/bash -e

# ── GPU health check ────────────────────────────────────────────────────────
echo "Rank $SLURM_PROCID - $(taskset -p $$) $ROCR_VISIBLE_DEVICES"
if [ $SLURM_LOCALID -eq 0 ]; then
    rocm-smi
fi
sleep 2

# ── MIOpen cache (per-node temp dir) ────────────────────────────────────────
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
if [ $SLURM_LOCALID -eq 0 ]; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

# ── RCCL / NCCL settings ───────────────────────────────────────────────────
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

# Do NOT set CUDA_LAUNCH_BLOCKING=1 or HIP_LAUNCH_BLOCKING=1;
# synchronous GPU ops deadlock RCCL collectives.
export CUDA_LAUNCH_BLOCKING=0
export HIP_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

# ── Distributed environment ────────────────────────────────────────────────
export MASTER_ADDR=$(/workdir/get-master "$SLURM_NODELIST")
export MASTER_PORT=$((6000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID

# ── HuggingFace cache ──────────────────────────────────────────────────────
export HF_HOME=/workdir
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
# For gated models (e.g. LLaMA), set HF_TOKEN:
# export HF_TOKEN=your_token_here

# ── Launch ──────────────────────────────────────────────────────────────────
set -x

NNODES=${SLURM_NNODES:-2}

export LAUNCHER="python -u -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts 0 \
    --tee 3"

export CMD="/workdir/bloom-ds-zero-inference-torch-launcher.py \
    --name bigscience/bloom --deepspeed"

$LAUNCHER --node_rank $SLURM_PROCID $CMD 2>&1 | tee /workdir/zero-inference-torch-launcher=1.txt
