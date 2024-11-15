#!/bin/bash -e
# Make sure GPUs are up
echo "Rank $SLURM_PROCID - $(taskset -p $$) $ROCR_VISIBLE_DEVICES"
# Make sure GPUs are up, this seems to sometimes be necessary on lumi... 
if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi 
fi
sleep 2

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Set MIOpen cache to a temporary folder.
if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

# Report affinity
echo "Rank $SLURM_PROCID --> $(taskset -p $$)"

# Start conda environment inside the container
$WITH_CONDA
# . /workdir/env_rocm5.7/bin/activate
export PYTHONPATH=/workdir/env_bloom/lib/python3.10/site-packages
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
# Set interfaces to be used by RCCL.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# export NCCL_NET_GDR_LEVEL=3
export LOGLEVEL=DEBUG

# Set environment for the app
export MASTER_ADDR=$(/workdir/get-master "$SLURM_NODELIST")
export MASTER_PORT=6000
export WORLD_SIZE=$SLURM_NPROCS
export RUNID="34567"
export RANK=$SLURM_PROCID
echo $MASTER_ADDR
echo "rank"
echo $WORLD_SIZE
echo "nodes\:-$SLURM_NODELIST"
echo "visible\: -$ROCR_VISIBLE_DEVICES"

# use the project directory as Huggingface cache folder
export HF_HOME=/workdir
export HF_TOKEN=yourtoken

# use cached/predownloaded model weights and datasets
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=0
# echo 
set -x
export HIP_LAUNCH_BLOCKING=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
echo "slurmprocid"
echo $SLURM_PROCID
export LAUNCHER="python -u -m torch.distributed.run \
        --nnodes=2 \
        --nproc_per_node=8 \
        --rdzv_id=$SLURM_JOBID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --max_restarts 0 \
        --tee 3
    "

# export CMD="/workdir/bloom-ds-inference.py \
#     --name tiiuae/falcon-180B-chat --deepspeed
#     "

# export CMD="/workdir/bloom-ds-zero-inference-torch-launcher.py \
#     --name meta-llama/Llama-2-70b-chat-hf --deepspeed
#      "
export CMD="/workdir/bloom-ds-zero-inference-torch-launcher.py \
    --name bigscience/bloom --deepspeed
     "   

$LAUNCHER --node_rank $SLURM_PROCID $CMD 2>&1 | tee /workdir/zero-inference-torch-launcher=1.txt