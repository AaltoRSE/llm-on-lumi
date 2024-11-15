#!/bin/bash
# SBATCH --job-name=dpoExample
#SBATCH --account=project_462000365
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH -o ming%j.out
#SBATCH -e ming%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch
export HF_HOME=/scratch/project_462000365/tianyu12/triton-llm-tutorial/lumi-examples/multi-node
export HF_TOKEN=yourtokenmo
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# use cached/predownloaded model weights and datasets
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=0
export OMP_NUM_THREADS=10
# export NCCL_DEBUG=INFO
# echo "Hostfile created: $(cat $hostfile)"
srun python3 bloom-ds-zero-inference-mpi.py --name bigscience/bloom --deepspeed 2>&1 | tee bloom-ds-zero-inference_bs=1.txt
# srun python3 bloom-ds-zero-inference-mpi.py --name  meta-llama/Llama-2-70b-chat-hf  --batch_size 1  2>&1 | tee bloom-ds-zero-inference_bs=1.txt
