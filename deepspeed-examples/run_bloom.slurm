#!/bin/bash
#SBATCH --job-name=dsExample
#SBATCH --account=project_462000365
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=7
#SBATCH --gpus-per-node=8
# SBATCH --hint=nomultithread
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH -o ting%j.out
#SBATCH -e ting%j.err

set -o pipefail
wd=$(realpath .)
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.1.0.sif
# SIF=../pytorch_rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1.sif
# SIF=deepspeed_latest.sif

export SINGULARITYENV_NCCL_DEBUG=INFO
# export SINGULARITYENV_NCCL_DEBUG_SUBSYS=INIT,COLL
Nodes=$SLURM_NNODES
c=fe
export MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

cd $wd

srun --jobid $SLURM_JOBID\
     -N $((Nodes)) \
     -n $((Nodes)) \
     -c $((8*7)) \
    --cpu-bind=mask_cpu:$MYMASKS \
    --gpus $(($Nodes*8)) \
    singularity exec \
        -B "$wd:/workdir" \
        -B /var/spool/slurmd \
        -B /opt/cray \
        -B /usr/lib64/libcxi.so.1 \
        -B /usr/lib64/libjansson.so.4 \
        $SIF /workdir/run-bloom.sh
