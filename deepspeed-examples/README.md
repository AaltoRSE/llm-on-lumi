# Multi gpu/node inference on LUMI
## Create environment
See [USE_LUMI.md](../USE_LUMI.md) for detailed instructions.

## Run the inference

Then run the following script to start the model inference.
```
sbatch run_bloom.slurm
```

## Launching distributed environment on LUMI: Rendezvous mechanism

We use the **rendezvous mechanism in PyTorch Distributed** to launch multi-node jobs on LUMI, since it is not possible to ssh to other nodes (which the DeepSpeed launcher would require).

See `run_bloom.slurm` for the setup.

CSC’s tutorial has examples for this mechanism:  
https://docs.csc.fi/support/tutorials/ml-multi/#pytorch-ddp

DeepSpeed discussion on using this approach:  
https://github.com/microsoft/DeepSpeed/issues/1603
