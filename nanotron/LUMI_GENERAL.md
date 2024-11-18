

## Submit sbatch job on LUMI (use nanoton as an example)
### step 1: find a container to use:
we can use this one for nanotron: /appl/local/containers/easybuild-sif-images/lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0-dockerhash-187f41102477.sif


### step 2: install extra python libraries as explained:
 https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#extending-the-containers-with-virtual-environment-support

In the nanotron case, we need to install a different version of torch:
`pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121`

To be distinguished from the torch in the system site package, we can installed it in a non-standard target folder like this:
`pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121 --target /scratch/$project_id/$USER/nanotron/preview_torch` and then modify the PYTHONPATH in the `
run_train.sh` script to force the usage of this preview version torch.

### step 3: submit job
For our case, three scripts are used:
- .slurm script to submit job (run_train.slurm)
- .sh script to set up environment for distributed training (run_train.sh)
- .py script, the entrypoint for model training (run_train.py)
After preparing these scripts, you can run:
`sbatch run_train.slurm`
to submit the training job.


