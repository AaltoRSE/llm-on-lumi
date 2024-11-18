

## Submit sbatch job on LUMI (use nanoton as an example)
### find a container to use:


### step 1: install extra python libraries as explained:
 https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#extending-the-containers-with-virtual-environment-support

For our case, we need to install a different version of torch:
`pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121`

to be distinguished from the torch in the system site package, we can installed it in a non-standard target folder like this:
`pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121 --target /scratch/$project_id/$USER/nanotron/preview_torch` and modify the pythonpath to force the usage of this preview version torch

### step 3: submit job
For our case, three scripts are used:
- .slurm script to submit job (run_train.slurm)
- .sh script to set up environment (run_train.sh)
- .py script, the entrypoint for model training (run_train.py)
After preparing these scripts, you can run:
`sbatch run_train.slurm`
to submit the job.

## Interactive use of LUMI
LUMI provides a web interface https://docs.lumi-supercomputer.eu/runjobs/webui/ for interactive usage.
In order to launch a jupyter notebook with a customized kernel, follow the steps:
- pip install jupyterlab (and any other extra libraries you need) in a interactive shell session of the container LUMI provides, see detailed explanation here: https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#extending-the-containers-with-virtual-environment-support

run `head -n 1 $(which jupyter-lab)` to check if jupyter-lab command matches the correct Python interpreter, the one in the container

- create a "init" script like this `./python4jupyter/script4jupyter.sh`
- create a wrapper script that acts as a custom python:`./python4jupyter/python`
- choose Advanced settings when launching jupyter, add the path to the "init" script there


overwrite layernorm!
srun?
use nightly torch!

mask?

# check if jupyter-lab command matches the correct Python interpreter
head -n 1 $(which jupyter-lab)
