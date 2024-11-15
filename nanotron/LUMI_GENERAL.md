

## Submit sbatch job on LUMI (use nanoton as an example)

### Step 1: Obtain the Singularity Container with EasyBuild

1. **Modify Installation Paths**:  
   Begin by setting up the installation paths for EasyBuild. Follow the preparation instructions in the [LUMI EasyBuild documentation](https://docs.lumi-supercomputer.eu/software/installing/easybuild/#preparation-set-the-location-for-your-easybuild-installation) to specify where to save the container and module files.

2. **Load Necessary Modules**:  
   Load the required modules for EasyBuild and Singularity:
   ```bash
   module load LUMI partition/container EasyBuild-user
   ```
3. **Install the container**:
    ```bash
    eb PyTorch-2.2.0-rocm-5.6.1-python-3.10-singularity-20240315.eb
    ```
Additional container options with different PyTorch, ROCm, or Python versions can be found in the:
https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#singularity-containers-with-modules-for-binding-and-extras

### step 2: install extra python libraries:
 https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#extending-the-containers-with-virtual-environment-support

For our case, extra libraries are listed in extra_requirements.txt (some might be omitted)

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