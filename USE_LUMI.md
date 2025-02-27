# PyTorch on LUMI with Singularity Containers

## Step 1: Obtain the Singularity Container with EasyBuild

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

## step 2: install extra python libraries:
 https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#extending-the-containers-with-virtual-environment-support

For our case, extra libraries are listed in extra_requirements.txt (some might be omitted)

## step 3: submit job
For our case, three scripts are used:
- .slurm script to submit job (run_imagenet_training.slurm)
- .sh script to set up environment (train_imagenet.sh)
- .py script, the entrypoint for model training (pl_eval_navit.py)

## Main modification of the python script:
PyTorch Lightning typically manages the distributed environment setup automatically. However, on LUMI, it is necessary to include the following line in your script for proper initialization (not sure if it is due to the slurm config):
```python
 torch.distributed.init_process_group(backend="nccl")
```

## Some other modifications can be done
Create a helper function to print only on rank 0 to avoid redundancy. 
```python
def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)
```
## Sharding datasets on LUMI
https://github.com/YuTian8328/dataset-sharding

On LUMI (in general any cluster based distributed file system) transfering big number of small files to compute nodes is very slow.
To optimize performance, it’s preferable to use larger files, which reduces transfer overhead and improve read speeds. However, working with a single large file also has its challenges. In such cases, sharding the dataset into larger chunks can provide a balance.
Sharding a dataset, which involves splitting a massive dataset into large subsets (typically 1-10GB per shard), offers several advantages:

- Increased I/O Efficiency: With sharded data, multiple processors can read different shards simultaneously, significantly improving read speeds and reducing I/O bottlenecks.
- Enhanced Dataset Randomization: Reading from randomly selected shards helps to avoid always presenting the same data first, thus achieving a level of dataset randomization. Although this is not true randomization, it is far better than sequentially reading from a single, large file.


## Webdataset
webdataset offers a convenient way to load large and sharded datasets into pytorch, it implements the iterable dataset interface of pytorch, and can thus be used like any other pytorch dataset.
A simple example can be found in the dataset-sharding repo.
