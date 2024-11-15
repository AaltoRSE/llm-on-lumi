# Multi gpu/node inference on LUMI
## Create environment
This environment is based on a container provided by LUMI. The container contains pytorch,torchvision, torchaudio.

Run the following script to obtain the container image and install additional requirements by using the requirements.txt file in the folder. 

```
sbatch buildEnv.sh
```
Before you run this, modify the buildEnv.sh file to use your project id.


Then run the following script to start the model inference.
```
sbatch run_bloom.slurm
```


## To launch distributed environment on LUMI, we have two choices:
### MPI via CSC's modules:
```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch
```
see bloom_mpi.slurm
### Rendevous mechanism in PyTorch Distributed 

see run_bloom.slurm

## deepspeed on LUMI:
Unfortunately, it is not possible to ssh to other nodes on LUMI, which is what
the DeepSpeed launcher is trying to do. DeepSpeed provides another method to
launch multi-node jobs based on mpirun, as explained here:
https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.deepspeed.ai%2Fgetting-started%2F%23mpi-and-azureml-compatibility&data=05%7C02%7Cyu.tian%40aalto.fi%7Caf9de594130d405e805a08dc43323ba0%7Cae1a772440414462a6dc538cb199707e%7C1%7C0%7C638459130234449909%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=Mvll9mRzGXuiBjG5ZDBO5vEgkewBcY2j0geZK%2BDeIhA%3D&reserved=0

This is a viable option if you are using the CSC PyTorch module on LUMI.

The PyTorch containers of LUMI currently do not include MPI.
If you are using those containers, you can consider using the rendevous
mechanism in PyTorch Distributed to launch your multi-node training jobs. The
following tutorial (written by CSC) includes examples on how to use that
mechanism: https://docs.csc.fi/support/tutorials/ml-multi/#pytorch-ddp

This method is also further discussed in the following issue on DeepSpeed's
GitHub repository: https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2Fmicrosoft%2FDeepSpeed%2Fissues%2F1603&data=05%7C02%7Cyu.tian%40aalto.fi%7Caf9de594130d405e805a08dc43323ba0%7Cae1a772440414462a6dc538cb199707e%7C1%7C0%7C638459130234458052%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=5JwAsu8CJ%2Febic%2B11mXToobuATSspcFnrTVm85MMOSQ%3D&reserved=0

