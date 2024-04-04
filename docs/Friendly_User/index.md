# Getting Started

## About Kestrel GPU Nodes



## Running on GPU Nodes
Kestrel will have 132 GPU nodes added to the existing CPU capacity, with each GPU node hosting 4 NVIDIA H100 GPUs. 


## Additional Resources

* [Kestrel System Configuration](https://www.nrel.gov/hpc/kestrel-system-configuration.html)
* A collection of sample makefiles, source codes, and scripts for Kestrel can be found in the [Kestrel repo](https://github.com/NREL/HPC/tree/master/kestrel). 


## Running Jobs

salloc --time=2:00:00 --reservation=<friendly user reservation> --partition=gpu-h100  --account=<project handle>  --nodes=1  --gres=gpu:h100:<# of GPUS per node>

```
salloc -A <account> -p gpu-h100 -t 01:00:00 -N 1 --gres=gpu:1
```

You can verify whether the GPU device is found by running `nvidia-smi` after landing on the node. If you receive any kind of output other than a `No devices were found` message, there is a GPU waiting to be used by your software.

!!! note
    If you submit to the `gpu-h100` partition without including `--gres=gpu:<# of GPUs per node>`, your job will *not* allocate any GPU cards to your session, preventing you from using the resource at all.

To start an interactive session:

1. Allocate the node(s):<br>
    ```salloc --nodes=N --ntasks-per-node=npn --time=1:00:00 ```
1. 
```srun -n np --mpi=pmi2 ./executable``` <br>
where "np" is N*npn, and npn=104 if requesting a whole node. 

!!! warning
    If the argument --mpi=pmi2 is not used, the executable will be launched np times instead of being launched once using np cores. 

There are example job submission scripts in the [Environments Tutorial](../Environments/tutorial.md) page. 
