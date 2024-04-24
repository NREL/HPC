# Getting Started

Kestrel has 132 GPU nodes. Each GPU node has 4 NVIDIA H100 GPUs (80 GB), 128 CPU cores, and 384GB RAM. 

GPU nodes are shared, meaning that less than a full node may be requested for a job, leaving the remainder of the node for use by other jobs concurrently.

To request use of a GPU, use the flag `--gres=gpu:<quantity>` with sbatch, srun, or salloc, or add it as an #SBATCH directive in your sbatch submit script, where `<quantity>` is a number from 1 to 4.

## Running on GPU Nodes

An example GPU job allocation command: 

```
salloc --time=2:00:00 --reservation=<friendly user reservation> --partition=gpu-h100  --account=<project handle>  --nodes=1  -n 1 --mem-per-cpu=8G --gres=gpu:h100:<# of GPUS per node>
```

You're automatically given access to all of the memory for the GPU or GPUs requested (80GB per GPU). Note that you'll need to use `-n` to request the number of CPU cores needed, and `--mem` or `--mem-per-cpu` to request the amount of CPU memory needed. You can use `--exclusive` to requst all of the resources on the GPU node.  

You can verify whether the GPU device is found by running `nvidia-smi` after landing on the node. If you receive any kind of output other than a `No devices were found` message, there is a GPU waiting to be used by your software.

!!! warning
    If you submit to the `gpu-h100` partition without including `--gres=gpu:<# of GPUs per node>`, your job will *not* allocate any GPU cards to your session, preventing you from using the resource at all.


## Additional Resources

* [Kestrel System Configuration](https://www.nrel.gov/hpc/kestrel-system-configuration.html)
* A collection of sample makefiles, source codes, and scripts for Kestrel can be found in the [Kestrel repo](https://github.com/NREL/HPC/tree/master/kestrel). 

