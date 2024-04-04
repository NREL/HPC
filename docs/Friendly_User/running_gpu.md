# Running on Kestrel GPU Nodes



A major consideration you should keep in mind when pulling or building GPU-enabled images is the version of both the GPU drivers and [CUDA](https://developer.nvidia.com/cuda-toolkit) on the host system. You can obtain this information from the host by running the command `nvidia-smi` after allocating a GPU within a Slurm job on your desired system. Alternatively, you could simply consult the table below. Note that the GPU drivers/CUDA versions depend on a given GPU partition. If your running container is installed with a different GPU driver/CUDA version than what is listed below for your target system, you will either run into a fatal error, or the software will bypass the GPU and run on the CPU, slowing computation. For example, an [image containing GPU-accelerated TensorFlow](../../Machine_Learning/Containerized_TensorFlow/index.md) might work perfectly fine on Eagle, but its drivers would likely be too old for either of Kestrel's GPU partitions.

| System   | Partition name | GPU type<br>(cards per node) | `nvidia-smi`<br>GPU driver version | CUDA Version |
|:--------:|:--------------:|:----------------------------:|:----------------------------------:|:------------:|
| Kestrel  | gpu-a100 (TBD) | A100 (4)                     | xxx.xx.xx                          | xx.x         |
| Kestrel  | gpu-h100       | H100 (4)                     | 545.23.08                          | 12.3         |
| Swift    | gpu            | A100 (4)                     | 545.23.08                          | 12.3         |
| Eagle    | gpu            | V100 (2)                     | 460.32.03                          | 11.2         |
| Vermilion| gpu (TBD)      | A100 (1)                     | xxx.xx.xx                          | xx.x         |



##### Allocate a Slurm job with a GPU resource

 You will need to submit the job to a GPU partition and explicitly request how many GPU cards (per node) you wish to run on with the `--gres=gpu:<number>` option. As an example, this uses `salloc` to request an interactive job that allocates one NVIDIA H100 GPU card (`--gres=gpu:1`) on one node (`-N 1`) on Kestrel:

```
salloc -A <account> -p gpu-h100 -t 01:00:00 -N 1 --gres=gpu:1
```

You can verify whether the GPU device is found by running `nvidia-smi` after landing on the node. If you receive any kind of output other than a `No devices were found` message, there is a GPU waiting to be used by your software.

!!! note
    If you submit to the `gpu-h100` partition without including `--gres=gpu:<# of GPUs per node>`, your job will *not* allocate any GPU cards to your session, preventing you from using the resource at all.