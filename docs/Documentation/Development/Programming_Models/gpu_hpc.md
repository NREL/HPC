This page documents how NREL HPC users can utilize GPUs, from submitting the right kind of job to Slurm to examples of creating custom CUDA kernels from Python.

## Submitting GPU jobs to Slurm

### Example scripts

The following examples are generic templates that NREL HPC users can adapt for their own GPU job scripts for a given system. Be sure to replace `<allocation>` with the name of your HPC allocation. Note that Kestrel and Swift's GPU partitions have sharable nodes, allowing for multiple jobs to run on one node simultaneously. Since there are four GPU cards on each node on these systems, each node can theoretically accommodate four GPU-driven jobs at once. As such, example scripts for those systems are tailored for requesting one-quarter of a node by default. Although Vermilion's GPUs are technically "shared" in the sense that multiple (CPU) jobs can run on one node, there is only one GPU per node. As such the Vermilion example requests the entire node. Please refer to the [system-specific pages](../../Systems/index.md) for more information on the GPUs available on each cluster and how AUs are charged accordingly.

!!! note
    When launching a GPU job on Kestrel, be sure to do so from [one of its dedicated GPU login nodes](https://nrel.github.io/HPC/Documentation/Systems/Kestrel/#accessing-kestrel).

!!! note
    Be aware that `--mem` in Slurm ALWAYS refers to CPU, not GPU, memory. You are automatically given all of the GPU memory in a Slurm job.

??? example "Kestrel"
    ```bash
    #!/bin/bash 
    #SBATCH --account=<allocation>
    #SBATCH --time=01:00:00
    #SBATCH --mem=80G
    #SBATCH --gpus=1
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=32
    #SBATCH --output=%j-%x.log

    # don't forget to submit this from a GPU login node!
    # note that you do not have to specify a partition on Kestrel;
    # your job will be sent to the appropriate gpu-h100 queue based
    # on your requested --time
    <GPU-enabled code to run>
    ```

??? example "Swift"
    ```bash
    #!/bin/bash
    #SBATCH --account=<allocation>
    #SBATCH --partition=gpu
    #SBATCH --time=01:00:00
    #SBATCH --mem=250G
    #SBATCH --gpus=1
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=24
    #SBATCH --output=%j-%x.log

    <GPU-enabled code to run>
    ```

??? example "Vermilion"
    ```bash
    #!/bin/bash
    #SBATCH --account=<allocation>
    #SBATCH --partition=gpu
    #SBATCH --time=01:00:00
    #SBATCH --mem=0
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=30
    #SBATCH --output=%j-%x.log
    #SBATCH --exclusive

    # Note that you do not have to explicitly request a GPU on Vermilion 
    # with `#SBATCH --gpus=1` or `#SBATCH --gres=gpu:1`.
    <GPU-enabled code to run>
    ```

### GPU-relevant environment variables

The following are some GPU-relevant environment variables you can set in your submission scripts to Slurm.

| Variable             | Description |
| :--                  | :--         |
| SLURM_GPUS_ON_NODE   | Quantity of GPU devices available to a Slurm job.|
| SLURM_JOB_GPUS       | GPU device ID(s) available to a Slurm job. Starts with `0`.|
| CUDA_VISIBLE_DEVICES | GPU device ID(s) available to a CUDA process. Starts with `0`. |

### Software containers

Please refer to our [dedicated documentation](../Containers/apptainer.md#utilizing-gpu-resources-with-apptainer-images) on using GPUs from software containers for more information.

## Migrating workflows from CPU to GPU 

GPUs contain hundreds or thousands of cores and can considerably speed up certain operations when compared to CPUs. However, unless you are already using a GPU-accelerated application with built-in CUDA kernels (such as some versions of PyTorch), your custom code will likely require significant changes to be able to effectively use a GPU device. This is even more true if your intent is to parallelize your code over multiple GPU devices. Further, some algorithms or routines are much better suited for GPU computation than others. As such, the first question you should always ask yourself is whether it makes sense to invest the time and effort needed to refactor your CPU-driven code for GPU computation. The following subsections describe key points to consider when you want to take the plunge into GPU computing, ending with an example using the `numba` package to refactor Python functions for Kestrel's H100 GPUs.

### Ensure your algorithm is suited for GPU computation

Not all algorithms are created equal when it comes to being able to effectively utilize a GPU. In general, GPUs best accommodate large numbers of relatively small, simulataneous operations ("massive parallelism"); canonical algorithmic examples of this include graphics processing (reflecting the "G" in "GPU") and many linear algebra computations (e.g., "matrix-matrix math" like [BLAS3](https://www.netlib.org/blas/#_level_3) routines). Algorithms that would likely perform poorly on a GPU without significant modification are those that launch serial tasks (think for-loops or `apply` statements in Python) that may each require a significant amount of RAM and/or write to the filesystem directly. 

### Minimize data transfer between CPU and GPU devices

Without even considering the characteristics of the algorithm itself, one of the largest bottlenecks in GPU computing is copying data from the CPU to the GPU device(s). In many cases, copying data between devices can easily take longer than the execution of the algorithm. As such, to maximize an algorithm's performance on a GPU, it is imperative to consider employing application-specific routines to minimize the total amount of data transferred during runtime. In other words, the goal with effective GPU computing often comes down to designing the code to transfer as little data as possible as infrequently as possible.

### Ways to compile CUDA code

CUDA is a low-level API distributed by NVIDIA that allows applications to parallelize on NVIDIA GPUs, such as the H100s available on Kestrel or the A100s on Swift. Because of this, any GPU-driven code gets compiled into a *CUDA kernel*, which is essentially a *function* translated to machine code for the GPU. There are two CUDA-aware compilers available from NVIDIA: [`nvcc`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/), a CUDA analog to the more generic `cc`, and [`nvrtc`](https://docs.nvidia.com/cuda/nvrtc/index.html), which is NVIDIA's runtime compiler for "just-in-time" (JIT) compilation.

See [this page](../../Systems/Kestrel/Environments/gpubuildandrun.md) for specific GPU code compilation examples on Kestrel, which include both CUDA and OpenAcc (an open-source alternative) implementations.

### Example: Create a custom CUDA kernel in Python with `numba`

To demonstrate some of the concepts described here, we will use `numba` to refactor an algorithm that initially performs poorly on a GPU due to how its input/output data are copied between devices. `numba` is a Python package for creating custom CUDA kernels from Python functions working with numeric data. It has a simple interface to CUDA that feels comfortable for most Python users, though advanced GPU programmers may consider building GPU-accelerated applications with "pure" CUDA. For such examples of creating custom CUDA kernels outside of Python, please [see here](../../Systems/Kestrel/Environments/gpubuildandrun.md). 

This example is written assuming you have access to Kestrel, but it should be able to run on any system with at least one GPU node.

#### Install `numba` from Anaconda

The `numba` package is easily installable through Anaconda/mamba. For any GPU-enabled application, the biggest concern during installation is whether the application version matches the GPU drivers. At the time this page was written, the GPU drivers on Kestrel reflect `CUDA 12.4`, and so we must ensure that our version of numba can work with that. In conda, we can control this by explicitly passing the corresponding `cuda-version=CUDA_VERSION` from conda-forge and asking for a `cuda-toolkit` from the `nvidia/label/cuda-CUDA_VERSION` channel. When we do this, we will force a compatible version of `numba` to install into the `$CONDA_ENVIRONMENT` we define (which is in /scratch to save space). We will also install `numpy` to work with numeric data, as well as `pandas` for data manipulation tasks:

!!! note
    It is best to create this environment on a node with at least one available NVIDIA GPU. On any such node, you can run the command `nvidia-smi` to display the current GPU driver version (as well as any running GPU processes).

```bash
ml mamba
CONDA_ENVIRONMENT=/scratch/$USER/.conda-envs/numba-cuda124
mamba create --prefix=$CONDA_ENVIRONMENT \
  conda-forge::numba \
  conda-forge::numpy \
  conda-forge::pandas \
  conda-forge::cuda-version=12.4 \
  nvidia/label/cuda-12.4.0::cuda-toolkit \
  --yes
conda activate $CONDA_ENVIRONMENT
```

#### Example numba code

Consider the script `numba-mat.py` below. This script demonstrates the importance of deciding *when* and *how often* one should copy data to and from the GPU device to optimize runtime performance.

!!! note
    This example requires approximately 40GB of CPU RAM to complete successfully. Be sure to run this on a GPU compute node from a Slurm job accordingly, with the defined `$CONDA_ENVIRONMENT` activated.

??? example "`numba-mat.py`: Matrix multiplication with numba"
    ```python
    # Define and JIT-compile a CUDA function (kernel) with numba for simple
    # matrix multiplication. This script demonstrates the importance of 
    # balancing the cost of copying data from the host CPU to GPU device in 
    # terms of runtime performance.

    # Please contact Matt.Selensky@nrel.gov with any questions.

    import numba
    from numba import vectorize
    from numba import cuda
    import pandas as pd
    import numpy as np
    from time import time

    # Note that you must define the dtype (float32 is preferred over 
    # float64) and target device type ('cuda' for GPU)
    @vectorize(['float32(float32, float32)'], target='cuda')
    def gpu_mult(x, y):
        z = x ** y
        return z


    # create random arrays as input data
    asize = pow(10, 9)
    array_a = np.float32(np.random.rand(asize))
    array_b = np.float32(np.random.rand(asize))
    array_c = np.float32(np.random.rand(asize))
    matrix_a = ([array_a], [array_b], [array_c])
    matrix_b = ([array_c], [array_b], [array_a])

    # define number of function loops to run for each test case
    nloops = 10

    ### numpy - CPU
    # Test Case 1: Here, we just use pure numpy to perform matrix multiplication on the CPU.
    t0 = time()
    for i in np.arange(nloops):
        np.multiply(matrix_a, matrix_b)
    cpu_time = time()-t0
    print("numpy on CPU required", cpu_time, "seconds for", nloops, "function loops")

    ### numba - GPU
    # Test Case 2: Here, we copy arrays to GPU device __during__ the execution of gpu_mult()
    t0 = time()
    for i in np.arange(nloops):
        gpu_mult(matrix_a, matrix_b)
    gpu_time0 = time()-t0
    print("numba on GPU required", gpu_time0, "seconds for", nloops, "function loops (data are actively copied to GPU device)")

    # Test Case 3: Here, we copy arrays to GPU device __before__ the execution of gpu_mult()
    # output is then copied back to GPU
    matrix_a_on_gpu = cuda.to_device(matrix_a)
    matrix_b_on_gpu = cuda.to_device(matrix_b)
    t0 = time()
    for i in np.arange(nloops):
        gpu_mult(matrix_a_on_gpu, matrix_b_on_gpu)
    gpu_time1 = time()-t0
    print("numba on GPU required", gpu_time1, "seconds for", nloops, "function loops (data were pre-copied to GPU device; output is copied back to CPU)")

    # Test Case 4: Here, we copy arrays to GPU device __before__ the execution of gpu_mult()
    # output remains on GPU unless we copy it back with out_device.copy_to_host()
    matrix_a_on_gpu = cuda.to_device(matrix_a)
    matrix_b_on_gpu = cuda.to_device(matrix_b)
    out_device = cuda.device_array(shape=(asize,len(matrix_a)), dtype=np.float32)  # does not initialize the contents, like np.empty()
    t0 = time()
    for i in np.arange(nloops):
        gpu_mult(matrix_a_on_gpu, matrix_b_on_gpu, out=out_device)
    gpu_time2 = time()-t0
    print("numba on GPU required", gpu_time2, "seconds for", nloops, "function loops (data were pre-copied to GPU device; output remains on GPU)")
    # out_device.copy_to_host() # what you would run if you needed to bring this back to the CPU non-GPU work

    # format runtime data as output table
    d = {'device_used': ['CPU', 'GPU', 'GPU', 'GPU'],
        'input_precopied_to_gpu': [np.nan, False, True, True],
        'output_copied_from_gpu': [np.nan, True, True, False],
        'seconds_required': [cpu_time, gpu_time0, gpu_time1, gpu_time2]}
    df = pd.DataFrame(d)
    print("")
    print(df)
    print("")
    df.to_csv('numba-runtimes.csv', index=False)
    ```

This script runs through four cases of multiplying two large random matrices, each with dimensions (10<sup>9</sup>, 3). For each test case, 10 loops of the function are executed, and the time required reflects the time it takes for all 10 loops. `Test Case 1` is the CPU speed baseline to which we will compare our various GPU runtimes. Matrix multiplication using pure `numpy.multiply()`, which does not invoke the GPU and runs entirely on the CPU, requires approximately 39.86 seconds. The remaining Test Cases will all run on the GPU, but have dramatically different runtime performances depending on how frequently data are copied between the CPU and GPU devices. 

Note that to use the GPU in this script, we define the function `gpu_mult()`, which is vectorized with a `numba` decorator that also tells the device to operate on `float32` values, and defines `cuda` as the runtime target device. Following these instructions, `numba` JIT-compiles `gpu_mult()` into a CUDA kernel that can execute on a GPU.

!!! note
    In general, computing on numeric `float32` data performs substantially better compared to `float64` on GPUs.

In `Test Case 2`, we simply call the vectorized `gpu_mult()`, which actually has much slower performance (55.67 seconds) than the CPU test case! On the surface, this is counterintuitive (aren't GPUs supposed to be faster?!), however a deeper examination of the code explains why we observe this. Becuase we initialized `matrix_a` and `matrix_b` on the CPU (a normal use case), we have to copy each object to the GPU before they can be multiplied together. After `gpu_mult()` is executed, the output matrix is then copied back to the CPU. Without some extra effort on our part, `numba` will default to copying these data *before* and *after* the execution of `gpu_mult()`. By contrast, since everything is already on the CPU, `numpy` simply does not have to deal with this, so it runs faster.

`Test Case 3` reflects a situation in which we pre-copy `matrix_a` and `matrix_b` to GPU memory before executing `gpu_mult()`. We do this with the `numba` command `cuda.to_device()`, which allows the input data to only be copied between devices once, even though we perform 10 executions on them. With this simple change, we observe a dramatic decrease in runtime to only ~0.8 seconds. However, because we do not specify an 'output device' in our vectorized `gpu_mult()`, the output matrix is actually copied back to CPU memory after each execution. However, with a bit of extra code, we can keep the output on the GPU, which would make sense if we wanted to do more work on it there later in the script. 

To that end, `Test Case 4` squeezes all possible performance out of `gpu_mult()` by both pre-copying the input data to the GPU *and* leaving the output matrix on the same device. The blazing-fast runtime of this test case (only about a millisecond) measures the GPU computation itself, without the clutter of copying data between devices. When compared to the runtime of `Test Case 1`, which also does not include any kind of data copying step, `Test Case 4` shows a roughly *24,000X* speedup in multiplying two matrices of this size, allowing us to appreciate the true power of the GPU.

This table summarizes the results and reflect runtimes of ten function loops on a node from Kestrel's `gpu-h100` partition.

| Test Case | Input pre-copied to GPU | Output copied from GPU | Time required (seconds) |
| :--       | :-- | :-- | :--      |
| 1 (CPU)   | NaN | NaN | 39.860077|
| 2 (GPU)   |False|True | 55.670377|
| 3 (GPU)   |True |True | 0.797287 |
| 4 (GPU)   |True |False| 0.001643 |

To be sure, there are many more considerations to have when developing a highly performant custom CUDA kernel, and there are many other packages that can do similar things. However, minimizing the amount of data copied between the CPU and GPU devices is a relatively easy approach that introductory GPU programmers can implement in their kernels to see immediate paybacks in performance regardless of computing platform. 

## Extra resources

* ["Preparing your Python code for Perlmutter's GPUs"](https://docs.nersc.gov/development/languages/python/perlmutter-prep/) (NERSC)
* [Another `numba` example](https://github.com/NREL/HPC/blob/master/languages/python/numba/numba_demo.ipynb) (NREL)
* ["Just-in-time (JIT) compilation"](https://docs.nvidia.com/cuda/cutensor/latest/just_in_time_compilation.html) (NVIDIA)
* [Numba documentation](https://numba.readthedocs.io/en/stable/) (Numba)
