# What is Pixi?

[Pixi](https://pixi.prefix.dev/latest/) is a package management tool that attempts to unify the workflows of existing package managers such as [conda](conda.md) or [pip](https://pip.pypa.io/en/stable/) for a smoother and more robust user experience. Pixi uses the [rattler](https://github.com/prefix-dev/rattler-build/tree/main) library, a high-performance implementation of core conda functionalities (such as dependency tree solving) written in [Rust](https://rust-lang.org), leading to Pixi being significantly faster than "traditional" or "pure" conda. Pixi facilitates the management of project-specific environments which may contain a mix of packages from Python and other languages. Pixi handles both *environment creation* and *package installation*, replacing the need to use conda for the former and pip for the latter. 

# Using Pixi on Kestrel

Pixi is available as a module on both the CPU and GPU nodes on [Kestrel](../../Systems/Kestrel/index.md):

```
$ ml help pixi
------------------ Module Specific Help for "pixi/0.65.0" ------------------
Name   : Pixi
Version: 0.65.0 (built 27 February 2026)
Source : https://github.com/prefix-dev/pixi
Docs   : https://pixi.prefix.dev

Pixi is a cross-platform, multi-language package manager and workflow tool 
built on the foundation of the conda ecosystem. It provides developers with 
an exceptional experience similar to popular package managers like cargo or 
npm, but for any language.
```

Pixi is mainly designed to create environments for a specific project/working directory. Two minimal examples (one for CPU nodes, and another for GPU nodes) of creating a Pixi environment and running a script from each are provided below. Please consult the [Pixi documentation](https://pixi.prefix.dev) for more information on how to get the most from Pixi.

## Minimal environment example on Kestrel - CPU

The following scripts represent a minimal example of using the Pixi module to 1. create a simple Pixi environment containing the `numpy` package (named `numpy-workspace`) and then 2. execute a Python script that performs a matrix multiplication 10 times (`numpy-test.py`). **Be sure to run this on a CPU node:**

??? "Example: Using Pixi to create an environment and execute `numpy-test.py`"
    Ensure that `numpy-test.py` (found in the next drop-down menu) exists one directory above `numpy-workspace` for this example.

    ```bash
    #!/bin/bash
    # Load Pixi module
    ml pixi
    # Initialize Pixi environment
    pixi init numpy-workspace
    # Note that we navigate to the Pixi environment folder to add packages and eventually execute the Python script
    cd numpy-workspace
    # Add numpy and Python as dependencies
    pixi add numpy python=3.11
    # The Python script we wish to execute is found one directory above 'numpy-workspace'
    pixi run python ../numpy-test.py

    # Optional - cleanup numpy-workspace and PIXI_CACHE_DIR
    #echo "Removing numpy-workspace..."
    #cd .. && rm -rf numpy-workspace
    #echo "Removing PIXI_CACHE_DIR..."
    #rm -rf $PIXI_CACHE_DIR
    ```

??? "`numpy-test.py`"
    ```python
    import numpy as np
    from time import time
    from time import sleep
    import os

    print(f"Running Python script using the Pixi environment '{os.getcwd()}'")

    # create random arrays as input data
    asize = pow(10, 6)
    array_a = np.float32(np.random.rand(asize))
    array_b = np.float32(np.random.rand(asize))
    array_c = np.float32(np.random.rand(asize))

    matrix_a = ([array_a], [array_b], [array_c])
    matrix_b = ([array_c], [array_b], [array_a])

    # numpy - CPU
    nloops = 10
    t0 = time()
    for i in np.arange(nloops):
        np.multiply(matrix_a, matrix_b)
    cpu_time = time()-t0
    print("numpy on CPU required", round(cpu_time, 2), "seconds for multiplying two matrices each of size", 3*asize, "a total number of", nloops, "times.")
    ```

Note that the Python script intended to be run by this environment (`numpy-test.py`) is executed from the `numpy-workspace` folder via `pixi run python ../numpy-test.py`. After the environment is created and you navigate to the environment folder, providing the call to Python with the `pixi run ...` prefix will use the version of Python and its associated packages from the `numpy-workspace` environment. 

## Minimal environment example on Kestrel - GPU

The following examples represent a minimal example of using the Pixi module to 1. create a simple Pixi environment containing a GPU-enabled version of `torch` (named `cuda-workspace`) and then 2. run a simple Python command that verifies whether this environment's `torch` can see a GPU device. **Be sure to run this on a GPU node.**

### RHEL9

This example uses `cuda/13.2` on [Kestrel nodes upgraded to RHEL9](../../../RHEL9_upgrade/index.md).

??? "Example: Using Pixi to create a GPU-enabled PyTorch environment on Kestrel (RHEL9)"

    ```bash
    #!/bin/bash
    # Load Pixi module
    ml pixi
    # Initialize Pixi environment
    pixi init cuda-workspace
    # Note that we navigate to the Pixi environment folder to add packages and eventually execute the Python script
    cd cuda-workspace

    # Manually create pixi.toml
    cat <<EOF > pixi.toml
    [workspace]
    channels = ["https://prefix.dev/conda-forge"]
    name = "pytorch-conda-forge"
    platforms = [ {platform = "linux-64", cuda = "13.2"} ]

    [dependencies]
    pytorch-gpu = "*"
    cowpy = "*"
    python = "3.11.*"
    EOF
    pixi run cowpy "MUUUUUUDA!"
    pixi run python -c "import torch; print('Can pixi find a GPU? -->', torch.cuda.is_available(), '\n', 'Using CUDA version:', torch.version.cuda)"
    ```


### RHEL8

This example uses `cuda/12.4` on "regular" Kestrel GPU nodes that have not been [upgraded to RHEL9](../../../RHEL9_upgrade/index.md) yet. Note that the toml file syntax is different from the RHEL9 example above due to the difference in default `pixi` versions between RHEL8 and RHEL9.

??? "Example: Using Pixi to create a GPU-enabled PyTorch environment on Kestrel (RHEL8)"

    ```bash
    #!/bin/bash
    # Load Pixi module
    ml pixi
    # Initialize Pixi environment
    pixi init cuda-workspace
    # Note that we navigate to the Pixi environment folder to add packages and eventually execute the Python script
    cd cuda-workspace

    # Manually create pixi.toml
    cat <<EOF > pixi.toml
    [workspace]
    channels = ["https://prefix.dev/conda-forge"]
    name = "pytorch-conda-forge"
    platforms = ["linux-64",]

    [system-requirements]
    cuda = "12.4"

    [dependencies]
    pytorch-gpu = "*"
    cuda-version = ">=12.4"
    cowpy = "*"
    python = "3.11.*"
    EOF
    pixi run cowpy "MUUUUUUDA!"
    pixi run python -c "import torch; print('Can pixi find a GPU? -->', torch.cuda.is_available(), '\n', 'Using CUDA version:', torch.version.cuda)"
    ```

Note that in this example, we specify `cuda = "12.4"` under `[system-requirements]` in the `pixi.toml`. This will allow Pixi to install a GPU-enabled version of PyTorch; without this, Pixi would install a CPU-only version of `torch`. Additionally, when creating environments from a custom `pixi.toml`, note that anything under `[dependencies]` is functionally equivalent to `pixi add <package1> <package2> ... <packageN>` as written in the [CPU example above](#minimal-environment-example-on-kestrel---cpu). At the time of writing, the GPU drivers on Kestrel are compatible with `cuda/12.4+`, so we pin `cuda-version = ">=12.4"` as a dependency accordingly as an extra insurance that we pull a compatible version of PyTorch.

!!! warning "A note on performant, multi-node PyTorch on Kestrel's GPU nodes"
    Note that installing PyTorch with the aim for good communication performance across multiple GPU nodes on Kestrel requires special considerations that are not covered in this page. See [our dedicated documentation on the topic](../../Machine_Learning/index.md#installing-pytorch-on-kestrel-with-multi-node-and-gpu-support) for more information.

## Package caching location - Kestrel

On Kestrel, the Pixi modules are designed to cache downloaded packages to `/scratch/${USER}/.cache/rattler` by default. This saves storage space in `/home` or `/projects` folders, though this may be overridden by modifying and exporting the `PIXI_CACHE_DIR` environment variable after loading the module.

To save space in your personal `/scratch`, you may safely run `rm -rf /scratch/${USER}/.cache/rattler` at any time to clear this cache directory.

# Using Pixi on Gila

Pixi is available as a module for both `arm` and `x86` node architectures on [Gila](../../Systems/Gila/index.md). 

!!! Note
    Pixi environments built on Gila will only work for either `x86` or `arm` architectures depending on which node was used to create them. In other words, an environment created on an `x86`-based node hosting A100s would not be expected to work on an `arm`-based Grace Hopper node (and vice versa). This is generally true for all software managed on Gila. As such, always ensure you are using an environment that was created on the same node architecture you plan to run it on.

## Minimal environment example - Gila

Note that the [Kestrel CPU example above](#minimal-environment-example-on-kestrel---cpu) should work on any type of node on Gila without modification. Some modifications are required for adapting the [Kestrel GPU example](#minimal-environment-example-on-kestrel---gpu) to Gila's GPUs, namely adjusting the `cuda` version (as well as adding 'linux-aarch64' to `platforms` to enable the use of a Gracehopper node). Below we provide the same example used for Kestrel for Gila's A100 and Gracehopper GPU nodes.

??? "Example: Using Pixi to create a GPU-enabled PyTorch environment on any Gila GPU node"
    ```bash
    #!/bin/bash
    # Load Pixi module
    ml pixi
    # Initialize Pixi environment
    pixi init cuda-workspace
    # Note that we navigate to the Pixi environment folder to add packages and eventually execute the Python script
    cd cuda-workspace

    # Manually create pixi.toml
    cat <<EOF > pixi.toml
    [workspace]
    channels = ["https://prefix.dev/conda-forge"]
    name = "pytorch-conda-forge"
    # Note that we use the 'linux-64' platform for an 'x86' node like an A100 with Intel CPUs
    # and the 'linux-aarch64' platform for an 'arm' node like a Gracehopper.
    platforms = ["linux-64", "linux-aarch64"]

    [system-requirements]
    cuda = "13.2"

    [dependencies]
    pytorch-gpu = "*"
    cuda-version = ">=13.2"
    cowpy = "*"
    python = "3.11.*"
    EOF
    pixi run cowpy "MUUUUUUDA!"
    pixi run python -c "import torch; print('Can pixi find a GPU? -->', torch.cuda.is_available(), '\n', 'Using CUDA version:', torch.version.cuda)"
    ```

!!! Note
    It is **highly** recommended to explicitly set `platforms = ["linux-64", "linux-aarch64"]` in a Gila project's `pixi.toml`. This facilitates switching between `arm` and `x86` nodes with ease. Pixi will automatically recreate the appropriate environment for the given chip architecture regardless of which architecture was used to originally create the workspace.

## Package caching locations - Gila

On Gila, the Pixi modules cache downloaded packages to either `/scratch/${USER}/.cache/x86/rattler` or `/scratch/${USER}/.cache/arm/rattler` depending on the architecture of the node you are connected to. 

# Useful links

- [Managing Python Environments with Pixi-From Laptop to HPC](https://nrel-my.sharepoint.com/:v:/g/personal/chschwin_nrel_gov/IQAr0Zstq-bQSZ5nkfcd-eXDAT_GOjhkcPXWHa6S2XRympU) (NLR HPC Tutorial series - requires access to CSC Tutorials Teams channel)
- [Switching from Conda/Mamba to Pixi](https://pixi.prefix.dev/latest/switching_from/conda/) (external site)
- [PyTorch installation with Pixi](https://pixi.prefix.dev/latest/python/pytorch/) (external site)
- [Building custom packages with Pixi](https://pixi.prefix.dev/latest/build/getting_started/) (external site)