# FEniCS/FEniCSx

**Documentation:** [FEniCS 2019.1.0](https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/), [FEniCSx](https://docs.fenicsproject.org/dolfinx/v0.6.0/python/)

*FEniCS is a collection of open-source software components designed to enable the automated solution of differential equations by the finite element method.*

!!! note
    There are two version of FEniCS. The original FEniCS ended in 2019 with version 2019.1.0 and development began on a complete refactor known as FEniCSx. FEniCS 2019.1.0 is still actively used and the main focus of this documentation. Since FEniCSx is in pre-release, HPC support is a work in progress.

## Getting Started

FEniCS is organized as a collection of interoperable components that together form the FEniCS Project. These components include the problem-solving environment *DOLFIN*, the form compiler *FFC*, the finite element tabulator *FIAT*, the just-in-time compiler *Instant*, the form language *UFL*, and a range of additional components.

FEniCS can be programmed both in C++ and Python, but Python programming is the simplest approach to exploring FEniCS and can give high performance.

Currently, FEniCS is supported through Anaconda. Users are required to build their own FEniCS environment with the following commands after loading the conda/anaconda module (see Example Job Scripts):

```
module load conda
conda create -n myfenics -c conda-forge fenics  matplotlib scipy jupyter 
```

The packages `matplotlib`, `scipy`, and `jupyter` are not required, but they are very handy to have. 

These commands will create a new environment named `myfenics` which contains all necessary packages as well as some commonly-used packages for programming FEniCS simulations. By default, this Conda environment will be installed in the directory `/home/<username>/.conda-envs/myfenics`. It will take roughly 3 GB of storage. Please make sure you **have enough storage quota** in the **home** directory before installation by running the `du -hs ~` command (which will take a minute or two to complete). 

FEniCSx can also be installed via conda using: 
```
conda create -n myfenics -c conda-forge fenics-dolfinx
``` 

### Example Job Scripts

??? example "Kestrel CPU"

    ```slurm
    #!/bin/bash

    # This test file is designed to run the Poisson demo on one node with a 4 cores

    #SBATCH --time=01:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=4
    #SBATCH --partition=standard
    #SBATCH --account=

    module purge
    module load conda

    # This is to prevent FEniCS from unnecessarily attempting to multi-thread
    export OMP_NUM_THREADS=1

    cd /scratch/USERNAME/poisson_demo/
    srun -n 4 python poisson_demo.py

    ```

??? example "Vermillion"

    ```slurm
    #!/bin/bash

    # This test file is designed to run the Poisson demo on one node with a 4 cores

    #SBATCH --time=01:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=4
    #SBATCH --partition=standard
    #SBATCH --account=

    module purge
    module load anaconda3

    # This is to prevent FEniCS from unnecessarily attempting to multi-thread
    export OMP_NUM_THREADS=1

    cd /scratch/USERNAME/poisson_demo/
    srun -n 4 python poisson_demo.py

    ```

??? example "Swift"

    ```slurm
    #!/bin/bash

    # This test file is designed to run the Poisson demo on one node with a 4 cores

    #SBATCH --time=01:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=4
    #SBATCH --partition=standard
    #SBATCH --account=

    module purge
    module load conda

    # This is to prevent FEniCS from unnecessarily attempting to multi-thread
    export OMP_NUM_THREADS=1

    cd /home/USERNAME/poisson_demo/
    srun -n 4 python poisson_demo.py

    ```

To run this script, first download the Poisson demo [ here ](https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/poisson/demo_poisson.py.html) and place it either in a folder titled "poisson_demo" in your scratch directory (home for Swift). Then save the script as "demo_script.sh" and submit it with `sbatch demo_script.sh`. This demo is only supported by FEniCS 2019.1.0 and not FEniCSx. 

## Supported Versions

| Kestrel | Vermillion | Swift |
|:-------:|:----------:|:-----:|
| 2019.1.0 | 2019.1.0 | 2019.1.0 |


[]: # (TODO: add spack build)