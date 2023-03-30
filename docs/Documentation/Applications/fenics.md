# Using FEniCS Software 

*Learn how to use FEniCS software on the Eagle and Swift system.*

FEniCS is a collection of open-source software components designed to enable the automated solution of differential equations by the finite element method.  For documentation, forums, and FAQs, see the [FEniCS website](https://fenicsproject.org/).

FEniCS is organized as a collection of interoperable components that together form the FEniCS Project. These components include the problem-solving environment *DOLFIN*, the form compiler *FFC*, the finite element tabulator *FIAT*, the just-in-time compiler *Instant*, the form language *UFL*, and a range of additional components.

FEniCS can be programmed both in C++ and Python, but Python programming is the simplest approach to exploring FEniCS and can give high performance.

Currently, FEniCS is supported through Anaconda. Users are required to install their own FEniCS environment with the following commands:

```
module purge
module load conda
conda create -n myfenics -c conda-forge fenics mshr matplotlib scipy jupyter 
```

These commands will create a new environment named `myfenics` which contains all necessary packages as well as some commonly-used packages for programming FEniCS simulations. By default, this Conda environment will be installed in the directory `/home/<username>/.conda-envs/myfenics`. It will take roughly 3 GB of storage. Please make sure you **have enough storage quota** in the **home** directory before installation by running the `du -hs ~` command (which will take a minute or two to complete).   

To run FEniCS, execute the following commands using an interactive bash shell or in a Slurm batch submission script with `#!/bin/bash` in the first line:

```
module purge
module load conda
source activate myfenics
```

For detailed instructions on how to use and write codes using FEniCS, see the [FEniCS Documentation page](https://fenicsproject.org/documentation/).
