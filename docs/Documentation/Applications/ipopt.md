---
title: IPOPT
postdate: October 27, 2020
layout: default
author: Jonathan Maack, Kinshuk Panda
description: How to install and use IPOPT with different programming languages
parent: Libraries
grand_parent: Development
---

# IPOPT

*IPOPT (Interior Point OPTimizer, pronounced "Eye-Pea-Opt") is an open-source non-linear optimizer using the interior point method.*

IPOPT is commonly used in solving power flow, e.g., AC Optimal Power Flow, and controls problems. Please refer to their [project website](https://github.com/coin-or/Ipopt) for the source code. The documentation can be found [here](https://coin-or.github.io/Ipopt/index.html).

!!! note
    IPOPT with HSL linear solvers is available as a module on Kestrel. Please see [IDAES Solvers](./idaes_solvers.md) for additional details. We recommend using the system module for ease-of-use and only build if the module does not meet your needs.


## Installation from source

!!! info
    We advise building all applications on a compute node using an interactive session. Please see [Running Interactive Jobs](../Slurm/interactive_jobs.md#running-interactive-jobs) for additional details.

### Optional Pre-requisites

We will build IPOPT using all prerequisites mentioned below. Users may pick and
choose depending on their needs.

#### Metis

It is highly recommended to install [Metis](https://github.com/KarypisLab/METIS.git)
- Serial Graph Partitioning and Fill-reducing Matrix Ordering software to 
improve the performance of linear solvers such as MUMPS and HSL.

!!! warning
    Using HSL linear solvers requires installing Metis. Metis is optional for MUMPS.

We will install Metis using Anaconda. However, it can also be installed from source.
To install using Anaconda, we will create a clean environment with only Metis.
The conda environment is being constructed within a directory in the `hpcapps` project on 
Kestrel. Users can create a conda environment in any place of their choice.

```bash
module load conda
conda create -p /projects/hpcapps/kpanda/conda-envs/metis python
conda activate /projects/hpcapps/kpanda/conda-envs/metis
conda install conda-forge::metis
```

#### Coinbrew

[Coinbrew](https://github.com/coin-or/coinbrew) is a package manager to install
COIN-OR tools. It makes installing IPOPT and its dependencies easier. However, it 
is not necessary to the installation if one clones the repositories individually.
A user can download `coinbrew` by running the following command

```bash
wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
```

#### Intel oneAPI MKL

Intel oneAPI MKL provides BLAS and LAPACK libraries for efficient linear algebra.
Additionally, it also provides access to oneMKL PARDISO linear solver that is 
compatible with IPOPT.

!!! note
    oneMKL PARDISO is not available on Kestrel GPU nodes since they consist of AMD processors.

#### HSL

[HSL (Harwell Subroutine Library)](http://hsl.rl.ac.uk/ipopt) is a set of linear solvers 
that can greatly accelerate the speed of optimization over other linear solvers, e.g., MUMPS.
HSL can be installed separately as well using [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL).
Please see [here](../Development/Libraries/hsl.md) for installation on Kestrel.

### Installation

In this demonstration, we will install Ipopt within
`/projects/msoc/kpanda/apps/Ipopt/install`. However, one is free to set their
install directory as they wish. Starting with the base working directory
`/projects/msoc/kpanda/apps/` we will do the following

```bash
cd /projects/msoc/kpanda/apps/ # go into the base working directory
wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew # install coinbrew
coinbrew fetch Ipopt # Fetch Ipopt and its dependencies
```

This will download 2 additional directories `Ipopt` and `ThirdParty`.
`ThirdParty`, furthermore, contains 3 subdirectories `ASL`, `HSL`, and `Mumps`.
The source code of all but `HSL` will be downloaded. 

Next, we will create our install directories and subdirectories

```bash
mkdir -p /projects/msoc/kpanda/apps/Ipopt/install # create the install directory
cd /projects/msoc/kpanda/apps/Ipopt/install # enter the directory
mkdir bin lib include # create some subdirectories
```

We then add symbolic links to Metis in the install directory. 

!!! note
    If `libmetis.so` is in your `LD_LIBRARY_PATH` you do not need to do this step.

```bash
cd /projects/msoc/kpanda/apps/Ipopt/install/lib
ln -s /projects/hpcapps/kpanda/conda-envs/metis/lib/libmetis.so libmetis.so
cd ../include
ln -s /projects/hpcapps/kpanda/conda-envs/metis/include/metis.h metis.h
cd /projects/msoc/kpanda/apps/ # go back base directory
```

This has two advantages.
First, we don't need to add `/projects/hpcapps/kpanda/conda-envs/metis/lib/` to
the `LD_LIBRARY_PATH`.  The second advantage is that anaconda puts all the 
environments libraries and include files in the same directories with
`libmetis.so` and `metis.h`.  Many of these libraries overlap with those used
by HSL, Mumps and IPOPT but are not necessarily the same versions.  Loading a
different version of a library than those compiled against can cause unexpected behavior.

Next, we will load additional modules. If users require oneMKL PARDISO or would
like to leverage intel performance optimization, run the following commands

```bash
module load intel-oneapi-mkl
```

Alternatively, users can load the open source Netlib LAPACK using the command

```bash
module load netlib-lapack # Please ensure you do not have intel-oneapi-mkl loaded
```

We will now copy the HSL source code tarball into 
`/projects/msoc/kpanda/apps/ThirdParty/HSL/`, unpack it, and rename or (create a 
symbolic link to the unpacked directory) as `coinhsl`. 

We are now ready to install IPOPT and its dependencies. We will use the default
compilers available in the Kestrel programming environment. Going back to the base 
directory, we will run the following commands

```bash
cd /projects/msoc/kpanda/apps/ # go back base directory
./coinbrew build Ipopt --disable-java \
--prefix=/kfs2/projects/msoc/kpanda/apps/Ipopt/install \
--with-metis \
--with-metis-cflags=-I/projects/hpcapps/kpanda/conda-envs/metis/include \
--with-metis-lflags="-L/projects/hpcapps/kpanda/conda-envs/metis/lib -lmetis" \
--parallel-jobs 4 \
--verbosity 4 \
--reconfigure
```

## Usage

The installed Ipopt is now ready to be used. We need to update our `PATH` AND 
`LD_LIBRARY_PATH` environment variables. In our demonstrations this will be

```bash
export PATH=/projects/msoc/kpanda/apps/Ipopt/install/bin:${PATH}
export LD_LIBRARY_PATH=/projects/msoc/kpanda/apps/Ipopt/install/lib:${LD_LIBRARY_PATH}
```

!!! note
    Do not forget to load `intel-oneapi-mkl` or `netlib-lapack` before running IPOPT else your runs will fail.

### Using Custom IPOPT with JuMP

To use our custom installation of IPOPT with `Ipopt.jl`, we do the following:

1. Open the Julia REPL and activate an environment that has IPOPT installed
2. Tell Julia and `Ipopt.jl` the location of our IPOPT library and executable
    ```julia
    ENV["JULIA_IPOPT_LIBRARY_PATH"] = ENV["/projects/msoc/kpanda/apps/Ipopt/install/lib"]
    ENV["JULIA_IPOPT_EXECUTABLE_PATH"] = ENV["/projects/msoc/kpanda/apps/Ipopt/install/bin"]
    ```
3. Rebuild `Ipopt.jl` with the above environment variables set to pick up the new library and executable
    ```julia
    using Pkg; Pkg.build("Ipopt");
    ```
4. Print the path `Ipopt.jl` has stored for `libipopt.so`. This should be the location of your compiled version.
    ```julia
    using Ipopt; println(Ipopt.libipopt_path)
    ```

!!! info
    The IPOPT build that comes with `Ipopt.jl` seems to expect the HSL library to have the name `libhsl.so`. The repo ThirdParty-HSL builds the library `libcoinhsl.so`.  The simplest fix is to do the following:

    ```bash
    cd /projects/msoc/kpanda/apps/Ipopt/install/lib # install directory
    # Create a symbolic link called libhsl.so
    ln -s libcoinhsl.so libhsl.so
    ```

The following Julia code is useful for testing the HSL linear solvers are working

```julia
using JuMP, IPOPT

m = JuMP.Model(()->IPOPT.Optimizer(linear_solver="ma97"))
@variable(m, x)
@objective(m, Min, x^2)
JuMP.optimize!(m)
```