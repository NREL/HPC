---
title: HSL
postdate: October 27, 2020
layout: default
author: Jonathan Maack, Kinshuk Panda
description: How to install and use HSL with with IPOPT
parent: Libraries
grand_parent: Development
---

# HSL for IPOPT

*HSL (Harwell Subroutine Library) for IPOPT are a set of linear solvers that can greatly accelerate the speed of the optimization over the default MUMPS solver.*

## Installation

Go to the [HSL](http://www.hsl.rl.ac.uk/ipopt/) site and follow the instructions to request the source code for all the available solvers. Note that the solver MA27 is free to obtain, but MA27 is a serial solver. Other solvers will require a license. Please request a license that applies to your use case.

!!! info
    If you are building IPOPT along with HSL, please follow the instructions [here](../../Applications/ipopt.md#installation-from-source).

We need to be careful regarding the selection of linear algebra libraries when installing HSL.
The default version of IPOPT distributed with `Ipopt.jl` on Linux links to the OpenBLAS library. This causes issues when linking the HSL library to the Intel oneAPI MKL libraries.  For this reason, to use HSL linear solvers with IPOPT on Kestrel, either we must compile IPOPT from scratch or compile HSL with OpenBLAS and NetLib LAPACK instead of Intel oneAPI MKL. We demonstrated IPOPT + HSL installation with Intel oneAPI MKL [here](../../Applications/ipopt.md#installation-from-source).

The following provides detailed instructions for building HSL using OpenBLAS and Netlib LAPACK ON HPC.

#### Pre-requisites

##### Metis

Metis is a serial graph partitioning and fill-reducing matrix ordering software that helps the HSL solvers perform better. Therefore, it is recommended that you also install or build the Metis library.  If you do want to install Metis, it must be done before compiling the HSL library.

The easiest way to install Metis is to use anaconda:

!!! warning
    Using HSL linear solvers requires installing Metis. Metis is optional for MUMPS.

We will install Metis using Anaconda, however, it can also be installed from source.
To install using Anaconda, we will create a clean environment with nothing but Metis.
The conda environment is being constructed within a directory in `hpcapps` project on 
Kestrel. 

```bash
module load conda
conda create -p /projects/hpcapps/kpanda/conda-envs/metis python
conda activate /projects/hpcapps/kpanda/conda-envs/metis
conda install conda-forge::metis
```

!!! info 
    `module load conda` loads the default anaconda module. You may use a different conda module based on your needs.

!!! note
    Anaconda packages sometimes have issues when they come from different channels.  We tend to pull everything from `conda-forge` hence the channel choice above.

The Metis library and header files are placed in `/projects/hpcapps/kpanda/conda-envs/metis/lib/` and `/projects/hpcapps/kpanda/conda-envs/metis/include/`, respectively

##### Compilers

We will be using the GNU compiler suite (`gcc` and `gfortran`).  These can be accessed on the cluster by loading the appropriate module.  This should work with any version of the GNU compilers. We use the default `gcc` and `gfortran` that are available on the CPU compute nodes.


#### Setting up the Environment 

We will install HSL in `/kfs2/projects/msoc/kpanda/apps/Ipopt/install` for this demonstration. This can be set to whatever location you wish to install.
Let's create the requisite installation directories

```bash
mkdir -p /kfs2/projects/msoc/kpanda/apps/Ipopt/install
cd /kfs2/projects/msoc/kpanda/apps/Ipopt/install
mkdir lib include
cd ..
```

We will make use of the following environment variables.

```bash
# Location of metis.h
export METIS_HEADER=/projects/hpcapps/kpanda/conda-envs/metis/include
# Location of metis library
export METIS_LIBRARY=/projects/hpcapps/kpanda/conda-envs/metis/lib

# Directory for keeping source code and build products
export MYAPPS=/kfs2/projects/msoc/kpanda/apps/Ipopt/install
# Location of header files
export MYINC=${MYAPPS}/include
# Location of static and dynamic libraries
export MYLIB=${MYAPPS}/lib
```

These can be added to the `.bash_profile` file (or equivalent for other shells).  Remember after adding these to source `.bash_profile` (or equivalent) or to open a new terminal and do all building there.
Alternatively, to make the Metis header and dynamic library easily accessible to the HSL, MUMPS and IPOPT libraries, we will put symbolic links in the `${MYINC}` and `${MYLIB}` directories.  Do this by doing the following:

```bash
cd ${MYINC}
ln -s ${METIS_HEADER}/metis.h metis.h
cd ${MYLIB}
ln -s ${METIS_LIBRARY}/libmetis.so libmetis.so
```

This has two advantages.
First, we don't need to add `/projects/hpcapps/kpanda/conda-envs/metis/lib/` to
the `LD_LIBRARY_PATH`.  The second advantage is that anaconda puts all the 
environments libraries and include files in the same directories with
`libmetis.so` and `metis.h`.  Many of these libraries overlap with those used
by HSL, Mumps and IPOPT but are not necessarily the same versions.  Loading a
different version of a library than those compiled against can cause unexpected behavior.

#### Configure and Install

We will clone `ThirdParty-HSL` and configure and install HSL in a working directory

```bash
git clone git@github.com:coin-or-tools/ThirdParty-HSL.git
```

Copy the HSL source code tarball into `/projects/msoc/kpanda/apps/ThirdParty/HSL/`, 
unpack it, and rename or (create a symbolic link to the unpacked directory) as `coinhsl`.

Run the following commands to configure

```bash
cd ThirdParty-HSL
module load netlib-lapack
./configure --prefix=${MYAPPS} \
--with-metis \
--with-metis-cflags=-I${METIS_HEADER} \
--with-metis-lflags="-L${METIS_LIBRARY} -lmetis"
make && make install
```

This should install the HSL libraries in `${MYAPPS}`. Finally, add `MYLIB` to your `LD_LIBRARY_PATH`. You can append the following line to your `.bash_profile` to make it permanent or call it every time you need to run IPOPT with HSL solvers.

```bash
export export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MYAPPS}/lib
```

## Usage

IPOPT has a feature called the linear solver loader (read about it [here](https://coin-or.github.io/Ipopt/INSTALL.html#LINEARSOLVERLOADER)). This allows for loading linear solvers from a dynamic library at run time.  We will use this feature to use the HSL solvers.

The only thing you have to do is to make the HSL dynamic library findable.  This is done by adding the directory containing the HSL library to the environment variable `LD_LIBRARY_PATH`. To use the new linear solvers just use the `linear_solver="<solver>"` argument to `IPOPT.Optimizer`.

!!! info
    The IPOPT build that comes with `Ipopt.jl` seems to expect the HSL library to have the name `libhsl.so`. The repo ThirdParty-HSL builds the library `libcoinhsl.so`.  The simplest fix is to do the following:

    ```bash
    cd ${MYLIB}
    # Create a symbolic link called libhsl.dylib
    ln -s libcoinhsl.dylib libhsl.dylib
    ```

Alternatively, users can follow the instructions mentioned [here for Julia JuMP](https://github.com/jump-dev/Ipopt.jl?tab=readme-ov-file#hsl)