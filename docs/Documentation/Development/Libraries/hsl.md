---
title: HSL
postdate: October 27, 2020
layout: default
author: Jonathan Maack
description: How to install and use HSL with with IPOPT
parent: Libraries
grand_parent: Development
---

# HSL for Ipopt

*HSL (Harwell Subroutine Library) for Ipopt are a set of linear solvers that can greatly accelerate the speed of the optimization over the default MUMPS solver.*

## Installation

Go to the [HSL for Ipopt](http://www.hsl.rl.ac.uk/ipopt/) site and follow the instructions to request the source code for all the available solvers. Note that the solver MA27 is free to obtain, but MA27 is a serial solver. Other solvers will require a license. Please request a license that applies to your use case.

### Eagle

The default version of Ipopt distributed with `Ipopt.jl` on Linux links to the OpenBLAS library.  This causes issues when linking the HSL library to the MKL libraries.  For this reason, to use HSL linear solvers with Ipopt on Eagle, either we must compile Ipopt from scratch or compile HSL with OpenBLAS instead of MKL.  For performance reasons, we have elected to compile Ipopt from scratch so that we can use the MKL libraries.

The following provides detailed instructions for compiling Ipopt with HSL and Mumps on Eagle.

#### Pre-requisites

##### Metis

Metis helps the HSL solvers perform better.  Therefore, it is recommended that you also install or build the Metis library.  If you do want to install Metis, it must be done before compiling the HSL library.

On Eagle, the easiest way to install Metis is to use anaconda:

```bash
module load conda
conda create -n <conda_environment>
conda activate <conda_environment>
conda install -c conda-forge metis
```
!!! info 
    `module load conda` loads the default anaconda module. You may use a different conda module based on your needs.

!!! note
    Anaconda packages sometimes have issues when they come from different channels.  We tend to pull everything from `conda-forge` hence the channel choice above.

##### pkg-config

[pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/) is a helper tool for specifying compiler options while building your code. It is available by default on Eagle.

##### Compilers

We will be using the GNU compiler suite (`gcc` and `gfortran`).  These can be accessed on Eagle by loading the appropriate module.  This should work with any version of the GNU compilers. We use version 8.4.0 here.  These can be loaded by typing `module load gcc/8.4.0`.


#### Setting up the Environment 

We will make use of the following environment variables.

```bash
# Location of metis.h
export METIS_HEADER=${HOME}/.conda-envs/<conda_environment>/include
# Location of metis library
export METIS_LIBRARY=${HOME}/.conda-envs/<conda_environment>/lib

# Directory for keeping source code and build products
export MYAPPS=${HOME}/apps
# Location of header files
export MYINC=${MYAPPS}/include
# Location of static and dynamic libraries
export MYLIB=${MYAPPS}/lib
```

These can be added to the .bash_profile file (or equivalent for other shells).  Remember after adding these to source `.bash_profile` (or equivalent) or to open a new terminal and do all building there.
To make the Metis header and dynamic library easily accessible to the HSL, MUMPS and Ipopt libraries, we will put symbolic links in the `${MYINC}` and `${MYLIB}` directories.  Do this by doing the following:

```bash
cd ${MYINC}
ln -s ${METIS_HEADER}/metis.h
cd ${MYLIB}
ln -s ${METIS_LIBRARY}/libmetis.so
```

This has a couple of advantages.  First, the `coinbrew` build will automatically add the `${MYLIB}` directory to the rpath of all constructed libraries and executables.  This means that we don't need to add `${MYLIB}` to the LD_LIBRARY_PATH.  The other advantage is that anaconda puts all the environments libraries and include files in the same directories with `libmetis.so` and `metis.h`.  Many of these libraries overlap with those used by HSL, Mumps and Ipopt but are not necessarily the same versions.  Loading a different version of a library than those compiled against can cause unexpected behavior.

#### Configure and Install

Follow the [Ipopt installation instructions here](ipopt.md#eagle) to finish the installation of HSL solvers on Eagle.

### MacOS

The following installation has been tested on Apple's M1 ARM based processors.

#### Pre-requisites

We will use [Homebrew](https://brew.sh) and [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL) to install HSL libraries (and IPOPT). As per the [default IPOPT installation instructions](https://coin-or.github.io/Ipopt/INSTALL.html), we will rely on GNU compilers for the installation. Run the following commands

```bash
# Update homebrew and download packages
brew update
brew install bash gcc metis pkg-config
# Create a directory of your choice to install HSL
mkdir -p {$HOME}/UserApps/IPOPT/HSL/hsl_install
cd {$HOME}/UserApps
# Clone ThirdParty-HSL
git clone git@github.com:coin-or-tools/ThirdParty-HSL.git
cd ThirdParty-HSL
# Place the HSL source code here
cp -r ${HSL_SOURCE_CODE_LOCATION} coinhsl
```

#### Setting up the Environment 

Assuming that you allow Homebrew to install to its default locations, we will declare the following environment variables

```bash
# Location of metis.h
export METIS_HEADER=/opt/homebrew/Cellar/metis/5.1.0/include
# Location of metis library
export METIS_LIBRARY=/opt/homebrew/Cellar/metis/5.1.0/lib
# Directory for keeping source code and build products
mkdir -p {$HOME}/UserApps/IPOPT/HSL/hsl_install
export MYAPPS={$HOME}/UserApps/IPOPT/HSL/hsl_install
# Location of static and dynamic libraries
mkdir -p ${MYAPPS}/lib
export MYLIB=${MYAPPS}/lib
```

#### Configure and Install

Go to the requisite directory and run the following commands

```bash
cd {$HOME}/UserApps/ThirdParty-HSL/
mkdir build && cd build
../configure F77=gfortran-12 FC=gfortran-12 CC=gcc-12 --prefix="${MYAPPS}" \
--with-metis --with-metis-lflags="-L${METIS_LIBRARY} -lmetis" \
--with-metis-cflags="-I${METIS_HEADER}"
make && make install
```

This should install the HSL libraries in `${MYAPPS}`. Finally add `MYLIB` to your `DYLD_LIBRARY_PATH`. You can append the following line to your `.bash_profile` to make it permanent or call it every time you need to run Ipopt with HSL solvers.

```bash
export export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${MYLIB}/UserApps/IPOPT/HSL/hsl_install/lib
```

## Usage

Ipopt has a feature called the linear solver loader (read about it [here](https://coin-or.github.io/Ipopt/INSTALL.html#LINEARSOLVERLOADER)). This allows for loading linear solvers from a dynamic library at run time.  We will use this feature to use the HSL solvers.

The only thing you have to do is to make the HSL dynamic library findable.  This is done by adding the directory containing the HSL library to the environment variable `DYLD_LIBRARY_PATH` in MacOS and `LD_LIBRARY_PATH` on Linux-based systems. See above for MacOS and [here](ipopt.md#using-custom-ipopt-with-jump) for NREL systems. To use the new linear solvers just use the `linear_solver="<solver>"` argument to `Ipopt.Optimizer`.

!!! info
    The Ipopt build that comes with `Ipopt.jl` seems to expect the HSL library to have the name `libhsl.dylib` on MacOS. The repo ThirdParty-HSL builds the library `libcoinhsl.dylib`.  The simplest fix is to do the following:

    ```bash
    cd ${MYLIB}
    # Create a symbolic link called libhsl.dylib
    ln -s libcoinhsl.dylib libhsl.dylib
    ```

The following Julia code is useful for testing the HSL linear solvers are working

```julia
using JuMP, Ipopt

m = JuMP.Model(()->Ipopt.Optimizer(linear_solver="ma97"))
@variable(m, x)
@objective(m, Min, x^2)
JuMP.optimize!(m)
```