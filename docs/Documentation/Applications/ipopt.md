---
title: IPOPT
postdate: October 27, 2020
layout: default
author: Jonathan Maack
description: How to install and use IPOPT with different programming languages
parent: Libraries
grand_parent: Development
---

# IPOPT

*IPOPT (Interior Point OPTimizer, pronounced "Eye-Pea-Opt") is an open-source non-linear optimizer using the interior point method.*

IPOPT is commonly used in solving power flow, e.g., AC Optimal Power Flow, and controls problems. Please refer to their [project website](https://github.com/coin-or/Ipopt) for the source code. The documentation can be found [here](https://coin-or.github.io/Ipopt/index.html).

!!! info
    IPOPT with HSL linear solvers is available as a module on Kestrel. Please see [IDAES Solvers](./idaes_solvers.md) for additional details.


## Installation from source

!!! warning
    The following installation instructions are for NREL's older cluster Eagle. We recommend users use the [IPOPT modules available on Kestrel](./idaes_solvers.md). Users will have to tweak the instructions below to build IPOPT from source on Kestrel.  

The default installation instructions can be found in the [IPOPT documentation here](https://coin-or.github.io/Ipopt/INSTALL.html). The remainder of the page describes what has worked for NREL HPC users.

We will use COIN-OR's [coinbrew](https://github.com/coin-or/coinbrew) repo to build IPOPT along with the dependencies ASL, HSL and Mumps libraries. 

!!! note
    Follow the [instructions to setup the environment](../Development/Libraries/hsl.md#hpc) for HSL before proceeding with the steps below.

1. `module load gcc/8.4.0 mkl`
2. Clone (or download) the [coinbrew](https://github.com/coin-or/coinbrew) repo. If you download the repo you may have to change the permissions on the `coinbrew` *script* before using it: `chmod u+x coinbrew/coinbrew`
3. `cd` into the directory
4. `./coinbrew fetch Ipopt@stable/3.13`
    * This fetches the branch `stable/3.13` of the IPOPT repository as well as the dependencies COIN-OR repositories `ThirdParty-ASL`, `ThirdParty-HSL` and `ThirdParty-Mumps` (other versions of IPOPT can also be downloaded in this manner)
5. `cd ThirdParty/HSL`
6. Copy the HSL source code to the current directory and unpack it
7. Create a link called `coinhsl` that points to the HSL source code (or rename the directory)
8. Go back to coinbrew root directory: `cd ../..`
9. Configure and build everything:

    ```bash
    ./coinbrew build Ipopt --disable-java --prefix="${MYAPPS}" --with-metis-cflags="-I${MYINC}" --with-metis-lflags="-L${MYLIB} -lmetis" --with-lapack-lflags="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl" --with-lapack-cflags="-m64 -I${MKLROOT}/include" ADD_CFLAGS="-march=skylake-avx512" ADD_FCFLAGS="-march=skylake-avx512" ADD_FFLAGS="-march=skylake-avx512"
    ```

    * `build Ipopt` tells `coinbrew` to configure and build IPOPT and its dependencies
    * `--disable-java` says to build IPOPT without the java interface
    * `--prefix` says to install the library in "${MYAPPS}"
    * `--with-metis-cflags` gives the compiler the location of the metis header "metis.h"
    * `--with-metis-lflags` gives the linker the location and name of the metis library
    * `--with-lapack-lflags` gives the location of LAPACK and BLAS libraries as well as the needed linker lines.  Here we are using Intel's single dynamic library interface (google "mkl single dynamic library" for more details on this).
    * `ADD_CFLAGS`, `ADD_FCFLAGS` and `ADD_FFLAGS` say to use those extra flags when compiling C and fortran code, respectively.

!!! tip
    When linking with MKL libraries, Intel's [link line advisor](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html) is extremely helpful.

!!! note 
    When compiling Julia with MKL libraries, the single dynamic library interface is used to link against.  This is why we are also using that linking method.  Using a different linking method will cause unusual behaviors when using IPOPT with Julia (e.g. through JuMP).

## Usage

### Using Custom IPOPT with JuMP

!!! note
    When running your custom IPOPT build on Kestrel, you will need to do two things:

    1. Load the same MKL module you compiled against:
        ```bash
        module load mkl
        ```
    2. Add the directory containing IPOPT and HSL libraries to your LD_LIBRARY_PATH
        ```bash
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MYLIB}
        ```


To use our custom installation of IPOPT with `Ipopt.jl`, we do the following:

1. Open the Julia REPL and activate an environment that has IPOPT installed
2. Tell Julia and `Ipopt.jl` the location of our IPOPT library and executable
    ```julia
    ENV["JULIA_IPOPT_LIBRARY_PATH"] = ENV["MYLIB"]
    ENV["JULIA_IPOPT_EXECUTABLE_PATH"] = ENV["MYBIN"]
    ```
3. Rebuild `Ipopt.jl` with the above environment variables set to pick up the new library and executable
    ```julia
    using Pkg; Pkg.build("Ipopt");
    ```
4. Print the path `Ipopt.jl` has stored for `libipopt.so`. This should be the location of your compiled version.
    ```julia
    using Ipopt; println(Ipopt.libipopt_path)
    ```