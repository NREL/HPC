---
layout: default
title: Julia
has_children: true
parent: Programming Languages
---

# Julia

*Julia is a dynamic programming language that offers high performance while being easy to learn and develop code in.*

This section contains demos (in the form of scripts and notebooks) and how-to guides for doing various things with Julia on NREL HPC environments.

## Available modules

| Eagle          | Swift              | Vermilion  | Kestrel |
|:--------------:|:------------------:|:----------:|:----------:|
|                |julia/1.6.2-ocsfign |||  
julia/1.7.2      |julia/1.7.2-gdp7a25 |julia/1.7.2 | julia/1.7.2 |
|                |                    |            | julia/1.8.5-generic-linux |
julia/1.9.0      |                    |            ||



!!! Bug "Julia 1.9.x does not work well on Sapphire Rapids"

    We advise against installing and using Julia 1.9.x on Kestrel as packages can fail to precompile and result in a segmentation fault. This is a known issue with Julia 1.9.x on Sapphire Rapids processors, possibly due to an LLVM issue. Julia 1.10 will be installed as a module once a stable release is available. Until then, please use Julia 1.7 or 1.8.



## Contents

1. [Installing Julia](julia_install.md)
2. [Tour of Julia](julia_tour.md)
3. [Parallel Computing in Julia](julia_parallel.md)
4. [Calling Python, C, and FORTRAN from Julia](julia_ccall_pycall.md)

## Demo Scripts and Notebooks

The following scripts and notebooks are available on the [`master` branch of NREL/HPC](https://github.com/NREL/HPC) to download and run,

* [Julia Tour](https://github.com/NREL/HPC/blob/master/languages/julia/julia-tutorial/source-notebooks/Julia-Tuor.ipynb)
* [Julia Parallel Computing](https://github.com/NREL/HPC/blob/master/languages/julia/julia-tutorial/source-notebooks/Julia-Parallel-Computing.ipynb)
* [Calling Python, C, and FORTRAN from Julia](https://github.com/NREL/HPC/blob/master/languages/julia/julia-tutorial/source-notebooks/Julia-Calling-Python-C-Tutorial.ipynb)
* PyJulia -- calling Julia from Python ([`PyJulia_Demo.ipynb`](https://github.com/NREL/HPC/tree/master/languages/julia/demos/notebooks))
* Integrating `mpi4py` and `MPI.jl`
    * [Hello World](https://github.com/NREL/HPC/blob/master/languages/julia/demos/scripts/mpi_jl_hello_world.py)
    * [Self contained approximation of pi](https://github.com/NREL/HPC/blob/master/languages/julia/demos/scripts/mpi_jl_pi.py)
    * [Approximation of pi as "library" call](https://github.com/NREL/HPC/blob/master/languages/julia/demos/scripts/mpi_jl_pi_as_lib.py)
    * [Comparing different Control Variates to approximate pi](https://github.com/NREL/HPC/blob/master/languages/julia/demos/scripts/mpi_jl_cv_pi.py) -- uses MPI Split
    * [Example batch script for all of the above](https://github.com/NREL/HPC/blob/master/languages/julia/demos/scripts/run_demo.sh)

### Requirements and Installation

Running the demos requires the python modules `mpi4py` and `julia`. For details on installing these modules, see the 'Environment Setup' section of the [README found in the demos/scripts directory](https://github.com/NREL/HPC/blob/master/languages/julia/demos/scripts/README.md).

For more information on mpi4py, see the [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/)

For more information on PyJulia, see the [PyJulia documentation](https://pyjulia.readthedocs.io/en/latest/installation.html).
