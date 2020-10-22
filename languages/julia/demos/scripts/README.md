This directory contains example scripts for using Julia (and Python) on HPC platforms.  See the Contents section for more details.  Running the demos contained in this directory requires a specialized environment.  To create the needed environment, follow the instructions in the Environment Set Up section.

### Contents

 1. *mpi_jl_hello_world.py* -- Basic example of integration of Python's mpi4py and MPI.jl
 2. *mpi_jl_pi.py* -- Approximate pi with Monte Carlo integration using MPI to scale the task.  This is self-contained.
 3. *mpi_jl_pi_as_lib.py* -- Same as *mpi_jl_pi.py* but with the Julia part treated as a "library" call
 4. *mpi_jl_cv_pi.py* -- Compares two control variate variance reductions to approximate pi through Monte Carlo integration.  Uses MPI to scale the task.
 5. *pi_func.jl* -- "Library" functions for *mpi_jl_pi_as_lib.py* and *mpi_jl_cv_pi.py*. 

### Environment Set Up
You will need to have Julia installed with the `julia` executable in your path prior to starting the following steps to setup the needed environment. For details on how to do this, see the [Julia bulid instructions](../../how-to-guides/install-Julia.md).

These instructions are given for python 3.  The mpi4py and pyjulia modules do (as of this writing) support python 2. If you decide to use python 2, you will need to alter the below commands appropriately.  It is also likely that the scripts in this directory will require alterations to work with python 2.  However, given that python 2 is [no longer supported](https://www.python.org/doc/sunset-python-2/), it is recommended that you use python 3.
 
 1. Create a conda environment with python, numpy installed and activate it:
 ```
 conda create -c conda-forge --name py-jl-mpi python=3 numpy
 source activate py-jl-mpi
 ```
 2. Load the MPI module of your choice.  For example, OpenMPI 3.1.6 built with gcc 8.4.0:
 ```
 module load openmpi/3.1.6/gcc-8.4.0
 ```
 3. Check that the loaded MPI is the one that will actually be used
 ```
 which mpicc
 ```
 The returned path should be consistent with the loaded MPI module.  For the above module, the return is
 ```
 /nopt/nrel/apps/openmpi/3.1.6-centos77/bin/mpicc
 ```
 4. Check that julia is accessible from the command line
 ```
 which julia
 ```
 This should return the path the julia executable you desire to use with PyJulia.
 5. Install pyjulia and mpi4py:
 ```
 pip3 install julia mpi4py
 ```
 The install process for mpi4py picks out the MPI library it uses based on the `mpicc` executable it finds (hence step 3).  The PyJulia module requires that your python installation must be able to call command line program julia (hence step 4).
 
 6. Start the python interpreter and install the required julia packages:
  ```
  python
  import julia
  julia.install()
```
 7. Set environment variables to point MPI.jl at the same module loaded MPI:
```
export JULIA_MPI_BINARY=system
export JULIA_MPI_PATH=/nopt/nrel/apps/openmpi/3.1.6-centos77
```
8. Go to demo/scripts directory and start up Julia REPL:
```
cd demos/scripts
julia --project
```
9. Go to the package manager and install MPI.jl:
```
]
instantiate
```

### Running
To run any of the demos, you can make use of the `run_demo.sh` script.  There are instructions in the script about needed changes.  If you followed the Environment Setup instructions, most of the requirements will be met already.

> Written with [StackEdit](https://stackedit.io/).
