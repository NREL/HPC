# Julia_HPC_Guides

This repo contains demos (in the form of scripts and notebooks) and how to guides for doing various things with Julia on Eagle (and to some extent any HPC environment).

### Contents

All how to guides are found in the `how_to_guides` directory. Demos are found in the `demos` directory which is divided into scripts and jupyter notebooks.

Below is a brief list of the demos and how to guides along with the path of the file relative the root of this repository in parentheses.

**1. Demos**
  * pyjulia -- calling Julia from Python (demos/notebooks/PyJulia_Demo.ipynb)
  * integrating mpi4py and MPI.jl
    * Hello World (demos/scripts/mpi_jl_hello_world.py)
    * Self contained approximation of pi (demos/scripts/mpi_jl_pi.py)
    * Approximation of pi as "library" call (demos/scripts/mpi_jl_pi_as_lib.py)
    * Comparing different Control Variates to approximate pi -- uses MPI Split (demos/scripts/mpi_jl_cv_pi.py)
  * example batch script (demos/scripts/run_demo.sh)

**2. How To Guides**
  * building Julia on Eagle
  * building Ipopt with Harwell Subroutine Library (HSL) on Eagle (includes instructions for Mac OSX as well)

### Requirements and Installation

Running the demos requires the python modules `mpi4py` and `julia`.  These can both be installed with `pip`:
```
conda create -n julia_hpc_demos python=3 # create a conda environment for this
conda activate julia_hpc_demos
pip install mpi4py
pip install julia # Julia must already be installed
```
Note that `mpi4py` requires that MPI is already installed and accessible.  On Eagle, this can be easily done:
```
module load mpi
mpicc --version # test to make sure it worked
```
For more information, see the [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/)

The `julia` module requires that `Julia` (the executable) is already installed.  Once `Julia` and `julia` are both installed, you need to install some julia packages as well.  This can be done through python:
```
python
# in python interpreter window
import julia
julia.install()
```
For more information as well as any issues that arise, look at the [PyJulia documentation](https://pyjulia.readthedocs.io/en/latest/installation.html).
