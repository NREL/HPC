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

Running the demos requires the python modules `mpi4py` and `julia`. For details on installing these modules, see the 'Environment Setup' section of the [README found in the demos/scripts directory](demos/scripts/README.md).

For more information on mpi4py, see the [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/)

For more information on PyJulia, see the [PyJulia documentation](https://pyjulia.readthedocs.io/en/latest/installation.html).
