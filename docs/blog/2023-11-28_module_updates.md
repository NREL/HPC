# Module Updates on Kestrel for November 2023

## OpenFOAM

* OpenFOAM versions 10, 11, and 2306 are now available as modules. All are compiled with GCC and OpenMPI.
* The module `openfoam/v2306-openmpi-gcc` is an installation of OpenFOAM obtained from the .com OpenFOAM website.
* OpenFOAM 10 and 11 are from the .org website which is cloned from the OpenFOAM repo. If there is a need for the dev version of OpenFOAM on Github, please let us know via [HPC-Help@nrel.gov](mailto:HPC-Help@nrel.gov) and we can work on making it available. 

## Anaconda

* `anaconda3/2022.05` was lately noticed to generate error message. This is now rectified, and the 2022 version should be working properly.
* The modules `anaconda3/2023` and `mamba/1.4.2` are also available.

## NetCDF 

* `netcdf/4.9.2-intel-oneapi-mpi-intel` is now available as a module. The module contains netcdfc, netcdfcxx, and netcdf-fortran compiled with Intel in one single module as opposed to having them in separate modules.