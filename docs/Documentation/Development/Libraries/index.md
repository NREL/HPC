# Technical Software Libraries

NREL's Computational Science Center supports installations of standard numerical, I/O, and language extension libraries.

## Libraries List

|Name	|Description|
|:------------------------------------:|:--------------:|
|[Boost](boost.md)|	Useful and widely used extensions to the C++ language standard. See [Boost website](https://www.boost.org/).|
|[FFTW](fftw.md) |	C  subroutine library for computing the discrete Fourier transform (DFT) in one or more dimensions, of arbitrary input size, and of both real and complex data. FFTW supports parallelization through multi-threading and MPI domain decomposition. See [FFTW website](http://www.fftw.org/).|
|[HDF5](hdf5.md)|	Data model, library, and file format for storing and managing hierarchically structured data. See [HDF5 website](https://www.hdfgroup.org/solutions/hdf5/).|
|Math Kernel Library ([MKL](mkl.md))	|This is our primary mechanism to support BLAS, LAPACK, BLACS for OpenMPI/Intel MPI/HPE MPI, and ScaLAPACK on Eagle for the Intel and GCC compilers. See "module show mkl" output for example link commands for various options. |
|[NetCDF](netcdf.md)| Interfaces for array-oriented data access, for C and Fortran. See [Unidata website](https://www.unidata.ucar.edu/software/netcdf/).|
|PnetCDF| High-performance access to netCDF file formats CDF-1, CDF-2, and CDF-5, from Argonne National Laboratory and Northwestern University. See [PnetCDF site](https://parallel-netcdf.github.io/). |

???+ tip "Note" 

    If you are mixing an Anaconda environment with modules to build, always activate the conda environment before loading any library modules, especially Boost and MKL. CMake discovery, for example, is very sensitive to the order in which these actions are taken. If you want to use libraries from the conda environment, do not use modules for those libraries.
