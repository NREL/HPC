# Scientific Libraries Overview

Scientific math libraries are a collection of highly optimized software tools that provide functions and algorithms for performing mathematical operations commonly used in scientific applications. They provide developers with a variety of tools for solving complex problems. These libraries are highly optimized for performance and generally designed to be portable across different platforms and operating systems. 

We support some of the most widely used scientific math libraries including:

* MKL 
* LibSci (Kestrel only)
* FFTW 
* LAPACK
* scaLAPACK
* HDF5 

For details on how to build an application with scientific libraries, see our [how-to guide](howto.md)

## MKL
**Documentation:** [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html)

MKL is a collection of highly optimized mathematical libraries provided by Intel for use in scientific, engineering, and financial applications. The library is designed to take full advantage of the latest Intel processors, including multi-core processors, and can significantly improve the performance of numerical applications. Core math functions include: 

* BLAS (Basic Linear Algebra Subroutines) 
* LAPACK (Linear Algebra routines) 
* ScaLAPACK (parallel Linear Algebra routines) 
* Sparse Solvers 
* Fast Fourier Transforms 
* Vector Math 

## LibSci

**Documentation:** [LibSci](https://support.hpe.com/hpesc/public/docDisplay?docId=a00115110en_us&docLocale=en_US&page=Cray_Scientific_and_Math_Libraries_CSML_.html)

LibSci is a collection of numerical libraries developed by AMD for scientific and engineering computing. LibSci is optimized for performance on AMD processors, including multi-core processors, and supports both single-precision and double-precision arithmetic. It also includes multithreading support for parallel execution on shared-memory systems. Like MKL, LibSci includes the following math functions: 

* BLAS (Basic Linear Algebra Subroutines) 
* CBLAS (C interface to the legacy BLAS) Note: not sure if this is also in MKL? 
* BLACS (Basic Linear Algebra Communication Subprograms) 
* LAPACK (Linear Algebra routines) 
* ScaLAPACK (parallel Linear Algebra routines) 

And additionally, libraries that are unique to Cray systems including: 
* IRT (Iterative Refinement Toolkit) - a library of solvers and tools that provides solutions to linear systems using single-precision factorizations while preserving accuracy through mixed-precision iterative refinement. 
* CrayBLAS - a library of BLAS routines autotuned for Cray XC series systems through extensive optimization and runtime adaptation.  

## FFTW

**Documentation:** [FFTW](https://www.fftw.org/)

FFTW is a C library for computing discrete Fourier transforms of arbitrary input sizes and dimensions. It is optimized for speed and can perform discrete Fourier transforms up to several orders of magnitude faster than other commonly available Fourier transform libraries. FFTW supports both single-precision and double-precision transforms, as well as multithreading for parallel execution on shared-memory systems.


## LAPACK and scaLAPACK

**Documentation:** [LAPACK](https://netlib.org/lapack/), [scaLAPACK](https://netlib.org/scalapack/)

LAPACK is a highly optimized library of linear algebra routines written in Fortran 90. These routines include matrix multiplication, factorization (LU, Cholesky, QR, etc.) least squares solutions of linear systems, eigenvalue problems, and many others. LAPACK routines are available in both single and double precision, and for complex and real numbers.

LAPACK depends on [BLAS](https://netlib.org/lapack/faq.html#_blas) (Basic Linear Algebra Subprograms).

ScaLAPACK is a parallel-distributed version of LAPACK (i.e., scalaPACK is MPI-parallel)

Both LAPACK and ScaLAPACK are available as either standalone libraries (`netlib-lapack`), or as part of the "package-of-packages" libraries [MKL](#mkl) and [LibSci](#libsci).

## HDF5

**Documentation:** [HDF5](https://portal.hdfgroup.org/display/HDF5/HDF5)

HDF5 is a versatile data storage and management library designed for storing and exchanging large and complex data collections. It provides a powerful and flexible data model for representing and organizing data, as well as a variety of high-level programming interfaces for accessing and manipulating data. HDF5 supports a wide range of data types and can handle data sets of virtually unlimited size.

## Additional Resources

For a detailed guide on how to include scientific libraries when compiling programs, see [our guide](howto.md).
