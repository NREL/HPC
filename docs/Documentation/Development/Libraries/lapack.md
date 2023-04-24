# LAPACK and scaLAPACK

**Documentation:** [LAPACK](https://netlib.org/lapack/), [scaLAPACK](https://netlib.org/scalapack/)

LAPACK is a highly optimized library of linear algebra routines written in Fortran 90. These routines include matrix multiplication, factorization (LU, Cholesky, QR, etc.) least squares solutions of linear systems, eigenvalue problems, and many others. LAPACK routines are available in both single and double precision, and for complex and real numbers.

LAPACK depends on [BLAS](https://netlib.org/lapack/faq.html#_blas) (Basic Linear Algebra Subprograms).

ScaLAPACK is a parallel-distributed version of LAPACK (i.e., scalaPACK is MPI-parallel)

Both LAPACK and ScaLAPACK are available as either standalone libraries (`netlib-lapack`), or as part of the "package-of-packages" libraries [MKL](mkl.md) and [LibSci](libsci.md).
