# PETSc

**Documentation:** [PETSC](https://petsc.org)

*PETSc is a suite of data structures and routines for the scalable (parallel) solution of scientific applications modeled by partial differential equations.*

On kestrel, PETSc is provided under multiple toolchains

```
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
  petsc:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
     Versions:
        petsc/3.14.6-cray-mpich-intel
        petsc/3.19.3-intel-oneapi-mpi-intel
        petsc/3.19.3-openmpi-gcc

```

`petsc/3.14.6-cray-mpich-intel` is a PETSc installation that uses HPE provided `Prgenv-Intel`. 
Therefore, the mpi used here is *cray-mpich* and the compiler is *intel/2023*.

`petsc/3.19.3-intel-oneapi-mpi-intel` is a PETSc installation that uses *intel-oneapi-compilers* and *intel-oneapi-mpi* for the compilers and mpi, respectively.

`petsc/3.19.3-openmpi-gcc` is a PETSc installation that uses *gcc/10.1.0* and *openmpi/4.1.5-gcc* for the compilers and mpi, respectively.
