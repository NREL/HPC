# Kestrel Programming Environments Overview

## Definitions

**Toolchain**: a combination of a compiler and an mpi library. Sometimes associated scientific libraries (scalapack, blas, etc.) or bundles of scientific libraries (MKL, libsci, etc.) are considered part of the toolchain.

**Environment**: a set of modules, including a toolchain. A "build environment" refers to the set of modules (including compiler and MPI library) used to compile a code. A "run-time environment" is the set of modules used to execute a code. The two typically, but not always, match.

## Environments

There are three types of module-based Toolchains available on Kestrel:

1. "PrgEnv-" Environments, shipped with Kestrel
2. NREL-built Environments
3. NREL-built Environments with `cray-mpich-abi`

The "PrgEnv-" environments are new on Kestrel. PrgEnv stands for "programming environment," and Kestrel ships with several of these. There are advantages to using a PrgEnv environment, as these environments are tailored for some of the Cray-specific features of Kestrel. For example, Cray MPICH utilizes Kestrel's Cray Slingshot network more effectively than OpenMPI or Intel MPI, so it runs noticeably faster than the other two for jobs that require two or more nodes. All `PrgEnv-` environments utilize Cray MPICH by default.

The NREL-built environments function similarly to those on Eagle, and it is up to the user to load all necessary modules to build and run their applications. These modules can be accessed by running `source /nopt/nrel/apps/env.sh`.

NREL-built environments can make use of Cray MPICH via the `cray-mpich-abi`. As long as program is compiled with an MPICH-based MPI (e.g., Intel MPI but *not* Open MPI), the `cray-mpich-abi` can be loaded at runtime, which causes the program to use Cray MPICH for dynamically built binaries.


## A note on OpenMPI

Currently, OpenMPI does not run performantly or stably on Kestrel. You should do your best to avoid using OpenMPI. Please reach out to hpc-help@nrel.gov if you need help working around OpenMPI.

## Summary of available compiler environments

Note: to access compilers not included in the default Cray modules (i.e., compilers within the NREL-built environment), you must run the command `source /nopt/nrel/apps/env.sh`.

* (Cray) denotes that the module belongs to the default Cray module set.
* (NREL) denotes that the module belongs to the NREL-built module set. If a compiler module is denoted (NREL), then the corresponding MPI module is also (NREL).

### GNU

| PrgEnv|Compiler Module| MPI module |Language|Wrapper|Compiler|MPI|
|-------|---------------|------------|--------|-------|--------|---|
| gnu   | gcc (Cray)    | cray-mpich | Fortran| ftn   |gfortran| Cray MPICH |
| gnu   | gcc (Cray)    | cray-mpich | C      | cc    | gcc    | Cray MPICH |
| gnu   | gcc (Cray)    | cray-mpich | C++    | CC    | g++    | Cray MPICH |
| n/a   | gcc (NREL)   | openmpi/4.1.5-gcc | Fortran| mpifort| gfortran| Open MPI|
| n/a   | gcc (NREL)   | openmpi/4.1.5-gcc | C      | mpicc | gcc | Open MPI|
| n/a   | gcc (NREL)   | openmpi/4.1.5-gcc | C++    | mpic++ | g++ | Open MPI|

### Cray 

| PrgEnv|Compiler Module| MPI module|Language|Wrapper|Compiler|MPI|
|-------|---------------|-----------|--------|-------|--------|---|
| cray  | cce (Cray)   |cray-mpich  |Fortran| ftn   | crayftn| Cray MPICH |
| cray  | cce (Cray)   |cray-mpich  |C      | cc    | craycc | Cray MPICH |
| cray  | cce (Cray)   |cray-mpich  |C++    | CC    | crayCC | Cray MPICH |

### Intel

| PrgEnv|Compiler Module| MPI Module|Language|Wrapper|Compiler|MPI|
|-------|---------------|------|--------|-------|--------|---|
| intel | intel (Cray)  | cray-mpich    | Fortran| ftn   | ifort  | Cray MPICH |
| intel | intel (Cray)  | cray-mpich    | C      | cc    | icc    | Cray MPICH |
| intel | intel (Cray)  | cray-mpich    | C++    | CC    | icpc    | Cray MPICH |
| n/a   | intel-oneapi (NREL) |intel-oneapi-mpi  | Fortran| mpiifort | ifort  | intel MPI|
| n/a   | intel-oneapi (NREL) |intel-oneapi-mpi  | C      | mpiicc   | icc    | intel MPI|
| n/a   | intel-oneapi (NREL) |intel-oneapi-mpi  | C++    | mpiicpc  | icpc   | intel MPI|

Note: 

The `Cray MPICH` used for each different `PrgEnv-` is pointing to a different instance of MPICH, E.g. for `PrgEnv-intel` the MPICH used is located under `/opt/cray/pe/mpich/8.1.21/ofi/intel/19.0` and for `PrgEnv-cray` the MPICH used is located under `/opt/cray/pe/mpich/8.1.20/ofi/crayclang/10.0`.

## PrgEnv- Programming Environments

### Introduction

These environments come packaged with:

1. A compiler, which corresponds to the name of the environment. E.g., `PrgEnv-intel` uses intel compilers
2. Cray MPICH
3. Cray LibSci, which can be used in place of MKL
4. Additional communication and network libraries

Upon logging into the machine, the `PrgEnv-cray` is loaded by default. If we `module list`, we can see the modules associated with `PrgEnv-cray`. If we `module unload PrgEnv-cray` then we can see a few lingering modules. These are `craype-x86-spr` and `perftools-base/22.09` where the first dictates the architecture of the processors and is used to optimize the build step for the given hardware and the latter is a perfomance software that can be used to profile codes.   

We can swap between programming environments using the `module swap` command. For example, if `PrgEnv-cray` is loaded but we want to use the GNU programming environment instead, we can `module swap PrgEnv-cray PrgEnv-gnu`.

### What is a PrgEnv module doing?

PrgEnv modules can seem a bit mysterious. We can check out the inner workings of a PrgEnv module with the `module show` command. For example, for `PrgEnv-gnu` we can:

`module show PrgEnv-gnu`

Which outputs:

```
/opt/cray/pe/modulefiles/PrgEnv-gnu/8.3.3:

conflict	 PrgEnv-amd 
conflict	 PrgEnv-aocc 
conflict	 PrgEnv-cray 
conflict	 PrgEnv-gnu 
conflict	 PrgEnv-intel 
conflict	 PrgEnv-nvidia 
setenv		 PE_ENV GNU 
setenv		 gcc_already_loaded 1 
module		 swap gcc/12.1.0 
module		 switch cray-libsci cray-libsci/22.10.1.2 
module		 switch cray-mpich cray-mpich/8.1.20 
module		 load craype 
module		 load cray-dsmml 
module		 load craype-network-ofi 
module		 load cray-mpich 
module		 load cray-libsci 
setenv		 CRAY_PRGENVGNU loaded 
```

This tells us that PrgEnv-gnu conflicts with all other PrgEnvs. The modulefile sets some environment variables (the `setenv` lines), and loads the modules associated with the programming environment.

For most intents and purposes, we could re-construct and utilize the same programming environment by individually loading the associated modules:

```
module load gcc/12.1.0
module load craype
module load cray-mpich
module load cray-libsci
module load craype-network-ofi
module load cray-dsmml
```

We can use the `module whatis` command to give us a brief summary of a module. For example, the command:

`module whatis craype`

outputs:

`craype               : Setup for Cray PE driver set and targeting modules`

We mentioned previously that the different PrgEnvs use different locations for Cray-MPICH. We can see this by using `module show cray-mpich` in each different PrgEnv, and examining (for example) the `CRAY_LD_LIBRARY_PATH` environment variable. 


### Compiling inside a PrgEnv: ftn, cc, and CC

`ftn`, `cc`, and `CC` are the Cray compiler wrappers for Fortran, C, and C++, respectively, which are part of the `craype` module. When a particular `PrgEnv-` programming environment is loaded, these wrappers will make use of the corresponding compiler. For example, if we load PrgEnv-gnu with:

```
module swap PrgEnv-cray PrgEnv-gnu
```

we would expect `ftn` to wrap around gfortran, the GNU fortran compiler. We can test this with:

`ftn --version`

Which outputs:

```
GNU Fortran (GCC) 12.1.0 20220506 (HPE)
Copyright (C) 2022 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

As expected. We can also `which ftn`:
```
/opt/cray/pe/craype/2.7.17/bin/ftn
```
Note1: In contrast with mpich, the location of the wrappers `cc`, `CC` and `ftn` is always the same `/opt/cray/pe/craype/2.7.17/bin/ftn` and does NOT depend on the loaded PrgEnv.

Note2: `cc`, `CC` and `ftn` are also wrappers around their mpi couterparts. For mpi codes, the wrappers call the necessary mpi compilers depending on which PrgEnv is loaded. 

Note3: When changing between PrgEnvs, it is better to use `module swap [current prgenv] [new prgenv]` instead of `module purge; module load [new prgenv]` due to the way the environments set some environment variables.

`ftn` is part of the `craype` module. If we `module unload craype` and then type `which ftn` we find:
```
/usr/bin/which: no ftn in (/opt/cray/pe/mpich/8.1.20/ofi/gnu/9.1/bin:/opt/cray/pe/mpich/8.1.20/bin:/opt/cray/libfabric/1.15.2.0/bin:/opt/cray/pe/gcc/12.1.0/bin:/home/ohull/.local/bin:/home/ohull/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/nopt/xalt/xalt/bin:/nopt/nrel/utils/bin:/nopt/slurm/current/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/c3/bin:/sbin:/bin)
```

I.e., it can't find the path to `ftn`, because `craype` is not loaded into the environment.

What happens if we `module swap PrgEnv-gnu PrgEnv-cray`, so that we're now using PrgEnv-cray, and then check `ftn`?

```
[ohull@eyas1 ~]$ ftn --version
Cray Fortran : Version 14.0.4
```

`ftn` is now using Cray Fortran under the hood.

Note: you can still directly access the underlying compiler. For example, if we're using PrgEnv-gnu (so our compilers are the GCC compilers), we can use `ftn`, or we can use `gfortran` or `mpifort` directly. It is considered best practice to use the Cray wrappers (`ftn`, `cc`, `CC`) on a Cray machine like Kestrel.

In fact, the use of `mpifort` can be quite confusing. Inside the PrgEnv-gnu environment, we might assume that `mpifort` is a wrapper around OpenMPI. This is not correct, as `mpifort` wraps around Cray MPICH inside PrgEnv-gnu. If we `module unload PrgEnv-gnu` and then `module load openmpi`, then `mpifort` will wrap around OpenMPI. Using the Cray wrappers (`ftn`, `cc`, `CC`) helps avoid this confusion.

### Submitting a job within a PrgEnv

Submitting a Slurm job using a PrgEnv environment is no different than how you would normally submit a job. In your slurm script, below the #SBATCH directives, include:

```
module swap PrgEnv-cray [new PrgEnv]
```

We swap from `PrgEnv-cray` because this is the default PrgEnv that is loaded when logging onto Kestrel.

`[new PrgEnv]` can be `PrgEnv-gnu` or `PrgEnv-intel`.

Depending on the software you're trying to run, you may need to load additional modules like `cray-hdf5` or `cray-fftw`.


## NREL-built environments

The NREL build modules are similar to Eagle, where the module are separate and no dependecy is created between modules. 

The modules are grouped by type `compilers_mpis` `utilities_libraries` and `applications`, and a module can be loaded using `module load $module_name`.

The modules are optimized for Kestrel architecture and will be updated/upgraded every 6/12months or upon request. If there is a module you need but is not available, email hpc-help@nrel.gov


## NREL-built environments with cray-mpich-abi

For binaries dyanamically built with an MPICH-based MPI such as intel-mpi, the user can choose to use `cray-mpich-abi` at runtime to leverage its optimization for Kestrel. To check if your executable was dynamically built with intel MPI, you can `ldd [your program name] | grep mpi`.

the module `cray-mpich-abi` will cause the program to run with Cray MPICH at runtime instead of Intel MPI. In your slurm submit script, you must include the two lines:

`module load craype`
`module load cray-mpich-abi`

in order for the Cray MPICH abi to work properly.

**Note**: If your code depends on libmpicxx, the Cray MPICH ABI is unlikely to work. You can check this by `ldd [your program name] | grep mpicxx`.
