# Building and Running on Kestrel's H100 GPU nodes.
This page describes how to build and run on Kestrel's GPU nodes using several programming paradigms.  There are pure Cuda programs, Cuda aware MPI programs, MPI programs without Cuda, MPI programs with Cuda, MPI programs with Openacc, and pure Openacc programs. 

Please clone the repo with the run scripts and examples:

On Kestrel:

```
git clone git@github.nrel.gov:tkaiser2/h100buildrun.git
cd h100buildrun
```

The [about.md](https://github.com/NREL/HPC/blob/master/h100buildrun/about.md) file contains condensed version of this page.  

The [README.md](https://github.com/NREL/HPC/blob/master/h100buildrun/README.md) file contains a tree style list of all of the tests.  These are all known to work on April 5th, 2024.  As the system evolves some module names might need to be changed. 

Each bottom level directory contains a file "doit" which if sourced will build and run the example.  **It is assumed that this is run on 2 H100 GPU nodes because some of the codes require a specific task count.**

Each run script starts with the commands:

```
: Start from a known module state, the default
module restore

: Enable a newer environment
source /nopt/nrel/apps/gpu_stack/env_cpe23.sh
: Load modules
module purge
ml craype-x86-genoa
```

This gives us a standard, known programming environment from which to start.  The "source" line give us access to a new set of modules. ml craype-x86-genoa sets optimizations for the cpus on the GPU nodes.


After this header we load module specific to the example.  We then compile and run the example.  

**If the file "runall" is sourced (after getting an interactive session on 2 gpu nodes) all examples will are run.  This takes about 20 minutes.  You can also sbatch script.**

 

Here is a brief description of all of the tests.  Click on the links to see the test runs script and source files.

## Cuda
This section contains examples of building straight Cuda codes, that is without MPI or Openacc.  The example is run in a loop, targeting each node in our allocation and each GPU on the node independently.

### [source](https://github.com/NREL/HPC/tree/master/h100buildrun/cuda/src)
The example stream.cu is the STREAM benchmark implementation in Cuda.  It measures the bandwidth or the GPUs processors to their memory by performing simple math operations: Copy, Scale, Add, and Triad on vectors.

The file stream.cu and the collection of files cuda.cu, extra.h and normal.c produce the same executable.  The original stream.cu was split to show how gcc can be used to compile routines that do not contain Cuda and then combined with Cuda enabled routines. : Load modules
module purge
ml craype-x86-genoa
ml gcc-native/12.1
ml nvhpc-native

The original code was modified to take the target GPU number on the command line.


### cray [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/cuda/cray)

```
: Load modules
ml craype-x86-genoa
ml gcc-native
ml PrgEnv-nvidia
ml cray-libsci/23.05.1.4 
```

Here we use the module PrgEnv-nvidia.  This is one of the "standard" Cray programming environments.  It gives us the MPI wrappers cc, CC, and ftn which can be used in place of the normal C, C++, and Fortran compilers.  In this case since our programs do not contain MPI they are compiled without the library.  The PrgEnv-nvidia pulls in the Nvidia Cuda compilers instead of the regular Cray compilers.  The regular Nvidia compilers, nvc, nvcc, nvfortran are also available.

### nvidia [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/cuda/nvidia)

```
: Load modules
module purge
ml craype-x86-genoa
ml nvhpc-nompi/24.1
```

There are several Nvidia related modules.  The module nvhpc-nompi/24.1 gives us the Nvidia Cuda compilers without MPI.  Here we compile with a "normal" Nvidia compiler nvcc.  


### gccalso [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/cuda/gccalso)
The module nvhpc-nompi/24.1 gives us the Nvidia Cuda compilers without MPI.  Here we compile with a "normal" Nvidia compiler nvcc.  Here we build the program in two parts. gcc is used to compile routines that do not contain Cuda an nvcc is used to compile Cuda routines and link with the gcc build routines.

```
: Load modules
module purge
ml craype-x86-genoa
ml gcc-native
ml nvhpc-nompi/24.1
```


## Openacc 

This section contains examples of building openacc codes, that is without MPI or Openacc.  The example is run in a loop, targeting each node in our allocation and each GPU on the node independently.

### [source](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/openacc/src)
This is a Nvidia written example that does an nbody calculation using openacc.  It also runs the same calculation without the GPU for a time comparison.  

### cray [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/openacc/cray) 

```
: Load modules
ml craype-x86-genoa 
ml gcc-native
ml PrgEnv-nvhpc
ml cray-libsci/23.05.1.4
```

Here we use the module PrgEnv-nvidia.  This is one of the "standard" Cray programming environment.  It gives us the MPI wrappers cc, CC, and ftn which can be used in place of the normal C, C++, and Fortran compilers.  In this case since our programs do not contain MPI they are compiled without the library.  The PrgEnv-nvidia pulls in the Nvidia Cuda compilers instead of the regular Cray compilers.  The regular Nvidia compilers, nvc, nvcc, nvfortran are also available.

The compile flat -acc enables openacc.  The flags -Minline -Minfo produce compile time information.

### nvidia [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/openacc/nvidia) 

The module nvhpc-native gives us the Nvidia Cuda compilers with MPI.  Here we compile with a "normal" Nvidia compiler nvcc.  We don't use MPI in this case.  This will also work with nvhpc-nompi/24.1 instead of nvhpc-native.

* nrelopenmpi [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/openacc/nvidia/nrelopenmpi)

```
 : Load modules
module purge
ml craype-x86-genoa 
ml gcc-native 
ml openmpi/4.1.6-nvhpc
ml nvhpc-nompi/24.1
```


* nvidia [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/openacc/nvidia/nvidia)

```
: Load modules
module purge
ml craype-x86-genoa 
ml gcc-native 
ml nvhpc-hpcx-cuda12/24.1
```





## MPI

Each example uses a different source code.

1. cudaaware: Pingpong directly from one GPU to another using MPI.  
1. normal: Hello world in both Fortran and C.  This program also prints the version of MPI that is being used.
1. openacc: Nvidia example that does Jacobi iterations using OpenACC and MPI
1. withcuda: Pingpong from one GPU to another using MPI.  Data is first transferred to/from the GPUs to the CPU and then sent via MPI.  It does not have a Cuda kernel but does show all of the allocation and memory transfer routines.


### Cuda aware MPI [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/cudaaware) [source](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/cudaaware/src)
Here we use the module PrgEnv-nvidia. This is one of the "standard" Cray programming environment. It gives us the MPI wrappers cc, CC, and ftn which can be used in place of the normal C, C++, and Fortran compilers.  PrgEnv-nvidia pulls in the Nvidia Cuda compilers instead of the regular Cray compilers. The regular Nvidia compilers, nvc, nvcc, nvfortran are also available.  The output from this version of pingpong should be compared to the "withcuda" version.  The second version is much slower.

```
: Load modules
ml craype-x86-genoa 
ml gcc-native
ml PrgEnv-nvhpc
ml cray-libsci/23.05.1.4  
``` 

### Normal MPI (no GPUs) [source](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/normal/src) 
* cray [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/normal/cray)

```
: Load modules
ml craype-x86-genoa 
```

Here we use the default programming environment which contains PrgEnv-cray/8.5.0.  It gives us the MPI wrappers cc, CC, and ftn which can be used in place of the normal C, C++, and Fortran compilers.  PrgEnv-cray uses the Cray backend C, C++ and Fortran compilers.

Note the output from the hello world example contains the MPI library version: *MPI VERSION    : CRAY MPICH version 8.1.28.15 (ANL base 3.4a2)*

* intel+abi [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/normal/intel%2Babi)

```
ml craype-x86-genoa 
ml gcc-native
ml intel-oneapi-mpi
ml intel-oneapi-compilers
```

This program is compiled using the modules intel-oneapi-mpi and intel-oneapi-compilers and compiled with mpiicx and mpifc.  These pull in the newer versions of Intel backend compilers and Intel MPI.

The program is run and it returns the MPI version: *Intel(R) MPI Library 2021.11 for Linux* OS*



The program is run a second time after loading the module cray-mpich-abi.  This module causes Intel MPI to be replaced with Cray MPI at runtime without recompiling or linking.  After this module is loaded the program returns the MPI version: *MPI VERSION    : CRAY MPICH version 8.1.28.15 (ANL base 3.4a2)* just as if we had built against Cray MPI.  In general Cray MPI will perform better than Intel MPI.

```
ml cray-mpich-abi
```


* nvidia
    * nrelopenmpi [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/normal/nvidia/nrelopenmpi)

```
: Load modules
module purge
ml craype-x86-genoa 
ml gcc-native 
ml openmpi/4.1.6-nvhpc
ml nvhpc-nompi/24.1
```

This version is compiled using the modules openmpi/4.1.6-nvhpc and nvhpc-nompi/24.1.  nvhpc-nompi provides the nvidia compilers but not MPI.  This particular openmpi module provides MPI built with Nvidia's backend compilers.  One advantage of this set is that programs can be launched with srun.  The program reports that it is running MPI version: *Open MPI v4.1.6*



* nvidia
    * nvidiampi [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/openacc/nvidia/nvidiaopenmpi)

```
: Load modules
module purge
ml craype-x86-genoa 
ml gcc-native 
ml nvhpc-hpcx-cuda12/24.1
```
    

This version is compiled using the module nvhpc-hpcx-cuda12/24.1. This provides Nvidia's compilers and Nvidia's MPI. This particular openmpi requires programs to be launched with mpirun instead of srun.  The program reports that it is running MPI version: *Open MPI v4.1.7a1*



### MPI with openacc [source](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/openacc/src)

This is a Nvidia example that does Jacobi iterations using OpenACC and MPI.  

* cray [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/openacc/cray)

Here we use the module PrgEnv-nvhpc. This is one of the "standard" Cray programming environment. It gives us the MPI wrappers cc, CC, and ftn which can be used in place of the normal C, C++, and Fortran compilers.  PrgEnv-nvhpc pulls in the Nvidia Cuda compilers instead of the regular Cray compilers. The regular Nvidia compilers, nvc, nvcc, nvfortran are also available.  The compile line option -acc enabled openacc and the option -Minfo=accel reports information about the compile.  This program is run using srun with 4 tasks-per-node.

It is possible to build this application to target CPUs instead of GPUs.  This is discussed in the source.  


* nvidia
    * nrelopenmpi [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/openacc/nvidia/nrelopenmpi)

This version is compiled using the modules openmpi/4.1.6-nvhpc and nvhpc-nompi/24.1.  nvhpc-nompi provides the nvidia compilers but not MPI.  This particular openmpi module provides MPI built by NREL using Nvidia's backend compilers.  One advantage of this set is that programs can be launched with srun.

* nvidia
    * nvidiaopenmpi [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/openacc/nvidia/nvidiaopenmpi)


This version is compiled using the module nvhpc-hpcx-cuda12/24.1. This provides Nvidia's compilers and Nvidia's MPI. This particular openmpi requires programs to be launched with mpirun instead of srun. The compile line option -acc enabled openacc and the option -Minfo=accel reports information about the compile.  The Nvidia version of MPI must be launched using mpirun instead of srun.


### MPI withcuda [source](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/withcuda/src)

This example is similar to cudaaware MPI example.  It does a pingpong between two MPI tasks.  The difference is this version copies data to/from the cpus and GPUs before sending it via MPI while the cudaaware version bypasses to cpus.  This version is much slower.  This program is run using srun with 2 tasks on a single node or 1 task on each of two nodes.


* cray [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/withcuda/cray)

```
: Load modules
ml craype-x86-genoa 
ml gcc-native
ml PrgEnv-nvhpc
ml cray-libsci/23.05.1.4
```

Here we use the module PrgEnv-nvhpc. This is one of the "standard" Cray programming environment. It gives us the MPI wrappers cc, CC, and ftn which can be used in place of the normal C, C++, and Fortran compilers.  PrgEnv-nvhpc pulls in the Nvidia Cuda compilers instead of the regular Cray compilers. The regular Nvidia compilers, nvc, nvcc, nvfortran are also available.  The compile line option -gpu=cc90 says to build for the h100 gpus.

* nvidia
    * nrelopenmpi [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/withcuda/nvidia/nrelopenmpi)

```
: Load modules
module purgenvc,nvcc
ml craype-x86-genoa 
ml gcc-native 
ml openmpi/4.1.6-nvhpc
ml nvhpc-nompi/24.1
```

This version is compiled using the modules openmpi/4.1.6-nvhpc and nvhpc-nompi/24.1.  nvhpc-nompi provides the nvidia compilers but not MPI.  This particular openmpi module provides MPI built by NREL using Nvidia's backend compilers.  One advantage of this set is that programs can be launched with srun.

* nvidia
    * nvidiaopenmpi [runscript](https://github.com/NREL/HPC/blob/master/h100buildrun/mpi/withcuda/nvidia/nvidiaopenmpi)

```
: Load modules
module purge
ml craype-x86-genoa 
ml gcc-native 
ml nvhpc-hpcx-cuda12/24.1
#ml nvhpc-native/24.1
```

This version is compiled using the module nvhpc-hpcx-cuda12/24.1. This provides Nvidia's compilers and Nvidia's MPI. This particular openmpi requires programs to be launched with mpirun instead of srun. The compile line option -acc enabled openacc and the option -Minfo=accel reports information about the compile.  The Nvidia version of MPI must be launched using mpirun instead of srun.

## Cudalib (Cuda enabled libraries)

### factor [runscript and source](https://github.com/NREL/HPC/blob/master/h100buildrun/cudalib/factor)

```
: Load modules
module purge
ml craype-x86-genoa
ml gcc-native/12.1
ml PrgEnv-gnu/8.5.0
```

This directory contains two programs, cpu.C and cusolver_getrf_example.cu.  They both do LU factorization to solve a linear system.
The first uses LAPACK and the second the Cuda library cusolver.  The parameter MYSIZE in the "doit" file sets the matrix.  The LAPACK
version is built/run with MKL and libsci. MYSIZE is 4500.  Setting it to 45000 works and the run time for the cpu versions is 45 seconds
with the GPU version a few seconds.

### fft [runscript and source](https://github.com/NREL/HPC/blob/master/h100buildrun/cudalib/fft)

```
: Load modules
module purge
ml craype-x86-genoa
ml gcc-native/12.1
ml nvhpc-native
```

The examples does a 3d complex fft on a given grid (cube) size.  It does it twice.  It does it using a single GPU and 4 GPUs.  The
second command line argument dictates which it does first.  1 implies run the single GPU version first.  Order influences timings.
This should work up to cube sizes of 2048. There is also 3d fftw example that is built with cc, PrgEnv-cray, and cray-fftw.

## Source credits





*  cuda/src/stream.cu - https://github.com/bcumming/cuda-stream with mods
*  openacc/src/nbodyacc2.c - Nvidia, part of the nvhpc distribution
*  mpi/openacc/src/acc_c3.c - Nvidia, part of the nvhpc distribution
*  mpi/normal/src/helloc.c, hellof.f90 - Timothy.Kaiser@nrel.gov
*  mpi/cudaaware/src/ping_pong_cuda_aware.cu, ping_pong_cuda_staged.cu [https://github.com/olcf-tutorials/MPI_ping_pong](https://github.com/olcf-tutorials/MPI_ping_pong)
*  cudalib/factor/cpu.C - Multiple sources with significant mods 
*  cudalib/factor/cusolver_getrf_example.cu - https://github.com/NVIDIA/CUDALibrarySamples.git with significant mods
*  cudalib/fft/3d_mgpu_c2c_example.cpp - https://github.com/NVIDIA/CUDALibrarySamples.git
*  cudalib/fft/fftw3d.c - Timothy.Kaiser@nrel.gov

