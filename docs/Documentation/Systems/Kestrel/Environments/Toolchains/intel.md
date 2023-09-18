# Compile and run: *Intel1API compilers & MPI*

### Introduction
This page shows how to compile and run programs using Intel's 1API tool chain.  We'll look at building using their MPI and Fortran and C compilers.  It is possible to build programs using Intel's MPI libraries but actually compile using gfortran and gcc. This is also covered.  

Intel's C compiler icc has been around for many years.  It is being retired and replaced with icx.  As of summer of 2023 you can still use icc but it is scheduled to be removed by the end of the year.  Building with icc produces a warning message.  We'll discuss how to surpress the warning and more importantly, build using icx.

Our example programs are hybrid MPI/Openmp so we'll show commands for building hybrid programs.  If your program is pure MPI the only change you need to make to the build process is to remove the compile line option -fopenmp.  

Sample makefile, source codes, and runscript for on Kestrel can be found in our [Kestrel Repo](https://github.com/NREL/HPC/tree/master/kestrel)  under the Toolchains folder.  There are individual directories for source,makefiles, and scripts or you can download the intel.tgz file containing all required files.


### module loads for compile

These are the module you will need for compiles:

```
module load  intel-oneapi-compilers 
module load intel-oneapi-mpi        
module load gcc                     
```

Intel compilers use some gcc functionality so  we load gcc to give a newer version of that compiler.

### module loads for run
Normally, builds are static, meaning that an application "knows" where to find its libraries.  Thus, we don't need to load the Intel modules at runtime  Unless you have some other external libaries that require a module load the only module lines you will need are:

```
module purge
module load libfabric
```


### Building programs

As discussed above we can build with Intel (ifort, icc, icx) or GNU (gcc, gfortran) underlying compilers.  The 5 options are:

1. Fortran with: Intel MPI and Intel Fortran compiler
1. C with: Intel MPI and Intel C compiler, older compiler (icc) 
1. C with: Intel MPI and Intel C compiler, newer compiler (icx)
1. Fortran with: Intel MPI with gfortran Fortran compiler
1. C with: Intel MPI with gcc C compiler

Here's what the compile lines should be where we add the -fopenmp option for Opnemp and the optimization flag -O3.

#### 1. Fortran with: Intel MPI and Intel Fortran compiler

```
mpiifort -O3 -g -fopenmp  ex1.f90  
```

#### 2. C with: Intel MPI and Intel C compiler, older compiler (icc) 
```
mpiicc -O3 -g -fopenmp  ex1.c  -o ex_c
```

This will produce the warning message *icc: remark #10441: The Intel(R) C++ Compiler Classic (ICC) is deprecated and will be removed from product release in the second half of 2023. The Intel(R) oneAPI DPC++/C++ Compiler (ICX) is the recommended compiler moving forward. Please transition to use this compiler. Use '-diag-disable=10441' to disable this message*

We can compile with the extra flag.

```
mpiicc -diag-disable=10441 -O3 -g -fopenmp  ex1.c   -o gex_c
```

#### 3. C with: Intel MPI and Intel C compiler, newer compiler (icx)

```
export I_MPI_CC=icx
mpiicc -O3 -g -fopenmp  ex1.c  -o ex_c
```
Setting the environmental variable tells mpiicc to use icx (the newer Intel compiler) instead of icc.

### mpicc and mpif90 may not give you what you expect.  

The commands mpicc and mpif90 actually call gcc and gfortran instead of the Intel compilers. If you consider these the default way to compile programs the "by default" Intel MPI does not use Intel compilers.  

#### 4. Fortran with: Intel MPI with gfortran Fortran compiler

```
mpif90 -O3 -g -fopenmp  ex1.f90 
```
#### 5. C with: Intel MPI with gcc C compiler
```
mpicc -O3 -g -fopenmp  ex1.f90 
```


Example programs
We have two example MPI/OpenMP programs, ex1.c and ex1.f90.  They are more or less identical in function.  They first print MPI Library and compiler information.  For example the fortran example compiled with mpiifort reports:

```
  Fortran MPI TASKS            4
 Intel(R) MPI Library 2021.8 for Linux* OS

 Intel(R) Fortran Intel(R) 64 Compiler Classic for applications running on Intel
```

For mpif90 we get:

```
  Fortran MPI TASKS            4
 Intel(R) MPI Library 2021.8 for Linux* OS

 GCC version 13.1.0
```

Note in these cases we have the same MPI library but different compilers.

The programs call a routine, *triad*. It keeps the cores busy for about 4 seconds.  This allows the OS to settle down.  Then for each MPI task and each openmp thread we get a line of the form:

```
task 0001 is running on x9000c3s2b0n0 thread=   2 of   3 is on core  054
```

This is saying that MPI task 1 is running on node x9000c3s2b0n0.  The task has 3 openmp threads and the second is running on core 54.




### Example makefile

The triad.c file containes the routines that keeps the cores busy for 4 seconds.  This is common to both the fortran and C versions of our codes. As discussed above our main codes are ex1.c and ex1.f90.  Our makefile will build for 

#### 1. Fortran with: Intel MPI and Intel Fortran compiler
#### 3. C with: Intel MPI and Intel C compiler, newer compiler (icx)
#### 4. Fortran with: Intel MPI with gfortran Fortran compiler
#### 5. C with: Intel MPI with gcc C compiler

There are comments in the makefile to show how to build with

#### 2. C with: Intel MPI and Intel C compiler, older compiler (icc) 

The makefile has an intresting "trick".  The default target is recurse.  This target loads the modules then calls make again using the same makefile but with the targets  intel and gnu.  By using this "trick" you don't have to load modules before the make.  

The targets intel and gnu each have a dependency to compile triad with either Intel or gcc compilers.  Then the final applications are built with Intel MPI and again the either Intel or gnu.

The final MPI codes are: 

* gex_c : gcc
* gex_f : gfortran
* ex_c  : Intel C (icx)
* ex_f  : Intel Fortran (ifort)


### Run script

1. Makes a new directory, copies the requred files and goes there
2. Does a make with output going into make.log
3. Sets the number of MPI tasks and openmp threads
4. Sets some environmental variables to control and report on threads (discussed below)
5. module commands
	1. module purge
	2. module load libfabric
6. Creates a string with all of our srun options (discussed below)
7. Calls srun on each version of our program
	1. output goes to *.out
	2. Report on thread placement goes to *.info

Our script sets these openmp related variables.  The first is familiar. KMP_AFFINITY is unique to Intel compilers.  In this case we are telling the OS to scatter (spread) out our threads.  OMP_PROC_BIND=spread does the same thing but it is not unique to Intel compilers. So in this case KMP_AFFINITY is actually redundent.  

```
  export OMP_NUM_THREADS=3
  export KMP_AFFINITY=scatter
  export OMP_PROC_BIND=spread
```

The next line 

```
export BIND="--cpu-bind=v,cores"
```

is not technically used as an environmental variable but it will be used to create the srun command line.  Passing --cpu-bind=v to srun will casue it to report threading information.  The "cores" option tells srun to "Automatically generate masks binding tasks to cores."  There are many other binding options as described in the srun man page. This setting works well for many programs.


Our srun command line options for 2 tasks per node and 3 threads per task are:

```
--mpi=pmi2 --cpu-bind=v,cores --threads-per-core=1 --tasks-per-node=2 --cpus-per-task=3
```

* --mpi=pmi2 : tells srun to use a particular launcher 
* --cpu-bind=v,cores : discussed above
* --threads-per-core=1 : don't allow multiple threads to run on the same core.  Without this option it is possible for multiple threads to end up on the same core, decreasing performance.  
* --cpus-per-task=3 : The cpus-per-task should always be equal to OMP\_NUM\_THREADS.


The final thing the script does is produce a results report.  This is just a list of mapping of mpi tasks and threads.  There should not be any repeats in the list.  There will be "repeats" of cores but on different nodes.   There will be "repeats" of nodes but with different cores.

You can change the values for --cpu-bind, OMP\_PROC\_BIND, and threads-per-core to see if this list changes.
