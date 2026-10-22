# Fortran

*Despite its age, Fortran is still a common language in scientific computing on account of its speed and ease of use in writing numerical computing-centric code.*


## Getting Started
This section walks through how to compile and run a basic Fortran code, and then a basic Fortran MPI code, adapted from [here](https://github.com/NatLabRockies/HPC/tree/master/languages/fortran). See [Compilers and Toolchains](#compilers-and-toolchains) for compiler and programming environment information on NLR HPC systems.  See [External Resources](#external-resources) for general Fortran language tutorials and Fortran-MPI tutorials.  

### Hello World

Create a file named hello.f90, and save the following text to the file:

```
PROGRAM hello

write(*,*) "Hello World"

END PROGRAM hello
```

Now, we must choose the compiler with which to compile our program. We can choose between the GNU, Intel, Nvidia, and Cray compilers, depending on which system we're on (see [Compilers and Toolchains](#compilers-and-toolchains)). 

To see available versions of a chosen compiler, use `module avail`. For this example, we'll use gfortran, which is part of GNU's `gcc` package:

```
module spider gcc 
...
...
    Versions:
      gcc/14.2.0
      gcc/15.2.0
    Other possible modules matches:
      gcc-native  gcc-native-mixed  
...
...
   gcc/10.3.0          gcc/11.2.0          gcc/12.1.0(default)
```

We'll use gcc/14.2.0

```
module load gcc/14.2.0
```


Now, we can compile the program with the following command:

`gfortran hello.f90 -o hello`

This creates an executable named `hello`. Execute it by typing the following into your terminal:

`./hello`

It should return the following output:

`Hello World`

### Hello World in MPI Parallel

The purpose of Fortran today is to run large scale computations fast. For the "large scale" part, we use MPI. Now that we have a working Hello World program, let's modify it to run on multiple MPI tasks.

On Kestrel, there are multiple implementations of MPI available. We can choose between OpenMPI, Intel MPI, MPICH, and Cray MPICH. These MPI implementations are associated with an underlying Fortran compiler. There are vendor supplied versions of MPI which give good performance.  To enable these we first:

`ml cpe-stack`

Then, to the list we:

`module spider PrgEnv`



We find  `PrgEnv-intel` is available which gives us Cray MPI with Intel compilers.  

Let's choose the PrgEnv-intel combination:

`module load PrgEnv-intel`

Now, create a new file named `hello_mpi.f90` and save the following contents to the file:

```
PROGRAM hello_mpi
include 'mpif.h'

integer :: ierr, my_rank, number_of_ranks

call MPI_INIT(ierr)
call MPI_COMM_SIZE(MPI_COMM_WORLD, number_of_ranks, ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, ierr)

write(*,*) "Hello World from MPI task: ", my_rank, "out of ", number_of_ranks

call MPI_FINALIZE(ierr)

END PROGRAM hello_mpi
```

To compile this program, type:

`mpif90 hello_mpi.f90 -o hello_mpi`

for the vendor supplied version of MPI there are aliases for mpif90, mpicc, and mpiCC.

|Normal compiler command | PrgEnv-* alias|
|:---------------:|:-------------------:|
|mpif90| ftn|
|mpicc|cc|
|mpiCC|CC|

We can run run this from the command line below. Be sure to replace the `<your account here>` with your account name.

`srun -n 4 --time=00:01:00 --partition=debug --account=<your account here> ./hello_mpi
`

This might look like you are running on the login node but you are actually telling the scheduler to grab a node and run this single program.  You might have to wait for a node to become available.  

You should receive a similar output to the following (the rank ordering may differ):

```
 Hello World from MPI task:            1 out of            4
 Hello World from MPI task:            2 out of            4
 Hello World from MPI task:            3 out of            4
 Hello World from MPI task:            0 out of            4
```

Let's submit this as a job to the scheduler. Create a file named `job.in` and modify the file to contain the following:

```
#!/bin/bash

#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=debug
#SBATCH --account=<your account here>

#Some programs require you to load you modules.
#We do that here, just in case.
ml cpe-stack
ml PrgEnv-intel

srun -n 4 ./hello_mpi > hello.out

```
Be sure to replace the `<your account here>` with your account name.

Submit the job:

`sbatch job.in`

When the job is done, the file hello.out should contain the same output as you found before (the ordering of ranks may differ).

## Compilers and Toolchains

### Fortran compilers

| Compiler        | Compiler Executable | Module Command | Systems available on |
|:---------------:|:-------------------:|:------------:|:--------------------:|
| gcc             | gfortran            | ml gcc          | Kestrel<br>Gila|
| intel           | ifx                 | ml oneapi       | Kestrel<br>Gila|
| Intel Fortran    | ftn | ml cpe-stack<br>ml PrgEnv-intel | Kestrel|
| Cray Fortran    | ftn | ml cpe-stack<br>ml PrgEnv-cray | Kestrel|
| nvidia Fortran    | ftn | ml cpe-stack<br>ml PrgEnv-nvidia | Kestrel|
| nvidia Fortran    | nvfortran<br>pgfortran<br>pgf77<br>pgf90<br>pgf95 | ml nvidia | Kestrel<br>Gila|

### Some Fortran-MPI Toolchains

| Compiler       | MPI     | Compiler Executable | Module Command                    | Systems available on |
|:--------------:|:-------:|:-------------------:|:-------------------------------:|:--------------------:|
| gcc            | openmpi | mpifort             | ml gcc<br>mlopenmpi                         | Kestrel<br>Gila
| Intel          | Intel | mpif90             |ml oneapi <br> ml intel-oneapi-mpi             | Kestrel<br>Gila |
| intel          | Intel   | mpiifx            | ml oneapi <br> ml intel-oneapi-mpi                | Kestrel<br>Gila
| gcc            | MPICH   | mpifort             | ml gcc<br>ml mpich                           | Kestrel<br>Gila
| intel          | MPICH   | mpifort             | ml oneapi<br>ml mpich               | Kestrel<br>Gila
| Intel Fortran    |Cray|  ftn | ml cpe-stack<br>ml PrgEnv-intel | Kestrel|
| Cray Fortran    |Cray| ftn | ml cpe-stack<br>ml PrgEnv-cray | Kestrel|
| nvidia Fortran   |Cray| ftn | ml cpe-stack<br>ml PrgEnv-nvidia | Kestrel|

## External Resources

* [Basic Fortran Tutorial](https://pages.mtu.edu/~shene/COURSES/cs201/NOTES/fortran.html)
