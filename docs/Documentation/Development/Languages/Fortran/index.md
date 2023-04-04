# Fortran

*Despite its age, Fortran is still a common language in scientific computing on account of its speed and ease of use in writing numerical computing-centric code.*


## Getting Started
This section walks through how to compile and run a basic Fortran code, and then a basic Fortran MPI code, adapted from [here](https://github.com/NREL/HPC/tree/master/languages/fortran). See [Compilers and Toolchains](#compilers-and-toolchains) for compiler and programming environment information on NREL HPC systems. For an extensive guide to Fortran 90, see our page on [Advanced Fortran](f90_advanced.md). See [External Resources](#external-resources) for general Fortran language tutorials and Fortran-MPI tutorials.  

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
module avail gcc 
   gcc/10.3.0          gcc/11.2.0          gcc/12.1.0(default)
```

We'll use gcc/12.1.0:

```
module load gcc/12.1.0
```


Now, we can compile the program with the following command:

`gfortran hello.f90 -o hello`

This creates an executable named `hello`. Execute it by typing the following into your terminal:

`./hello`

It should return the following output:

`Hello World`

### Hello World in MPI Parallel

The purpose of Fortran today is to run large scale computations fast. For the "large scale" part, we use MPI. Now that we have a working Hello World program, let's modify it to run on multiple MPI tasks.

On Kestrel, there are multiple implementations of MPI available. We can choose between OpenMPI, Intel MPI, MPICH, and Cray MPICH. These MPI implementations are associated with an underlying Fortran compiler. For example, if we type:

`module avail openmpi`

we find that both `openmpi/4.1.4-gcc` and `openmpi/4.1.4-intel` are available.

Let's choose the openmpi/gcc combination:

`module load openmpi/4.1.4-gcc`

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

To run this code on the login node, type:

`mpirun -n 4 ./hello_mpi`

You should receive a similar output to the following (the rank ordering may differ):

```
 Hello World from MPI task:            1 out of            4
 Hello World from MPI task:            2 out of            4
 Hello World from MPI task:            3 out of            4
 Hello World from MPI task:            0 out of            4
```

Generally, we don't want to run MPI programs on the login node! Let's submit this as a job to the scheduler. Create a file named `job.in` and modify the file to contain the following:

```
#!/bin/bash

#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=standard
#SBATCH --account=<your account here>

module load openmpi/4.1.4-gcc

srun -n 4 ./hello_mpi &> hello.out

```
Be sure to replace the `<your account here>` with your account name.

Submit the job:

`sbatch job.in`

When the job is done, the file hello.out should contain the same output as you found before (the ordering of ranks may differ).

## Compilers and Toolchains

### Fortran compilers

| Compiler        | Compiler Executable | Module Avail | Systems available on |
|:---------------:|:-------------------:|:------------:|:--------------------:|
| gcc             | gfortran            | gcc          | Kestrel(Eagle), Swift, Vermilion
| intel           | ifort               | intel-oneapi | Kestrel(Eagle), Swift, Vermilion
| intel           | ifort               | intel-classic| Kestrel

### Fortran-MPI Toolchains

| Compiler       | MPI     | Compiler Executable | Module Avail                    | Systems available on |
|:--------------:|:-------:|:-------------------:|:-------------------------------:|:--------------------:|
| gcc            | openmpi | mpifort             | openmpi                         | Kestrel(Eagle), Swift, Vermilion
| intel          | openmpi | mpifort             | openmpi/4.1.x-intel             | Kestrel(Eagle)
| intel          | intel   | mpiifort            | intel-oneapi-mpi                | Kestrel, Swift, Vermilion
| gcc            | MPICH   | mpifort             | mpich                           | Kestrel, Swift, Vermilion
| intel          | MPICH   | mpifort             | mpich/4.0.2-intel               | Kestrel only
| cray           | MPICH   | ftn                 | cray-mpich                      | Kestrel only

## External Resources

* [Comprehensive treatise on Fortran 90](f90_advanced.md)
* [Basic Fortran Tutorial](https://pages.mtu.edu/~shene/COURSES/cs201/NOTES/fortran.html)
* [Detailed Fortran Tutorial](https://github.com/NREL/HPC/blob/gh-pages/docs/Documentation/languages/fortran/f90.md)
* [Fortran/MPI on an HPC Tutorial](https://curc.readthedocs.io/en/latest/programming/MPI-Fortran.html)

