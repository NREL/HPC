---
 layout: default
 title: Running on Vermilion
 parent: Vermilion
 grand_parent: Systems
---

# Running on Vermilion

*This page discusses the compute nodes, partitions, and gives some examples of building and running applications.*

The page [Modules](./modules.md) discuses how to activate and use the modules on Vermilion. Modules are not available by default and must be activated.  Please see the [Modules](./modules.md) page for more information about setting up your environment and loading modules. 
## Compute hosts
Vermilion is a collection of physical nodes with each regular node containing Dual AMD EPYC 7532 Rome CPUs.  However, each node is virtualized.  That is it is split up into virtual nodes with each virtual node having a portion of the cores and memory of the physical node.  Similar virtual nodes are then assigned slurm partitions as shown below.  

## Shared file systems

Vermilion's home directories are shared across all nodes.  There is also /scratch/$USER and /projects spaces seen across all nodes.

## Partitions

Partitions are flexible and fluid on Vermilion.  A list of partitions can be found by running the `sinfo` command.  Here are the partitions as of 10/20/2022.

| Partition Name                          | Qty | RAM    | Cores/node | /var/scratch <br>1K-blocks | AU Charge Factor | 
| :--:                               | :--: | :--:    | :--:             | :--:   | :--: |                         
| gpu<br>*1 x NVIDIA Tesla A100*      |  17  | 114 GB |   30            |  6,240,805,336| 12 |       
| lg                                 | 39  | 229 GB |   60            |   1,031,070,000| 7 |
| std                                | 60  | 114 GB |   30            |     515,010,816| 3.5 |
| sm                                 | 28  |  61 GB |   16            |     256,981,000| 0.875 |
| t                                  | 15  |  16 GB |   4             |      61,665,000| 0.4375 |

## Allocation Unit (AU) Charges
The equation for calculating the AU cost of a job on Vermilion is: 

```AU cost = (Walltime in hours * Number of Nodes * Charge Factor)```

The Walltime is the actual length of time that the job runs, in hours or fractions thereof.

The **Charge Factor** for each partition is listed in the table above. 

## Operating Software
The Vermilion HPC cluster runs fairly current versions of OpenHPC and SLURM on top of OpenStack.


## Software Environments and Example Files
Environments are provided with a number of commonly used compilers, common build tools, specific optimized libraries, and some analysis tools. Environments must be enabled before modules can be seen.  This is discussed in detail on the page [Modules](./modules.md).

You can use the "standard" environment by running the command:

```
source /nopt/nrel/apps/210929a/myenv.2110041605
```

The examples on this page uses the environment enabled by this command.   You may want to add this command to your .bashrc file so you have a useful environment when you login.  

In the directory **/nopt/nrel/apps/210929a** you will see a subdirectory **example**.  This contains a makefile for a simple hello world program written in both Fortran and C and several run scripts. The README.md file contains additional information, some of which is replicated here. 

It is suggested you copy the directory to run the examples:

```
cp -r /nopt/nrel/apps/210929a/example ~/example
cd ~/example
```

## Simple batch script

Here is a sample batch script, *runopenmpi*, for running the hello world examples. 

!!! warning "Note"
    You must build the applications before running this script. Please see [Building hello world first](running.md#building-hello-world-first) below.

```
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=t
#SBATCH --time=00:01:00

cat $0

source /nopt/nrel/apps/210929a/myenv*
ml gcc   openmpi 

export OMP_NUM_THREADS=2
srun --mpi=pmi2 -n 2 ./fhostone -F
srun --mpi=pmi2 -n 2 ./phostone -F

```


The submission command is:

```
sbatch --partition=sm --account=MY_HPC_ACCOUNT runopenmpi
```

where MY\_HPC\_ACCOUNT is your account. 

## Building hello world first

For the script given above to work you must first build the application.  You need to:

1. Load the environment
2. Load the modules
3. make

#### Loading the environment
Loading the environment is just a matter of sourcing the file

```
source /nopt/nrel/apps/210929a/myenv.2110041605

```

#### Loading the modules.

We are going to use gnu compilers with OpenMPI.

```
module load gcc 
module load openmpi
```

#### Run make

```
make
```

## Full procedure screen dump

```
[joeuser@vs-login-1 ~]$ cp -r /nopt/nrel/apps/210929a/example ~/example
[joeuser@vs-login-1 ~]$ cd example/
[joeuser@vs-login-1 example]$ source /nopt/nrel/apps/210929a/myenv.2110041605
[joeuser@vs-login-1 example]$ module load gcc
[joeuser@vs-login-1 example]$ module load openmpi
[joeuser@vs-login-1 example]$ make
mpif90 -Wno-argument-mismatch -g -fopenmp fhostone.f90  -o fhostone 
rm getit.mod  mympi.mod  numz.mod
mpicc -g -fopenmp phostone.c -o phostone
[joeuser@vs-login-1 example]$ cat runopenmpi 
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=t
#SBATCH --time=00:01:00

cat $0

source /nopt/nrel/apps/210929a/myenv*
ml gcc   openmpi 

export OMP_NUM_THREADS=2
srun --mpi=pmi2 -n 2 ./fhostone -F
srun --mpi=pmi2 -n 2 ./phostone -F


[joeuser@vs-login-1 example]$ sbatch --account=MY_HPC_ACCOUNT runopenmpi 
Submitted batch job 50031771
[joeuser@vs-login-1 example]$ 
```

### Results

```
[joeuser@vs example]$ cat slurm-187.out
[joeuser@vs-login-1 example]$ cat slurm-50031771.out
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=t
#SBATCH --time=00:01:00

cat $0

source /nopt/nrel/apps/210929a/myenv*
ml gcc   openmpi 

export OMP_NUM_THREADS=2
srun --mpi=pmi2 -n 2 ./fhostone -F
srun --mpi=pmi2 -n 2 ./phostone -F

SRUN --mpi=pmi2 -n 2 ./fhostone -F

MPI Version:Open MPI v4.1.1, package: Open MPI joeuser@vs-sm-0001 Distribution, ident: 4.1.1, repo rev: v4.1.1, Apr 24, 2021
task    thread             node name  first task    # on node  core
0000      0000    vs-t-0012.vs.hpc.n        0000         0000   002
0000      0001    vs-t-0012.vs.hpc.n        0000         0000   003
0001      0000    vs-t-0013.vs.hpc.n        0001         0000   003
0001      0001    vs-t-0013.vs.hpc.n        0001         0000   002
SRUN --mpi=pmi2 -n 2 ./phostone -F

MPI VERSION Open MPI v4.1.1, package: Open MPI joeuser@vs-sm-0001 Distribution, ident: 4.1.1, repo rev: v4.1.1, Apr 24, 2021
task    thread             node name  first task    # on node  core
0000      0000    vs-t-0012.vs.hpc.nrel.gov        0000         0000  0003
0000      0001    vs-t-0012.vs.hpc.nrel.gov        0000         0000  0002
0001      0000    vs-t-0013.vs.hpc.nrel.gov        0001         0000  0003
0001      0001    vs-t-0013.vs.hpc.nrel.gov        0001         0000  0000
[joeuser@vs-login-1 example]$ 

```


Many programs can be built/run with OpenMPI and with icc/ifort as the backend compilers or built/run with the Intel version of MPI with either gcc/gfortran or icc/ifort as the backend compilers.  These options are discussed below.

## Building with Intel Fortran or Intel C and OpenMPI


You can build parallel programs using OpenMPI and the Intel Fortran *ifort* and Intel C *icc* compilers.



If you want to use the Intel compilers you first do a module load.

```
ml intel-oneapi-compilers
```

Then we can set the variables *OMPI_FC=ifort* and *OMPI_CC=icc*.  Then recompile.

```
[joeuser@vs example]$ export OMPI_FC=ifort
[joeuser@vs example]$ export OMPI_CC=icc
[joeuser@vs example]$ mpif90 -fopenmp fhostone.f90 -o fhostone
[joeuser@vs example]$ mpicc -fopenmp phostone.c -o phostone

```

If you do a *ls -l* on the executable files you will note the size of the files change with different compiler versions.  You can also see the difference by running the commands

```
nm fhostone | grep intel | wc
nm phostone | grep intel | wc
```

on the two versions of the program.  It will show how many calls to Intel routines are in each, 51 and 36 compared to 0 for the gnu versions.


## Building and Running with Intel MPI

We can build with the Intel versions of MPI and with icc and ifort as the backend compilers.  We load the modules:

```
ml gcc
ml intel-oneapi-compilers
ml intel-oneapi-mpi
```

Then, building and running the same example as above:

```
make clean
make PFC=mpiifort PCC=mpiicc
```

The actual compile lines produced by make are:

```
mpiifort -g -fopenmp fhostone.f90  -o fhostone 
mpiicc   -g -fopenmp phostone.c    -o phostone
```

For running, we need to make some changes to our batch script.  Replace the load of openmpi with:

```
ml intel-oneapi-compilers
ml intel-oneapi-mpi
```

Launch with the srun command:

```

srun --mpi=pmi2  ./a.out -F

```

Our IntelMPI batch script is:


```
[joeuser@vs-login-1 example]$ cat runintel 
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=lg
#SBATCH --time=00:01:00

cat $0

source /nopt/nrel/apps/210929a/myenv*
ml intel-oneapi-mpi intel-oneapi-compilers gcc

export OMP_NUM_THREADS=2
srun --mpi=pmi2 -n 2 ./fhostone -F
srun --mpi=pmi2 -n 2 ./phostone -F



```

With output

```
MPI Version:Intel(R) MPI Library 2021.3 for Linux* OS

task    thread             node name  first task    # on node  core
0000      0000                 c1-32        0000         0000   127
0000      0001                 c1-32        0000         0000   097
0001      0000                 c1-32        0000         0001   062
0001      0001                 c1-32        0000         0001   099

MPI VERSION Intel(R) MPI Library 2021.3 for Linux* OS

task    thread             node name  first task    # on node  core
0000      0000                 c1-32        0000         0000  0127
0000      0001                 c1-32        0000         0000  0097
0001      0000                 c1-32        0000         0001  0127
0001      0001                 c1-32        0000         0001  0099
```

## Linking Intel's MKL library.

The environment defined by sourcing the file /nopt/nrel/apps/210929a/myenv.2110041605
enables loading of many other modules, including one for Intel's MKL 
library. Then to build against MKL using the Intel compilers
icc or ifort you normally just need to add the flag **-mkl**.

There are examples in the directory /nopt/nrel/apps/210929a/example/mkl.
There is a Readme.md file that explains in a bit more detail.

Assuming you copied the example directory to you home directory the mkl examples will be in ~example/mkl

The short version is that you can:

```
[joeuser@vs-login-1 mkl]$ cd ~/example/mkl
[joeuser@vs-login-1 mkl]$ source /nopt/nrel/apps/210929a/myenv.2110041605
[joeuser@vs-login-1 mkl]$ module purge
[joeuser@vs-login-1 mkl]$ module load intel-oneapi-compilers
[joeuser@vs-login-1 mkl]$ module load intel-oneapi-mkl
[joeuser@vs-login-1 mkl]$ module load gcc

[joeuser@vs-login-1 mkl]$ icc   -O3 -o mklc mkl.c   -mkl
[joeuser@vs-login-1 mkl]$ ifort -O3 -o mklf mkl.f90 -mkl

```
or to build and run the examples using make instead directly calling icc and ifort you can:

```
make run
```


## Running VASP on Vermilion
Please see the [VASP page](../../Applications/vasp.md) for detailed information and recommendations for running VASP on Vermilion. 





