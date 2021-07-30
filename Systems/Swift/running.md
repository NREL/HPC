---
layout: default
title: Running on Swift
parent: Swift
grand_parent: Systems
---

# Running on Swift
Please see the section "Modules" for information about setting up your environment, loading modules, and current know limitations.  

## Example 
Environments are provided with a number of commonly used modules including compilers, common build tools, specific AMD optimized libraries, and some analysis tools. The environments are in date stamped subdirectory under in the directory /nopt/nrel/apps.  Each environemnt directory has a file myenv.\*.   Sourcing that file will enable the environment.

In the directory for an environment you will see a subdirectory **example**.  This contains a makefile for a simple hello world program written in both Fortran and C.  The README.md file contains additional information, most of which is replicated here.  It is suggested you

```
cp -r example ~/example
cd ~/example
```

## Simple batch script

Here is a sample batch script for running the hello world examples *runopenmpi*. 


```
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --time=00:01:00


cat $0

PATH=/nopt/nrel/slurm/bin:$PATH

source /nopt/nrel/apps/210729a/myenv*
ml gcc   openmpi

export OMP_NUM_THREADS=2
mpirun -n 2 ./fhostone -F
mpirun -n 2 ./phostone -F
```

To run this you must first ensure that slurm is in your path by running:

```
PATH=/nopt/nrel/slurm/bin:$PATH
```

Then 

```
sbatch --partition=test runopenmpi
```

## Building hello world first

Obviously for the script given above to work you must first build the application.  You need to:

1. Load the environment
2. Load the modules
3. make

#### Loading the environment
Loading the environemnt is just a matter of sourcing the file 

```
source /nopt/nrel/apps/210729a/myenv*
```

Note that **210729** is a date stamp showing when the environment was built.  You may have a different value as environments evolve.


#### Loading the modules.

We are going to use gnu compilers with OpenMPI.

```
ml gcc openmpi
```

#### Run make

```
make
```

## Full procedure

```
[tkaiser2@eaglet 210729a]$ cd /nopt/nrel/apps/210729a
[tkaiser2@eaglet 210729a]$ cp -r example ~/example
[tkaiser2@eaglet 210729a]$ cd ~/example
[tkaiser2@eaglet example]$ cat runopenmpi 
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --time=00:01:00


cat $0

PATH=/nopt/nrel/slurm/bin:$PATH

source /nopt/nrel/apps/210729a/myenv*
ml gcc   openmpi

export OMP_NUM_THREADS=2
mpirun -n 2 ./fhostone -F
mpirun -n 2 ./phostone -F

[tkaiser2@eaglet example]$ PATH=/nopt/nrel/slurm/bin:$PATH
[tkaiser2@eaglet example]$ source /nopt/nrel/apps/210729a/myenv*
[tkaiser2@eaglet example]$ ml gcc   openmpi
[tkaiser2@eaglet example]$ make
mpif90 -fopenmp fhostone.f90 -o fhostone
rm getit.mod  mympi.mod  numz.mod
mpicc -fopenmp phostone.c -o phostone
[tkaiser2@eaglet example]$ sbatch --partition=test runopenmpi
Submitted batch job 187
[tkaiser2@eaglet example]$ 
```

### Results

```
[tkaiser2@eaglet example]$ cat slurm-187.out 
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --time=00:01:00


cat $0

PATH=/nopt/nrel/slurm/bin:$PATH

source /nopt/nrel/apps/210729a/myenv*
ml gcc   openmpi

export OMP_NUM_THREADS=2
mpirun -n 2 ./fhostone -F
mpirun -n 2 ./phostone -F

MPI Version:Open MPI v4.1.1, package: Open MPI tkaiser2@c1-32 Distribution, ident: 4.1.1, repo rev: v4.1.1, Apr 24, 2021
task    thread             node name  first task    # on node  core
0000      0000                 c1-32        0000         0000   000
0000      0001                 c1-32        0000         0000   064
0001      0000                 c1-32        0000         0001   065
0001      0001                 c1-32        0000         0001   001
MPI VERSION Open MPI v4.1.1, package: Open MPI tkaiser2@c1-32 Distribution, ident: 4.1.1, repo rev: v4.1.1, Apr 24, 2021
task    thread             node name  first task    # on node  core
0000      0000                 c1-32        0000         0000  0000
0000      0001                 c1-32        0000         0000  0064
0001      0001                 c1-32        0000         0001  0065
0001      0000                 c1-32        0000         0001  0001
[tkaiser2@eaglet example]$ 

```

## Building with Intel Fortran or Intel C and OpenMPI

You can build parallel programs using OpenMPI and the Intel Fortran *ifort* and Intel C *icc* compilers.

We have the example programs build with gnu compilers and OpenMP using  the lines:

```
[tkaiser2@eaglet example]$ mpif90 -fopenmp fhostone.f90 -o fhostone
[tkaiser2@eaglet example]$ mpicc -fopenmp phostone.c -o phostone
```

This gives us:

```
[tkaiser2@eaglet example]$ ls -l fhostone
-rwxrwxr-x. 1 tkaiser2 tkaiser2 36880 Jul 30 13:36 fhostone
[tkaiser2@eaglet example]$ ls -l phostone
-rwxrwxr-x. 1 tkaiser2 tkaiser2 27536 Jul 30 13:36 phostone

```
Note the size of the executable files.  

Then we can set the variables *OMPI_FC=ifort* and *OMPI_CC=icc*.  Then recompile.

```
[tkaiser2@eaglet example]$ export OMPI_FC=ifort
[tkaiser2@eaglet example]$ export OMPI_CC=icc
[tkaiser2@eaglet example]$ mpif90 -fopenmp fhostone.f90 -o fhostone
[tkaiser2@eaglet example]$ mpicc -fopenmp phostone.c -o phostone


[tkaiser2@eaglet example]$ ls -lt fhostone
-rwxrwxr-x. 1 tkaiser2 tkaiser2 951448 Jul 30 13:37 fhostone
[tkaiser2@eaglet example]$ ls -lt phostone
-rwxrwxr-x. 1 tkaiser2 tkaiser2 155856 Jul 30 13:37 phostone
[tkaiser2@eaglet example]$ 
```

Note the size of the executable files have changed.  You can also see the difference by running the commands

```
nm fhostone | grep intel | wc
nm phostone | grep intel | wc
```

on the two versions of the program.  It will show how many calls to Intel routines are in each, 51 and 36 compared to 0.



## Running VASP

The batch script given above can be modifed to run VASP.  You need to add

```
ml vasp
```

This will give you:

```

[tkaiser2@eaglet example]$ which vasp_gam
/nopt/nrel/apps/210729a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_gam
[tkaiser2@eaglet example]$ which vasp_ncl
/nopt/nrel/apps/210729a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_ncl
[tkaiser2@eaglet example]$ which vasp_std
/nopt/nrel/apps/210729a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_std
[tkaiser2@eaglet example]$ 
```

Note the directory might be different.

Then you need to add calls in your script to set up / point do your data files.  So your final script will look something like the following. Here we download data from NREL's benchmark repository.



```
#!/bin/bash
#SBATCH --job-name=b2.4
#SBATCH --nodes=1
#SBATCH --time=8:00:00
##SBATCH --error=std.err
##SBATCH --output=std.out
#SBATCH --partition=test
#SBATCH --exclusive

cat $0

hostname

PATH=/nopt/nrel/slurm/bin:$PATH

module purge
source /nopt/nrel/apps/210728a/myenv*
ml openmpi gcc
ml vasp

#### wget is needed to download data
ml wget

#### get input and set it up
#### No warranty about the data or setup
#### This is from an old benchmark test

input_path=input
rm -rf $input_path
mkdir $input_path

wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/INCAR?token=AAAALJZRV4QFFTS7RC6LLGLBBV67M   -q -O input/INCAR
wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/POTCAR?token=AAAALJ6E7KHVTGWQMR4RKYTBBV7SC  -q -O input/POTCAR
wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/POSCAR?token=AAAALJ5WKM2QKC3D44SXIQTBBV7P2  -q -O input/POSCAR
wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/KPOINTS?token=AAAALJ5YTSCJFDHUUZMZY63BBV7NU -q -O input/KPOINTS

for dir in 00 01 02 03 ; do
  rm -rf $dir
  mkdir $dir
  cp  $input_path/POSCAR $dir
done

cp $input_path/KPOINTS  .
cp $input_path/INCAR  .
cp $input_path/POTCAR  .

export OMP_NUM_THREADS=4

mpirun  -n 16 vasp_std 

```


