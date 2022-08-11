---
 layout: default
 title: Running on Vermillion
 parent: Vermillion
 grand_parent: Systems
---

# Running on Vermilion

Please see the [Modules](./modules.md) page for information about setting up your environment and loading modules. This page uses the environment at /nopt/nrel/apps/210929a/myenv.2110041605 as a placeholder since this environment is guaranteed to work, but if a newer environment and set of modules exists, it will be documented in the [Modules](./modules.md) page.


## Partitions

Partitions are flexible and fluid on Vermilion.  A list of partitions can be returned by running the `sinfo` command.
Currently, all systems are connected via bonded 25GbE (50Gb combined) with OFED/RDMA installed.

| Part Name | Qty | RAM    | processors/node | /opt/scratch | Description     |
| :--:      | --: | --:    | --:   | --:   | :--                    |
| GPU       |  5  | 114 GB |   1   |       | Dual NVIDIA Tesla V100s @ 40 GBs |
| lg        | 18  | 229 GB |   60  |       |      |
| std       | 62  | 114 GB |   30  |       |      |
| sm        | 31  |  61 GB |   16  |       |      |
| t         | 15  |  16 GB |   4   |       |      |

## Operating Software
The Vermilion HPC cluster runs fairly current versions of OpenHPC and SLURM on top of OpenStack.

## Hardware

### Compute hosts
Each host runs Dual AMD EPYC 7532 Rome CPUs, 64 cores per host. Large nodes use all the CPUs except those held back for systems operations.

### GPU nodes
5 nodes Single A100


## Example
Environments are provided with a number of commonly used modules including compilers, common build tools, specific AMD optimized libraries, and some analysis tools. The environments are in date stamped subdirectory under in the directory /nopt/nrel/apps.  Each environment directory has a file myenv.\*.   Sourcing that file will enable the environment.

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
#SBATCH --partition=t
#SBATCH --time=00:01:00


cat $0

source /nopt/nrel/apps/210929a/myenv.2110041605
module load slurm

ml gcc   openmpi

export OMP_NUM_THREADS=2
srun     -n 2 ./fhostone.I -F
srun     -n 2 ./phostone.I -F
```

To run this you must first ensure that slurm is in your path by running:

```
source /nopt/nrel/apps/210929a/myenv.2110041605
module load slurm
```

Then

```
sbatch --partition=sm runopenmpi
```

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
ml gcc openmpi
```

#### Run make

```
make
```

## Full procedure

```
[myuser@~]$ cd /nopt/nrel/apps/210929a
[myuser@vs:/nopt/nrel/apps/210929a]$ cp -r example ~/example
[myuser@vs:]$ cd ~/example

[myuser@vs:~/example]$ cat runopenmpi
    #!/usr/bin/env bash
    #SBATCH --job-name="install"
    #SBATCH --nodes=1
    #SBATCH --exclusive
    #SBATCH --partition=t
    #SBATCH --time=00:01:00


cat $0

source /nopt/nrel/apps/210929a/myenv.2110041605
module load slurm

ml gcc   openmpi

export OMP_NUM_THREADS=2
mpirun -n 2 ./fhostone -F
mpirun -n 2 ./phostone -F

[myuser@vs example]$ PATH=/nopt/nrel/slurm/bin:$PATH
[myuser@vs example]$ source /nopt/nrel/apps/210929a/myenv*
[myuser@vs example]$ ml gcc   openmpi
[myuser@vs example]$ make

mpif90 -fopenmp fhostone.f90 -o fhostone
rm getit.mod  mympi.mod  numz.mod
mpicc -fopenmp phostone.c -o phostone
[myuser@vs example]$ sbatch --partition=t runopenmpi
Submitted batch job 187
[myuser@vs example]$
```

### Results

```
[myuser@vs example]$ cat slurm-187.out
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=t
#SBATCH --time=00:01:00


cat $0

PATH=/nopt/nrel/slurm/bin:$PATH

source /nopt/nrel/apps/210929a/myenv*
ml gcc   openmpi

export OMP_NUM_THREADS=2
mpirun -n 2 ./fhostone -F
mpirun -n 2 ./phostone -F

MPI Version:Open MPI v4.1.1, package: Open MPI myuser@c1-32 Distribution, ident: 4.1.1, repo rev: v4.1.1, Apr 24, 2021
task    thread             node name  first task    # on node  core
0000      0000                 c1-32        0000         0000   000
0000      0001                 c1-32        0000         0000   064
0001      0000                 c1-32        0000         0001   065
0001      0001                 c1-32        0000         0001   001
MPI VERSION Open MPI v4.1.1, package: Open MPI myuser@c1-32 Distribution, ident: 4.1.1, repo rev: v4.1.1, Apr 24, 2021
task    thread             node name  first task    # on node  core
0000      0000                 c1-32        0000         0000  0000
0000      0001                 c1-32        0000         0000  0064
0001      0001                 c1-32        0000         0001  0065
0001      0000                 c1-32        0000         0001  0001
[myuser@vs example]$

```

## Building with Intel Fortran or Intel C and OpenMPI

You can build parallel programs using OpenMPI and the Intel Fortran *ifort* and Intel C *icc* compilers.

We have the example programs build with gnu compilers and OpenMP using  the lines:

```
[myuser@vs example]$ mpif90 -fopenmp fhostone.f90 -o fhostone
[myuser@vs example]$ mpicc -fopenmp phostone.c -o phostone
```

This gives us:

```
[myuser@vs example]$ ls -l fhostone
-rwxrwxr-x. 1 myuser myuser 36880 Jul 30 13:36 fhostone
[myuser@vs example]$ ls -l phostone
-rwxrwxr-x. 1 myuser myuser 27536 Jul 30 13:36 phostone

```
Note the size of the executable files.

If you want to use the Intel compilers you first do a module load.

```
ml intel-oneapi-compilers
```

Then we can set the variables *OMPI_FC=ifort* and *OMPI_CC=icc*.  Then recompile.

```
[myuser@vs example]$ export OMPI_FC=ifort
[myuser@vs example]$ export OMPI_CC=icc
[myuser@vs example]$ mpif90 -fopenmp fhostone.f90 -o fhostone
[myuser@vs example]$ mpicc -fopenmp phostone.c -o phostone


[myuser@vs example]$ ls -lt fhostone
-rwxrwxr-x. 1 myuser myuser 951448 Jul 30 13:37 fhostone
[myuser@vs example]$ ls -lt phostone
-rwxrwxr-x. 1 myuser myuser 155856 Jul 30 13:37 phostone
[myuser@vs example]$
```

Note the size of the executable files have changed.  You can also see the difference by running the commands

```
nm fhostone | grep intel | wc
nm phostone | grep intel | wc
```

on the two versions of the program.  It will show how many calls to Intel routines are in each, 51 and 36 compared to 0 for the gnu versions.


## Building and Running with Intel MPI

We can build with the Intel versions of MPI.  We assume we will want to build with icc and ifort as the backend compilers.  We load the modules:

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

Giving us:

```
[myuser@swift-login-1 example]$ ls -lt fhostone phostone
-rwxrwxr-x. 1 myuser hpcapps 155696 Aug  5 16:14 phostone
-rwxrwxr-x. 1 myuser hpcapps 947112 Aug  5 16:14 fhostone
[myuser@swift-login-1 example]$
```

We need to make some changes to our batch script.  Add the lines:

```
ml intel-oneapi-compilers
ml intel-oneapi-mpi
export I_MPI_PMI_LIBRARY=/nopt/nrel/apps/210929a/level01/gcc-9.4.0/slurm-21-08-1-1/lib/libpmi2.so
```

Launch with the srun command:

```

srun --mpi=pmi2  ./a.out -F

```

Our IntelMPI batch script is:


```
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=t
#SBATCH --time=00:01:00


cat $0

PATH=/nopt/nrel/slurm/bin:$PATH
source /nopt/nrel/apps/210929a/myenv*
ml intel-oneapi-mpi intel-oneapi-compilers gcc
export I_MPI_PMI_LIBRARY=/nopt/nrel/apps/210929a/level01/gcc-9.4.0/slurm-21-08-1-1/lib/libpmi2.so


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


## Running VASP on CPUs

The batch script given above can be modified to run VASP. VASP with Open MPI is recommended.

To load a build of VASP that is compatible with Open MPI:

```
source /nopt/nrel/apps/210929a/myenv.2110041605
ml vasp
```

This will give you:

```
[myuser@vs example]$ which vasp_gam
/nopt/nrel/apps/123456a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_gam
[myuser@vs example]$ which vasp_ncl
/nopt/nrel/apps/123456a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_ncl
[myuser@vs example]$ which vasp_std
/nopt/nrel/apps/123456a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_std
```

Note the directory might be different.

Issues have been reported running VASP on multiple nodes. The most reliable solution is to use a different build of the openmpi module and set the OMPI_MCA_param variable, as shown below. This configuration has shown good results up to 4 nodes, but is not guaranteed to make VASP run successfully on 8 nodes. 

```
module use /nopt/nrel/apps/220525b/level01/modules/lmod/linux-rocky8-x86_64/gcc/12.1.0
module load openmpi
OMPI_MCA_param="btl_tcp_if_include ens7"
```

Then you need to add calls in your script to set up and point to your data files.  So your final script will look something like the following. Here we download data from NREL's benchmark repository.

```
#!/bin/bash
#SBATCH --job-name=vasp
#SBATCH --nodes=1
#SBATCH --time=8:00:00
##SBATCH --error=std.err
##SBATCH --output=std.out
#SBATCH --partition=sm
#SBATCH --exclusive

cat $0

hostname

source /nopt/nrel/apps/210929a/myenv.2110041605

module purge
ml gcc
ml vasp

module use /nopt/nrel/apps/220525b/level01/modules/lmod/linux-rocky8-x86_64/gcc/12.1.0
module load openmpi
OMPI_MCA_param="btl_tcp_if_include ens7"

#### wget is needed to download data
ml wget

#### get input and set it up
#### This is from an old benchmark test
#### see https://github.nrel.gov/ESIF-Benchmarks/VASP/tree/master/bench2


mkdir input

wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/INCAR?token=AAAALJZRV4QFFTS7RC6LLGLBBV67M   -q -O INCAR
wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/POTCAR?token=AAAALJ6E7KHVTGWQMR4RKYTBBV7SC  -q -O POTCAR
wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/POSCAR?token=AAAALJ5WKM2QKC3D44SXIQTBBV7P2  -q -O POSCAR
wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/KPOINTS?token=AAAALJ5YTSCJFDHUUZMZY63BBV7NU -q -O KPOINTS

srun --mpi=pmi2 -n 16 vasp_std

```

## Running VASP on GPUs

VASP can also be run on Vermilion's GPUs. To do this we need to add a few #SBATCH lines at the top of the script to assign the job to run in the gpu partition and to set the gpu binding. The --gpu-bind flag requires 1 set of "0,1" for each node used. 

```
#SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --gpu-bind=map_gpu:0,1,0,1
```

A gpu build of VASP can be accessed by adding the following path to your PATH variable.

```
export PATH=/projects/hpcapps/tkaiser2/vasp/6.3.1/nvhpc_acc:$PATH
```

This will give you:

```
[myuser@vs example]$ which vasp_gam
/projects/hpcapps/tkaiser2/vasp/6.3.1/nvhpc_acc/vasp_gam
[myuser@vs example]$ which vasp_ncl
/projects/hpcapps/tkaiser2/vasp/6.3.1/nvhpc_acc/vasp_ncl
[myuser@vs example]$ which vasp_std
/projects/hpcapps/tkaiser2/vasp/6.3.1/nvhpc_acc/vasp_std
```

Instead of srun, use mpirun to run VASP on GPUs. Since Vermilion only has 1 GPU per node, it's important to make sure you are only requesting 1 task per node by setting -npernode 1. 

```
mpirun -npernode 1 vasp_std > vasp.$SLURM_JOB_ID
```

There's a few more modules needed to run VASP on GPUs, and two library variables need to be set. We can modify the VASP CPU script to include lines to load the modules, set library variables and make the changes outlined above. The final script will look something like this.

```
#!/bin/bash
#SBATCH --job-name=vasp
#SBATCH --nodes=2
#SBATCH --time=1:00:00
##SBATCH --error=std.err
##SBATCH --output=std.out
#SBATCH --partition=gpu
#SBATCH --gpu-bind=map_gpu:0,1,0,1
#SBATCH --exclusive

cat $0

hostname

#load necessary modules and set library paths
module use  /nopt/nrel/apps/220421a/modules/lmod/linux-rocky8-x86_64/gcc/11.3.0/
ml nvhpc
ml gcc
ml fftw
export LD_LIBRARY_PATH=/nopt/nrel/apps//220421a/install/opt/spack/linux-rocky8-zen2/gcc-11.3.0/nvhpc-22.2-ruzrtpyewnnrif6s7w7rehvpk7jimdrd/Linux_x86_64/22.2/compilers/extras/qd/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/nopt/nrel/apps//220421a/install/opt/spack/linux-rocky8-zen2/gcc-11.3.0/gcc-11.3.0-c3u46uvtuljfuqimb4bgywoz6oynridg/lib64:$LD_LIBRARY_PATH

#add a path to the gpu build of VASP to your script
export PATH=/projects/hpcapps/tkaiser2/vasp/6.3.1/nvhpc_acc:$PATH

#### wget is needed to download data
ml wget

#### get input and set it up
#### This is from an old benchmark test
#### see https://github.nrel.gov/ESIF-Benchmarks/VASP/tree/master/bench2


mkdir input

wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/INCAR?token=AAAALJZRV4QFFTS7RC6LLGLBBV67M   -q -O INCAR
wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/POTCAR?token=AAAALJ6E7KHVTGWQMR4RKYTBBV7SC  -q -O POTCAR
wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/POSCAR?token=AAAALJ5WKM2QKC3D44SXIQTBBV7P2  -q -O POSCAR
wget https://github.nrel.gov/raw/ESIF-Benchmarks/VASP/master/bench2/input/KPOINTS?token=AAAALJ5YTSCJFDHUUZMZY63BBV7NU -q -O KPOINTS

mpirun -npernode 1 vasp_std > vasp.$SLURM_JOB_ID

```
