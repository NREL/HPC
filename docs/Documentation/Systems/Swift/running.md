---
layout: default
title: Running on Swift
parent: Swift
grand_parent: Systems
---

# Running on Swift

Please see the [Modules](./modules.md) page for information about setting up your environment and loading modules. 

## Login nodes

```
swift.hpc.nrel.gov
swift-login-1.hpc.nrel.gov
```

`swift.hpc.nrel.gov` is a round-robin alias that will connect you to any available login node.

## SSH Keys

User accounts have a default set of keys `cluster` and `cluster.pub`. The `config` file will use these even if you generate a new keypair using `ssh-keygen`. If you are adding your keys to Github or elsewhere you should either use `cluster.pub` or will have to modify the `config` file.

## Slurm and Partitions

The most up to date list of partitions can always be found by running the `sinfo` command on the cluster.

| Partition | Description |
|-----------|-------------|
| long      | jobs up to ten days of walltime |
| standard  | jobs up to two days of walltime | 
| gpu  |  Nodes with four NVIDIA A100 40 GB Computational Accelerators, up to two days of walltime |
| parallel  | optimized for large parallel jobs, up to two days of walltime |
| debug     | two nodes reserved for short tests, up to four hours of walltime |

Each partition also has a matching `-standby` partition. Allocations which have consumed all awarded AUs for the year may only submit jobs to these partitions, and their default QoS will be set to `standby`. Jobs in standby partitions will be scheduled when there are otherwise idle cycles and no other non-standby jobs are available. 

Any allocation may submit a job to a standby QoS, even if there are unspent AUs.

By default, nodes can be shared between users.  To get exclusive access to a node use the `--exclusive` flag in your sbatch script or on the sbatch command line.

!!! tip "Important"
    Use `--cpus-per-task` with srun/sbatch otherwise some applications may only utilize a single core. This behavior differs from Eagle.

## Allocation Unit (AU) Charges

The equation for calculating the AU cost of a job on Swift is:

`AU cost = (Walltime in hours * Number of Nodes * QoS Factor * Charge Factor)`

The **Walltime** is the actual length of time that the job runs, in hours or fractions thereof.

The **Number of nodes** can be whole nodes or fractions of a node. See below for more information.

The **Charge Factor** for Swift CPU nodes is **5**. 

The **Charge Factor** for Swift GPU nodes is **50**, or **12.5 per GPU**.

The **QoS Factor** for *normal priority* jobs is **1**. 

The **QoS Factor** for *high-priority* jobs is **2**.

The **QoS Factor** for *standby priority* jobs is **0**. There is no AU cost for standby jobs.

One CPU node for one hour of walltime at *normal priority* costs **5 AU** total.

One CPU node for one hour of walltime at *high priority* costs **10 AU** total.

One GPU for one hour of walltime at *normal priority* costs **12.5 AU** total.

Four GPUs for one hour of walltime at *normal priority* costs **50 AU** total.

### Shared/Fractional CPU Nodes

Swift allows jobs to share nodes, meaning fractional allocations are possible. 

Standard (CPU) compute nodes have 128 CPU cores and 256GB RAM.

When a job only requests part of a node, usage is tracked on the basis of: 

1 core = 2GB RAM = 1/128th of a node

Using all resources on a single node, whether CPU, RAM, or both, will max out at 128/128 per node = 1.

**The highest quantity of resource requested will determine the total AU charge.**

For example, a job that requests 64 cores and 128GB RAM (one half of a node) would be: 

1 hour walltime * 0.5 nodes * 1 QoS Factor * 5 Charge Factor = **2.5** AU per node-hour.

### Shared/Fractional GPU Nodes

Jobs on Swift may also share GPU nodes.

Standard GPU nodes have 96 CPU cores, four NVIDIA A100 40GB GPUs, and 1TB RAM.

You may request 1, 2, 3, or 4 GPUs per GPU node, as well as any additional CPU and RAM required. 

Usage is tracked on the basis of: 

1 GPU = 25% of total cores (24/96) = 25% of total RAM (256GB/1TB) = 25% of a node 

**The highest quantity of resource requested will determine the total AU charge.**

For example, a request of 1 GPU, up to 24 CPU cores, and up to 256GB RAM will be charged at 12.5 AU/hr.

A request of 1 GPU, 48 CPU cores, and 100GB RAM will be charged at 25 AU/hr:

* 1/4 GPUs = 25% total GPUs = 50 AU * 0.25 = 12.5 AU (ignored)
* 48/96 cores = 50% total cores = 50 AU * 0.5 = **25 AU** (this is what will be charged)
* 100GB/1TB = 10% total RAM = 50 AU * 0.10 = 5 AU (ignored)

A request of 2 GPUs, 55 CPU cores, and 200GB RAM will be charged at approximately 28.7 AU/hr:

* 2/4 GPUs = 0.5 * 50 = 25 AU (ignored)
* 55/96 cores ~= 57.3% of total cores, 50 * .573 = **28.65 AU** (this is what will be charged)
* 200GB/1TB = 0.2 * 50 = 10 AU (ignored)

RAM usage may be calculated in a similar fashion.


## Software Environments and Example Files

Multiple software environments are available on Swift, with a number of commonly used modules including compilers, common build tools, specific AMD optimized libraries, and some analysis tools. The environments are in date stamped subdirectories, in the directory /nopt/nrel/apps. Each environment directory has a file myenv.\*.  Sourcing that file will enable the environment.

When you login you will have access to the default environments and the *myenv* file will have been sourced for you. You can see the directory containing the environment by running the `module avail` command.  

In the directory for an environment you will see a subdirectory **example**. This contains a makefile for a simple hello world program written in both Fortran and C. The README.md file contains additional information, most of which is replicated here. It is suggested that you copy the example directory to your own /home for experimentation:

```
cp -r example ~/example
cd ~/example
```
#### Conda
There is a very basic version of conda in the "anaconda" directory in each  /nopt/nrel/apps/YYMMDDa directory. However, there is a more complete environment pointed to by the module under /nopt/nrel/apps/modules. This is set up like Eagle. Please see our [Conda Documentation](../../Environment/Customization/conda.md) for more information.

## Simple batch script

Here is a sample batch script for running the 'hello world' example program, *runopenmpi*. 


```bash
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --exclusive
#SBATCH --account=<myaccount>
#SBATCH --partition=debug
#SBATCH --time=00:01:00


cat $0

#These should be loaded before doing a make
module load gcc  openmpi 

export OMP_NUM_THREADS=2
srun  -n 4 ./fhostone -F
srun  -n 4 ./phostone -F
```

To run this you need to replace `<myaccount>` with the appropriate account and ensure that slurm is in your path by running:

```
module load slurm
```

Then submit the sbatch script with: 

```
sbatch --partition=test runopenmpi
```

## Building the 'hello world' example 

Obviously for the script given above to work you must first build the application. You need to:

1. Load the modules
2. make

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
```bash
[nrmc2l@swift-login-1 ~]$ cd ~
[nrmc2l@swift-login-1 ~]$ mkdir example
[nrmc2l@swift-login-1 ~]$ cd ~/example
[nrmc2l@swift-login-1 ~]$ cp -r /nopt/nrel/apps/210928a/example/* .

[nrmc2l@swift-login-1 ~ example]$ cat runopenmpi 
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --exclusive
#SBATCH --account=<myaccount>
#SBATCH --partition=debug
#SBATCH --time=00:01:00


cat $0

#These should be loaded before doing a make:
module load gcc  openmpi 

export OMP_NUM_THREADS=2
srun  -n 4 ./fhostone -F
srun  -n 4 ./phostone -F


[nrmc2l@swift-login-1 ~ example]$ module load gcc  openmpi
[nrmc2l@swift-login-1 ~ example]$ make
mpif90 -fopenmp fhostone.f90 -o fhostone
rm getit.mod  mympi.mod  numz.mod
mpicc -fopenmp phostone.c -o phostone
[nrmc2l@swift-login-1 ~ example]$ sbatch runopenmpi
Submitted batch job 187
[nrmc2l@swift-login-1 ~ example]$ 
```

### Results

```bash
[nrmc2l@swift-login-1 example]$ cat *312985*
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --time=00:01:00


cat $0

#These should be loaded before doing a make
module load gcc  openmpi 

export OMP_NUM_THREADS=2
srun  -n 4 ./fhostone -F
srun  -n 4 ./phostone -F

MPI Version:Open MPI v4.1.1, package: Open MPI nrmc2l@swift-login-1.swift.hpc.nrel.gov Distribution, ident: 4.1.1, repo rev: v4.1.1, Apr 24, 2021
task    thread             node name  first task    # on node  core
0002      0000                 c1-31        0002         0000   018
0000      0000                 c1-30        0000         0000   072
0000      0001                 c1-30        0000         0000   095
0001      0000                 c1-30        0000         0001   096
0001      0001                 c1-30        0000         0001   099
0002      0001                 c1-31        0002         0000   085
0003      0000                 c1-31        0002         0001   063
0003      0001                 c1-31        0002         0001   099
0001      0000                 c1-30        0000         0001  0097
0001      0001                 c1-30        0000         0001  0103
0003      0000                 c1-31        0002         0001  0062
0003      0001                 c1-31        0002         0001  0103
MPI VERSION Open MPI v4.1.1, package: Open MPI nrmc2l@swift-login-1.swift.hpc.nrel.gov Distribution, ident: 4.1.1, repo rev: v4.1.1, Apr 24, 2021
task    thread             node name  first task    # on node  core
0000      0000                 c1-30        0000         0000  0072
0000      0001                 c1-30        0000         0000  0020
0002      0000                 c1-31        0002         0000  0000
0002      0001                 c1-31        0002         0000  0067
[nrmc2l@swift-login-1 example]$ 
```

## Building with Intel Fortran or Intel C and OpenMPI

You can build parallel programs using OpenMPI and the Intel Fortran *ifort* and Intel C *icc* compilers.

We have the example programs build with gnu compilers and OpenMP using the lines:

```bash
[nrmc2l@swift-login-1 ~ example]$ mpif90 -fopenmp fhostone.f90 -o fhostone
[nrmc2l@swift-login-1 ~ example]$ mpicc -fopenmp phostone.c -o phostone
```

This gives us:

```bash
[nrmc2l@swift-login-1 ~ example]$ ls -l fhostone
-rwxrwxr-x. 1 nrmc2l nrmc2l 42128 Jul 30 13:36 fhostone
[nrmc2l@swift-login-1 ~ example]$ ls -l phostone
-rwxrwxr-x. 1 nrmc2l nrmc2l 32784 Jul 30 13:36 phostone

```
Note the size of the executable files.  

If you want to use the Intel compilers, first load the appropriate modules:

```bash
module load openmpi intel-oneapi-compilers gcc
```

Then we can set the variables *OMPI_FC=ifort* and *OMPI_CC=icc*, and recompile:

```bash
[nrmc2l@swift-login-1 ~ example]$ export OMPI_FC=ifort
[nrmc2l@swift-login-1 ~ example]$ export OMPI_CC=icc
[nrmc2l@swift-login-1 ~ example]$ mpif90 -fopenmp fhostone.f90 -o fhostone
[nrmc2l@swift-login-1 ~ example]$ mpicc -fopenmp phostone.c -o phostone


[nrmc2l@swift-login-1 ~ example]$ ls -lt fhostone
-rwxrwxr-x. 1 nrmc2l nrmc2l 41376 Jul 30 13:37 fhostone
[nrmc2l@swift-login-1 ~ example]$ ls -lt phostone
-rwxrwxr-x. 1 nrmc2l nrmc2l 32200 Jul 30 13:37 phostone
[nrmc2l@swift-login-1 ~ example]$ 
```

Note the size of the executable files have changed. You can also see the difference by running the commands:

```bash
nm fhostone | grep intel | wc
nm phostone | grep intel | wc
```

on the two versions of the program. It will show how many calls to Intel routines are in each, 51 and 36 compared to 0 for the gnu versions.


## Building and Running with Intel MPI

We can build with the Intel versions of MPI. We assume we will want to build with icc and ifort as the backend compilers. We load the modules:

```bash
ml gcc
ml intel-oneapi-compilers
ml intel-oneapi-mpi
```

Then, build and run the same example as above:

```bash
make clean
make PFC=mpiifort PCC=mpiicc 
```

Giving us:

```bash
[nrmc2l@swift-login-1 example]$ ls -lt fhostone phostone
-rwxrwxr-x. 1 nrmc2l hpcapps 160944 Aug  5 16:14 phostone
-rwxrwxr-x. 1 nrmc2l hpcapps 952352 Aug  5 16:14 fhostone
[nrmc2l@swift-login-1 example]$ 
```

We need to make some changes to our batch script.  Replace the module load line with:

```bash
module load intel-oneapi-mpi intel-oneapi-compilers gcc

```

Our IntelMPI batch script, *runintel* under */example*, is:


```bash
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --exclusive
#SBATCH --account=<myaccount>
#SBATCH --partition=debug
#SBATCH --time=00:01:00


cat $0

#These should be loaded before doing a make
module load intel-oneapi-mpi intel-oneapi-compilers gcc

export OMP_NUM_THREADS=2
srun  -n 4 ./fhostone -F
srun  -n 4 ./phostone -F


```

Which produces the following output:

```bash
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

## VASP, Jupyter, Julia, and Other Applications on Swift

Please see the relevant page in the [Applications](https://nrel.github.io/HPC/Documentation/Applications/) section for more information on using applications on Swift and other NREL clusters.

