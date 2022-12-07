---
 layout: default
 title: Running on Vermillion
 parent: Vermillion
 grand_parent: Systems
---

The page [Modules](./modules.md) discuses how to activate and use the modules on Vermilion. Modules are not available by default and must be activated.  Please see the [Modules](./modules.md) page for more information about setting up your environment and loading modules. 

# Running on Vermilion

This page discusses the compute nodes, partitions and gives some examples of building and running applications including running Vasp.


## Compute hosts
Vermilion is a collection of physical nodes with each regular node containing Dual AMD EPYC 7532 Rome CPUs.  However, each node is virtualized.  That is it is split up into virtual nodes with each virtual node having a portion of the cores and memory of the physical node.  Similar virtual nodes are then assigned slurm partitions as shown below.  

## Shared file systems

Vermilion's home directories are shared across all nodes.  There is also /scratch/$USER and /projects spaces seen across all nodes.

## Partitions

Partitions are flexible and fluid on Vermilion.  A list of partitions can be found by running the `sinfo` command.  Here are the partitions as of 10/20/2022.

| Partition Name                          | Qty | RAM    | Cores/node | /var/scratch <br>1K-blocks |
| :--:                               | --: | --:    | --:             | --:   |                    
| gpu<br>*1 x NVIDIA Tesla A100*      |  5  | 114 GB |   30            |  6,240,805,336|        
| lg                                 | 18  | 229 GB |   60            |   1,031,070,000| 
| std                                | 62  | 114 GB |   30            |     515,010,816| 
| sm                                 | 31  |  61 GB |   16            |     256,981,000| 
| t                                  | 15  |  16 GB |   4             |      61,665,000| 

## Operating Software
The Vermilion HPC cluster runs fairly current versions of OpenHPC and SLURM on top of OpenStack.


## Example
Environments are provided with a number of commonly used compilers, common build tools, specific optimized libraries, and some analysis tools. Environments must be enabled before modules can be seen.  This is discussed in detail on the page [Modules](./modules.md)

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

Here is a sample batch script, *runopenmpi*, for running the hello world examples .  **NOTE: You must build the applications before running this script.**   Please see **Building hello world first** below.

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

# Running VASP on Vermilion

A few different versions of VASP are available on Vermilion:
- [VASP 5 (Intel MPI)](#Running-VASP-5-with-IntelMPI-on-CPUs)
- [VASP 6 (Intel MPI)](#Running-VASP-6-with-IntelMPI-on-CPUs)
- [VASP 6 (Open MPI)](#Running-VASP-6-with-OpenMPI-on-CPUs)
- [VASP 6 on GPUs](#Running-VASP-6-on-GPUs)

Running VASP with Open MPI shows a small improvement compared to running with Intel MPI, and running VASP on GPUs shows an even larger improvement.

VASP runs faster on 1 node than on 2 nodes. In some cases, VASP run times on 2 nodes have been observed to be double (or more) the run times on a single node. Many issues have been reported for running VASP on multiple nodes. In order for MPI to work successfully on Vermilion, it is necessary to specify the interconnect network that Vermilion should use to communicate between nodes. This is documented in each of the scripts below. The documented recommendations for setting the interconnect network have been shown to work well for multi-node jobs on 2 nodes, but aren't guaranteed to produce succesful multi-node runs on 4 nodes. 

If many cores are needed for your VASP calcualtion, it is recommended to run VASP on a singe node in the lg partition (60 cores/node), which provides the largest numbers of cores per node. 

## Running VASP 5 with IntelMPI on CPUs

To load a build of VASP 5 that is compatible with Intel MPI (and other necessary modules):

```
module use  /nopt/nrel/apps/220525b/level01/modules/lmod/linux-rocky8-x86_64/gcc/12.1.0/
ml vasp/5.5.4
ml intel-oneapi-mkl
ml intel-oneapi-compilers
ml intel-oneapi-mpi
```

This will give you:

```
[myuser@vs example]$ which vasp_gam
/nopt/nrel/apps/220525b/level01/install/opt/spack/linux-rocky8-zen2/gcc-12.1.0/vasp544/bin/vasp_gam
[myuser@vs example]$ which vasp_ncl
/nopt/nrel/apps/220525b/level01/install/opt/spack/linux-rocky8-zen2/gcc-12.1.0/vasp544/bin/vasp_ncl
[myuser@vs example]$ which vasp_std
/nopt/nrel/apps/220525b/level01/install/opt/spack/linux-rocky8-zen2/gcc-12.1.0/vasp544/bin/vasp_std
```

Note the directory might be different. 

In order to run on more than one node, we need to specify the network interconnect. To do so, use mpirun instead of srun. We want to use "ens7" as the interconnect. The mpirun command looks like this. 

```
I_MPI_OFI_PROVIDER=tcp mpirun -iface ens7 -np 16 vasp_std
```

For VASP calculations on a single node, srun is sufficient. However, srun and mpirun produce similar run times. To run with srun for single node calculations, use the following line.

```
srun -n 16 vasp_std
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
module use  /nopt/nrel/apps/220525b/level01/modules/lmod/linux-rocky8-x86_64/gcc/12.1.0/
ml vasp/5.5.4
ml intel-oneapi-mkl
ml intel-oneapi-compilers
ml intel-oneapi-mpi

# some extra lines that have been shown to improve VASP reliability on Vermilion
ulimit -s unlimited
export UCX_TLS=tcp,self
export OMP_NUM_THREADS=1

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

# mpirun is recommended (necessary for multi-node calculations)
I_MPI_OFI_PROVIDER=tcp mpirun -iface ens7 -np 16 vasp_std

# srun can be used instead of mpirun for sinlge node calculations
# srun -n 16 vasp_std

```

## Running VASP 6 with IntelMPI on CPUs

To load a build of VASP 6 that is compatible with Intel MPI (and other necessary modules):

```
source /nopt/nrel/apps/210929a/myenv.2110041605 
ml vaspintel
ml intel-oneapi-mkl
ml intel-oneapi-compilers
ml intel-oneapi-mpi
```

This will give you:

```
[myuser@vs example]$ which vasp_gam
/nopt/nrel/apps/210929a/level01/linux-centos8-zen2/gcc-9.4.0/vaspintel-1.0-dwljo4wr6xcrgxqaq7pz35yqfxdxxsq4/bin/vasp_gam
[myuser@vs example]$ which vasp_ncl
/nopt/nrel/apps/210929a/level01/linux-centos8-zen2/gcc-9.4.0/vaspintel-1.0-dwljo4wr6xcrgxqaq7pz35yqfxdxxsq4/bin/vasp_ncl
[myuser@vs example]$ which vasp_std
/nopt/nrel/apps/210929a/level01/linux-centos8-zen2/gcc-9.4.0/vaspintel-1.0-dwljo4wr6xcrgxqaq7pz35yqfxdxxsq4/bin/vasp_std
```

Note the directory might be different. 

In order to run on more than one node, we need to specify the network interconnect. To do so, use mpirun instead of srun. We want to use "ens7" as the interconnect. The mpirun command looks like this. 

```
I_MPI_OFI_PROVIDER=tcp mpirun -iface ens7 -np 16 vasp_std
```

For VASP calculations on a single node, srun is sufficient. However, srun and mpirun produce similar run times. To run with srun for single node calculations, use the following line.

```
srun -n 16 vasp_std
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
source /nopt/nrel/apps/210929a/myenv.2110041605 
ml intel-oneapi-mkl
ml intel-oneapi-compilers
ml intel-oneapi-mpi
ml vaspintel

# some extra lines that have been shown to improve VASP reliability on Vermilion
ulimit -s unlimited
export UCX_TLS=tcp,self
export OMP_NUM_THREADS=1

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

# mpirun is recommended (necessary for multi-node calculations)
I_MPI_OFI_PROVIDER=tcp mpirun -iface ens7 -np 16 vasp_std

# srun can be used instead of mpirun for sinlge node calculations
# srun -n 16 vasp_std

```

## Running VASP 6 with OpenMPI on CPUs

To load a build of VASP 6 that is compatible with Open MPI:

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

In order to specify the network interconnect, we need to set the OMPI_MCA_param variable. We want to use "ens7" as the interconnect.

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

# some extra lines that have been shown to improve VASP reliability on Vermilion
ulimit -s unlimited
export UCX_TLS=tcp,self
export OMP_NUM_THREADS=1

# lines to set "ens7" as the interconnect network
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

## Running VASP 6 on GPUs

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

mpirun -npernode 1 vasp_std > vasp.$SLURM_JOB_ID.

```


