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
| GPU       | jobs up to two days of walltime |
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

The **Charge Factor** for Swift is **5**. 

The **QoS Factor** for *normal priority* jobs is **1**. 

The **QoS Factor** for *high-priority* jobs is **2**.

The **QoS Factor** for *standby priority* jobs is **0**. There is no AU cost for standby jobs.

One node for one hour of walltime at *normal priority* costs **5 AU** total.

One node for one hour of walltime at *high priority* costs **10 AU** total.

### Fractional Nodes

Swift allows jobs to share nodes, meaning fractional allocations are possible. 

Standard compute nodes have 128 CPU cores and 256GB RAM.

When a job only requests part of a node, usage is tracked on the basis of: 

1 core = 2GB RAM = 1/128th of a node

Using all resources on a single node, whether CPU, RAM, or both, will max out at 128/128 per node = 1.

For example, a job that requests 64 cores and 128GB RAM (one half of a node) would be: 

1 hour walltime * 0.5 nodes * 1 QoS Factor * 5 Charge Factor = **2.5** AU per node-hour.

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

To run this you need replace `<myaccount>` with the appropriate account and ensure that slurm is in your path by running:

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
module load intel-oneapi-mpi intel-oneapi-compilers gcc
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

Launch with the srun command:

```bash
srun   ./a.out -F
```

Our IntelMPI batch script, *runintel* under */example*, is (remember to replace `<myaccount>` with the appropriate account):


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


## Running VASP

The batch script given above can be modified to run VASP. To do so, load the VASP module, as well:

```bash
ml vasp
```

This will give you:

```bash

[nrmc2l@swift-login-1 ~ example]$ which vasp_gam
/nopt/nrel/apps/210928a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_gam
[nrmc2l@swift-login-1 ~ example]$ which vasp_ncl
/nopt/nrel/apps/210928a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_ncl
[nrmc2l@swift-login-1 ~ example]$ which vasp_std
/nopt/nrel/apps/210928a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_std
[nrmc2l@swift-login-1 ~ example]$ 
```

Note the directory might be different.

Then you need to add calls in your script to set up / point do your data files. So your final script will look something like the following. Here we use data downloaded from NREL's benchmark repository and it is also included in the copied subdirectory *\example* named with *runvasp*:

```bash
#!/bin/bash
#SBATCH --job-name=b2_4
#SBATCH --nodes=1
#SBATCH --time=4:00:00
##SBATCH --error=std.err
##SBATCH --output=std.out
#SBATCH --account=<myaccount>
#SBATCH --partition=debug
#SBATCH --exclusive

cat $0

hostname

module purge
ml slurm openmpi gcc vasp 

#### get input and set it up
#### This is from an old benchmark test
#### see https://github.nrel.gov/ESIF-Benchmarks/VASP/tree/master/bench2

mkdir $SLURM_JOB_ID
cp input/* $SLURM_JOB_ID
cd $SLURM_JOB_ID



srun   -n 16 vasp_std > vasp.$SLURM_JOB_ID

```
This will run a version of Vasp built with openmpi and gfortran/gcc. You can run a version of Vasp built with the Intel toolchain replacing the *ml* line with the following module load as shown in *runvaspintel* under */example*:

 ```ml vaspintel intel-oneapi-mpi intel-oneapi-compilers intel-oneapi-mkl```


## Running Jupyter / Jupyter-lab

Jupyter and Jupyter-lab are available by loading the module "python/3.10.0-wwsaj4n" or "python/3.9.6-mydisst". If load "python/3.10.0-wwsaj4n":

```bash

[nrmc2l@swift-login-1 ~]$ ml python/3.10.0-wwsaj4n
[nrmc2l@swift-login-1 ~]$ which python
/nopt/nrel/apps/210928a/level00/gcc-9.4.0/python-3.10.0/bin/python
[nrmc2l@swift-login-1 ~]$ which jupyter
/nopt/nrel/apps/210928a/level00/gcc-9.4.0/python-3.10.0/bin/jupyter
[nrmc2l@swift-login-1 ~]$ which jupyter-lab
/nopt/nrel/apps/210928a/level00/gcc-9.4.0/python-3.10.0/bin/jupyter-lab
[nrmc2l@swift-login-1 ~]$ 
```

It is recommended that you use the --no-browser option and connect to your notebook from your desktop using a ssh tunnel and web browser.

On Swift enter the command below, and note the URLs in the output:  

```bash
[nrmc2l@swift-login-1 ~]$ jupyter-lab --no-browser
[I 2022-03-30 07:54:25.937 ServerApp] jupyterlab | extension was successfully linked.
[I 2022-03-30 07:54:26.224 ServerApp] nbclassic | extension was successfully linked.
[I 2022-03-30 07:54:26.255 ServerApp] nbclassic | extension was successfully loaded.
[I 2022-03-30 07:54:26.257 LabApp] JupyterLab extension loaded from /nopt/nrel/apps/210928a/level00/gcc-9.4.0/python-3.10.0/lib/python3.10/site-packages/jupyterlab
[I 2022-03-30 07:54:26.257 LabApp] JupyterLab application directory is /nopt/nrel/apps/210928a/level00/gcc-9.4.0/python-3.10.0/share/jupyter/lab
[I 2022-03-30 07:54:26.260 ServerApp] jupyterlab | extension was successfully loaded.
[I 2022-03-30 07:54:26.261 ServerApp] Serving notebooks from local directory: /home/nrmc2l
[I 2022-03-30 07:54:26.261 ServerApp] Jupyter Server 1.11.1 is running at:
[I 2022-03-30 07:54:26.261 ServerApp] http://localhost:8888/lab?token=183d33c61bb136f8d04b83c70c4257a976060dd84afc9156
[I 2022-03-30 07:54:26.261 ServerApp]  or http://127.0.0.1:8888/lab?token=183d33c61bb136f8d04b83c70c4257a976060dd84afc9156
[I 2022-03-30 07:54:26.261 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2022-03-30 07:54:26.266 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///home/nrmc2l/.local/share/jupyter/runtime/jpserver-2056000-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=183d33c61bb136f8d04b83c70c4257a976060dd84afc9156
     or http://127.0.0.1:8888/lab?token=183d33c61bb136f8d04b83c70c4257a976060dd84afc9156
```
Note the *8888* in the URL it might be different. On your desktop in a new terminal window enter the command:

```bash
ssh -t -L 8888:localhost:8888 swift-login-1.hpc.nrel.gov
```

replacing 8888 with the number in the URL if it is different.

Then in a web browser window, paste the URL to bring up a new notebook.

## Running Jupyter / Jupyter-lab on a compute node

You can get an interactive session on a compute node with the salloc command, as in the following example:

```bash
[nrmc2l@swift-login-1 ~]$ salloc  --account=hpcapps   --exclusive    --time=01:00:00   --ntasks=16           --nodes=1 --partition=debug
```

but replacing *hpcapps* with your account. After you get a session on a node, `module load python` and run as shown above.

```bash
[nrmc2l@swift-login-1 ~]$ salloc  --account=hpcapps   --exclusive    --time=01:00:00   --ntasks=16           --nodes=1 --partition=debug
salloc: Pending job allocation 313001
salloc: job 313001 queued and waiting for resources
salloc: job 313001 has been allocated resources
salloc: Granted job allocation 313001
[nrmc2l@c1-28 ~]$ 
[nrmc2l@c1-28 ~]$ module load python
[nrmc2l@c1-28 ~]$ 

[nrmc2l@c1-28 ~]$ jupyter-lab --no-browser
[I 2022-03-30 08:04:28.063 ServerApp] jupyterlab | extension was successfully linked.
[I 2022-03-30 08:04:28.468 ServerApp] nbclassic | extension was successfully linked.
[I 2022-03-30 08:04:28.508 ServerApp] nbclassic | extension was successfully loaded.
[I 2022-03-30 08:04:28.509 LabApp] JupyterLab extension loaded from /nopt/nrel/apps/210928a/level00/gcc-9.4.0/python-3.10.0/lib/python3.10/site-packages/jupyterlab
[I 2022-03-30 08:04:28.509 LabApp] JupyterLab application directory is /nopt/nrel/apps/210928a/level00/gcc-9.4.0/python-3.10.0/share/jupyter/lab
[I 2022-03-30 08:04:28.513 ServerApp] jupyterlab | extension was successfully loaded.
[I 2022-03-30 08:04:28.513 ServerApp] Serving notebooks from local directory: /home/nrmc2l
[I 2022-03-30 08:04:28.514 ServerApp] Jupyter Server 1.11.1 is running at:
[I 2022-03-30 08:04:28.514 ServerApp] http://localhost:8888/lab?token=cd101872959be54aea33082a8af350fc7e1484e47a9fdfbf
[I 2022-03-30 08:04:28.514 ServerApp]  or http://127.0.0.1:8888/lab?token=cd101872959be54aea33082a8af350fc7e1484e47a9fdfbf
[I 2022-03-30 08:04:28.514 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2022-03-30 08:04:28.519 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///home/nrmc2l/.local/share/jupyter/runtime/jpserver-3375148-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=cd101872959be54aea33082a8af350fc7e1484e47a9fdfbf
     or http://127.0.0.1:8888/lab?token=cd101872959be54aea33082a8af350fc7e1484e47a9fdfbf
```


On your desktop run the command:

```bash
ssh -t -L 8888:localhost:8475 swift-login-1 ssh -L 8475:localhost:8888 c1-28
```

replacing *8888* with the value in the URL if needed and c1-28 with the name of the compute node on which you are running. Then again paste the URL in a web browser. You should get a notebook running on the compute node.


## Running Julia 

Julia is also available via a module.  

```bash
[nrmc2l@swift-login-1:~ ] $ module spider julia
...
     Versions:
        julia/1.6.2-ocsfign
        julia/1.7.2-gdp7a25
...
[nrmc2l@swift-login-1:~ ] $ 

[nrmc2l@swift-login-1:~/examples/spack ] $ module load julia/1.7.2-gdp7a25 
[nrmc2l@swift-login-1:~/examples/spack ] $ which julia
/nopt/nrel/apps/210928a/level03/install/linux-rocky8-zen2/gcc-9.4.0/julia-1.7.2-gdp7a253nsglyzssybqknos2n5amkvqm/bin/julia
[nrmc2l@swift-login-1:~/examples/spack ] $ 

```
Julia can be run in a Jupyter notebook as discussed above. However, before doing so you will need to run the following commands in each Julia version you are using:  

```bash
julia> using Pkg
julia> Pkg.add("IJulia")

```

Please see [https://datatofish.com/add-julia-to-jupyter/](https://datatofish.com/add-julia-to-jupyter/) for more information.

If you would like to install your own copy of Julia complete with Jupyter-lab, contact Tim Kaiser **tkaiser2@nrel.gov** for a script to do so.

