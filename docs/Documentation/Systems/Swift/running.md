--
layout: default
title: Running on Swift
parent: Swift
grand_parent: Systems
---

# Running on Swift
Please see the [Modules](./modules.md) page for information about setting up your environment and loading modules. 

**There are currently a number of known issues on Swift please check [Known issues](./known.md) for a complete list**

## Login nodes

```
swift.hpc.nrel.gov
swift-login-1.hpc.nrel.gov
swift-login-2.hpc.nrel.gov
```

## Slurm and Partitions

As more of Swift is brought on line different partitions will be created. A list of partitions can be returned by sunning the `sinfo` command.  




## Example 
Environments are provided with a number of commonly used modules including compilers, common build tools, specific AMD optimized libraries, and some analysis tools. The environments are in date stamped subdirectories under in the directory /nopt/nrel/apps.  Each environment directory has a file myenv.\*.   Sourcing that file will enable the environment.

When you login you will have access to the default environments and the *myenv*
file will have been sourced for you.  You can see the directory containing the environment by runnint the `module avail` command.  

In the directory for an environment you will see a subdirectory **example**.  This contains a makefile for a simple hello world program written in both Fortran and C.  The README.md file contains additional information, most of which is replicated here.  It is suggested you

```
cp -r example ~/example
cd ~/example
```

## Simple batch script

Here is a sample batch script for running the hello world examples *runopenmpi*. 


```bash
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
```

To run this you must first ensure that slurm is in your path by running:

```
module load slurm
```

Then 

```
sbatch --partition=test runopenmpi
```

## Building hello world first

Obviously for the script given above to work you must first build the application.  You need to:

2. Load the modules
3. make



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
#SBATCH --partition=debug
#SBATCH --time=00:01:00


cat $0

#These should be loaded before doing a make
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

We have the example programs build with gnu compilers and OpenMP using  the lines:

```
[nrmc2l@swift-login-1 ~ example]$ mpif90 -fopenmp fhostone.f90 -o fhostone
[nrmc2l@swift-login-1 ~ example]$ mpicc -fopenmp phostone.c -o phostone
```

This gives us:

```
[nrmc2l@swift-login-1 ~ example]$ ls -l fhostone
-rwxrwxr-x. 1 nrmc2l nrmc2l 36880 Jul 30 13:36 fhostone
[nrmc2l@swift-login-1 ~ example]$ ls -l phostone
-rwxrwxr-x. 1 nrmc2l nrmc2l 27536 Jul 30 13:36 phostone

```
Note the size of the executable files.  

If you want to use the Intel compilers you first do a module load.

```
module load intel-oneapi-mpi intel-oneapi-compilers gcc
```

Then we can set the variables *OMPI_FC=ifort* and *OMPI_CC=icc*.  Then recompile.

```bash
[nrmc2l@swift-login-1 ~ example]$ export OMPI_FC=ifort
[nrmc2l@swift-login-1 ~ example]$ export OMPI_CC=icc
[nrmc2l@swift-login-1 ~ example]$ mpif90 -fopenmp fhostone.f90 -o fhostone
[nrmc2l@swift-login-1 ~ example]$ mpicc -fopenmp phostone.c -o phostone


[nrmc2l@swift-login-1 ~ example]$ ls -lt fhostone
-rwxrwxr-x. 1 nrmc2l nrmc2l 951448 Jul 30 13:37 fhostone
[nrmc2l@swift-login-1 ~ example]$ ls -lt phostone
-rwxrwxr-x. 1 nrmc2l nrmc2l 155856 Jul 30 13:37 phostone
[nrmc2l@swift-login-1 ~ example]$ 
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
[nrmc2l@swift-login-1 example]$ ls -lt fhostone phostone
-rwxrwxr-x. 1 nrmc2l hpcapps 155696 Aug  5 16:14 phostone
-rwxrwxr-x. 1 nrmc2l hpcapps 947112 Aug  5 16:14 fhostone
[nrmc2l@swift-login-1 example]$ 
```

We need to make some changes to our batch script.  Replace the module load line with :

```
module load intel-oneapi-mpi intel-oneapi-compilers gcc

```

Launch with the srun command:

```
srun   ./a.out -F
```

Our IntelMPI batch script is:


```bash
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --time=00:01:00


cat $0

#These should be loaded before doing a make
module load intel-oneapi-mpi intel-oneapi-compilers gcc

export OMP_NUM_THREADS=2
srun  -n 4 ./fhostone -F
srun  -n 4 ./phostone -F


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


## Running VASP

The batch script given above can be modified to run VASP.  You need to add

```
ml vasp
```

This will give you:

```bash

[nrmc2l@swift-login-1 ~ example]$ which vasp_gam
/nopt/nrel/apps/210728a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_gam
[nrmc2l@swift-login-1 ~ example]$ which vasp_ncl
/nopt/nrel/apps/210728a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_ncl
[nrmc2l@swift-login-1 ~ example]$ which vasp_std
/nopt/nrel/apps/210728a/level02/gcc-9.4.0/vasp-6.1.1/bin/vasp_std
[nrmc2l@swift-login-1 ~ example]$ 
```

Note the directory might be different.

Then you need to add calls in your script to set up / point do your data files.  So your final script will look something like the following. Here we use data downloaded from NREL's benchmark repository.



```bash
#!/bin/bash
#SBATCH --job-name=b2_4
#SBATCH --nodes=1
#SBATCH --time=4:00:00
##SBATCH --error=std.err
##SBATCH --output=std.out
#SBATCH --partition=debug
#SBATCH --exclusive

cat $0

hostname

module purge
ml openmpi gcc vasp 

#### get input and set it up
#### This is from an old benchmark test
#### see https://github.nrel.gov/ESIF-Benchmarks/VASP/tree/master/bench2

mkdir $SLURM_JOB_ID
cp input/* $SLURM_JOB_ID
cd $SLURM_JOB_ID



srun   -n 16 vasp_std > vasp.$SLURM_JOB_ID

```
This will run a version of Vasp built with openmpi and gfortran/gcc.  You can run a version of Vasp built with the Intel toolchain replacing the *ml* line with

 ```ml vaspintel intel-oneapi-mpi intel-oneapi-compilers intel-oneapi-mkl```


## Running Jupyter / Jupyter-lab

Jupyter and Jupyter-lab are available by loading the module "python"

```

[nrmc2l@swift-login-1 ~]$ ml python
[nrmc2l@swift-login-1 ~]$ which python
/nopt/nrel/apps/210928a/level00/gcc-9.4.0/python-3.10.0/bin/python
[nrmc2l@swift-login-1 ~]$ which jupyter
/nopt/nrel/apps/210928a/level00/gcc-9.4.0/python-3.10.0/bin/jupyter
[nrmc2l@swift-login-1 ~]$ which jupyter-lab
/nopt/nrel/apps/210928a/level00/gcc-9.4.0/python-3.10.0/bin/jupyter-lab
[nrmc2l@swift-login-1 ~]$ 
```

It is recomended that you use the --no-browser option and connect to your notebook from your desktop using a ssh tunnel and web browser.

On Swift enter the command and note the URLs.  

```
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
Note the *8888* in the URL it might be different.  On your desktop in a new terminal window enter the command:

```
ssh -t -L 8888:localhost:8888 swift-login-1.hpc.nrel.gov
```

replacing 8888 with the number in the URL if it is different.

Then in a web browser window past the URL.  This should bring up a new notebook.

## Running Jupyter / Jupyter-lab on a compute node

You can get an interactive session on a compute node by running the command

```
[nrmc2l@swift-login-1 ~]$ salloc  --account=hpcapps   --exclusive    --time=01:00:00   --ntasks=16           --nodes=1 --partition=debug
```

but replacing *hpcapps* with your account.  After you get an session module load python and run as shown above.

```
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

```
ssh -t -L 8888:localhost:8475 swift-login-1 ssh -L 8475:localhost:8888 c1-28
```

replacing *8888* with the value in the URL if needed and c1-28 with the name of the compute node on which you are running.  Then again paste the URL in a web browser.  You should get a notebook running on the compute node.


## Running Julia 

Julia is also available via a module.  

```
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
Julia can be run in a Jupyter notebook as discussed above.  However, before doing so you will need to run the following commands in each Julia version you are using.  

```
julia> using Pkg
julia> Pkg.add("IJulia")

```


See [https://datatofish.com/add-julia-to-jupyter/](https://datatofish.com/add-julia-to-jupyter/) more more information.

If you would like to install your own copy of Juila complete with Jupyter-lab contact Tim Kaiser **tkaiser2@nrel.gov** for a script to do so.

