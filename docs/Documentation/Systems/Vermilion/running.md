---
 layout: default
 title: Running on Vermilion
 parent: Vermilion
 grand_parent: Systems
---

# Running on Vermilion

*This page discusses the compute nodes, partitions and gives some examples of building and running applications including running Vasp.*

The page [Modules](./modules.md) discuses how to activate and use the modules on Vermilion. Modules are not available by default and must be activated.  Please see the [Modules](./modules.md) page for more information about setting up your environment and loading modules. 
## Compute hosts
Vermilion is a collection of physical nodes with each regular node containing Dual AMD EPYC 7532 Rome CPUs.  However, each node is virtualized.  That is it is split up into virtual nodes with each virtual node having a portion of the cores and memory of the physical node.  Similar virtual nodes are then assigned slurm partitions as shown below.  

## Shared file systems

Vermilion's home directories are shared across all nodes.  There is also /scratch/$USER and /projects spaces seen across all nodes.

## Partitions

Partitions are flexible and fluid on Vermilion.  A list of partitions can be found by running the `sinfo` command.  Here are the partitions as of 10/20/2022.

| Partition Name                          | Qty | RAM    | Cores/node | /var/scratch <br>1K-blocks |
| :--:                               | --: | --:    | --:             | --:   |                    
| gpu<br>*1 x NVIDIA Tesla A100*      |  17  | 114 GB |   30            |  6,240,805,336|        
| lg                                 | 39  | 229 GB |   60            |   1,031,070,000| 
| std                                | 60  | 114 GB |   30            |     515,010,816| 
| sm                                 | 28  |  61 GB |   16            |     256,981,000| 
| t                                  | 15  |  16 GB |   4             |      61,665,000| 

## Operating Software
The Vermilion HPC cluster runs fairly current versions of OpenHPC and SLURM on top of OpenStack.


## Examples: Build and run simple applications

This section discusses how to compile and run a simple MPI application, as well as how to link against the Intel MKL library.

Environments are provided with a number of commonly used compilers, common build tools, specific optimized libraries, and some analysis tools. Environments must be enabled before modules can be seen.  This is discussed in detail on the page [Modules](./modules.md)

You can use the "standard" environment by running the command:

```
source /nopt/nrel/apps/210929a/myenv.2110041605
```

The examples on this page uses the environment enabled by this command.   You may want to add this command to your `.bashrc` file so you have a useful environment when you login.

In the directory **/nopt/nrel/apps/210929a** you will see a subdirectory **example**.  This contains a makefile for a simple hello world program written in both Fortran and C and several run scripts. The README.md file contains additional information, some of which is replicated here. 

We will begin by creating a new directory and copying the source for a simple MPI test program.  More details about the test program are available in the README.md file that accompanies it.  Run the following commands to create a new directory and make a copy of the source code:

```bash
mkdir example
cd example
cp /nopt/nrel/apps/210929a/example/phostone.c .
```

### Compile and run with Intel MPI

First we will look at how to compile and run the application using Intel MPI.  To build the application, we load the necessary Intel modules.  Execute the following commands to load the modules and build the application, naming the output `phost.intelmpi`.  Note that this application uses OpenMP as well as MPI, so we provide the `-fopenmp` flag to link against the OpenMP libraries.

```bash
source /nopt/nrel/apps/210929a/myenv.2110041605
ml intel-oneapi-mpi intel-oneapi-compilers gcc
mpiicc -fopenmp phostone.c -o phost.intelmpi
```

The following batch script is an example that runs the job using two MPI ranks across two nodes and two threads per rank.  Save this script to a file such as `submit_intel.sh`, replace `<myaccount>` with the appropriate account, and submit using `sbatch submit_intel.sh`.  Feel free to experiment with different numbers of tasks, threads, and nodes.  Note that multi-node jobs on Vermilion can be finicky, and applications may not scale as well as they do on other systems.  If you experience problems with a multi-node job, start by first making sure that you application can run on a single node.


??? example "Intel MPI submission script"

    ```bash
    #!/bin/bash
    #SBATCH --nodes=2
    #SBATCH --exclusive
    #SBATCH --time=00:01:00
    #SBATCH --account=<myaccount>

    source /nopt/nrel/apps/210929a/myenv.2110041605
    ml intel-oneapi-mpi intel-oneapi-compilers gcc

    export OMP_NUM_THREADS=2
    export I_MPI_OFI_PROVIDER=tcp
    srun --mpi=pmi2 --cpus-per-task 2 -n 2 ./phost.intelmpi -F
    ```
    
Your output should look similar to the following:

```
MPI VERSION Intel(R) MPI Library 2021.9 for Linux* OS

task    thread             node name  first task    # on node  core
0000      0000           vs-std-0001        0000         0000  0001
0000      0001           vs-std-0001        0000         0000  0000
0001      0000           vs-std-0002        0001         0000  0001
0001      0001           vs-std-0002        0001         0000  0000
```

### Link Intel's MKL library

The environment defined by sourcing the file `/nopt/nrel/apps/210929a/myenv.2110041605`
enables loading of many other modules, including one for Intel's MKL
library. Then to build against MKL using the Intel compilers
icc or ifort you normally just need to add the flag `-qmkl`.

There are examples in the directory `/nopt/nrel/apps/210929a/example/mkl`.
There is a Readme.md file that explains in a bit more detail.

To compile a simple test program that links against MKL, run:

```bash
cp /nopt/nrel/apps/210929a/example/mkl/mkl.c .

source /nopt/nrel/apps/210929a/myenv.2110041605
ml intel-oneapi-mkl intel-oneapi-compilers gcc
icc -O3 -qmkl mkl.c -o mkl
```

An example submission script is:

??? example "Intel MKL submission script"

    ```bash
    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --exclusive
    #SBATCH --time=00:01:00
    #SBATCH --account=<myaccount>

    source /nopt/nrel/apps/210929a/myenv.2110041605
    ml intel-oneapi-mkl intel-oneapi-compilers gcc

    ./mkl
    ```


### Compile and run with Open MPI

Please note that multi-node Open MPI jobs are not currently functioning properly.  If running on multiple nodes is needed, it is advised to use Intel MPI or try to run your jobs on a different system.

Use the following commands to load the Open MPI modules and compile the test program into an executable named `phost.openmpi`:

```bash
source /nopt/nrel/apps/210929a/myenv.2110041605
ml gcc openmpi
mpicc -fopenmp phostone.c -o phost.openmpi
```

The following is an example script that runs two tasks on a single node, with two threads per task:


??? example "Open MPI submission script"

    ```bash
    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --exclusive
    #SBATCH --time=00:01:00
    #SBATCH --account=<myaccount>

    source /nopt/nrel/apps/210929a/myenv.2110041605
    ml gcc openmpi

    export OMP_NUM_THREADS=2
    mpirun -np 2 --map-by socket:PE=2 ./phost.openmpi -F
    ```



## Running VASP on Vermilion

The batch script given above can be modified to run VASP.

There are actually several builds of Vasp on Vermilion, including builds of VASP 5 and VASP 6.  There are scripts for running them in the directory /nopt/nrel/apps/210929a/example/vasp.  Some of these version use different environments from the one discussed above.  The script example/vasp/runvasp_4 will runs the a GPU enabled version of Vasp on 4 Vermilion GPU nodes.  This version of vasp needs to be launched using mpirun instead of srun.  

The run times and additional information can be found in the file /nopt/nrel/apps/210929a/example/vasp/versions.  The run on the GPU nodes is considerably faster than the CPU node runs.  

The data set for these runs is from a standard NREL vasp benchmark. See [https://github.nrel.gov/ESIF-Benchmarks/VASP/tree/master/bench2]() This is a system of 519 atoms (Ag504C4H10S1).

There is a NREL report that discuss running the this test case and also a smaller test case with with various setting of nodes, tasks-per-nodes and OMP_NUM_THREADS.  It can be found at: [https://github.com/NREL/HPC/tree/master/applications/vasp/Performance%20Study%202](https://github.com/NREL/HPC/tree/master/applications/vasp/Performance%20Study%202)

### Running multi-node VASP jobs on Vermilion

VASP runs faster on 1 node than on 2 nodes. In some cases, VASP runtimes on 2 nodes have been observed to be double (or more) the run times on a single node. Many issues have been reported for running VASP on multiple nodes, especially when requesting all available cores in each node. In order for MPI to work reliably on Vermilion, it is necessary to specify the interconnect network that Vermilion should use to communicate between nodes. This is documented in each of the scripts below. Different solutions exists for Open MPI and Intel MPI. The documented recommendations for setting the interconnect network have been shown to work well for multi-node jobs on 2 nodes, but aren't guaranteed to produce succesful multi-node runs on 4 nodes. 

If many cores are needed for your VASP calcualtion, it is recommended to run VASP on a singe node in the lg partition (60 cores/node), which provides the largest numbers of cores per node. 

### Setting up VASP sbatch scripts

The following sections walk through building sbatch scripts for running VASP on Vermilion, including explanations of necessary tweaks to run multi-node jobs reliably. 

- [VASP 5 (Intel MPI)](#running-vasp-5-with-intelmpi-on-cpus)
- [VASP 6 (Intel MPI)](#running-vasp-6-with-intelmpi-on-cpus)
- [VASP 6 (Open MPI)](#running-vasp-6-with-openmpi-on-cpus)
- [VASP 6 on GPUs](#running-vasp-6-on-gpus)

#### Running VASP 5 with IntelMPI on CPUs

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

#### Running VASP 6 with IntelMPI on CPUs

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

#### Running VASP 6 with OpenMPI on CPUs

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

#### Running VASP 6 on GPUs

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


