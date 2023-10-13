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


## Software Environments

Environments are provided with a number of commonly used compilers, common build tools, specific optimized libraries, and some analysis tools. Environments must be enabled before modules can be seen.  This is discussed in detail on the page [Modules](./modules.md).

You can use the "standard" environment by running the command:

```
source /nopt/nrel/apps/210929a/myenv.2110041605
```

The examples on this page uses the environment enabled by this command.   You may want to add this command to your `.bashrc` file so you have a useful environment when you login.


## Examples: Build and run simple applications

This section discusses how to compile and run a simple MPI application, as well as how to link against the Intel MKL library.

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

Please note that multi-node jobs with Open MPI may not function properly.  If running on multiple nodes is needed, it is advised to use Intel MPI or try to run your jobs on a different system.

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

Please see the [VASP page](../../Applications/vasp.md) for detailed information and recommendations for running VASP on Vermilion.
