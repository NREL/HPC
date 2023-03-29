# MPI Frameworks


## OpenMPI

**Documentation:** [OpenMPI](https://www.open-mpi.org)

*The Open MPI Project is an open source Message Passing Interface implementation that is developed and maintained by a consortium of academic, research, and industry partners. Open MPI is therefore able to combine the expertise, technologies, and resources from all across the High Performance Computing community in order to build the best MPI library available. Open MPI offers advantages for system and software vendors, application developers and computer science researchers.*

The Open MPI framework is a free and open-source communications library that is commonly developed against by many programmers. As an open-source package with strong academic support, the latest ideas may appear as implementations here prior to commercial MPI libraries.

Note that the Slurm-integrated builds of OpenMPI do not create the mpirun or mpiexec wrapper scripts that you may be used to. Ideally you should use srun (to take advantage of Slurm integration), but you can also use OpenMPI's native job launcher orterun. Some have also had success simply symlinking mpirun to orterun.

OpenMPI implements two Byte Transfer Layers for data transport between ranks in the same physical memory space: `sm` and `vader`. 
Both use a memory-mapped file, which by default is placed in `/tmp`. 
On Eagle, the node-local `/tmp` filesystem is quite small, and it is easy to fill this and crash or hang your job. 
Non-default locations of this file may be set through the `OMPI_TMPDIR` environment variable.

    *If you are running only a few ranks per node with modest buffer space requirements, consider setting `OMPI_TMPDIR` to `/dev/shm` in your job script.

    *If you are running many nodes per rank, you should set i`OMPI_TMPDIR` to `/tmp/scratch`, which holds at least 1 TB depending on Eagle node type.


### Supported Versions

| Eagle                                | Swift          | Vermilion |
|:------------------------------------:|:--------------:|:----------------:|
openmpi/1.10.7/gcc-8.4.0               |openmpi/4.1.1-6vr2flz |openmpi/4.1.4-gcc |   
openmpi/3.1.6/gcc-8.4.0                |                |                  |
openmpi/4.0.4/gcc-8.4.0                |                |                  |
openmpi/4.1.1/gcc+cuda                 |||
openmpi/4.1.2/gcc                      |||
openmpi/4.1.2/intel                    |||
openmpi/4.1.3/gcc-11.3.0-cuda-11.7     |||
openmpi/4.1.0/gcc-8.4.0                |||


## IntelMpi
**Documentation:** [IntelMPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html)

*Intel® MPI Library is a multifabric message-passing library that implements the open source MPICH specification. Use the library to create, maintain, and test advanced, complex applications that perform better on HPC clusters based on Intel® and compatible processors.*

Intel's MPI library enables tight interoperability with its processors and software development framework, and is a solid choice for most HPC applications.

### Supported Versions

| Eagle                                | Swift          | Vermilion |
|:------------------------------------:|:--------------:|:----------------:|
|intel-mpi/2020.1.217                  |intel-oneapi-mpi/2021.3.0-hcp2lkf  |intel-oneapi-mpi/2021.7.1-intel |   


## MPT
**Documentation:** [MPT](https://support.hpe.com/hpesc/public/docDisplay?docId=a00105727en_us&docLocale=en_US)

*HPE's Message Passing Interface (MPI) is a component of the HPE Message Passing Toolkit (MPT), a software package that supports parallel programming across a network of computer systems through a technique known as message passing.*

Hewlett-Packard Enterprise (HPE)—Eagle's creator—offers a very performant MPI library as well, built on top of and colloquially known via its underlying Message Passing Toolkit high-performance communications component as "MPT."
### Supported Versions

| Eagle                                | Swift          | Vermilion |
|:------------------------------------:|:--------------:|:----------------:|
|mpt/2.23                              |                |                  |   
|mpt/2.22                              |                |                  |


## Mpich
**Documentation:** [Mpich](https://www.mpich.org)

*MPICH is a high performance and widely portable implementation of the Message Passing Interface (MPI) standard. 
MPICH and its derivatives form the most widely used implementations of MPI in the world. They are used exclusively on nine of the top 10 supercomputers (June 2016 ranking), including the world’s fastest supercomputer: Taihu Light.*

### Supported Versions

| Eagle                                | Swift          | Vermilion |
|:------------------------------------:|:--------------:|:----------------:|
|                                      |mpich/3.4.2-h2s5tru | mpich/4.0.2-gcc  |   

## Running MPI Jobs on Eagle GPUs

To run MPI (message-passing interface) jobs on the Eagle system's NVidia GPUs, the MPI library must be "CUDA-aware."
A suitable OpenMPI build has been made available via the openmpi/4.0.4/gcc+cuda module. 
This module is currently in test.

### Interactive Use

srun does not work with this OpenMPI build when running interactively, so please use orterun instead. 
However, OpenMPI is cognizant of the Slurm environment, so one should request the resources needed via srun (for example, the number of available "slots" is determined by the number of tasks requested via srun). 
Ranks are mapped round-robin to the GPUs on a node. 
nvidia-smi shows, for example,

Processes:                                                              

|  GPU   |     PID |Type |  Process name                      |GPU Memory Usage|
|:------:|:-------:|:---:|:----------------------------------:|:-------------:|
|    0   |  24625  |    C|   ./jacobi                            |         803MiB |
|    0   |  24627  |    C|   ./jacobi                            |         803MiB |
|    1   |  24626  |    C|   ./jacobi                            |         803MiB |

when oversubscribing 3 ranks onto the 2 GPUs via the commands

```bash
srun --nodes=1 --ntasks-per-node=3 --account=<allocation_id> --time=10:00 --gres=gpu:2 --pty $SHELL
...<getting node>...
orterun -np 3 ./jacobi
```

If more ranks are desired than were originally requested via srun, the OpenMPI flag --oversubscribe could be added to the orterun command.

Batch Use
An example batch script to run 4 MPI ranks across two nodes is as follows.

```bash 
#!/bin/bash --login
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=2:00
#SBATCH --gres=gpu:2
#SBATCH --job-name=GPU_MPItest
#SBATCH --account=<allocation_id>
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out

ml use -a /nopt/nrel/apps/modules/test/modulefiles
ml gcc/8.4.0 cuda/10.2.89 openmpi/4.0.4/gcc+cuda

cd $SLURM_SUBMIT_DIR
srun ./jacobi
```

### Multi-Process Service

To run multiple ranks per GPU, you may find it beneficial to run NVidia's Multi-Process Service. This process management service can increase GPU utilization, reduce on-GPU storage requirements, and reduce context switching. To do so, include the following functionality in your Slurm script or interactive session:

### MPS setup

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/scratch/nvidia-mps
if [ -d $CUDA_MPS_PIPE_DIRECTORY ]
then
   rm -rf $CUDA_MPS_PIPE_DIRECTORY
fi
mkdir $CUDA_MPS_PIPE_DIRECTORY

export CUDA_MPS_LOG_DIRECTORY=/tmp/scratch/nvidia-log
if [ -d $CUDA_MPS_LOG_DIRECTORY ]
then
   rm -rf $CUDA_MPS_LOG_DIRECTORY
fi
mkdir $CUDA_MPS_LOG_DIRECTORY

# Start user-space daemon
nvidia-cuda-mps-control -d

# Run OpenMPI job.
orterun ...

# To clean up afterward, shut down daemon, remove directories, and unset variables
echo quit | nvidia-cuda-mps-control
for i in `env | grep CUDA_MPS | sed 's/=.*//'`; do rm -rf ${!i}; unset $i; done
```

For more information on MPS, see the [NVidia guide](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf).
