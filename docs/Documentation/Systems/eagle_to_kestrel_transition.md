---
title: Transitioning from Eagle to Kestrel
---

## Overview of Steps

This page is meant to provide all necessary information to transition a project from Eagle to Kestrel. Transitioning a project can be broken down into five steps:

1. Accessing Kestrel
2. Transferring Data from Eagle to Kestrel
3. Understanding the options for running your software on Kestrel

    a. How to check if your software is available as a module on Kestrel

    b. How to build your own software on Kestrel

    c. General environment recommendations

4. Submitting your jobs on Kestrel
5. Review performance recommendations if scalability or performance is worse than expected

If you find yourself stuck on any of the above steps, please reach out to [hpc-help@nrel.gov](mailto:hpc-help@nrel.gov) as soon as possible.

## 1. Accessing Kestrel

Access to Kestrel requires an NREL HPC account and access to an active project allocation on Kestrel. You can use [Lex](https://hpcprojects.nrel.gov/login/?next=/) to check your allocations. 

The steps to login to Kestrel are very similar to Eagle. For example, you can log into Kestrel by opening a terminal and ssh'ing to the machine:

```bash
# NREL Employees 
ssh <your username>@kestrel.hpc.nrel.gov

# External Collaborators 
ssh <your username>@kestrel.nrel.gov

```
For more detailed information on accessing Kestrel, please see [this page](./Kestrel/index.md). 

## 2. Transferring Data from Eagle to Kestrel


Please see our page on [transferring files](../Managing_Data/Transferring_Files/index.md) for detailed information. Essentially, you should use the command-line `rsync` tool for small transfers (<100 GB), and [Globus](../Managing_Data/Transferring_Files/globus.md) for large transfers. 

### Filesystems

Data storage polices and the filesystems layout on Kestrel is similar to Eagle. Kestrel has a **95 PB** ClusterStor Lustre file system. Unlike on Eagle, the Parallel Filesystem (PFS) consists of a ProjectFS and a ScratchFS which have different configurations. ScratchFS uses a Lustre file system in a hybrid flash-disk configuration providing a total of **27 petabytes** (PB) of capacity with **354 gigabytes (GB)/s** of IOR bandwidth. ProjectFS has **68 PB** of capacity with **200 GB/s** of IOR bandwidth. We advise running jobs out of `/scratch` and moving data to `/projects` for long term storage. Like on Eagle, `/scratch` will have a 28 day purge policy with no exceptions.

The Home File System (HFS) on Kestrel is part of the ClusterStor used for PFS, providing highly reliable storage for user home directories and NREL-specific software. HFS will provide 1.2 PB of capacity. Snapshots of files on the HFS will be available up to 30 days after change/deletion. `/home` directories have a quota of 50 GB. 


Please see the [Kestrel Filesystem page](./Kestrel/filesystems.md) for more information. 

## 3. Understanding the options for running your software on Kestrel

### How to check if your software is available as a module on Kestrel

If you are used to using your software as an NREL-maintained module on Eagle, first check the availability of that software on Kestrel:

`module avail [your software name]`

If nothing shows up, please email [hpc-help@nrel.gov](mailto:hpc-help@nrel.gov) to get the module set up on Kestrel.

If the module exists, then you simply need to `module load [your software name]`, the same as you would do on Eagle.

### How to build your own software on Kestrel

If you need to build your own software on Kestrel, and NOT use an already-existing module, then the steps can be a bit different than Eagle. For a general software-building procedure, please see our [Libraries How-To](../Development/Libraries/howto.md#summary-of-steps) tutorial.

In general, on Kestrel we recommend using the `PrgEnv-cray` or `PrgEnv-intel` environments to build your code. For detailed descriptions on these environments, see our [Environments](./Kestrel/Environments/index.md) page. For a tutorial walkthrough of building a simple code (IMB) within these environments, see our [Environments Tutorial](./Kestrel/Environments/tutorial.md) page. Note that `PrgEnv-` environments on Kestrel are different than environments on Eagle. Loading a `PrgEnv` loads a number of modules at once that together constitute a consistent environment. 

!!! danger
	OpenMPI currently does not work well on Kestrel, and thus it is **strongly** recommended to NOT use OpenMPI. If you require assistance in building your code with an MPI other than 		
    OpenMPI, please reach out to hpc-help@nrel.gov. The issue with OpenMPI is at the networking layer, and building your own OpenMPI will not fix the issue.

!!! tip
    Some MPI codes, especially old legacy scientific software, may be difficult to build with Cray MPICH. In these cases, if it is possible to build the code with Intel MPI or a different MPICH implementation, then Cray MPICH can be utilized at run-time via use of the `cray-mpich-abi` module (note that OpenMPI is *NOT* an implementation of MPICH, and you cannot use the `cray-mpich-abi` if you built with OpenMPI). A detailed example of building with Intel MPI but running with Cray MPICH can be found on our [VASP application page](../Applications/vasp.md). 

## 4. Running your Jobs on Kestrel

See our page on submitting jobs on Kestrel [here](./Kestrel/running.md).

Like Eagle, Kestrel uses the [Slurm job scheduler](../Slurm/index.md). If the application you need to run can be found under our [Applications tab](../Applications/index.md), then there may be example Kestrel submission scripts on the application page. Otherwise, our [VASP documentation page](../Applications/vasp.md#vasp-on-kestrel) contains a variety of sample submit scripts that you can modify to fit your own purposes.

For information on the Kestrel hardware configuration, see our [Kestrel System Configuration](https://www.nrel.gov/hpc/kestrel-system-configuration.html) page. One key difference from Eagle is that not all of the Kestrel nodes have a local disk. If you need local disk space, you will need to request that in your job submission script with the `--tmp` option. For more detailed information on this, please see [this page](./Kestrel/filesystems.md#node-file-system).


### Shared Partition

Note that each Kestrel standard CPU node contains 104 CPU cores and 256 GB memory. Some applications or application use-cases may not scale well to this many CPU cores. In these cases, it is recommended to submit your jobs to the shared partition. A job submitted to the shared partition  will be charged AUs proportionate to whichever resource you require more of, between CPUs and memory.

The following is an example shared-partition submission script:

```
#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --partition=shared         
#SBATCH --time=2:00:00    
#SBATCH --ntasks=26 # CPUs requested for job 
#SBATCH --mem-per-cpu=2000 # Request 2GB per core.
#SBATCH --account=<allocation handle>

cd /scratch/$USER 
srun ./my_progam # Use your application's commands here  
```

For more information on the shared partition and an example AU-accounting calculation, see [here](./Kestrel/running.md#shared-node-partition).

## 5. Performance Recommendations

### OpenMP

If you are running a code with OpenMP enabled, we recommend manually setting one of the following environment variables:

```
export OMP_PROC_BIND=spread # for non-intel built codes

export KMP_AFFINITY=balanced # for codes built with intel compilers
```

You may need to export these variables even if you are not running your job with threading, i.e., with `OMP_NUM_THREADS=1`

### MPI

Currently, some applications on Kestrel are not scaling with the expected performance. We are actively working with the vendor's engineers to resolve these issues. For now, for these applications, we have compiled a set of recommendations that may help with performance. Note that any given recommendation may or may not apply to your specific application. We strongly recommend conducting your own performance and scalability tests on your performance-critical codes.

1. Use Cray MPICH over OpenMPI or Intel MPI. If you need help rebuilding your code so that it uses Cray MPICH, please reach out to hpc-help@nrel.gov

2. For MPI collectives-heavy applications, setting the following environment variables (for Cray MPICH):
```
export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier 
export MPICH_COLL_OPT_OFF=mpi_allreduce 
```
These environment variables turn off some collective optimizations that we have seen can cause slowdowns. For more information on these environment variables, visit HPE's documentation site [here](https://cpe.ext.hpe.com/docs/mpt/mpich/intro_mpi_ucx.html).

4. For hybrid MPI/OpenMP codes, requesting more threads per task than you tend to request on Eagle. This may yield performance improvements.

6. ONLY if you are running on 10 or more nodes and are experiencing scalability issues, you can try half-packing the nodes you request, i.e., requesting 52 ranks per node instead of 104 ranks per node, then spreading these ranks evenly across the two sockets. This can be accomplished by including the following in your srun command:   
```
--ntasks-per-node=52 --distribution=cyclic:cyclic --cpu_bind=cores
```

Please note that all of these recommendations are subject to change as we continue to improve the system.

## Kestrel Release Notes

Release notes for Kestrel after major upgrades can be found [here](./Kestrel/kestrel_release_notes.md).

## Resources

1. [Accessing Kestrel](./Kestrel/index.md)
2. [Transferring Files between Filesystems on the NREL Network](../Managing_Data/Transferring_Files/index.md)
3. [Using Globus to move data from Eagle to Kestrel](../Managing_Data/Transferring_Files/globus.md)
4. [General software building tutorial](../Development/Libraries/howto.md)
5. [Environments Overview](./Kestrel/Environments/index.md)
6. [Environments Tutorial](./Kestrel/Environments/tutorial.md)

Please reach out to [hpc-help@nrel.gov](mailto:hpc-help@nrel.gov) for assistance with any topic on this page.
