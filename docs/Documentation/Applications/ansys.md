---
title: Ansys
parent: Applications
---

# Ansys Fluent 

The NREL Computational Science Center (CSC) maintains an Ansys Fluent computational fluid dynamics (CFD) license pool for general use, including two seats of CFD (`cfd_base`) and four Ansys HPC Packs (`anshpc_pack`) to support running a model on many cores/parallel solves.

The main workflow that we support has two stages. The first is interactive graphical usage, e.g., for interactively building meshes or visualizing boundary geometry. For this, Ansys should be run on a [FastX desktop](https://kestrel-dav.hpc.nrel.gov/session/). The second stage is batch (i.e., non-interactive) parallel processing, which should be run on compute nodes via a Slurm job script. Of course, if you have Ansys input from another location ready to run in batch mode, the first stage is not needed. We unfortunately cannot support running parallel jobs on the DAV nodes, nor launching parallel jobs from interactive sessions on compute nodes.

!!! tip "Shared License Etiquette"
     Ansys uses network floating licenses that are a shared resource. Whenever you open an Ansys Fluent window, a license is pulled from the pool and becomes unavailable to other users. *Please do not keep idle windows open if you are not actively using the application*, close it and return the associated licenses to the pool. Excessive retention of software licenses falls under the inappropriate use policy.

## Building Models in the Ansys GUI
GUI access is provided through [FastX desktops](https://kestrel-dav.hpc.nrel.gov/session/). Open a terminal, load, and launch the Ansys Fluent environment with:

```
module load ansys/<version>
vglrun runwb2 &
```

where `<version>` will be replaced with an Ansys version/release e.g., `2023R1`. Press `tab` to auto-suggest all available versions. Because FastX desktop sessions are supported from DAV nodes shared between multiple HPC users, limits are placed on how much memory and compute resources can be consumed by a single user/job. For this reason, it is recommended that the GUI be primarily used to define the problem and run small-scale tests to validate its operation before moving the model to a compute node for larger-scale runs.

## Running a Fluent Model in Parallel Batch Mode

To launch Ansys Fluent jobs in parallel batch mode, you can build on the batch script presented below.

???+ example "Example Fluent Submission Script"
    ```bash
    #!/bin/bash
    ...
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=104

    cd $SLURM_SUBMIT_DIR
    module load PrgEnv-intel
    module load ansys/2023R1
    
    ...
    NODEFILE=${SLURM_SUBMIT_DIR}/slurmhosts.txt
    ...
    fluent 2ddp -pcheck -mpi=intel -t$SLURM_NTASKS -cnf=$NODEFILE -gu -driver null -i input_file.jou > output_file.jou
    ```

Once this script file (assumed to be named `submit_ansys_job.sh`) is saved, it can be submitted to the job scheduler with

```
[user@kl3 ~]$ sbatch submit_ansys_job.sh
```

In this example batch script, `2ddp` can be replaced with the version of FLUENT your job requires (`2d`, `3d`, `2ddp`, or `3ddp`), `-g` specifies that the job should run without the GUI, `-t` specifies the number of processors to use (in this example, 2 x 36 processors), `-cnf` specifies the hosts file (the list of nodes allocated to this job), `-mpi` and `-p<...>` specify the MPI implementation and interconnect, respectively, and`-i` is used to specify the job input file.  Note that generally speaking the generation of the hostname file,`slurmhosts.txt`, must be repeated in the beginning of each job since the allocated nodes will likely change for each run. 

!!! tip "A Note on Licenses and Job Scaling"
    HPC Pack licenses are used to distribute Ansys batch jobs to run in parallel across many compute cores. The HPC Pack model is designed to enable exponentially more computational resources per each additional license, roughly 2x4^(num_hpc_packs). A table summarizing this relationship is shown below.


    | HPC Pack Licenses Used | Total Cores Enabled           |
    |------------------------|-------------------------------|
    | 0                      | 4 (0 `hpc_pack` + 4 solver)     |
    | 1                      | 12 (8 `hpc_pack` + 4 solver)    |
    | 2                      | 36 (32 `hpc_pack` + 4 solver)   |
    | 3                      | 132 (128 `hpc_pack` + 4 solver) |
    | 4                      | 516 (512 `hpc_pack` + 4 solver) |

    Additionally, Fluent allows you to use up to four cores without consuming any of the HPC Pack licenses.  When scaling these jobs to more than four cores, the four cores are added to the total amount made available by the HPC Pack licenses. For example, a batch job designed to completely fill a node with 104 cores requires one `cfd_base` license and two HPC Pack licenses (100 + 4 cores enabled).



##Contact
For information about accessing licenses beyond CSC's base capability, please contact [Emily Cousineau.](mailto://Emily.Cousineau@nrel.gov)
