---
title: Ansys
parent: Applications
---

The NREL Computational Science Center (CSC) maintains an Ansys license pool for general use, including two seats of CFD, one seat of Ansys Mechanical, and four Ansys HPC Packs to support running a model on many cores/parallel solves.

The main workflow that we support has two stages. The first is interactive graphical usage, e.g., for interactively building meshes or visualizing boundary geometry. For this, Ansys should be run on a [FastX desktop](https://nrel.github.io/HPC/Documentation/Viz_Analytics/virtualgl_fastx/). The second stage is batch (i.e., non-interactive) parallel processing, which should be run on compute nodes via a Slurm job script. Of course, if you have Ansys input from another location ready to run in batch mode, the first stage is not needed. We unfortunately cannot support running parallel jobs on the DAV nodes, nor launching parallel jobs from interactive sessions on compute nodes.

!!! tip "Shared License Etiquette"
     License usage can be checked on Kestrel with the command `lmstat.ansys`. Network floating licenses are a shared resource. Whenever you open an Ansys Fluent window, a license is pulled from the pool and becomes unavailable to other users. *Please do not keep idle windows open if you are not actively using the application*, close it and return the associated licenses to the pool. Excessive retention of software licenses falls under the inappropriate use policy.

## Building Models in the Ansys GUI
GUI access is provided through [FastX desktops](https://nrel.github.io/HPC/Documentation/Viz_Analytics/virtualgl_fastx/). Open a terminal, load, and launch the Ansys Workbench with:

```
module load ansys/<version>
vglrun runwb2
```

where `<version>` will be replaced with an Ansys version/release e.g., `2024R1`. Press `tab` to auto-suggest all available versions. Because FastX desktop sessions are supported from DAV nodes shared between multiple HPC users, limits are placed on how much memory and compute resources can be consumed by a single user/job. For this reason, it is recommended that the GUI be primarily used to define the problem and run small-scale tests to validate its operation before moving the model to a compute node for larger-scale runs.

## Running Ansys Model in Parallel Batch Mode

### Ansys Fluent
To launch Ansys Fluent jobs in parallel batch mode, you can build on the batch script presented below.


```
bash
#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --account=your_account
#SBATCH -o fluent_%j.out
#SBATCH -e fluent_%j.err
#SBATCH --nodes=2
#SBATCH --time=1:00:00
#SBATCH --qos=high 
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=104
#SBATCH --exclusive

cd $SLURM_SUBMIT_DIR
module load ansys/<version>

export FLUENT_AFFINITY=0
export SLURM_ENABLED=1
export SCHEDULER_TIGHT_COUPLING=13

scontrol show hostnames > nodelist

FLUENT=`which fluent`
VERSION=3ddp
JOURNAL=journal_name.jou
LOGFILE=fluent.log
MPI=intel
 
OPTIONS="-i$JOURNAL -t$SLURM_NPROCS -mpi=$MPI -cnf=$PWD/nodelist"
 
nodelist > fluent.log
 
$FLUENT $VERSION -g $OPTIONS > $LOGFILE 2>&1
```

Once this script file (assumed to be named `ansys-job.slurm`) is saved, it can be submitted to the job scheduler with

```
[user@kl3 ~]$ sbatch ansys-job.slurm
```

In this example batch script, `2ddp` can be replaced with the version of FLUENT your job requires (`2d`, `3d`, `2ddp`, or `3ddp`), `-g` specifies that the job should run without the GUI, `-t` specifies the number of processors to use (in this example, 2 x 36 processors), `-cnf` specifies the hosts file (the list of nodes allocated to this job), `-mpi` and `-p<...>` specify the MPI implementation and interconnect, respectively, and`-i` is used to specify the job input file.  Note that generally speaking the generation of the hostname file,`myhosts.txt`, must be repeated in the beginning of each job since the allocated nodes will likely change for each run. 

!!! tip "A Note on Licenses and Job Scaling"
    HPC Pack licenses are used to distribute Ansys batch jobs to run in parallel across many compute cores.  The HPC Pack model is designed to enable exponentially more computational resources per each additional license, roughly 2x4^(num_hpc_packs).  A table summarizing this relationship is shown below.


    | HPC Pack Licenses Used | Total Cores Enabled           |
    |------------------------|-------------------------------|
    | 0                      | 4 (0 `hpc_pack` + 4 solver)     |
    | 1                      | 12 (8 `hpc_pack` + 4 solver)    |
    | 2                      | 36 (32 `hpc_pack` + 4 solver)   |
    | 3                      | 132 (128 `hpc_pack` + 4 solver) |
    | 4                      | 516 (512 `hpc_pack` + 4 solver) |

    Additionally, Fluent allows you to use up to four cores without consuming any of the HPC Pack licenses.  When scaling these jobs to more than four cores, the four cores are added to the total amount made available by the HPC Pack licenses. For example, a batch job designed to completely fill a node with 36 cores requires one `cfd_base` license and two HPC Pack licenses (32 + 4 cores enabled).



##Contact
For information about accessing licenses beyond CSC's base capability, please contact [Emily Cousineau.](mailto://Emily.Cousineau@nrel.gov)
