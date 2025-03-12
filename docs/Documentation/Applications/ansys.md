## Ansys
The current Ansys license is an unlimited license that covers all Ansys products, with no restrictions on quantities. However, since Ansys is unable to provide a license file that includes all products in unlimited quantities, we have requested licenses based on our anticipated needs. You can check the available licenses on Kestrel using the command `lmstat.ansys`. If the module you need is not listed, please submit a ticket by emailing [HPC-Help@nrel.gov](mailto:HPC-Help@nrel.gov) so that we can request an updated license to include the specific module you require.

The main workflow that we support has two stages. The first is interactive graphical usage, e.g., for interactively building meshes or visualizing boundary geometry. For this, Ansys should be run on a [FastX desktop](https://nrel.github.io/HPC/Documentation/Viz_Analytics/virtualgl_fastx/). The second stage is batch (i.e., non-interactive) parallel processing, which should be run on compute nodes via a Slurm job script. Of course, if you have Ansys input from another location ready to run in batch mode, the first stage is not needed. We unfortunately cannot support running parallel jobs on the DAV nodes, nor launching parallel jobs from interactive sessions on compute nodes.

### Shared License Etiquette
Network floating licenses are a shared resource. Whenever you open an Ansys Fluent window, a license is pulled from the pool and becomes unavailable to other users. *Please do not keep idle windows open if you are not actively using the application*, close it and return the associated licenses to the pool. Excessive retention of software licenses falls under the inappropriate use policy.

### A Note on Licenses and Job Scaling
HPC Pack licenses are used to distribute Ansys batch jobs to run in parallel across many compute cores. The HPC Pack model is designed to enable exponentially more computational resources per each additional license, roughly 2x4^(num_hpc_packs).  A table summarizing this relationship is shown below.


    | HPC Pack Licenses Used | Total Cores Enabled           |
    |------------------------|-------------------------------|
    | 0                      | 4 (0 `hpc_pack` + 4 solver)     |
    | 1                      | 12 (8 `hpc_pack` + 4 solver)    |
    | 2                      | 36 (32 `hpc_pack` + 4 solver)   |
    | 3                      | 132 (128 `hpc_pack` + 4 solver) |
    | 4                      | 516 (512 `hpc_pack` + 4 solver) |

Additionally, Ansys allows you to use up to four cores without consuming any of the HPC Pack licenses.  When scaling these jobs to more than four cores, the four cores are added to the total amount made available by the HPC Pack licenses. For example, a batch job designed to completely fill a node with 36 cores requires one `cfd_base` license and two HPC Pack licenses (32 + 4 cores enabled).

## Building Models in the Ansys GUI
GUI access is provided through [FastX desktops](https://nrel.github.io/HPC/Documentation/Viz_Analytics/virtualgl_fastx/). Open a terminal, load, and launch the Ansys Workbench with:

```
module load ansys/<version>
vglrun runwb2
```

where `<version>` will be replaced with an Ansys version/release e.g., `2024R1`. Press `tab` to auto-suggest all available versions. Because FastX desktop sessions are supported from DAV nodes shared between multiple HPC users, limits are placed on how much memory and compute resources can be consumed by a single user/job. For this reason, it is recommended that the GUI be primarily used to define the problem and run small-scale tests to validate its operation before moving the model to a compute node for larger-scale runs.

## Running Ansys Model in Parallel Batch Mode

### Ansys Fluent
Ansys Fluent is a general-purpose computational fluid dynamics (CFD) software used to model fluid flow, heat and mass transfer, chemical reactions, and more. It comes with the features of advanced physics modeling, turbulence modeling, single and multiphase flows, combustion, battery modeling, fluid-structure interaction.

To launch Ansys Fluent jobs in parallel batch mode, you can build on the batch script presented below.

```
bash
#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --account=<your_account>
#SBATCH -o fluent_%j.out
#SBATCH -e fluent_%j.err
#SBATCH --nodes=2
#SBATCH --time=1:00:00
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

In this example batch script, `3ddp` can be replaced with the version of FLUENT your job requires (`2d`, `3d`, `2ddp`, or `3ddp`), `-g` specifies that the job should run without the GUI, `-t` specifies the number of processors to use (in this example, 2 x 104 processors), `-cnf` specifies the hosts file (the list of nodes allocated to this job), `-mpi` specifies the MPI implementation (intel or openmpi, Ansys uses its own mpi comes with the package instead of the mpi installed on our cluster, the current Ansys version only supports intel or openmpi), and `-i` is used to specify the job input file. For more Fluent options, you can run `fluent -help` to show after load the Ansys module.

In addition, the following commands in the slurm script are included to make sure the right bootstrap is used:

```
export FLUENT_AFFINITY=0
export SLURM_ENABLED=1
export SCHEDULER_TIGHT_COUPLING=13
```


### Ansys Mechanical
Ansys Mechanical is a finite element analysis (FEA) software used to perform structural analysis using advanced solver options, including linear dynamics, nonlinearities, thermal analysis, materials, composites, hydrodynamic, explicit, and more. The slurm script for Ansys Mechanical jobs is presented as follows.

```
#!/bin/bash
#
#SBATCH --job-name=jobname
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=1:00:00
#SBATCH --exclusive
#SBATCH --account=<your_account>
#SBATCH --output="ansys-%j.out"
#SBATCH --error="ansys-%j.err"

cd $SLURM_SUBMIT_DIR

module load ansys

machines=$(srun hostname | sort | uniq -c | awk '{print $2 ":" $1}' | paste -s -
d ":" -)

ansys241 -dis -mpi intelmpi2018 -machines $machines -i inputfilename.
dat -o joboutput.out
```
In the slurm script, `ansys241` starts the Ansys mechanical module, `-dis` enables distributed-memory parallel processing, `-mpi` specifies the mpi to be used (intelmpi2018 or openmpi), `-machine` specifies the host names, `-i` is used to specify the job input file, and `-o` is used to specify the job output file.

### A Few Nodes

When running an Ansys job, the out of memory error (OOM) is commonly encountered. To overcome the out of memory issue, you can try the following:

If you are running on shared nodes, by default, your job will be allocated about 1G of RAM per core requested. To change this amount, you can use the `--mem` or `--mem-per-cpu` flag in your job submission. To allocate all of the memory available on a node, use the `--mem=0` flag (https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/).

Try to run the job on nodes with local disk by using the `--tmp` option in your job submission script (e.g. `--tmp=1600000` https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/)


## Connect to Your Own License

At NREL, a few groups own their own Ansys license. In order to connect to the private license, the user can set the environment variable `ANSYSLMD_LICENSE_FILE` (e.g. `export ANSYSLMD_LICENSE_FILE=1055@10.60.1.85`, replacing your corresponding port and license server hostname or IP address).  

## Contact
For information about accessing licenses beyond CSC's base capability, please contact [Emily Cousineau.](mailto://Emily.Cousineau@nrel.gov)
