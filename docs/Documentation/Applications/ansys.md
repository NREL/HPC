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

where `<version>` will be replaced with an Ansys version/release e.g., `2025R1`. Press `tab` to auto-suggest all available versions. If no version is specified, it will load the default version which is 2025R1 at this moment. Because FastX desktop sessions are supported from DAV nodes shared between multiple HPC users, limits are placed on how much memory and compute resources can be consumed by a single user/job. For this reason, it is recommended that the GUI be primarily used to define the problem and run small-scale tests to validate its operation before moving the model to a compute node for larger-scale runs.

## Running Ansys Model in Parallel Batch Mode

### Ansys Fluent
Ansys Fluent is a general-purpose computational fluid dynamics (CFD) software used to model fluid flow, heat and mass transfer, chemical reactions, and more. It comes with the features of advanced physics modeling, turbulence modeling, single and multiphase flows, combustion, battery modeling, fluid-structure interaction.

#### CPU Solver
To launch Ansys Fluent jobs in parallel batch mode on CPU nodes, you can build on the batch script presented below.

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
export I_MPI_HYDRA_BOOTSTRAP=slurm

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
export I_MPI_HYDRA_BOOTSTRAP=slurm
```
#### GPU Solver
Ansys Fluent supports native GPU solver which is a solver architecture specifically designed to run on GPUs. With native GPU solver, the pre-processing (meshing, defining BCs, etc.) and post-processing are usually still done on the CPU while the solver takes over and runs almost entirely on the GPU. 

Note that, not all Fluent features support GPU solver. Specifically, the GPU solver is not available in the Ansys Workbench environment and profiles in cylindrical coordinate systems, which includes those used for swirl inlets, are not supported. For more information about GPU solver limitation, please refer to Ansys documentation.

To launch Ansys Fluent jobs in parallel batch mode with GPU solver, you can build on the batch script presented below.

```
#!/bin/bash
#SBATCH --job-name=fluent_GPU
#SBATCH --account=<your_account>
#SBATCH -o fluent_GPU_%j.out
#SBATCH -e fluent_GPU_%j.err
#SBATCH --nodes=2
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
 
module load ansys/<version>
export FLUENT_AFFINITY=0
export SLURM_ENABLED=1
export SCHEDULER_TIGHT_COUPLING=13
export I_MPI_HYDRA_BOOTSTRAP=slurm
 
scontrol show hostnames > nodelist
 
FLUENT=`which fluent`
VERSION=3ddp
JOURNAL=inputjournal.jou
LOGFILE=fluent_GPU.log
MPI=openmpi
 
OPTIONS="-i$JOURNAL -t$SLURM_NTASKS -gpu -mpi=$MPI -cnf=$PWD/nodelist"
$FLUENT $VERSION -g $OPTIONS > $LOGFILE 2>&1
```
In this case, we are running this GPU jon on 2 GPU nodes, 1 GPU per node, and 1 CPU per node. The `-gpu` flag in the lauch command enables the GPU solver. `-t` specifies the CPU and GPU configurations for the GPU solver as follows (for `-tn`):

If only 1 GPU is available, 1 GPU + n CPUs;
    
If multiple GPUs are available and n is less than the number of available GPUs, n CPU processes and n GPUs.
    
If multiple GPUs are avilable and the value of n is greater than or equal to the number of available GPUs, n CPUs + all of the GPUs.


### Ansys Mechanical
Ansys Mechanical is a finite element analysis (FEA) software used to perform structural analysis using advanced solver options, including linear dynamics, nonlinearities, thermal analysis, materials, composites, hydrodynamic, explicit, and more. 

#### CPU Job

The slurm script for Ansys Mechanical CPU jobs is presented as follows.

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

export ANSYS_LOCK=OFF
export I_MPI_HYDRA_BOOTSTRAP=slurm

unset HYDRA_LAUNCHER_EXTRA_ARGS
unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
unset OMPI_MCA_plm_slurm_args
unset PRTE_MCA_plm_slurm_args

machines=$(srun hostname | sort | uniq -c | awk '{print $2 ":" $1}' | paste -s -
d ":" -)

ansys251 -dis -mpi intelmpi2018 -machines $machines -i inputfilename.
dat -o joboutput.out
```
In the slurm script, `ansys251` starts the Ansys mechanical module, `-dis` enables distributed-memory parallel processing, `-mpi` specifies the mpi to be used (intelmpi2018 or openmpi), `-machine` specifies the host names, `-i` is used to specify the job input file, and `-o` is used to specify the job output file.

#### GPU Acceleration
Ansys Mechanical supports GPU acceleration in which GPUs are used to assist the CPUs in solving parts of the simulation — it’s still mostly a CPU-based solver, but the GPU helps offload some of the heavy linear algebra (like matrix operations). In situations where the analysis type is not supported by the GPU accelerator capability, the solution will continue but GPU acceleration will not be used.

The slurm script for Ansys Mechanical with GPU acceleration is presented as follows.

```
#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --account=<your account>
#SBATCH --output=mechanical_GPGPU_%j.out
#SBATCH --error=mechanical_GPGPU_%j.err

module load ansys

export ANSYS_LOCK=OFF
export I_MPI_HYDRA_BOOTSTRAP=slurm

unset HYDRA_LAUNCHER_EXTRA_ARGS
unset I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS
unset OMPI_MCA_plm_slurm_args
unset PRTE_MCA_plm_slurm_args

machines=$(srun hostname | sort | uniq -c | awk '{print $2 ":" $1}' | paste -s -d ":" -)

input_file=test_mech.inp
output_file=test_mech_GPGPU.out

ansys251 -dis -mpi intelmpi2021 -acc nvidia -na 2 -machines $machines -i $input_file -o $output_file

```

In this example, the job will run on 2 GPU nodes with 2 GPUs per node and 2 CPUs per node. In the launching command, the flag `-acc nvidia` is an openacc flag to indicate Ansys Mechanical to compile and run on Nvidia GPUs. The flag `-na` specifies the number of GPU accelerator devices to use per node. 


### A Few Nodes

When running an Ansys job, the out of memory error (OOM) is commonly encountered. To overcome the out of memory issue, you can try the following:

If you are running on shared nodes, by default, your job will be allocated about 1G of RAM per core requested. To change this amount, you can use the `--mem` or `--mem-per-cpu` flag in your job submission.

Try to run the job on nodes with local disk by using the `--tmp` option in your job submission script (e.g. `--tmp=1600000` https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/)


## Connect to Your Own License

At NREL, a few groups own their own Ansys license. In order to connect to the private license, the user can set the environment variable `ANSYSLMD_LICENSE_FILE` (e.g. `export ANSYSLMD_LICENSE_FILE=1055@10.60.1.85`, replacing your corresponding port and license server hostname or IP address).  

## Contact
For information about accessing licenses beyond CSC's base capability, please contact [Emily Cousineau.](mailto://Emily.Cousineau@nrel.gov)
