#Using ANSYS 

*ANSYS is a licensed commercial application for the Eagle system.* 

The NREL Computation Science Center (CSC) maintains an ANSYS computational fluid dynamics (CFD) license pool for general use, including two seats of CFD and four ANSYS HPC Packs for parallel solves.

Note: Network floating licenses are a shared resource. Whenever you open an ANSYS window, a license is pulled from the pool and becomes unavailable to other Eagle users. Please do NOT keep idle windows open—if you are not actively using the application, close it and return the associated licenses to the pool. Excessive retention of software licenses falls under the inappropriate use policy.

The main workflow that we support has two stages. The first is interactive graphical usage for, e.g., building meshes. For this case, ANSYS should be run on the DAV nodes in serial, and preferably without invoking simulation capability. The second stage is batch (i.e., non-interactive) parallel processing, which should be run on compute nodes via a Slurm job script. Of course, if you have ANSYS input from another location ready to run in batch, the first stage is not needed. We unfortunately cannot support running parallel jobs on the DAV nodes, nor launching parallel jobs from interactive sessions on compute nodes.

License usage can be checked on Eagle with the command `lmstat.ansys`.

Running ANSYS Interactively
GUI access is provided through FastX desktops. Open the Terminal application to issue the commands below.

To enable the ANSYS Fluent environment, use:

```
module load ansys/<version>
```

To start Workbench, you can then issue the command `vglrun runwb2`.
Licenses and Scaling
HPC Pack licenses are used to distribute ANSYS batch jobs to run in parallel across many compute cores.  The HPC Pack model is designed to enable exponentially more computational resources per each additional license.  A table summarizing this relationship is shown below.

|HPC Pack Licenses Used	| Maximum Cores Enabled|
|-----------------------|----------------------|
|1	                    |8                     |
|2	                    |32                    |
|3	                    |128                   |
|4	                    |512                   |

Additionally, a number of ANSYS products allow you to use up to four cores without consuming any of the HPC Pack licenses.  For example, a Mechanical or Fluent job can be run on four cores and consume only the underlying physics license.  When scaling these jobs to more than four cores, the four cores are added to the total amount made available by the HPC Pack licenses. For example, a Mechanical or Fluent batch job designed to completely utilize an Eagle compute node (36 cores) requires one physics license and two HPC Pack licenses (4 + 32 cores enabled).

##Running ANSYS in Parallel Batch Mode

To initiate an ANSYS run that uses the HPC Packs, it is necessary to create a command line that contains the hosts and number of processes on each in a format host1:ppn_host1:host2:ppn_host2:.... In order to do this as illustrated below, you must set --ntasks-per-node and --nodes in your Slurm header. A partial example submit script might look as follows.

```bash
#!/bin/bash -l
...
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=36
...
cd $SLURM_SUBMIT_DIR
module purge  # purge everything else
module load ansys/19.2
module load intel-mpi/2018.0.3
...
unset I_MPI_PMI_LIBRARY
