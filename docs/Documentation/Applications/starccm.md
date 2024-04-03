# Running STAR-CCM+ Software


Simcenter STAR-CCM+ is a multiphysics CFD software that enables CFD engineers to model the complexity and explore the possibilities of products operating under real-world conditions.For information about the software's features, see the [STAR-CCM+
website](https://mdx.plm.automation.siemens.com/star-ccm-plus).

STAR-CCM+ are installed on both Eagle and Kestrel but it is not supported on Vermillion and Swift. The network
licenses are checked out from the license server running on 1wv11lic02.nrel.gov. 

STAR-CCM+ can be run interactively on both Eagle and Kestrel using X windows by running the following command in the terminal of a X window.

```
module load starccm
starccm+
```

To run STAR-CCM+ in batch mode, first, you need to build your simulation `<your_simulation.sim>` and
put it to your project directory:

```
ls /projects/your_project/sim_dir
your_simulation.sim
```

Then you need to create a Slurm script `<your_scriptfile>` as shown below to submit the job:

``` 
#!/bin/bash -l
#SBATCH --time=2:00:00             # walltime limit of 2 hours
#SBATCH --nodes=2                  # number of nodes
#SBATCH --ntasks-per-node=36       # number of tasks per node (<=36 on Eagle, <=104 on Kestrel)
#SBATCH --ntasks=72                # total number of tasks
#SBATCH --job-name=your_simulation # name of job
#SBATCH --account=<allocation-id>  # name of project allocation
#SBATCH --partition=standard       # partition

module load starccm                # load starccm module

rm -rf /projects/your_project/sim_dir/simulation.log   # remove the log file from last run
# Run Job

echo "------ Running Starccm+ ------"
    
starccm+ -np $SLURM_NTASKS -batch /projects/your_project/sim_dir/your_simulation.sim >> simulation.log

echo "------ End of the job ------"
```

Note that you must give the full path of your input file in the script.

By default, STAR-CCM+ uses open-MPI; However, the performance of open-MPI on Kestrel is poor wheen running on multiple nodes. Intel-MPI and cray-MPI are recommended for STAR-CCM+ on Kestrel, while Cray-MPI is expected to have a better performance than Intel-MPI. 

The simulation may be tested in an [interactive job](../Systems/Eagle/Running/interactive_jobs.md) before being submitted to the
batch queue.
After the interactive job is allocated, type the commands from the Slurm script
and make sure the job runs:

``` bash
module load starccm
export TMPDIR="/scratch/$USER/<sim_dir>"
...
echo $SLURM_JOB_NODELIST > nodelist
...
starccm+ -power -rsh "ssh -oStrictHostKeyChecking=no" -machinefile nodelist -np $SLURM_NTASKS -batch /scratch/$USER/<sim_dir>/your_simulation.sim >> simulation.log
```

If this succeeds, submit your job with:

```
sbatch <your_scriptfile>
```

When the job completes, the output files are stored in the `<sim_dir>` directory
with your_simulation.sim file:

```
ls /scratch/$USER/<sim_dir>
your_simulation.sim     simulation.log     slurm-12345.out
```
