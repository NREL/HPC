# Running STAR-CCM+ Software

*Learn how to run STAR-CCM+ software on the Eagle system.*

For information about the software's features, see the [STAR-CCM+
website](https://mdx.plm.automation.siemens.com/star-ccm-plus).

STAR-CCM+ can be run interactively on Eagle using X windows. The network
licenses are checked out from the license server running on wind-lic.nrel.gov.

First, build your simulation `<your_simulation.sim>` on your workstation and
copy to your `/scratch/$USER/<sim_dir>` directory on Eagle:

```
ls /scratch/$USER/sim_dir
your_simulation.sim
```

Create a Slurm script `<your_scriptfile>` as shown below:

??? example "Example Slurm script"

    ``` bash
    #!/bin/bash -l
    #SBATCH --time=2:00:00             # walltime limit of 2 hours
    #SBATCH --nodes=2                  # number of nodes
    #SBATCH --ntasks-per-node=36       # number of tasks per node
    #SBATCH --ntasks=72                # total number of tasks
    #SBATCH --job-name=your_simulation # name of job #
    #SBATCH --account=<allocation-id>  # name of project allocation

    export TMPDIR="/scratch/$USER/<sim_dir>"
    scontrol show hostnames > nodelist
    module load starccm \

    # Run Job

    echo "------ Running Starccm+ ------"

    date
    starccm+ -rsh "ssh -oStrictHostKeyChecking=no" -machinefile nodelist -np $SLURM_NTASKS -batch /scratch/$USER/<sim_dir>/your_simulation.sim >> simulation.log   
    rm nodelist
    date

    echo "------ End of the job ------"
    ```

Note that you must give the full path of your input file in the script.

<!-- TODO: link to docs for interactive job -->
The simulation may be tested in an interactive job before being submitted to the
batch queue.

After the interactive job is allocated, type the commands from the SLURM script
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
