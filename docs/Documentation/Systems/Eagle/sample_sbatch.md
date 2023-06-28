---
layout: default
title: Example SBATCH Scripts
has_children: false
---

# Example SBATCH Script Walkthrough
The primary method of submitting an unattended job to the Slurm scheduler queue is via a batch script. 


Many examples of sbatch scripts are available in the [HPC Repository Slurm Directory](https://github.com/NREL/HPC/tree/master/slurm) on Github. 

Here's an example script to get started.

```
#!/bin/bash
#SBATCH --account=<allocation>
#SBATCH --time=4:00:00
#SBATCH --job-name=job
#SBATCH --mail-user=your.email@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=job_output_filename.%j.out  # %j will be replaced with the job ID
module load myprogram
myprogram.sh
```

## Script Details

Here is a section-by-section breakdown of the sample sbatch script, to help you begin writing your own.

### Script Begin

`#!/bin/bash`

This denotes the start of the script, and that it is written in BASH shell language, the most common Linux environment. 

### SBATCH Directives

```
#SBATCH --account=<allocation>
#SBATCH --time=4:00:00
#SBATCH --job-name=job
#SBATCH --mail-user=your.email@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=job_output_filename.%j.out  # %j will be replaced with the job ID
```
Generalized form:

`#SBATCH --<command>=<value>` 

Command flags to the sbatch program are given via `#SBATCH` directives in the sbatch script. There are many flags available that can affect your job. See the official [Slurm documentation on sbatch](https://slurm.schedmd.com/sbatch.html) for a complete list, or view the man page on a login node with `man sbatch`. 

Sbatch directives must be at the beginning of your sbatch script. Once a line with any other non-directive content is detected, Slurm will no longer parse further directives.

Note that sbatch flags do not need to be issued via directives inside the script. They can also be issued via the commandline when submitting the job. Flags issued via commandline will supercede directives issued inside the script. For example:

`sbatch --account=csc000 --time=60 --partition=debug mytestjob.sh`

#### Job Instructions

After the sbatch directive block, you may then begin executing your job. The syntax is normal BASH shell scripting. You may load system modules for software, load virtual environments, define environment variables, and execute your software to perform work.

```
module load myprogram
myprogram.sh
```

See the "master" main branch of the [Github repository](https://www.github.com/NREL/HPC) for downloadable examples.

## Sample Batch Scripts for Running Jobs on the Eagle System

??? info "Sample batch script for a serial job in the debug queue"
    ```
    #!/bin/bash 
    #SBATCH --ntasks=4 # Tasks to be run 
    #SBATCH --nodes=1  # Run the tasks on the same node 
    #SBATCH --time=5   # Required, estimate 5 minutes 
    #SBATCH --account=<project_handle> # Required 
    #SBATCH --partition=debug 

    cd /scratch/$USER 

    srun $HOME/hpcapp -options 
    ```

??? info "Sample serial batch script with GPU and memory request"
    ```
    #!/bin/bash
    #SBATCH --nodes=2          # Use 2 nodes
    #SBATCH --time 00:20:00    # Set a 20 minute time limit
    #SBATCH --ntasks 2         # Maximum CPU cores for job 
    #SBATCH --gres=gpu:2       # GPU request 
    #SBATCH --mem=184000       # Standard partition (192GB nodes) 

    cd /scratch/$USER 
    srun my_graphics_intensive_scripting 
    ```

??? info "Sample batch script for a serial job in default (standard) queue"
    ```
    #!/bin/bash 
    #SBATCH --partition=standard       # Name of Partition 
    #SBATCH --ntasks=12                # CPU cores requested for job 
    #SBATCH --nodes=1                  # Keeep all cores on the same node
    #SBATCH --time=02-00:00:00         # Job should run for up to 2 days (for example) 

    cd /scratch/<userid>/mydir

    srun hpcapp -options /home/hpcuser/app/parameters  # use your application's commands 
    ```
    *For best scheduling functionality, it is not recommended to select a partition.*

??? info "Sample batch script to utilize Local Disk (/tmp/scratch)"
    ```
    #!/bin/bash 
    #SBATCH --ntasks=36                # CPU cores requested for job 
    #SBATCH --nodes=1                  # Keeep all cores on the same node 
    #SBATCH --time=01-00               # Job should run for up to 1 day (for example) 
    #SBATCH --tmp=20TB                 # Request minimum 20TB local disk 
    
    export TMPDIR=$LOCAL_SCRATCH 
    cp /scratch/<userid>/myfiles* $TMPDIR 

    srun ./my_parallel_readwrite_program -input-options $TMPDIR/myfiles  # use your application's commands  
    ```
    *If you or your application has a need for large local disk, please use /tmp/scratch. In the example above, environment variable $LOCAL_SCRATCH can be used in place of the size limited /tmp.*

??? info "Sample batch script for an MPI job"
    ```
    Eagle MPI (intel-mpi, hpe-mpi): 
    
    #!/bin/bash 
    #SBATCH --nodes=4                   # Number of nodes 
    #SBATCH --ntasks=100                # Request 100 CPU cores 
    #SBATCH --time=06:00:00             # Job should run for up to 6 hours 
    #SBATCH --account=<project handle>  # Where to charge NREL Hours 
    
    module purge
    module load mpi/intelmpi/18.0.3.222 
    srun ./compiled_mpi_binary          # srun will infer which mpirun to use
    ```
    *For best scheduling functionality, it is not recommended to select a partition.*

??? info "Sample batch script for high-priority job"
    ```
    #!/bin/sh
    #SBATCH --job-name=job_monitor
    #SBATCH -A <account>
    #SBATCH --time=00:05:00
    #SBATCH --qos=high
    #SBATCH --ntasks=2
    #SBATCH -N 2 
    #SBATCH --output=job_monitor.out 
    #SBATCH --exclusive
    
    srun ./my_job_monitoring.sh
    ```
