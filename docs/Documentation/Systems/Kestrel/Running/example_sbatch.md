---
layout: default
title: Example SBATCH Scripts
has_children: false
---
# Sample Batch Scripts for Running Jobs on the Kestrel System

For a walkthrough of the elements of an sbatch script, please see [Submitting Batch Jobs](/Documentation/Slurm/batch_jobs/).

Many more examples of sbatch scripts are available in the [HPC Repository Slurm Directory](https://github.com/NREL/HPC/tree/master/slurm) on Github. You may also check the individual [Systems](/Documentation/Systems) pages for details related to the cluster you're working on.


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

??? info "Sample batch script for a job in the shared partition"
    ```
    #!/bin/bash
    #SBATCH --nodes=1 
    #SBATCH --partition=shared         
    #SBATCH --time=2:00:00    
    #SBATCH --ntasks=26 # CPUs requested for job 
    #SBATCH --mem-per-cpu=2000 # Request 2G per core.
    #SBATCH --account=<allocation handle>

    cd /scratch/$USER 
    srun ./my_progam # Use your application's commands here  
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

??? info "Sample batch script to utilize Local Disk ($TMPDIR)"
    ```
    #!/bin/bash 
    #SBATCH --ntasks=36                # CPU cores requested for job 
    #SBATCH --nodes=1                  # Keeep all cores on the same node 
    #SBATCH --time=01-00               # Job should run for up to 1 day (for example) 
    #SBATCH --tmp=20TB                 # Request minimum 20TB local disk 
    
    # Copy files into $TMPDIR 
    cp /scratch/<userid>/myfiles* $TMPDIR 

    srun ./my_parallel_readwrite_program -input-options $TMPDIR/myfiles  # use your application's commands  
    ```
    *`$TMPDIR` is a preset variable that points to `/tmp/scratch/<JOB_ID>`. Be sure to use the flag `SBATCH --tmp=<LOCAL_DISK_REQUEST>` or your job will use RAM.*

??? info "Sample batch script for an MPI job"
    ```
    Eagle MPI (intel-mpi, hpe-mpi): 
    
    #!/bin/bash 
    #SBATCH --nodes=4                   # Number of nodes 
    #SBATCH --ntasks=100                # Request 100 CPU cores 
    #SBATCH --time=06:00:00             # Job should run for up to 6 hours 
    #SBATCH --account=<project handle>  # Where to charge NREL Hours 
    
    module swap PrgEnv-cray <new_PrgEnv>  # Line to run if software to run uses a different PrgEnv than the default
 
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


