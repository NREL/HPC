---
layout: default
title: Example SBATCH Scripts
has_children: false
---
# Sample Batch Scripts for Running Jobs on the Kestrel System

For a walkthrough of the elements of an sbatch script, please see [Submitting Batch Jobs](/Documentation/Slurm/batch_jobs/).

Many more examples of sbatch scripts are available in the [HPC Repository Slurm Directory](https://github.com/NREL/HPC/tree/master/slurm) on Github. You may also check the individual [Systems](/Documentation/Systems) pages for details related to the cluster you're working on.


??? example "Sample batch script for a serial job in the debug queue"
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

??? example "Sample batch script for a job in the shared partition"
    If your job does not need to utilize the full CPU (104 cores), you can run your job in the shared partition. When running on a shared partition, the default memory per CPU for users is 1G. To change this amount, use the `--mem-per-cpu=<MEM_REQUEST>` flag.

    ```
    #!/bin/bash
    #SBATCH --nodes=1 
    #SBATCH --partition=shared         
    #SBATCH --time=2:00:00    
    #SBATCH --ntasks=26                        # CPUs requested for job 
    #SBATCH --mem-per-cpu=2000                 # Requesting 2G per core.
    #SBATCH --account=<allocation handle>

    cd /scratch/$USER 
    srun ./my_progam # Use your application's commands here  
    ```

??? example "Sample serial batch script with full GPU request"
    Every node on Kestrel has 4 h100 GPUs. To run jobs on GPUs, your script should contain the `--gres=gpu:<NUM_GPUS>` flag in the SBATCH directives.

    ```
    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --partition=gpu-h100
    #SBATCH --time=00:20:00         # Setting a 20 minute time limit
    #SBATCH --ntasks-per-node=104   # Maximum CPU cores for job 
    #SBATCH --gres=gpu:4            # GPU request 

    export CUDA_VISIBLE_DEVICES=0,1,2,3

    # Enable access to new modules for running on GPUs
    source /nopt/nrel/apps/gpu_stack/env_cpe23.sh

    # Load modules
    module purge
    ml craype-x86-genoa  # Module to set optimizations for CPUs on GPU nodes

    # Run program
    cd /scratch/$USER 
    srun my_graphics_intensive_scripting 
    ```
    *Currently, `source /nopt/nrel/apps/gpu_stack/env_cpe23.sh` is necessary to access GPU modules. This is subject to change as the system improves.*

    The `CUDA_VISIBLE_DEVICES` environment variable specifies the GPU(s) your programs will run on, and is included for the sake of completeness. Read more about this environment variable [here](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/).

??? example "Sample serial batch script with partial GPU request"
    GPU nodes can be shared so you may request fewer than all 4 GPUs on a node. When doing so, you must also request appropriate CPU cores and memory with the `--ntasks-per-node=<NUM_CPUS>` and `--mem=<MEMORY_REQUEST>` flags, respectively.
    
    ```
    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --partition=gpu-h100
    #SBATCH --time=00:20:00         # Setting a 20 minute time limit
    #SBATCH --ntasks-per-node=104   # Maximum CPU cores for job 
    #SBATCH --gres=gpu:2            # GPU request 
    #SBATCH --mem=184000            # Standard partition (192GB nodes) 

    export CUDA_VISIBLE_DEVICES=0,1

    # Enable access to new modules for running on GPUs
    source /nopt/nrel/apps/gpu_stack/env_cpe23.sh

    # Load modules
    module purge
    ml craype-x86-genoa  # Module to set optimizations for CPUs on GPU nodes

    # Run program
    cd /scratch/$USER 
    srun my_graphics_intensive_scripting 
    ```

??? example "Sample batch script to utilize Local Disk ($TMPDIR) on standard compute nodes"
    On Kestrel, 256 of the standard compute nodes have 1.7TB of local disk space. Be sure to use the flag `SBATCH --tmp=<LOCAL_DISK_REQUEST>` to request a node with local disk space. If this is not included, there is a chance your job will get on a node without disk space, where it will use RAM instead.

    `$TMPDIR` is a preset variable that points to `/tmp/scratch/<JOB_ID>`. 

    ```
    #!/bin/bash 
    #SBATCH --ntasks=104               # CPU cores requested for job 
    #SBATCH --nodes=1                  # Keeep all cores on the same node 
    #SBATCH --time=01-00               # Job should run for up to 1 day (for example) 
    #SBATCH --tmp=1.7TB                # Request maximum 1.7TB local disk on standard nodes

    # Copy files into $TMPDIR 
    cp /scratch/<userid>/myfiles* $TMPDIR 

    srun ./my_parallel_readwrite_program -input-options $TMPDIR/myfiles  # use your application's commands  
    ```

??? example "Sample batch script to utilize Local Disk ($TMPDIR) on bigmem nodes"
    If your job requires more than 1.7TB of local disk, you can run your job on a bigmem node. Kestrel's bigmem nodes have 5.8TB of NVMe local disk.

    `$TMPDIR` is a preset variable that points to `/tmp/scratch/<JOB_ID>`. 

    ```
    #!/bin/bash 
    #SBATCH --ntasks=104               # CPU cores requested for job 
    #SBATCH --nodes=1                  # Keeep all cores on the same node 
    #SBATCH --time=01-00               # Job should run for up to 1 day (for example) 
    #SBATCH --tmp=5.8TB                # Request maximum 5.8TB local disk on bigmem nodes
    #SBATCH --partition=bigmem         # Run on bigmem nodes

    # Copy files into $TMPDIR 
    cp /scratch/<userid>/myfiles* $TMPDIR 

    srun ./my_parallel_readwrite_program -input-options $TMPDIR/myfiles  # use your application's commands  
    ```

??? example "Sample batch script for an MPI job on CPUs"
    The default module for running MPI jobs, PrgENV-cray is automatically loaded for all users.

    ```
    #!/bin/bash 
    #SBATCH --nodes=4                   # Number of nodes 
    #SBATCH --ntasks=104                # Request 104 CPU cores 
    #SBATCH --time=06:00:00             # Job should run for up to 6 hours 
    #SBATCH --account=<project handle>  # Where to charge NREL Hours 
    
    # module swap PrgEnv-cray <new_PrgEnv>  # Line to run if software to run uses a different PrgEnv than the default
 
    srun ./compiled_mpi_binary          # srun will infer which mpirun to use
    ```
    *For best scheduling functionality, it is not recommended to select a partition.*

??? example "Sample batch script for high-priority job"
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
