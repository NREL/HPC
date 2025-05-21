---
layout: default
title: Example Sbatch Scripts
has_children: false
---
# Sample Batch Scripts for Running Jobs on the Kestrel System

For a walkthrough of the elements of an sbatch script, please see [Submitting Batch Jobs](../../../Slurm/batch_jobs.md). For application specific recommendations and examples, please check the [Application pages](../../../Applications/index.md). 

??? example "Sample batch script for a CPU job in the debug queue"
    ```
    #!/bin/bash 
    #SBATCH --account=<allocation handle>   # Required
    #SBATCH --ntasks=104                    # Tasks to be run 
    #SBATCH --nodes=1                       # Run the tasks on the same node 
    #SBATCH --time=5                        # Required, maximum job duration 
    #SBATCH --partition=debug 

    cd /scratch/$USER 

    srun ./my_program.sh
    ```

??? example "Sample batch script with memory request"
    Standard Kestrel CPU nodes have about 250G of usable RAM. There are 10 bigmem nodes with 2TB of ram. 
    ```
    #!/bin/bash 
    #SBATCH --account=<allocation handle>   # Required 
    #SBATCH --ntasks=104                    # CPU cores requested for job 
    #SBATCH --time=01-00                    # Required, maximum job duration
    #SBATCH --mem=500G                      # Memory request


    cd /scratch/$USER 
    srun ./my_program.sh
 
    ```

??? example "Sample batch script for a job in the shared partition"
    If your job doesn't need a full CPU node (104 cores), you can run your job in the shared partition. When running on a shared node, the default memory per CPU is 1G. To change this amount, use the `--mem-per-cpu=<MEM_REQUEST>` flag.

    ```
    #!/bin/bash
    #SBATCH --nodes=1 
    #SBATCH --partition=shared         
    #SBATCH --time=2:00:00                  # Required, maximum job duration
    #SBATCH --ntasks=26                     # CPUs requested for job 
    #SBATCH --mem-per-cpu=2000              # Requesting 2G per core.
    #SBATCH --account=<allocation handle>   # Required 

    cd /scratch/$USER 
    srun ./my_progam # Use your application's commands here  
    ```



??? example "Sample batch script to utilize local disk"
    On Kestrel, 256 of the standard compute nodes have 1.7TB of NVMe node local storage. Use the flag `SBATCH --tmp=<LOCAL_DISK_REQUEST>` to request a node with local disk space. The storage may then be accessed inside the job by using the `$TMPDIR` environment variable.

    ```
    #!/bin/bash 
    #SBATCH --account=<allocation handle>      # Required 
    #SBATCH --ntasks=104                       # CPU cores requested for job 
    #SBATCH --nodes=1                  
    #SBATCH --time=01-00                       # Required, maximum job duration
    #SBATCH --partition=nvme                   # Request node with local disk

    # Copy files into $TMPDIR 
    cp /scratch/<userid>/myfiles* $TMPDIR 

    srun ./my_parallel_readwrite_program -input-options $TMPDIR/myfiles  # use your application's commands  
    ```

??? example "Sample batch script for high-priority job"
    A job may request high priority using `--qos=high`, which will give a small priority bump in the queue. This will charge your allocation at 2x the normal rate. 

    ```
    #!/bin/bash
    #SBATCH --job-name=job_monitor
    #SBATCH --account=<allocation handle>      # Required     
    #SBATCH --time=00:05:00                    # Required, maximum job duration
    #SBATCH --qos=high                         # Request high priority
    #SBATCH --ntasks=104
    #SBATCH -N 2 
    #SBATCH --output=job_monitor.out 
    
    cd /scratch/$USER 
    srun ./my_program.sh
    ```

??? example "Sample batch script for a GPU job in the debug queue"
    All GPU nodes in the debug queue are shared.  You are limited to two GPUs per job, across 1 or 2 nodes. 
    ```
    #!/bin/bash 
    #SBATCH --account=<allocation handle>   # Required
    #SBATCH --nodes=2  
    #SBATCH --gpus-per-node=1
    #SBATCH --mem=50G                       # Request system RAM per node. 
    #SBATCH --ntasks-per-node=2             # Request CPU cores per node
    #SBATCH --time=01:00:00                 # Required, maximum job duration 
    #SBATCH --partition=debug 

    cd /scratch/$USER 

    srun ./my_program.sh
    ```
??? example "Sample batch script for a full GPU node"
    Kestrel GPU nodes have 4 H100 GPUs. To run jobs on GPUs, your script should contain the `--gpus=<NUM_GPUS>` flag in the SBATCH directives.
    Submit GPU jobs from the [GPU login nodes](../index.md).

    ```
    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --account=<allocation handle>   # Required 
    #SBATCH --time=02:00:00                 # Required, maximum job duration
    #SBATCH --ntasks-per-node=128           # Maximum CPU cores for job 
    #SBATCH --gpus=4                        # GPU request 
    #SBATCH --exclusive                     # Request exclusive access to node. Allocates all CPU cores and GPUs by default.  
    #SBATCH --mem=360000                    # Request system RAM. If you need more memory, request up to 720000 to use the larger mem GPU nodes.

    # Load modules
    module load vasp

    # Run program
    cd /scratch/$USER 
    srun my_graphics_intensive_scripting 
    ```

??? example "Sample batch script for a partial GPU node"
    GPU nodes can be shared so you may request fewer than all 4 GPUs on a node. When doing so, you must also request appropriate CPU cores and memory with the `--ntasks-per-node=<NUM_CPUS>` and `--mem=<MEMORY_REQUEST>` flags, respectively. Submit GPU jobs from the [GPU login nodes](../index.md).
    
    ```
    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --account=<allocation handle>   # Required 
    #SBATCH --time=2:00:00                  # Required, maximum job duration
    #SBATCH --ntasks-per-node=20            # Request CPU cores 
    #SBATCH --gpus=2                        # GPU request 
    #SBATCH --mem=170G                      # Request CPU memory

    # Load modules
    
    # Run program
    cd /scratch/$USER 
    srun my_graphics_intensive_scripting 
    ```
