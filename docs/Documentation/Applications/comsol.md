---
title: Comsol
parent: Applications
---

# COMSOL Multiphysics 

*COMSOL Multiphysics is a versatile finite element analysis and simulation package. The COMSOL graphical user interface (GUI) environment is supported primarily for building and solving small models while operation in batch mode allows users to scale their models to larger, higher-fidelity studies. Currently, we host three floating network licenses and a number of additional modules. Two COMSOL versions are available on Kestrel, they are 6.2 and 6.3.*

## Building a COMSOL Model
Extensive documentation is available in the menu: **Help > Documentation**. For beginners, it is highly recommended to follow the steps in *Introduction to COMSOL Multiphysics* found in **Help > Documentation**.

For instructional videos, see the [COMSOL website](https://www.comsol.com) Video Gallery.

## Building Models in the COMSOL GUI
Before beginning, it is a good practice to check the license status. To do so, you need to run the following script command:
     
```
[user@kl3 ~]$ ./lmstat.comsol
```

When licenses are available, COMSOL can be used by starting the COMSOL GUI which allows you to build models, run the COMSOL computational engine, and analyze results. The COMSOL GUI can be accessed through a [FastX desktop](https://kestrel-dav.hpc.nrel.gov/auth/ssh/) by opening a terminal in a FastX window and running the following commands:

```
[user@kl3 ~]$ module load comsol
[user@kl3 ~]$ vglrun comsol
```

Because FastX desktop sessions are supported from DAV nodes shared between multiple HPC users, limits are placed on how much memory and compute resources can be consumed by a single user/job. For this reason, it is recommended that the GUI be primarily used to define the problem and run small-scale tests to validate its operation before moving the model to a compute node for larger-scale runs. For jobs that require both large-scale compute resources and GUI interactivity simultaneously, there is partial support for running the GUI from an X-enabled shell on a compute node by replacing the `vglrun comosl` command with:

```
[user@kl3 ~]$ comsol -3drend sw
```

However, the performance may be slow and certain display features may behave unexpectedly.

## Running a Single-Node COMSOL Model in Batch Mode
You can save your model built in FastX+GUI mode into a file such as `myinputfile.mph`. Once that's available, the following job script shows how to run a single process multithreaded job in batch mode:

    ```
    #!/bin/bash                                                                                                                                                                                     
    #SBATCH --job-name="comsol-batch-single-node"                                                                                                                                                   
    #SBATCH --nodes=1                                                                                                                                                                               
    #SBATCH --ntasks-per-node=104                                                                                                                                                                   
    #SBATCH --cpus-per-task=1                                                                                                                                                                       
    #SBATCH --time=00:10:0        
    #SBATCH --partition=debug
    #SBATCH --account=<allocation handle>
    #SBATCH --output="comsol-%j.out"
    #SBATCH --error="comsol-%j.err"

    # This helps ensure your job runs from the directory
    # from which you ran the sbatch command
    SLURM_SUBMIT_DIR=<your working directory>
    cd $SLURM_SUBMIT_DIR

    # Set up environment, and list to stdout for verification
    module load comsol
    echo " "
    module list
    echo " "

    inputfile=$SLURM_SUBMIT_DIR/myinputfile.mph
    outputfile=$SLURM_SUBMIT_DIR/myoutputfilename
    logfile=$SLURM_SUBMIT_DIR/mylogfilename

    # Run a COMSOL job with 104 threads.

    comsol batch -np 104 -inputfile $inputfile -outputfile $outputfile –batchlog $logfile
    ```

Once this script file (e.g., `submit_single_node_job.sh`) is saved, it can be submitted to the job scheduler with

```
[user@kl3 ~]$ sbatch ./submit_single_node_job.sh
```

## Running a Multi-Node COMSOL Model in Batch Mode
To configure a COMSOL job with multiple MPI ranks, required for any job where the number of nodes >1, you can build on the following template:
    
    ```
    #!/bin/bash                                                                                                                                                                                     
    #SBATCH --job-name="comsol-batch-multinode-hybrid"                                                                                                                                                  
    #SBATCH --nodes=4                                                                                                                                                                               
    #SBATCH --ntasks-per-node=8                                                                                                                                                                     
    #SBATCH --cpus-per-task=13                                                                                                                                                                      
    #SBATCH --time=00:10:0                                                                                                                                                                          
    #SBATCH --partition=debug                                                                                                                                                                       
    #SBATCH --exclusive                                                                                                                                                                             
    #SBATCH --account=<allocation handle>                                                                                                                                                                  
    #SBATCH --output="comsol-%j.out"                                                                                                                                                                
    #SBATCH --error="comsol-%j.err"                                                                                                                                                                 

    # This helps ensure your job runs from the directory                                                                                                                                            
    # from which you ran the sbatch command                                                                                                                                                         
    SLURM_SUBMIT_DIR= <your working directory>
    cd $SLURM_SUBMIT_DIR

    # Set up environment, and list to stdout for verification                                                                                                                                       
    module load comsol
    echo " "
    module list
    echo " "

    export SLURM_MPI_TYPE=pmi2
    export OMP_NUM_THREADS=13

    inputfile=$SLURM_SUBMIT_DIR/myinputfile.mph
    outputfile=$SLURM_SUBMIT_DIR/myoutputfilename
    logfile=$SLURM_SUBMIT_DIR/mylogfilename

    # Run a 4-node job with 32 MPI ranks and 13 OpenMP threads per each rank.                                                                                                                        
    comsol batch -mpibootstrap slurm -inputfile $inputfile -outputfile $outputfile –batchlog $logfile
    ```

The job script can be submitted to SLURM just the same as above for the single-node example. The option `-mpibootstrap slurm` helps COMSOL to deduce runtime parameters such as `-nn`, `-nnhost` and `-np`. For large jobs that require more than one node, this approach, which uses MPI and/or OpenMP, can be used to efficiently utilize the available resources. Note that in this case, we choose 32 MPI ranks, 8 per node, and each rank using 13 threads for demonstration purpose, but *not* as an optimal performance recommendation. The optimal configuration depends on your particular problem, workload, and choice of solver, so some experimentation may be required.

## Running COMSOL Model with GPU
In COMSOL Multiphysics®, GPU acceleration can significantly increase performance for time-dependent simulations that use the discontinuous Galerkin (dG) method, such as those using the Pressure Acoustics, Time Explicit interface, and for training deep neural network (DNN) surrogate models. The following is a job script example used to run COMSOL jobs on GPU nodes.

     ```
     #!/bin/bash
     #SBATCH --job-name=comsol-batch-GPUs
     #SBATCH --time=00:20:00
     #SBATCH --gres=gpu:1  # request 1 gpu per node, each gpu has 80 Gb of memory
     #SBATCH --mem-per-cpu=2G # requested memory per CPU core
     #SBATCH --ntasks-per-node=64
     #SBATCH --nodes=2
     #SBATCH --account=<allocation handle> 
     #SBATCH --output=comsol-%j.out
     #SBATCH --error=comsol-%j.err
     
     # This helps ensure your job runs from the directory
     # from which you ran the sbatch command
     cd $SLURM_SUBMIT_DIR
     
     # Set up environment, and list to stdout for verification
     module load comsol
     echo " "
     module list
     echo " "
     
     inputfile=$SLURM_SUBMIT_DIR/myinputfile.mph
     outputfile=$SLURM_SUBMIT_DIR/myoutputfilename
     logfile=$SLURM_SUBMIT_DIR/mylogfilename
     
     # Run a 2-node, 64-rank parallel COMSOL job with 1 threads for each rank and 1 gpu per node
     # -nn = total number of MPI ranks
     # -nnhost = number of MPI ranks per host
     # -np = number of threads per rank
     
     comsol –nn 128 -nnhost 64 batch -np 1 -inputfile $inputfile -outputfile $outputfile –batchlog $logfile
     ```

Note, When launching a GPU job on Kestrel, be sure to do so from one of its dedicated GPU login nodes (ssh to Kestrel from the NREL network using kestrel-gpu.hpc.nrel.gov).

The Complex Systems Simulation and Optimization group has hosted introductory and advanced COMSOL trainings. The introductory training covered how to use the COMSOL GUI and run COMSOL in batch mode on Kestrel. The advanced training showed how to do a parametric study using different sweeps (running an interactive session is also included) and introduced equation-based simulation and parameter estimation. To learn more about using COMSOL on Kestrel, please refer to the training. The recording can be accessed at [Computational Sciences Tutorials](https://nrel.sharepoint.com/sites/ComputationalSciencesTutorials/Lists/Computational%20Sciences%20Tutorial%20Recordings/AllItems.aspx?viewid=7b97e3fa%2Dedf6%2D48cd%2D91d6%2Df69848525ba4&playlistLayout=playback&itemId=75) and the slides and models used in the training can be downloaded from [Github](https://github.com/NREL/HPC/tree/master/applications/comsol/comsol-training).
