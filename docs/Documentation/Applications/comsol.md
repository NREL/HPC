---
title: Comsol
parent: Applications
---

# COMSOL Multiphysics 

*COMSOL Multiphysics is a versatile finite element analysis and simulation package. The COMSOL graphical user interface (GUI) environment is supported primarily for building and solving small models while operation in batch mode allows users to scale their models to larger, higher-fidelity studies. Currently, we host three floating network licenses and a number of additional modules.*

## Building a COMSOL Model
Extensive documentation is available in the menu: **Help > Documentation**. For beginners, it is highly recommended to follow the steps in *Introduction to COMSOL Multiphysics* found in **Help > Documentation**.

For instructional videos, see the [COMSOL website](https://www.comsol.com) Video Gallery.

## Building Models in the COMSOL GUI
Before beginning, it is a good practice to check the license status. To do so you need create a bash script file named `lmstat.comsol` in your working directory, add executable permission to the `lmstat.comsol` file, and execute it:

Copy and paste the following script to the created file named `lmstat.comsol`:
     
```bash
#!/bin/bash
COMSOL_LIC_DIR=/nopt/nrel/apps/software/comsol/6.1/comsol61/multiphysics/license/glnxa64
cd $COMSOL_LIC_DIR
./lmstat -a --no-user-info -c ../license.dat
```
Add executable permission:
```
[user@kl3 ~]$ chmod +x ./lmstat.comsol
```
Execute it:
```
[user@kl3 ~]$ ./lmstat.comsol
```

When licenses are available, COMSOL can be used by starting the COMSOL GUI which allows you to build models, run the COMSOL computational engine, and analyze results. The COMSOL GUI can be accessed through a [FastX desktop](https://kestrel-dav.hpc.nrel.gov/auth/ssh/) by opening a terminal in a FastX window and running the following commands:

```
[user@kl3 ~]$ module load comsol/6.1
[user@kl3 ~]$ vglrun comsol &
```

Because FastX desktop sessions are supported from DAV nodes shared between multiple HPC users, limits are placed on how much memory and compute resources can be consumed by a single user/job. For this reason, it is recommended that the GUI be primarily used to define the problem and run small-scale tests to validate its operation before moving the model to a compute node for larger-scale runs. For jobs that require both large-scale compute resources and GUI interactivity simultaneously, there is partial support for running the GUI from an X-enabled shell on a compute node by replacing the `vglrun comosl` command with:

```
[user@kl3 ~]$ comsol -3drend sw
```

However, the performance may be slow and certain display features may behave unexpectedly.

## Running a Single-Node COMSOL Model in Batch Mode
You can save your model built in FastX+GUI mode into a file such as `myinputfile.mph`. Once that's available, the following job script shows how to run a single process multithreaded job in batch mode:

???+ example "Example Submission Script"

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
    module load comsol/6.1
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

???+ example "Example Multiprocess Submission Script"
    
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
    module load comsol/6.1
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

The Complex Systems Simulation and Optimization group has hosted introductory and advanced COMSOL trainings. The introductory training covered how to use the COMSOL GUI and run COMSOL in batch mode on Kestrel. The advanced training showed how to do a parametric study using different sweeps (running an interactive session is also included) and introduced equation-based simulation and parameter estimation. To learn more about using COMSOL on Kestrel, please refer to the training. The recording can be accessed at [Computational Sciences Tutorials](https://nrel.sharepoint.com/sites/ComputationalSciencesTutorials/Lists/Computational%20Sciences%20Tutorial%20Recordings/AllItems.aspx?viewid=7b97e3fa%2Dedf6%2D48cd%2D91d6%2Df69848525ba4&playlistLayout=playback&itemId=75) and the slides and models used in the training can be downloaded from [Github](https://github.com/NREL/HPC/tree/master/applications/comsol/comsol-training).