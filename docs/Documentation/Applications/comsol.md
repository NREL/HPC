---
title: Comsol
parent: Applications
---

# COMSOL Multiphysics 
*COMSOL Multiphysics is a versatile finite element analysis and simulation package. The COMSOL graphical user interface (GUI) environment is supported primarily for building and solving small models while operation in batch mode allows users to scale their models to larger, higher-fidelity studies.*

Currently, we host three floating network licenses and a number of additional modules. Issue the command `lmstat.comsol` to see current license status and COMSOL modules available.

## Building a COMSOL Model
Extensive documentation is available in the menu: **Help > Documentation**. For beginners, it is highly recommended to follow the steps in *Introduction to COMSOL Multiphysics* found in **Help > Documentation**.

For instructional videos, see the [COMSOL website](https://www.comsol.com) Video Gallery.

## Building Models in the COMSOL GUI
Before beginning, it is good practice to check the license status with:

```
[user@el3 ~]$ lmstat.comsol
```

When licenses are available, COMSOL can be used by starting the COMSOL GUI which allows you to build models, run the COMSOL computational engine, and analyze results. The COMSOL GUI can be accessed through a [FastX desktop](https://eagle-dav.hpc.nrel.gov/session/) by opening a terminal and running the following commands:

```
[user@ed3 ~]$ module purge
[user@ed3 ~]$ module load comsol/6.0
[user@ed3 ~]$ vglrun comsol
```

Because FastX desktop sessions are supported from DAV node nodes shared between multiple HPC users, limits are placed on how much memory and compute resources can be consumed by a single user/job. For this reason, it is recommended that the GUI by primarily used to define the problem and run small-scale tests to validate its operation before moving the model to a compute node for larger-scale runs. For jobs that require both large-scale compute resources and GUI interactivity simultaneously, there is partial support for running the GUI from an X-enabled shell (ssh -Y ...) on a compute node if the `vglrun comosl` command is replaced with:

```
[user@r1i7n24 ~]$ comsol -3drend sw
```

However, the performance may be slow and certain display features may behave unexpectedly.

## Running a COMSOL Model in Batch Mode
You can save your model built in FastX+GUI mode into a file such as `myinputfile.mph`. Once that's available, the following job script shows how to run a single process multithreaded job in batch mode:

??? example "Example Submission Script"

    ```bash
    #!/bin/bash
    #SBATCH --job-name=comsol-batch-1proc
    #SBATCH --time=00:20:00
    #SBATCH --nodes=1
    #SBATCH --account=<your-allocation-id>
    #SBATCH --output=comsol-%j.out
    #SBATCH --error=comsol-%j.err

    # This helps ensure your job runs from the directory
    # from which you ran the sbatch command
    cd $SLURM_SUBMIT_DIR

    # Set up environment, and list to stdout for verification
    module purge
    module load comsol/6.0
    echo " "
    module list
    echo " "

    inputfile=$SLURM_SUBMIT_DIR/myinputfile.mph
    outputfile=$SLURM_SUBMIT_DIR/myoutputfilename
    logfile=$SLURM_SUBMIT_DIR/mylogfilename

    # Run a COMSOL job with 36 threads.
    # -np = number of threads per rank

    comsol batch -np 36 -inputfile $inputfile -outputfile $outputfile –batchlog $logfile
    ```

Once this script file (assumed to be named `script-comsol.slurm`) is saved, it can be submitted to the job scheduler with

```
[user@el3 ~]$ sbatch script-comsol.slurm
```

## Running a COMSOL Model in Batch Mode (with MPI)
To configure a COMSOL job with multiple MPI ranks, required for any job where the number of nodes >1, you can build on the following template:

??? example "Example Multiprocess Submission Script"
    ```bash
    #!/bin/bash
    #SBATCH --job-name=comsol-batch-4proc
    #SBATCH --time=00:20:00
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=2
    #SBATCH --account=<your-allocation-id>
    #SBATCH --output=comsol-%j.out
    #SBATCH --error=comsol-%j.err

    # This helps ensure your job runs from the directory
    # from which you ran the sbatch command
    cd $SLURM_SUBMIT_DIR

    # Set up environment, and list to stdout for verification
    module purge
    module load comsol/6.0
    echo " "
    module list
    echo " "

    inputfile=$SLURM_SUBMIT_DIR/myinputfile.mph
    outputfile=$SLURM_SUBMIT_DIR/myoutputfilename
    logfile=$SLURM_SUBMIT_DIR/mylogfilename

    # Run a 2-node, 4-rank parallel COMSOL job with 18 threads for each rank.
    # -nn = total number of MPI ranks
    # -nnhost = number of MPI ranks per "host", here equivalent to node
    # -np = number of threads per rank

    comsol –nn 4 -nnhost 2 batch -np 18 -inputfile $inputfile -outputfile $outputfile –batchlog $logfile
    ```

The job script is submitted to the scheduler just the same as above for the single-process example. For jobs that require >1 node, this approach, which uses multiple MPI ranks, must be used. Note that in this case, we choose 4 MPI ranks, 2 per node, each using 18 threads to demonstrate the available submission options *not* any optimal performance recommendation. A different arrangement, e.g., `-nn 2 --nnhost 1 batch -np 36`, which translates to 2 MPI ranks, 1 per node, each using 36 threads may perform better for your application. The optimal configuration depends on your particular problem and choice of solver, so some experimentation may be required.
