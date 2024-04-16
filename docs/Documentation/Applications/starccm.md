# Running STAR-CCM+ Software


Simcenter STAR-CCM+ is a multiphysics CFD software that enables CFD engineers to model the complexity and explore the possibilities of products operating under real-world conditions. For information about the software's features, see the [STAR-CCM+
website](https://mdx.plm.automation.siemens.com/star-ccm-plus).

STAR-CCM+ is installed on both Eagle and Kestrel but it is not supported on Vermilion or Swift. The network
licenses are checked out from the license server running on 1wv11lic02.nrel.gov. 

## Running STAR-CCM+ in GUI

STAR-CCM+ can be run interactively on both Eagle and Kestrel using X windows by running the following commands in the terminal of an X window.

```bash
module load starccm
starccm+
```

## Running STAR-CCM+ in Batch Mode

To run STAR-CCM+ in batch mode, first, you need to build your simulation `<your_simulation.sim>` and
put it in your project directory:

```bash
ls /projects/<your_project>/sim_dir
your_simulation.sim
```

Then you need to create a Slurm script `<your_scriptfile>` as shown below to submit the job:

???+ example "Example Submission Script"

    ```bash
    #!/bin/bash -l
    #SBATCH --time=2:00:00             # walltime limit of 2 hours
    #SBATCH --nodes=2                  # number of nodes
    #SBATCH --ntasks-per-node=36       # number of tasks per node (<=36 on Eagle, <=104 on Kestrel)
    #SBATCH --ntasks=72                # total number of tasks
    #SBATCH --job-name=your_simulation # name of job
    #SBATCH --account=<allocation-id>  # name of project allocation
    
    module load starccm                # load starccm module
    
    rm -rf /projects/<your_project>/sim_dir/simulation.log   # remove the log file from last run
    # Run Job
    
    echo "------ Running Starccm+ ------"
        
    starccm+ -np $SLURM_NTASKS -batch /projects/<your_project>/sim_dir/your_simulation.sim >> simulation.log
    
    echo "------ End of the job ------"
    ```

Note that you must give the full path of your input file in the script.

By default, STAR-CCM+ uses OpenMPI. However, the performance of OpenMPI on Kestrel is poor when running on multiple nodes. Intel MPI and Cray MPI are recommended for STAR-CCM+ on Kestrel.  Cray MPI is expected to have a better performance than Intel MPI. 

### Running STAR-CCM+ with Intel MPI

STAR-CCM+ comes with its own Intel MPI. To use the Intel MPI, the Slurm script should be modified to be:

???+ example "Example Intel MPI Submission Script"

    ```bash
    #!/bin/bash -l
    #SBATCH --time=2:00:00             # walltime limit of 2 hours
    #SBATCH --nodes=2                  # number of nodes
    #SBATCH --ntasks-per-node=36       # number of tasks per node (<=36 on Eagle, <=104 on Kestrel)
    #SBATCH --ntasks=72                # total number of tasks
    #SBATCH --job-name=your_simulation # name of job
    #SBATCH --account=<allocation-id>  # name of project allocation
    
    module load starccm                # load starccm module
    
    export UCX_TLS=tcp                 # telling IntelMPI to treat the network as ethernet (Kestrel Slingshot can be thought of as ethernet) 
                                       # by using the tcp protocol
    
    rm -rf /projects/<your_project>/sim_dir/simulation.log   # remove the log file from last run
    # Run Job
    
    echo "------ Running Starccm+ ------"
        
    starccm+ -mpi intel -np $SLURM_NTASKS -batch /projects/<your_project>/sim_dir/your_simulation.sim >> simulation.log
    
    echo "------ End of the job ------"
    ```

We are specifying the MPI to be Intel MPI in the launch command. By default, Intel MPI thinks the network on which it is running is Infiniband. Kestrel’s is Slingshot, which you can think of as ethernet on steroids. The command `export UCX_TLS=tcp` is telling Intel MPI to treat the network as ethernet by using the tcp protocol.

To modify the settings for built-in Intel MPI, users can refer to the documentation of STAR-CCM by running `starccm+ --help`.

### Running STAR-CCM+ with Cray MPI

STAR-CCM+ can run with Cray MPI. The following Slurm script submits STAR-CCM+ job to run with Cray MPI.

???+ example "Example Cray MPI Script"

    ```bash
    #!/bin/bash -l
    #SBATCH --time=2:00:00             # walltime limit of 2 hours
    #SBATCH --nodes=2                  # number of nodes
    #SBATCH --ntasks-per-node=36       # number of tasks per node (<=36 on Eagle, <=104 on Kestrel)
    #SBATCH --ntasks=72                # total number of tasks
    #SBATCH --job-name=your_simulation # name of job
    #SBATCH --account=<allocation-id>  # name of project allocation
    
    module load starccm                # load starccm module
    
    rm -rf /projects/<your_project>/sim_dir/simulation.log   # remove the log file from last run
    # Run Job
    
    echo "------ Running Starccm+ ------"
        
    starccm+ -mpi crayex -np $SLURM_NTASKS -batch /projects/<your_project>/sim_dir/your_simulation.sim >> simulation.log
    
    echo "------ End of the job ------"
    ```
