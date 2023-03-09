#Intel Trace Analyzer

Intel Trace Analyzer and Collector is a tool for understanding the behavior of MPI applications. Use this tool to visualize and understand MPI parallel application behavior, evaluate load balancing, learn more about communication patterns, and identify communication hot spots.

The recommended way to use this tool on Eagle is to collect data from the command line and view the data using the GUI.

Example batch script to collect MPI communication data:

```bash
#!/bin/bash --login
#SBATCH -J <job name>
#SBATCH -q <queue>
#SBATCH -N <nodes>
#SBATCH -t 00:30:00
#SBATCH -A <Allocation handle>

# set your tmpdir, and don't forget to clean it after your job
# completes.
export TMPDIR=/scratch/$USER/tmp

# load application specific modules
module load comp-intel
# Setup the environment to use parallel studio and load the Intel MPI module
module load intel-mpi
. /nopt/nrel/apps/compilers/intel/2019.5/parallel_studio_xe_2019/psxevars.sh

# to profile the executable, just append '-trace' to mpirun
mpirun -trace -n 4 ./executable.exe
# this generates a .stf file that can viewed using the GUI
```

GUI: `traceanalyzer`

???+ tip "Note"

    Before you begin, please make sure that your application is compiled with the debug flag (-g), to enable profiling and debugging.
    
    When using the suite of tools from Intel Parallel Studio on Eagle, we recommend that you set your TMPDIR to point to a location in your SCRATCH directory.
    
    export TMPDIR=/scratch/$USER/tmp
    Important: Please make sure that you clean up this directory after your job completes.    
