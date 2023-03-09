#Intel VTune

*Intel VTune Amplifier XE is a performance profiler for C, C++, C#, Fortran, Assembly and Java code. Hot spots analysis provides a sorted list of functions that use a lot of CPU time. Other features enable the user to quickly find common causes of slow performance in parallel programs, including waiting too long at locks and load imbalance among threads and processes. VTune Amplifier XE uses the Performance Monitoring Unit (PMU) on Intel processors to collect data with very low overhead.*

The recommended way to use this tool on Eagle, is to run the profiler from the command line and view the data using the GUI or generate a text report from the command line. 

You can list all the available profiling options for the machine you're profiling on, from the GUI or from the command line using amplxe-cl -collect-list.

Include the following in you batch script to get a HPC- characterization profile of you application:

```bash
#!/bin/bash --login
#SBATCH -J <job name>
#SBATCH -N <nodes>
#SBATCH -t 00:30:00
#SBATCH -A <Allocation handle>

# set your tmpdir, and don't forget to clean it after your job
# completes. 
export TMPDIR=/scratch/$USER/tmp
# load application specific modules
module load comp-intel
# Setup the environment to use parallel studio
. /nopt/nrel/apps/compilers/intel/2019.5/parallel_studio_xe_2019/psxevars.sh
# profile the executable
amplxe-cl --collect hpc-performance ./executable.exe
``` 


GUI: `amplxe-gui`


???+ tip "Note"

    Before you begin, please make sure that your application is compiled with the debug flag (-g), to enable profiling and debugging.
    
    When using the suite of tools from Intel Parallel Studio on Eagle, we recommend that you set your TMPDIR to point to a location in your SCRATCH directory.
    
    export TMPDIR=/scratch/$USER/tmp
    Important: Please make sure that you clean up this directory after your job completes.    
