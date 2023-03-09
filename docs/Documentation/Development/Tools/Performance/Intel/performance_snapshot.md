#Performance Snapshot

*The new Application Performance Snapshot merges the earlier MPI Performance Snapshot and Application Performance Snapshot Tech Preview. 
MPI Performance Snapshot is no longer available separately, but all of its capabilities and more are available in the new combined snapshot. 
This tool lets you take a quick look at your application's performance to see if it is well optimized for modern hardware. 
It also includes recommendations for further analysis if you need more in-depth information.*

Using This Tool on Eagle

```bash
# load application specific modules
module load comp-intel
# Setup the environment to use parallel studio
. /nopt/nrel/apps/compilers/intel/2019.5/parallel_studio_xe_2019/psxevars.sh

# serial/SMP executable
$ aps <executable> # this generates an aps result directory
# DMP executable
$ mpirun -n 4 aps <executable>
# this generates an aps result directory # to gerate text and /hmtl result files:
$ aps --report=<the generated results directory from the previous step>
# the result file can be viewed in a browser or text editor
```
???+ tip "Note"

    Before you begin, please make sure that your application is compiled with the debug flag (-g), to enable profiling and debugging.
    
    When using the suite of tools from Intel Parallel Studio on Eagle, we recommend that you set your TMPDIR to point to a location in your SCRATCH directory.
    
    export TMPDIR=/scratch/$USER/tmp
    Important: Please make sure that you clean up this directory after your job completes.    
