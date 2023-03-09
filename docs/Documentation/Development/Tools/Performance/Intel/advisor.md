#Intel Advisor

*Intel Advisor helps with vectorization and threading in your , C++ and Fortran Applications. This tool helps identify areas that would benefit the most from vectorization.*
It also helps with identifying what is blocking vectorization and gives insights to overcome it:


load application specific modules

`module load comp-intel`

Setup the environment to use parallel studio

`. /nopt/nrel/apps/compilers/intel/2019.5/parallel_studio_xe_2019/psxevars.sh`

set your tmpdir, and don't forget to clean it after your job completes.

`export TMPDIR=/scratch/$USER/tmp`

You can list all the available profiling options for the machine you're profiling on, from the GUI or from the command line using:

`advixe-cl -collect-list`

This tool has a lot of features that can be accessed from the GUI:

`advixe-gui`


???+ tip "Note"

    Before you begin, please make sure that your application is compiled with the debug flag (-g), to enable profiling and debugging.
    
    When using the suite of tools from Intel Parallel Studio on Eagle, we recommend that you set your TMPDIR to point to a location in your SCRATCH directory.
    
    export TMPDIR=/scratch/$USER/tmp
    Important: Please make sure that you clean up this directory after your job completes.    
