#Intel Inspector XE

*Intel Inspector XE is an easy to use memory checker and thread checker for serial and parallel applications written in C, C++, C#, F#, and Fortran. It takes you to the source locations of threading and memory errors and provides a call stack to help you determine how you got there. This tool has a GUI and a command line interface.*


load application specific modules

`module load comp-intel`

Setup the environment to use parallel studio

`. /nopt/nrel/apps/compilers/intel/2019.5/parallel_studio_xe_2019/psxevars.sh`

set your tmpdir, and don't forget to clean it after your job completes.

`export TMPDIR=/scratch/$USER/tmp`

You can list all the available profiling options for the machine you're running this tool on, from the GUI or from the command line using:

`inspxe-cl -collect-list`

This tool has a lot of features, that can be accessed from the GUI:

`inspxe-gui`
