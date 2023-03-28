# How to run MAP

## FastX Setup
In order to run MAP you will need to have FastX. Follow instructions here to download the and install the desktop client and connect to DAV nodes using it: [FastX](https://nrel.github.io/HPC/Documentation/Software_Tools/FastX/fastx/)


## Program Setup
ARM-MAP can show you how much time was spent on each line of code. To see the source code in map, you must use a version of your code that is compiled with the debug flag. For most compilers, this is `-g`. Note: You should not just use a debug build but should keep optimization flags `-O0` turned on when profiling. 

For more information, see the ARM-MAP documentation on getting started. In particular, if your program uses statically linked libraries, the map profiler libraries will not be automatically linked and you will need to do so yourself. 

Note: Ensure that your program is working before trying to run it in MAP

## Running map
Once you have FastX installed and an appropriate build of your program to profile, you can obtain profiling data through map with the following steps. We will profile VASP as an example.

1.	Start an xterm window from within FastX connected to a DAV node
2.	Start an interactive job session.  
    Use the debug or other partitions as appropriate.  
    `$ salloc --nodes=<N>  --time=<time> --account=<handle>`
3.	Load the arm module   
    Additionally load any other modules needed to run your program  
    `$ module load arm`  
    `$ module load mkl intel-mpi #for VASP`  
4.	Start a map session using the command `map`  
    Optionally, navigate to your working directory and give map the path to your exe  
    `$ cd PATH/TO/YOUR/WORKING/DIRECTORY`  
    `$ map PATH/TO/YOUR/PROGRAM/exe`  

You should now see the arm forge GUI appear and a submission box with some information filled out if you followed the optional directions. Otherwise use the GUI to input them now.

Make sure the path to the application includes your program exe.
Make sure your working directory includes your input files, or specify your stdin file and its path.
Adjust other parameters as needed for profiling.

5.	Start your map by clicking “Run”

You should now see the profiling data we described in the previous section [MAP](/Documentation/Development/Performance_Tools/index.md). Please refer to that page as well as the [ARM-MAP Documentation](https://developer.arm.com/documentation/102732/1910) for more details on what you can learn from such profiles.
