# Intel Parallel Studio

*Intel Parallel Studio is a set of tools that enable developing and optimizing software for the latest processor architectures.*

Some of the tools available as part of the Intel Parallel Studio include:

??? example "Intel VTune Amplifier XE"
	
	Intel VTune Amplifier XE is a performance profiler for C, C++, C#, Fortran, Assembly and Java code. Hot spots analysis provides a sorted list of functions that use a lot of CPU time. Other features enable the user to quickly find common causes of slow performance in parallel programs, including waiting too long at locks and load imbalance among threads and processes. VTune Amplifier XE uses the Performance Monitoring Unit (PMU) on Intel processors to collect data with very low overhead.
	
	The recommended way to use this tool is to run the profiler from the command line and view the data using the GUI or generate a text report from the command line. 
	
	You can list all the available profiling options for the machine you're profiling on, from the GUI or from the command line using `amplxe-cl -collect-list`.
	
	Include the following in you batch script to get a HPC-characterization profile of you application:
	
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
	module load intel-oneapi-vtune
	# profile the executable
	amplxe-cl --collect hpc-performance ./executable.exe
	```

	GUI:
	
	`amplxe-gui`

??? example "Intel Trace Analyzer XE"
	
	Intel Trace Analyzer and Collector is a tool for understanding the behavior of MPI applications. Use this tool to visualize and understand MPI parallel application behavior, evaluate load balancing, learn more about communication patterns, and identify communication hot spots.
	
	The recommended way to use this tool is to collect data from the command line and view the data using the GUI.
	
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
	module load intel-oneapi-trace
	
	# to profile the executable, just append '-trace' to mpirun
	mpirun -trace -n 4 ./executable.exe
	# this generates a .stf file that can viewed using the GUI
	```	

	GUI:
	
	`traceanalyzer`

??? example "Intel Advisor XE"
	
	
	
	Intel Advisor helps with vectorization and threading in your C++ and Fortran Applications. This tool helps identify areas that would benefit the most from vectorization. It also helps with identifying what is blocking vectorization and gives insights to overcome it:
	
	```bash
	# load application specific modules
	module load intel-oneapi-advisor
	
	# set your tmpdir, and don't forget to clean it after your job
	# completes.
	export TMPDIR=/scratch/$USER/tmp
	```

	You can list all the available profiling options for the machine you're profiling on, from the GUI or from the command line using:
	
	`advixe-cl -collect-list`
	
	This tool has a lot of features that can be accessed from the GUI:
	
	`advixe-gui`

??? example "Intel Inspector XE"
	
	Intel Inspector XE is an easy to use memory checker and thread checker for serial and parallel applications written in C, C++, C#, F#, and Fortran. It takes you to the source locations of threading and memory errors and provides a call stack to help you determine how you got there. This tool has a GUI and a command line interface.
	
	```bash
	# load application specific modules
	module load intel-oneapi-inspector
	
	# set your tmpdir, and don't forget to clean it after your job
	# completes.
	export TMPDIR=/scratch/$USER/tmp
	```	
	
	You can list all the available profiling options for the machine you're running this tool on, from the GUI or from the command line using:
	
	`inspxe-cl -collect-list`
	
	This tool has a lot of features that can be accessed from the GUI:
	
	`inspxe-gui`

??? example "Intel Application Performance Snapshot"
	
	The new Application Performance Snapshot merges the earlier MPI Performance Snapshot and Application Performance Snapshot Tech Preview. MPI Performance Snapshot is no longer available separately, but all of its capabilities and more are available in the new combined snapshot. This tool lets you take a quick look at your application's performance to see if it is well optimized for modern hardware. It also includes recommendations for further analysis if you need more in-depth information.

	Using This Tool:
	
	```bash
	# load application specific modules
	module load intel-oneapi-vtune
	
	# serial/SMP executable
	$ aps <executable> # this generates an aps result directory
	# DMP executable
	$ mpirun -n 4 aps <executable>
	# this generates an aps result directory # to gerate text and /hmtl result files:
	$ aps --report=<the generated results directory from the previous step> 
	# the result file can be viewed in a browser or text editor
	```

**Before you begin, please make sure that your application is compiled with the debug flag (-g), to enable profiling and debugging.**


`export TMPDIR=/scratch/$USER/tmp`

!!! tip "Important:"
	Please make sure that you clean up this directory after your job completes.
