# oneAPI Tools (formerly Intel Parallel Studio)

*oneAPI tools are a set of software that enable developing and optimizing software for the latest processor architectures.*

!!! note "Intel Parallel Studio is retired"
    Intel retired the Parallel Studio XE branding in favor of **Intel oneAPI** in 2020. The profiling and analysis tools formerly distributed as part of Parallel Studio (VTune, Trace Analyzer and Collector, Advisor, Inspector) are now distributed as individual oneAPI components. On Kestrel, these tools are made visible by first loading the `oneapi` toolchain module, then loading the specific tool module.

??? example "Intel VTune Profiler"
	
	Intel VTune Profiler (formerly VTune Amplifier XE) is a performance profiler for C, C++, C#, Fortran, Assembly and Java code. Hot spots analysis provides a sorted list of functions that use a lot of CPU time. Other features enable the user to quickly find common causes of slow performance in parallel programs, including waiting too long at locks and load imbalance among threads and processes. VTune uses the Performance Monitoring Unit (PMU) on Intel processors to collect data with very low overhead.
	
	The recommended way to use this tool is to run the profiler from the command line and view the data using the GUI or generate a text report from the command line. 
	
	You can list all the available profiling options for the machine you're profiling on, from the GUI or from the command line using `vtune -collect-list`.
	
	Include the following in you batch script to get a HPC-characterization profile of you application:
	
	```bash	
	#!/bin/bash --login
	#SBATCH -J <job name>
	#SBATCH -N <nodes>
	#SBATCH -t 00:30:00
	#SBATCH -A <Allocation handle>
	
	# $TMPDIR is already set to node-local scratch (/tmp/scratch/$SLURM_JOB_ID);
	# no need to export it yourself.
	# load the oneAPI toolchain, then the tool module it reveals
	module reset
	module load oneapi/2025.3.1
	module load intel-oneapi-vtune
	# profile the executable
	vtune -collect hpc-performance ./executable.exe
	```

	GUI:
	
	`vtune-gui`

??? example "Intel Trace Analyzer and Collector"
	
	Intel Trace Analyzer and Collector is a tool for understanding the behavior of MPI applications. Use this tool to visualize and understand MPI parallel application behavior, evaluate load balancing, learn more about communication patterns, and identify communication hot spots.
	
	The recommended way to use this tool is to collect data from the command line and view the data using the GUI.
	
	Example batch script to collect MPI communication data:
	
	```bash
	#!/bin/bash --login
	#SBATCH -J <job name>
	#SBATCH -N <nodes>
	#SBATCH -t 00:30:00
	#SBATCH -A <Allocation handle>
	
	module reset
	module load oneapi/2025.3.1
	module load intel-oneapi-trace
	
	# to profile the executable, just append '-trace' to mpirun
	mpirun -trace -n 4 ./executable.exe
	# this generates a .stf file that can viewed using the GUI
	```	

	GUI:
	
	`traceanalyzer`

??? example "Intel Advisor"
	
	Intel Advisor helps with vectorization and threading in your C++ and Fortran Applications. This tool helps identify areas that would benefit the most from vectorization. It also helps with identifying what is blocking vectorization and gives insights to overcome it:
	
	```bash
	module reset
	module load oneapi/2025.3.1
	module load intel-oneapi-advisor
	```

	You can list all the available profiling options for the machine you're profiling on, from the GUI or from the command line using:
	
	`advixe-cl -collect-list`
	
	This tool has a lot of features that can be accessed from the GUI:
	
	`advixe-gui`

??? example "Intel Inspector"
	
	Intel Inspector is an easy to use memory checker and thread checker for serial and parallel applications written in C, C++, C#, F#, and Fortran. It takes you to the source locations of threading and memory errors and provides a call stack to help you determine how you got there. This tool has a GUI and a command line interface.
	
	```bash
	module reset
	module load oneapi/2025.3.1
	module load intel-oneapi-inspector
	```	
	
	You can list all the available profiling options for the machine you're running this tool on, from the GUI or from the command line using:
	
	`inspxe-cl -collect-list`
	
	This tool has a lot of features that can be accessed from the GUI:
	
	`inspxe-gui`

??? example "Intel Application Performance Snapshot (APS)"
	
	Application Performance Snapshot merges the earlier MPI Performance Snapshot and Application Performance Snapshot Tech Preview. This tool lets you take a quick look at your application's performance to see if it is well optimized for modern hardware. It also includes recommendations for further analysis if you need more in-depth information. APS is distributed as part of the `intel-oneapi-vtune` module.

	Using This Tool:
	
	```bash
	module reset
	module load oneapi/2025.3.1
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

On Kestrel, `$TMPDIR` is automatically set to node-local scratch (`/tmp/scratch/$SLURM_JOB_ID`) for every job, so there is no need to export it yourself when using the oneAPI tools above.

!!! tip "Loading the tools"
	All of the oneAPI performance tools above require the `oneapi` toolchain module to be loaded first (`module reset; module load oneapi/2025.3.1`) before the tool-specific module (e.g., `intel-oneapi-vtune`) becomes visible. Use `module spider intel-oneapi-vtune` to confirm the exact version available.
