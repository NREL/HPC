# Template for an Application Page

**Documentation:** [Weather Research Framework (WRF) Model](https://www.mmm.ucar.edu/models/wrf)

*Write a brief description of the program here. Keep the italics.*
*[The WRF](https://www.mmm.ucar.edu/models/wrf) model is a state of the art mesoscale numerical weather prediction system designed for both atmospheric research and operational forecasting applications. 

## Getting Started

This section provides the minimum amount of information necessary to successfully run a WRF job on an NREL Cluster.
#This information should be as complete and self-contained as possible.

```
module avail wrf
     wrf/4.1.3/intel-20.1.217-mpi (D)    wrf/4.2.1/intel-20.1.217-mpi
     
```

The `module avail wrf` command shows that two wrf modules are
available for two different versions of wrf and built with the intel
compiler v20.1 toolchain. This command also shows that the version
4.1.3 is the default build which is automatically loaded with `module
load wrf`. Users are free to choose any of the module versions
available for use. Currently there are no modules to run wrf on GPUs, but
there is current effort underway to make that available on future systems.

Next, we look at how to use the wrf module. Below is an example job script

### Example Job Scripts

??? example "Kestrel CPU"

	```slurm
	#!/bin/bash

	# In a comment summarize the hardware requested, e.g. number of nodes, 
        # number of tasks per node

	#SBATCH --time=
	#SBATCH --nodes=
	#SBATCH --ntasks-per-node=
	#SBATCH --partition=
	#SBATCH --account=

	# include a section of relevant export and module load commands, e.g.:

	module purge
	module load intel-mpi
	module load netcdf-c/4.7.4/intel
	module load netcdf-f/4.5.3/intel-serial
	module load pnetcdf/1.12.1
	module load hdf5/1.12.0/intel-impi
	module load wrf/4.1.3

	export OMP_NUM_THREADS=1

	# include a sample srun command or similar
	srun wrf.exe

	```


Include instructions on how to submit the job script

To submit the above wrf jobscript named `submit_wrf.sh`, do ``` sbatch submit_wrf.sh ```

## Supported Versions

| Kestrel | Swift | Vermillion |
|:-------:|:-----:|:----------:|
| 0.0.0   | 0.0.0 | 0.0.0      |

## Advanced

Include advanced user information about the code here (see BerkeleyGW page for some examples)

One common "advanced case" might be that users want to build their own version of the code.

### Building From Source

Here, give detailed and step-by-step instructions on how to build the code, if this step is necessary. Include detailed instructions for how to do it on each applicable HPC system. Be explicit in your instructions. Ideally a user reading one of the build sections can follow along step-by-step
and have a functioning build by the end.

If building from source is not something anyone would reasonably want to do, remove this section.

Be sure to include where the user can download the source code

??? example "Building on Kestrel"

	Include here, for example, a Kestrel-specific makefile (see berkeleygw example page). This template assumes that we build the code with only one toolchain, which may not be the case. If someone might reasonably want to build with multiple toolchains, use the "Multiple toolchain instructions on Kestrel" template instead.
	
	```
	Include relevant commands in blocks.
	```
	or as in-line `blocks`

	Be sure to state how to set-up the necessary environment, e.g.:

	```
	module load gcc/8.4.0
	module load openmpi/3.1.6/gcc-8.4.0
	module load hdf5/1.10.6/gcc-ompi
	```

	Give instructions on compile commands. E.g., to view the available make targets, type `make`. To compile all program executables, type:

	```
	make cleanall
	make all
	```
	
??? example "Building on Vermillion"

	information on how to build on Vermillion

??? example "Building on Swift"

	information on how to build on Swift


## Troubleshooting

Include known problems and workarounds here, if applicable

