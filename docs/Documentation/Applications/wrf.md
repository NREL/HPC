# How to Use the WRF Application Software 

**Documentation:** [Weather Research Framework (WRF) Model](https://www.mmm.ucar.edu/models/wrf)

*The [WRF](https://www.mmm.ucar.edu/models/wrf) model is a state of the art mesoscale numerical weather prediction system designed for both atmospheric research and operational forecasting applications.*

## Getting Started

This section provides the minimum amount of information necessary to
successfully run a WRF job on the NREL Kestrel cluster. First, we show
how to use WRF given that we may have different versions of WRF
in different toolchains already built and available as modules.


```
% module avail wrf
     wrf/4.2.2-cray (D)    
     wrf/4.2.2-intel
     wrf/4.6.1-cray-mpich-gcc
```

The command `module avail wrf` displays the available WRF modules for various WRF versions and toolchains. The WRF version 4.2.2 uses the Cray and Intel toolchains, while version 4.6.1 uses the GNU toolchain. We suggest using the newest module because of its bug fixes, large domain handling capacity, enhanced parallel IO, and faster solution times. Since WRF doesn't currently support GPUs, no modules are available for running it on GPUs.

The following job script demonstrates the use of the latest WRF module. This job needs 8 nodes, each running 96 tasks, for a total of 3072 tasks. When running your job, only modify the node count, total core count, job name, runtime, partition, and account in the example. For optimal performance, configure the NIC policy to NUMA, use a single OMP thread, 96 tasks per node, employ block-block distribution, and bind tasks by rank to CPUs within NUMA nodes.

### Sample Job Script

??? example "Kestrel-CPU"

	```slurm
	#!/bin/bash
	
	#SBATCH --job-name=<"job-name">
	#SBATCH --nodes=8
	#SBATCH --ntasks-per-node=96
	#SBATCH --time=<hour:minute:second>
	#SBATCH --partition=<partition-name>
	#SBATCH --account=<account-name>
	#SBATCH --exclusive
	#SBATCH --mem=0
	
	module load PrgEnv-gnu/8.5.0
	module load cray-mpich/8.1.28
	module load cray-libsci/23.12.5
	module load wrf/4.6.1-cray-mpich-gcc
	module list

	export MPICH_OFI_NIC_POLICY=NUMA
	export OMP_NUM_THREADS=1

	srun -N 8 -n 3072 --ntasks-per-node=96 --distribution=block:block --cpu_bind=rank_ldom wrf.exe

	```

To submit this job script, named `submit_wrf.sh`, do ``` sbatch ./submit_wrf.sh ```

## Supported Version

| Kestrel |
|:-------:
| 4.6.1   |

## Advanced

### Build Instructions from Source

All WRF versions are available for download at this [link](https://github.com/wrf-model/WRF/releases). To build WRF, load the `netcdf` module; this automatically loads `hdf5`, `pnetcdf`, and other necessary dependencies. After completing the WRF build, download and build the WRF Pre-processing System (WPS) version from [here](https://github.com/wrf-model/WPS/releases). Building WPS requires loading the `jasper` module, which will automatically load `libpng`. The instructions below will guide you through installing your chosen WRF and WPS versions.

??? example "Building on Kestrel with the GNU Toolchain"
   	```
	# Get a compute node
	$ salloc --time=02:00:00 --account= <project account> --partition=shared --nodes=1 --ntasks-per-node=52

	# Load the netcdf and jasper modules
	$ module load PrgEnv-gnu/8.5.0
	$ module load cray-mpich/8.1.28
	$ module load cray-libsci/23.12.5
	$ module load netcdf/4.9.3-cray-mpich-gcc
	$ module load jasper/1.900.1-cray-mpich-gcc

	# Set the runtime environment
	$ export PATH="/usr/bin:${PATH}"
  	$ export LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH}"

	# Set paths to the WRF and WPS directories
	$ export WRF_DIR=<Path to WRF directory>
	$ export WPS_DIR=<Path to WPS directory>

	# Configure WRF
	$ cd ${WRF_DIR}
	$ ./configure
	$ Enter selection [1-83] : 35
	$ Compile for nesting? (1=basic, 2=preset moves, 3=vortex following) [default 1]:1

	# Compile WRF
	$ ./compile -j 48 em_real

	# Configure WPS
	$ cd ${WRF_DIR}
	$ ./configure
	$ Enter selection [1-44] : 3

	# Append “-fopenmp” to the WRF_LIB line in the configuration.wps file
	WRF_LIB         = -L$(WRF_DIR)/external/io_grib1 -lio_grib1 \
                        -L$(WRF_DIR)/external/io_grib_share -lio_grib_share \
                        -L$(WRF_DIR)/external/io_int -lwrfio_int \
                        -L$(WRF_DIR)/external/io_netcdf -lwrfio_nf \
                        -L$(NETCDF)/lib -lnetcdff -lnetcdf -fopenmp
        # Compile WPS
	$ ./compile
	
	```

## WRF Resources

The WRF community offers helpful resources, including [tutorials](https://www2.mmm.ucar.edu/wrf/OnLineTutorial/) and user [support](https://forum.mmm.ucar.edu/forums/frequen\
tly-asked-questions.115/).

