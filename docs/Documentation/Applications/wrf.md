# How to Use the WRF Application Software 

**Documentation:** [Weather Research Framework (WRF) Model](https://www.mmm.ucar.edu/models/wrf)

*The [WRF](https://www.mmm.ucar.edu/models/wrf) model is a state of the art mesoscale numerical weather prediction system designed for both atmospheric research and operational forecasting applications.*

## Getting Started

This section provides the minimum amount of information necessary to
successfully run a WRF job on the NLR Kestrel cluster. First, we show
how to use WRF given that we may have different versions of WRF
in different toolchains already built and available as modules.


```
% module avail wrf
     wrf/4.2.2-cray (D)    
     wrf/4.2.2-intel
     wrf/4.6.1-cray-mpich-gcc
```

The command `module avail wrf` displays the available WRF modules for various WRF versions and toolchains. The WRF version 4.2.2 uses the Cray and Intel toolchains, while version 4.6.1 uses the GNU toolchain. We suggest using the newest module because of its bug fixes, large domain handling capacity, enhanced parallel IO, and faster solution times. Since WRF doesn't currently support GPUs, no modules are available for running it on GPUs.

!!! note "RHEL 9"
	On Kestrel's RHEL 9 CPU nodes, `module avail wrf` shows a different set of
	modules built against the RHEL 9 software stack:

	```
	% module avail wrf
	     wrf/4.5.1-craype-gnu
	     wrf/4.5.1-craype-gnu-wps-pnetcdf
	     wrf/4.5.1-intel (D)
	     wrf/4.8.0-craype-gnu
	```

The following job script demonstrates the use of the latest WRF module. This job needs 8 nodes, each running 96 tasks, for a total of 3072 tasks. When running your job, only modify the node count, total core count, job name, runtime, partition, and account in the example. For optimal performance, configure the NIC policy to NUMA, use a single OMP thread, 96 tasks per node, employ block-block distribution, and bind tasks by rank to CPUs within NUMA nodes.

### Sample Job Script

??? example "Kestrel-CPU (RHEL 8)"

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

??? example "Kestrel-CPU (RHEL 9)"

	```slurm
	#!/bin/bash

	#SBATCH --job-name=<"job-name">
	#SBATCH --nodes=4
	#SBATCH --ntasks-per-node=104
	#SBATCH --time=<hour:minute:second>
	#SBATCH --partition=<partition-name>
	#SBATCH --account=<account-name>
	#SBATCH --exclusive
	#SBATCH --mem=0

	module load wrf/4.8.0-craype-gnu
	module list

	export MPICH_OFI_NIC_POLICY=NUMA
	export OMP_NUM_THREADS=1

	cd $SLURM_SUBMIT_DIR

	# 1. Build wrfinput/wrfbdy (use ideal.exe instead for idealised cases)
	srun --overlap --ntasks=$SLURM_NTASKS \
	     --ntasks-per-node=$SLURM_NTASKS_PER_NODE real.exe

	# 2. Main WRF integration
	srun --overlap --ntasks=$SLURM_NTASKS \
	     --ntasks-per-node=$SLURM_NTASKS_PER_NODE wrf.exe

	```

	The `--overlap` flag is required so the `srun` step can share the
	resources granted to the batch job. Cray MPICH has no `mpirun` — do not
	wrap `wrf.exe` in `mpirun` or `bash -c`. Run the case from a shared
	filesystem (`/scratch/$USER/...` or `/projects/<project>/...`).


To submit this job script, named `submit_wrf.sh`, do ``` sbatch ./submit_wrf.sh ```

## Supported Version

| Kestrel (RHEL 8) | Kestrel (RHEL 9) |
|:----------------:|:----------------:|
| 4.6.1, 4.5.1     | 4.8.0, 4.5.1     |

## Advanced

### RHEL 9 modules and toolchains

Four WRF modules are available on RHEL 9. Every module provides a complete
**WRF + WPS** install — `wrf.exe` / `real.exe` / `ndown.exe` / `tc.exe` plus
`geogrid.exe` / `metgrid.exe` / `ungrib.exe` — and sets `$WRF_DIR` and
`$WPS_DIR`.

| Module | WRF / WPS | Compiler / MPI | Notes |
|--------|-----------|----------------|-------|
| `wrf/4.8.0-craype-gnu` | WRF 4.8.0 / WPS 4.6.0 | gfortran 14.2.0 (spack) + Cray MPICH 8.1.32 |  Parallel netCDF I/O (PnetCDF 1.14.1, `io_form=11`) and parallel HDF5 1.14.6 linked in. |
| `wrf/4.5.1-craype-gnu` | WRF 4.5.1 / WPS 4.5 | gfortran 13.3.1 (PrgEnv-gnu) + Cray MPICH 8.1.32 | Supports PnetCDF (`io_form=11`). |
| `wrf/4.5.1-craype-gnu-wps-pnetcdf` | WRF 4.5.1 / WPS 4.5 | same as `4.5.1-craype-gnu` | Identical WRF binaries to `4.5.1-craype-gnu`; the **only** difference is the WPS `metgrid.exe`, a fork that writes `io_form_metgrid = 11` (parallel PnetCDF) across all MPI ranks. Use when metgrid I/O is a bottleneck at scale. |
| `wrf/4.5.1-intel` | WRF 4.5.1 / WPS 4.5 | Intel oneAPI 2025.3 (`ifx`) + Intel MPI 2021.17.2 | Default (`D`) on RHEL 9 when no version is specified. Supports PnetCDF (`io_form=11`). Launches with `mpirun`, or `srun` with `I_MPI_PMI_LIBRARY=/nopt/slurm/current/lib/libpmi2.so`. |

Each module prints a doc-file path on load (`module help <name>`). The
authoritative run instructions, namelist examples, and validated benchmarks
live in the in-built READMEs, e.g.
`/nopt/nlr/apps/kestrel-cpu/software/wrfRHEL9/wrf-4.8.0-craype-gnu/README.md`.

#### Parallel netCDF I/O (`io_form`)

Parallel I/O is opt-in — WRF defaults to `io_form_history = 2` (serial netCDF-3,
master-only writes) even though these builds link PnetCDF. To use the parallel
PnetCDF path, set in `&time_control` of `namelist.input`:

```
&time_control
  io_form_history      = 11    ! PnetCDF parallel writes (RECOMMENDED)
  io_form_restart      = 11    ! PnetCDF parallel restart
  io_form_input        = 2     ! read wrfinput serially (fine at load)
  io_form_boundary     = 2     ! read wrfbdy serially
/
```

`io_form=11` produces CDF-2/CDF-5 files that `ncdump`, `nco`, `xarray`, and
`netCDF4-python` read natively. Do **not** use `io_form = 13` (netCDF-4 /
parallel HDF5) on these builds — it is not linked in (4.5.1) and fails at
runtime with error 100 on 4.8.0; use `11` instead.

#### Parallel WPS metgrid

For the parallel-metgrid fork (`wrf/4.5.1-craype-gnu-wps-pnetcdf`), also set
`io_form_metgrid = 11` in `namelist.wps` (keep `io_form_geogrid = 2`). The
forked `metgrid.exe` has been verified bit-identical to stock output (89
variables, 1–32 ranks, single- and multi-node). Full recipes are in the
maintainer guide at
`/nopt/nlr/apps/kestrel-cpu/software/wrfRHEL9/wrf-4.5.1-wps-pnetcdf-guide.md`.

For the bundled WPS in each module: `geogrid.exe` and `metgrid.exe` are built
`dmpar` and can be launched with `srun` for parallelism; `ungrib.exe` is serial
by design. GRIB2 output is **not** compiled into WRF itself — GRIB2 is only
used on the WPS side (ungrib reading GRIB forcing via Jasper/libpng).

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

