---
search:
  exclude: true
---

# BerkeleyGW

**Documentation:** [BerkeleyGW](https://berkeleygw.org/documentation/)

*[BerkeleyGW](https://www.berkeleygw.org) is a massively parallel many-body perturbation theory code capable of performing RPA, GW, and GW-BSE calculations, which can be used to investigate properties of materials with high accuracy.*

## Getting Started

This section provides the minimum amount of information needed to run a BerkeleyGW job on an NREL cluster.

First, see which versions of BerkeleyGW are available with `module avail` and load your preferred version with `module load`:

```
module avail berkeleygw
   berkeleygw/3.0.1-cpu    berkeleygw/3.0.1-gpu
```
The `module avail berkeleygw` command shows that two BerkeleyGW modules are available. To select the GPU-enabled version of BerkeleyGW, for example, we use the `module load` command:

```
module load berkeleygw/3.0.1-gpu
```

Next, create a job script. Below are example job scripts for the available NREL systems. Continuing the above example, we would select the "Kestrel GPU" example script.

### Sample Job Scripts

??? example "Kestrel CPU"

	```slurm
	#!/bin/bash

	# This job requests 72 MPI tasks across 2 nodes (36 tasks/node) and no threading

	#SBATCH --time=01:00:00
	#SBATCH --nodes=2
	#SBATCH --ntasks-per-node=36
	#SBATCH --partition=standard
	#SBATCH --account=

	module load berkeleygw/3.0.1-cpu

	srun epsilon.cplx.x
	```
		
??? example "Kestrel GPU"

	Put job example here

??? example "Vermillion"

	Put job example here

??? example "Swift"

	Put job example here


Save the submit file as bgw.in, and submit with the command:

`sbatch bgw.in`

## Supported Versions

| Kestrel    | Swift | Vermillion |
|:----------:|:-----:|:----------:|
| 2.0, 3.0   | 0.0.0 | 0.0.0      |

## Advanced

### Wavefunction file: HDF5 vs Fortran binary

For large systems, the wavefunction binary file format yields significantly slower read-in times relative to an HDF5-format wavefunction file. The BerkeleyGW code includes utilities to convert wavefunction binary files to HDF5 format and vice-versa called hdf2wfn.x and wfn2hdf.x (see [documentation](
http://manual.berkeleygw.org/3.0/meanfield-utilities/#wfn2hdfx)). It is recommended to use HDF5-formatted wavefunction files where possible.

### Lustre File Striping

BerkeleyGW supports wavefunction files in HDF5 format and binary format. Wavefunction inputs to BerkeleyGW can become large depending on the system under investigation. Large (TODO: define large for Kestrel. Probably > 10 GB) HDF5 wavefunction files benefit from Lustre file striping, and the BerkeleyGW code can see major runtime speed-ups when using this feature.

!!! tip
	Binary format wavefunction files do not benefit from Lustre file striping

For more on Lustre file striping, see (TODO: documentation section on Lustre file striping?)

### Advanced submission script example

Because multiple executables in BerkeleyGW require the WFN input files (WFN and WFNq), we can streamline the file linking inside a submission script. We can also include the Lustre file striping step in our submission script. The below example script shows how this can be done for the BerkeleyGW epsilon executable.

??? example "Advanced submit script"

	This script assumes you build your own version of BerkeleyGW. If not, remove the `BGW=/path/to/where/you/built/BerkeleyGW/bin` and `ln -s $BGW/epsilon.cplx.x .` lines.

	Be sure to load the proper modules (see Getting Started if not building your own version.)

	```slurm
	#!/bin/bash
	#SBATCH -t 00:20:00
	#SBATCH -N 8
	#SBATCH --gpus-per-node=4
	#SBATCH -C gpu
	#SBATCH -o BGW_EPSILON_%j.out
	#SBATCH --account=
	
	BGW=/path/to/where/you/built/BerkeleyGW/bin
	WFN_folder=/path/to/folder/that/contains/WFN/and/WFNq
	
	mkdir BGW_EPSILON_$SLURM_JOBID
	lfs setstripe -c 60 BGW_EPSILON_$SLURM_JOBID
	cd    BGW_EPSILON_$SLURM_JOBID
	ln -s $BGW/epsilon.cplx.x .
	ln -s  ../epsilon.inp .
	ln -sfn  ${WFN_folder}/WFNq.h5      .   
	ln -sfn  ${WFN_folder}/WFN.h5   ./WFN.h5
	
	ulimit -s unlimited
	export OMP_PROC_BIND=true
	export OMP_PLACES=threads
	export BGW_WFN_HDF5_INDEPENDENT=1
	
	export OMP_NUM_THREADS=16
	srun -n 32 -c 32 --cpu-bind=cores epsilon.cplx.x
	```
	
	This script will create a directory "BGW_EPSILON_$SLURM_JOBID" (where `$SLURM_JOBID` will be a numeric ID), stripe the directory with a stripe count of 60, link the epsilon executable, WFNq, and WFN files to the directory, and run BerkeleyGW with 32 GPUs.

### Building Instructions

First, [download BerkeleyGW](https://berkeleygw.org/download/).

Then, follow the build instructions in the "building" drop-downs below for the cluster you will be running on.

??? example "Building on Kestrel"

	The following `arch.mk` file was used to build BerkeleyGW-3.0 on Kestrel on (date).
        Copy this arch.mk file into your BerkeleyGW directory.
	
	```make
	COMPFLAG  = -DGNU
	PARAFLAG  = -DMPI -DOMP
	MATHFLAG  = -DUSESCALAPACK -DUNPACKED -DUSEFFTW3 -DHDF5
	
	FCPP    = /usr/bin/cpp -C
	F90free = mpifort -ffree-form -ffree-line-length-none -fopenmp -fno-second-underscore -cpp
	LINK    = mpifort -fopenmp
	# FHJ: -funsafe-math-optimizations breaks Haydock and doesn't give any significant speedup
	FOPTS   = -O3 -funroll-loops 
	FNOOPTS = $(FOPTS)
	MOD_OPT = -J  
	INCFLAG = -I
	
	C_PARAFLAG  = -DPARA
	CC_COMP = mpiCC
	C_COMP  = mpicc
	C_LINK  = mpicc
	C_OPTS  = -O3 -ffast-math
	C_DEBUGFLAG = 
	
	REMOVE  = /bin/rm -f
	
	# Math Libraries                                                                                                                                                                                            
	FFTWPATH     =  /projects/scatter/mylibraries_CentOS77/
	#/nopt/nrel/apps/fftw/3.3.3-impi-intel/
	#FFTWLIB      = $(FFTWPATH)/lib/libfftw3.a
	FFTWLIB      =  $(FFTWPATH)/lib/libfftw3_omp.a $(FFTWPATH)/lib/libfftw3.a
	FFTWINCLUDE  =  $(FFTWPATH)/include
	
	LAPACKLIB = /projects/scatter/mylibraries_CentOS77/lib/libopenblas.a
	
	SCALAPACKLIB = /projects/scatter/mylibraries_CentOS77/lib/libscalapack.a
	
	HDF5PATH      = /nopt/nrel/apps/base/2020-05-12/spack/opt/spack/linux-centos7-x86_64/gcc-8.4.0/hdf5-1.10.6-dj4jq2ffttkdxksimqe47245ryklau4a
	HDF5LIB      =  ${HDF5PATH}/lib/libhdf5hl_fortran.a \
	                ${HDF5PATH}/lib/libhdf5_hl.a \
	                ${HDF5PATH}/lib/libhdf5_fortran.a \
	                ${HDF5PATH}/lib/libhdf5.a /home/ohull/.conda-envs/bgw/lib/libsz.a -lz -ldl
	HDF5INCLUDE  = ${HDF5PATH}/include

	PERFORMANCE  =

	TESTSCRIPT = 
	```

	Then, load the following modules:

	```
	module load gcc/8.4.0
	module load openmpi/3.1.6/gcc-8.4.0
	module load hdf5/1.10.6/gcc-ompi
	```

	Choose whether to use the real or complex flavor of BerkeleyGW by copying the corresponding file to flavor.mk. For example, for the complex version:

	`cp flavor_cplx.mk flavor.mk`

	Finally, compile the code. To view the available make targets, type `make`. To compile all BerkeleyGW executables, type:
	```
	make cleanall
	make all
	```

??? example "Building on Swift"

	TODO: add Swift build instructions

## Troubleshooting

Include known problems and workarounds here, if applicable

