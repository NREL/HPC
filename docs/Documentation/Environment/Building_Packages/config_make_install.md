---
layout: default
title: Config, Make, Install
parent: Building Packages
grand_parent: Intermediate
---

# Configuring your build


1. We will illustrate a package build that relies on the popular autotools system. 
Colloquially, this is the `configure; make; make install` process that is often encountered first by those new to package builds on Linux. 
Other build systems like [CMake](../../Development/Build_Tools/cmake.md) (which differ primarily in the configuration steps) won't be covered. 
If you need to build a package that relies on CMake, please contact hpc-help@nlr.gov for assistance.

2. We'll use *GCC* version 14.2.0 for this illustration, so load it first as the NLR toolchain base module (module reset to start clean, then load the compiler):

	```bash
	module reset
	module load gcc/14.2.0
	```

???+ note "Building on Kestrel"
     You can use any NLR-provided GCC version visible via `module spider gcc`, or the `oneapi`/`llvm` toolchains instead. 
     Do not mix a loaded NLR toolchain (`gcc`, `oneapi`, `llvm`) with the Cray Programming Environment (`cpe-stack`) in the same session.

3. Now that you've acquired and unpacked the package tarball and changed into the top-level directory of the package, you should see a script named "configure". 
In order to see all available options to an autotools configure script, use `./configure -h` (don't forget to include the `./` explicit path, otherwise the script will not be found in the default Linux search paths, or worse, a different script will be found).

	We will build with the following command: 
	```
	./configure --prefix=/scratch/$USER/openmpi/5.0.5-gcc-14.2.0 --with-slurm --with-pmi=$(dirname $(dirname $(which srun))) --with-gnu-ld --with-lustre --with-zlib --without-psm --without-psm2 --with-ucx --without-verbs --with-hwloc=external --with-hwloc-libdir=$(dirname $(module show hwloc 2>&1 | grep -m1 LD_LIBRARY_PATH | awk -F'"' '{print $2}')) --enable-cxx-exceptions --enable-mpi-cxx --enable-mpi-fortran --enable-static
	```
	These options are given for the following reasons.

	* `--prefix=` : This sets the location that "make install" will ultimately populate. If this isn't given, generally the default is to install into /usr or /usr/local, both of which require privileged access. We'll set up the environment using environment modules to point to this custom location.
	* `--with-slurm` : Enables the interface with the Slurm resource manager
	* `--with-pmi=` : Point to the Process Management Interface, the abstraction layer for MPI options
	* `--with-gnu-ld` : Letting the build system know that linking will be done with GNU's linker, rather than a commercial or alternative open one.
	* `--with-lustre` : Enable Lustre features
	* `--with-zlib` : Enable compression library
	* `--without-psm[2]` : Explicitly turn off interfaces to Intel's Performance Scaled Messaging for the now-defunct Omni-Path network
	* `--with-ucx=` : Point to UCX, an intermediate layer between the network drivers and MPI
	* `--without-verbs=` : For newer MPIs, communications go through UCX and/or libfabric, not directly to the Verbs layer
	* `--with-hwloc[-libdir]=` : Point to a separately built hardware localization library for process pinning. Use `module spider hwloc` to see which toolchains already provide one before building your own.
	* `--enable-cxx-exceptions`, `--enable-mpi-cxx` : Build the C++ interface for the libraries
	* `--enable-mpi-fortran` : Build the Fortran interface for the libraries
	* `--enable-static` : Build the .a archive files for static linking of applications

	After lots of messages scroll by, you should be returned to a prompt following a summary of options. 
    It's not a bad idea to glance through these, and make sure everything makes sense and is what you intended.

4. Now that the build is configured, you can "make" it. 
For packages that are well integrated with automake, you can speed the build up by parallelizing it over multiple processes with the `-j #` option. 
If you're building this on a compute node, feel free to set this option to the total number of cores available. 
On the other hand, if you're using a login node, be a good citizen and leave cores available for other users (_i.e._, don't use more than 4; Arbiter should limit access at any rate regardless of this setting).

	```
	make -j 4
	```

5. Try a `make check` and/or a `make test`. 
Not every package enables these tests, but if they do, it's a great idea to run these sanity checks to find if your build is perfect, maybe-good-enough, or totally wrong before building lots of other software on top of it.

6. Assuming checks passed if present, it's now time for `make install`. 
Assuming that completes without errors, you can move onto creating an environment module to use your new MPI library.

