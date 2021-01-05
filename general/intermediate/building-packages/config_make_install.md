## Configuring your build

1. We will illustrate a package build that relies on the popular autotools system. Colloquially, this is
the `configure; make; make install` process that is often encountered first by those new to package
builds on Linux. Other build systems like CMake (which differ primarily in the configuration steps)
won't be covered. If you need to build a package that relies on CMake, please contact hpc-help@nrel.gov
for assistance.

2. We'll use GCC version 8.4.0 for this illustration, so load the associated module first (_i.e._, `gcc/8.4.0`).

3. Now that you've acquired and unpacked the package tarball and changed into the top-level directory
of the package, you should see a script named "configure". In order to see all available options to
an autotools configure script, use `./configure -h` (don't forget to include the `./` explicit path,
otherwise the script will not be found in the default Linux search paths, or worse, a different script
will be found).

	We will build with the following command: 
	```
	./configure --prefix=/scratch/$USER/openmpi/4.1.0-gcc-8.4.0 --with-slurm --with-pmi=/nopt/slurm/current --with-gnu-ld --with-lustre --with-zlib --without-psm --without-psm2 --with-ucx --without-verbs --with-hwloc=external --with-hwloc-libdir=/nopt/nrel/apps/base/2020-05-12/spack/opt/spack/linux-centos7-x86_64/gcc-8.4.0/hwloc-1.11.11-mb5lwdajmllvrdtwltwe3r732aca76ny/lib --enable-cxx-exceptions --enable-mpi-cxx --enable-mpi-fortran --enable-static LDFLAGS="-L/nopt/nrel/apps/base/2020-05-12/spack/opt/spack/linux-centos7-x86_64/gcc-8.4.0/11.0.2-4x2ws7fkooqbrerbsnfbzs6wyr5xutdk/lib64 -L/nopt/nrel/apps/base/2020-05-12/spack/opt/spack/linux-centos7-x86_64/gcc-8.4.0/cuda-11.0.2-4x2ws7fkooqbrerbsnfbzs6wyr5xutdk/lib64 -Wl,-rpath=/nopt/nrel/apps/base/2020-05-12/spack/opt/spack/linux-centos7-x86_64/gcc-8.4.0/hwloc-1.11.11-mb5lwdajmllvrdtwltwe3r732aca76ny/lib -Wl,-rpath=/nopt/nrel/apps/base/2020-05-12/spack/opt/spack/linux-centos7-x86_64/gcc-8.4.0/cuda-11.0.2-4x2ws7fkooqbrerbsnfbzs6wyr5xutdk/lib64" CPPFLAGS=-I/nopt/nrel/apps/base/2020-05-12/spack/opt/spack/linux-centos7-x86_64/gcc-8.4.0/hwloc-1.11.11-mb5lwdajmllvrdtwltwe3r732aca76ny/include
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
	* `--with-hwloc[-libdir]=` : Point to a separately built hardware localization library for process pinning
	* `--enable-cxx-exceptions`, `--enable-mpi-cxx` : Build the C++ interface for the libraries
	* `--enable-mpi-fortran` : Build the Fortran interface for the libraries
	* `--enable-static` : Build the .a archive files for static linking of applications
	* `LDFLAGS` : -L options point to non-standard library locations. -Wl,-rpath options embed paths into the binaries, so that having LD_LIBRARY_PATH set correctly is not necessary (i.e., no separate module for these components).
	* `CPPFLAGS` : Point to header files in non-standard locations.

	NOTE: The CUDA paths are not needed for CUDA function per se, but the resulting MPI errors out without setting them. There appears to be a lack of modularity that sets up a seemingly unneeded dependency.

	After lots of messages scroll by, you should be returned to a prompt following a summary of options. It's not a bad idea to glance through these, and make sure everything makes sense and is what you intended.

4. Now that the build is configured, you can "make" it. For packages that are well integrated with automake, you can speed the build up by parallelizing it over multiple processes with the `-j #` option. If you're building this on a compute node, feel free to set this option to the total number of cores available. On the other hand, if you're using a login node, be a good citizen and leave cores available for other users (_i.e._, don't use more than 4; Arbiter should limit access at any rate regardless of this setting).

	```
	make -j 4
	```

5. Try a `make check` and/or a `make test`. Not every package enables these tests, but if they do, it's a great idea to run these sanity checks to find if your build is perfect, maybe-good-enough, or totally wrong before building lots of other software on top of it.

6. Assuming checks passed if present, it's now time for `make install`.

7. Now that the package has been installed to your preferred location, we can set up an environment module.

	a. If this is your first package, then you probably need to create a place to collect modulefiles. For example, `mkdir -p /scratch/$USER/modules/default/modulefiles`.

	b. You can look at the systems module collection(s), _e.g._, `/nopt/nrel/apps/modules/default/modulefiles`, to see how modules are organized from a filesystem perspective. In short, each library, application, or framework has its own directory in the `modulefiles` directory, and the modulefile itself sits either in this directory, or one level lower to accomodate additional versioning. In this example, there is the MPI version (4.1.0), as well as the compiler type and version (GCC 8.4.0) to keep track of. So, we'll make a `/scratch/$USER/modules/default/modulefiles/openmpi/4.1.0` directory, and name the file by the compiler version used to build (gcc-8.4.0). You're free to modify this scheme to suit your own intentions.

	c. In the `openmpi/4.1.0/gcc840` directory you just made, or whatever directory name you chose, goes the actual modulefile. It's much easier to copy an example from the system collection than to write one _de novo_, so you can do

	```
	cp /nopt/nrel/apps/modules/default/modulefiles/openmpi/4.0.4/gcc-8.4.0.lua /scratch/$USER/modules/default/modulefiles/openmpi/4.1.0/.
	```

	The Lmod modules system uses the Lua language natively for module code. It is not necessary for you to know the language to modify our examples. Tcl modules will also work under Lmod, but don't offer quite as much flexibility.
	
	d. For this example, (a) the OpenMPI version we're building is 4.1.0 instead of 4.0.4, and (b) the location is in `/scratch/$USER`, rather than `/nopt/nrel/apps`. So, edit `/scratch/$USER/modules/default/modulefiles/openmpi/4.1.0/gcc-8.4.0.lua` to make the required changes. Most of these changes only need to be made at the top of the file; variable definitions take care of the rest.

	e. Now you need to make a one-time change in order to see modules that you put in this collection (`/scratch/$USER/modules/default/modulefiles`). In your `$HOME/.bash_profile`, add the following line near the top:

	```
	module use /scratch/$USER/modules/default/modulefiles
	```

	Obviously, if you've built packages before and enabled them this way, you don't have to do this again!

8. Now logout, log back in, and you should see your personal modules collection with a brand new module.

	```
	[cchang@el1 ~]$ module avail
	
	---------------------------------- /scratch/cchang/modules/default/modulefiles -----------------------------------
	openmpi/4.1.0/gcc-8.4.0
	```
	
	Notice that the ".lua" extension does not appear--the converse is also true, if the extension is missing it will not appear via module commands!
	As a sanity check, it's a good idea to load the module, and check that an executable file you know exists there is in fact on your PATH:
	
	```
	[cchang@el1 ~]$ module load openmpi/4.1.0/gcc-8.4.0
	[cchang@el1 ~]$ which mpirun
	/scratch/cchang/openmpi/4.1.0-gcc-8.4.0/bin/mpirun
	```
	
