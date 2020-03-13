## Configuring your build

1. We will illustrate a package build that relies on the popular autotools system. Colloquially, this is
the `configure; make; make install` process that is often encountered first by those new to package
builds on Linux. Other build systems like CMake (which differ primarily in the configuration steps)
won't be covered. If you need to build a package that relies on CMake, please contact hpc-help@nrel.gov
for assistance.

2. Now that you've acquired and unpacked the package tarball and changed into the top-level directory
of the package, you should see a script named "configure". In order to see all available options to
an autotools configure script, use `./configure -h` (don't forget to include the `./` explicit path,
otherwise the script will not be found in the default Linux search paths, or worse, a different script
will be found).

	We will build with the following command: 
	```
	./configure --prefix=/scratch/$USER/openmpi/2.1.3-unthr-gcc-4.8.5 --with-gnu-ld --with-verbs=/usr --with-tm=/nopt/torque --enable-mpi-cxx --enable-mpi-fortran --without-scif --enable-static
	```
	These options are given for the following reasons.

	* `--prefix=` : This sets the location that "make install" will ultimately populate. If this isn't given, generally the default is to install into /usr or /usr/local, both of which require privileged access. We'll set up the environment using environment modules to point to this custom location.
	* `--with-gnu-ld` : Letting the build system know that linking will be done with GNU's linker, rather than a commercial or alternative open one.
	* `--with-verbs=` : This is how Infiniband is enabled. The relevant headers are in /usr/include, and libraries in /usr/lib64. Specifying the containing directory (_i.e._, /usr) is sufficient for the build system to find them.
	* `--with-tm=` : tm -> Torque Manager, this allows integration with the resource manager for setting certain default behaviors (_e.g._, to look for $PBS_* environment variables as a source of system information).
	* `--enable-mpi-cxx` : Build the C++ interface for the libraries
	* `--enable-mpi-fortran` : Build the Fortran interface for the libraries
	* `--without-scif` : SCIF is needed for Intel Xeon Phi support, otherwise just additional complexity
	* `--enable-static` : Build the .a archive files for static linking of applications

	NOTE: This build should work with the libxml2-devel package installed. As of May 10, 2018, this package is not present on CentOS 7 nodes. See [here](http://www.nordugrid.org/documents/rpm_for_everybody.html#11) for how to unpack RPM files for users, `man gcc` and the `-L` option to point the linker to alternative locations, and [here](https://centos.pkgs.org/7/centos-x86_64/libxml2-devel-2.9.1-6.el7_2.3.x86_64.rpm.html) for the required RPM package. You will have to change the symlink in `<wherever you install>/usr/lib64` to point to the system `/usr/lib64/libxml2.so.2.9.1` library.

	We'll use the system-provided GCC (version 4.8.5) for this illustration. Remember that if you want to use a different version of a compiler for your applications, load the associated module first (_e.g._, `gcc/7.2.0`).

	After lots of messages scroll by, you should be returned to a prompt following a summary of options. It's not a bad idea to glance through these, and make sure everything makes sense and is what you intended.

3. Now that the build is configured, you can "make" it. For packages that are well integrated with automake, you can speed the build up by parallelizing it over multiple processes with the `-j #` option. If you're building this on a compute node, feel free to set this option to the total number of cores available. On the other hand, if you're using a login node, be a good citizen and leave cores available for other users (_i.e._, don't use more than 4).

	```
	make -j 4
	```

4. Try a `make check` and/or a `make test`. Not every package enables these tests, but if they do, it's a great idea to run these sanity checks to find if your build is perfect, maybe-good-enough, or totally wrong before building lots of other software on top of it.

5. Assuming checks passed if present, it's now time for `make install`.

6. Now that the package has been installed to your preferred location, we can set up an environment module.

	a. If this is your first package, then you probably need to create a place to collect modulefiles. For example, `mkdir -p /scratch/$USER/modules/default/modulefiles`.

	b. You can look at the systems module collection(s), _e.g._, `/nopt/nrel/apps/modules/centos7/modulefiles`, to see how modules are organized from a filesystem perspective. In short, each library, application, or framework has its own directory in the `modulefiles` directory, and the modulefile itself sits in this directory, and is named as the package version. So, we'll make a `/scratch/$USER/modules/default/modulefiles/openmpi-gcc` directory. You're free to modify this scheme; for example, if you plan on having a software stack built on gcc 4.8.5 (the system version) AND on gcc 7.2.0 (via modules), then you might name this directory `openmpi-gcc485`, or `openmpi-gcc_system`. Or, make the distinction in the name of the actual modulefile.

	c. In the `openmpi-gcc` directory you just made, or whatever directory name you chose, goes the actual modulefile. It's much easier to copy an example from the system collection than to write one _de novo_, so you can do

	```
	cp /nopt/nrel/apps/modules/centos7/modulefiles/openmpi-gcc/2.1.2-4.8.5 /scratch/$USER/modules/default/modulefiles/openmpi-gcc/2.1.3-4.8.5
	```

	d. For this example, (a) the OpenMPI version we're building is 2.1.3 instead of 2.1.2, and (b) the location is in `/scratch/$USER`, rather than `/nopt/nrel/apps`. So, edit `/scratch/$USER/modules/default/modulefiles/openmpi-gcc/2.1.3-4.8.5` to make the required changes. Most of these changes only need to be made at the top of the file; variable definitions take care of the rest.

	e. Now you need to make a one-time change in order to see modules that you put in this collection (`/scratch/$USER/modules/default/modulefiles`). In your `$HOME/.bash_profile`, add the following line near the top:

	```
	module use /scratch/$USER/modules/default/modulefiles
	```

	Obviously, if you've built packages before and enabled them this way, you don't have to do this again!

7. Now logout, log back in, and you should see your personal modules collection with a brand new module.

	```
	[cchang@login4 01:57:13 /scratch/cchang]$ module avail
	
	---------------------------------- /scratch/cchang/modules/default/modulefiles -----------------------------------
	openmpi-gcc/2.1.3-4.8.5
	```
	
	As a sanity check, it's a good idea to load the module, and check that an executable file you know exists there is in fact on your PATH:
	
	```
	[cchang@login4 01:58:32 /scratch/cchang]$ module load openmpi-gcc/2.1.3-4.8.5
	[cchang@login4 02:00:26 /scratch/cchang]$ which mpirun
	/scratch/cchang/openmpi/2.1.3-unthr-gcc-4.8.5/bin/mpirun
	```
	
