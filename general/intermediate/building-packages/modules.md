## Setting up your module

1. Now that the package has been installed to your preferred location, we can set up an environment module.

	a. If this is your first package, then you probably need to create a place to collect modulefiles. For example, `mkdir -p /scratch/$USER/modules/default/modulefiles`.

	b. You can look at the systems module collection(s), _e.g._, `/nopt/nrel/apps/modules/centos7/modulefiles`, to see how modules are organized from a filesystem perspective. In short, each library, application, or framework has its own directory in the `modulefiles` directory. So, we'll make a `/scratch/$USER/modules/default/modulefiles/openmpi-gcc` directory. You're free to modify this scheme; for example, if you plan on having a software stack built on gcc 4.8.5 (the system version) AND on gcc 7.2.0 (via modules), then you might name this directory `openmpi-gcc485`, or `openmpi-gcc_system`. Or, make the distinction in the name of the actual modulefile.

	c. In the `openmpi-gcc` directory you just made, or whatever directory name you chose, goes the actual modulefile. It's much easier to copy an example from the system collection than to write one de novo, so you can do

	```
	cp /nopt/nrel/apps/modules/centos7/modulefiles/openmpi-gcc/2.1.2-4.8.5 /scratch/$USER/modules/default/modulefiles/openmpi-gcc/2.1.3-4.8.5
	```

	d. For this example, (a) the OpenMPI version we're building is 2.1.3 instead of 2.1.2, and (b) the location is in `/scratch/$USER`, rather than `/nopt/nrel/apps`. So, edit `/scratch/$USER/modules/default/modulefiles/openmpi-gcc/2.1.3-4.8.5` to make the required changes. Most of these changes only need to be made at the top of the file; variable definitions take care of the rest.

	e. Now you need to make a one-time change in order to see modules that you put in this collection (`/scratch/$USER/modules/default/modulefiles`). In your `$HOME/.bash_profile`, add the following line near the top:

	```
	module use /scratch/$USER/modules/default/modulefiles
	```

	Obviously, if you've built packages before and enabled them this way, you don't have to do this again!

2. Now logout, log back in, and you should see your personal modules collection with a brand new module.

	```
	[cchang@login4 01:57:13 /scratch/cchang]$ module avail
	
	---------------------------------- /scratch/cchang/modules/default/modulefiles -----------------------------------
	openmpi-gcc/2.1.3-4.8.5
	```

	As a sanity check, it's a good idea to load the module, and check that an executable file that you know should exist there, is in fact on your `PATH`:

	```
	[cchang@login4 01:58:32 /scratch/cchang]$ module load openmpi-gcc/2.1.3-4.8.5
	[cchang@login4 02:00:26 /scratch/cchang]$ which mpirun
	/scratch/cchang/openmpi/2.1.3-unthr-gcc-4.8.5/bin/mpirun
	```

