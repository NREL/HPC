---
layout: default
title: Create a Modulefile
parent: Building Packages
grand_parent: Intermediate
---
# Setting up your module

1. Now that the package has been installed to your preferred location, we can set up an environment module.

	a. If this is your first package, then you probably need to create a place to collect modulefiles. 
    For example, `mkdir -p /scratch/$USER/modules/default`.

	b. You can look at the systems module collection(s), _e.g._, `/nopt/nrel/apps/modules/default/modulefiles` on Eagle or `/nopt/nrel/apps/modules/default` on Kestrel, to see how modules are organized from a filesystem perspective. 
    In short, each library, application, or framework has its own directory in the `modulefiles` directory, and the modulefile itself sits either in this directory, or one level lower to accomodate additional versioning. 
    In this example, there is the MPI version (4.1.0), as well as the compiler type and version (GCC 8.4.0) to keep track of. 
    So, we'll make a `/scratch/$USER/modules/default/openmpi/4.1.0` directory, and name the file by the compiler version used to build (gcc-8.4.0). 
    You're free to modify this scheme to suit your own intentions.

	c. In the `openmpi/4.1.0/gcc840` directory you just made, or whatever directory name you chose, goes the actual modulefile. 
    It's much easier to copy an example from the system collection than to write one _de novo_, so you can do
    
    ???+ example "On Eagle"
	     ```
	     cp /nopt/nrel/apps/modules/default/modulefiles/openmpi/4.0.4/gcc-8.4.0.lua /scratch/$USER/modules/default/openmpi/4.1.0/.
	     ```

    ???+ example "On Eagle"
	     ```
	     cp /nopt/nrel/apps/modules/default/compilers_mpi/openmpi/4.1.5-gcc /scratch/$USER/modules/default/openmpi/4.1.0/.
	     ```
    
    ???+ warning "OpenMpi modulefile on Kestrel"
         Please note that the OpenMpi modulefile on Kestrel is of TCL type
         It is not necessary for you to know the language to modify our examples. 

	The Lmod modules system uses the Lua language natively for module code. 
    Tcl modules will also work under Lmod, but don't offer quite as much flexibility.
	
	d. For this example, (a) the OpenMPI version we're building is 4.1.0 instead of 4.0.4 on Eagle or 4.1.5 on Kestrel, and (b) the location is in `/scratch/$USER`, rather than `/nopt/nrel/apps`. 
    So, edit `/scratch/$USER/modules/default/openmpi/4.1.0/gcc-8.4.0.lua` to make the required changes. 
    Most of these changes only need to be made at the top of the file; variable definitions take care of the rest.

	e. Now you need to make a one-time change in order to see modules that you put in this collection (`/scratch/$USER/modules/default`). 
    In your `$HOME/.bash_profile`, add the following line near the top:

	```
	module use /scratch/$USER/modules/default
	```

	Obviously, if you've built packages before and enabled them this way, you don't have to do this again!

2. Now logout, log back in, and you should see your personal modules collection with a brand new module.

	```
	[$USER@el1 ~]$ module avail
	
	---------------------------------- /scratch/$USER/modules/default -----------------------------------
	openmpi/4.1.0/gcc-8.4.0
	```
	
	Notice that the ".lua" extension does not appear--the converse is also true, if the extension is missing it will not appear via module commands!
	As a sanity check, it's a good idea to load the module, and check that an executable file you know exists there is in fact on your PATH:
	
	```
	[$USER@el1 ~]$ module load openmpi/4.1.0/gcc-8.4.0
	[$USER@el1 ~]$ which mpirun
	/scratch/$USER/openmpi/4.1.0-gcc-8.4.0/bin/mpirun
	```

