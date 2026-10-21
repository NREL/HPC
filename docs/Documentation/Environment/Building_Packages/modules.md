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

	b. You can look at the system's module collection(s), _e.g._, `/nopt/nrel/apps/cpu_stack/modules/default`, to see how modules are organized from a filesystem perspective. 
    In short, each library, application, or framework has its own directory in the `modulefiles` directory, and the modulefile itself sits either in this directory, or one level lower to accommodate additional versioning. 
    In this example, there is the MPI version (5.0.5), as well as the compiler type and version (GCC 14.2.0) to keep track of. 
    So, we'll make a `/scratch/$USER/modules/default/openmpi/5.0.5` directory, and name the file by the compiler version used to build (gcc-14.2.0). 
    You're free to modify this scheme to suit your own intentions.

	c. In the `openmpi/5.0.5` directory you just made, or whatever directory name you chose, goes the actual modulefile. 
    It's much easier to copy an example from the system collection than to write one _de novo_, so you can find and copy a comparable existing modulefile:

    ```
    module --redirect show openmpi | grep -i ':\s*$' # locate the active modulefile path
    cp /nopt/nrel/apps/cpu_stack/modules/default/openmpi/5.0.3-gcc.lua /scratch/$USER/modules/default/openmpi/5.0.5/gcc-14.2.0.lua
    ```

	The module system on Kestrel is Lmod 8.7.37, a Lua-based hierarchical module system. 
    Lua modulefiles (`.lua`) are recommended, but Tcl modulefiles will also work under Lmod, just with somewhat less flexibility.

	d. Edit `/scratch/$USER/modules/default/openmpi/5.0.5/gcc-14.2.0.lua` to point at your own build's `--prefix` location instead of the system one, and adjust the version strings baked into the file. 
    Most of these changes only need to be made at the top of the file; variable definitions take care of the rest.

	e. Now you need to make a one-time change in order to see modules that you put in this collection (`/scratch/$USER/modules/default`). 
    In your `$HOME/.bash_profile`, add the following line near the top:

	```
	module use /scratch/$USER/modules/default
	```

	Obviously, if you've built packages before and enabled them this way, you don't have to do this again!

2. Now logout, log back in (or run `module use /scratch/$USER/modules/default` directly), and you should see your personal modules collection with a brand new module.

	```
	[$USER@kl3 ~]$ module avail
	
	---------------------------------- /scratch/$USER/modules/default -----------------------------------
	openmpi/5.0.5/gcc-14.2.0
	```
	
	Notice that the `.lua` extension does not appear--the converse is also true, if the extension is missing it will not appear via module commands!
	As a sanity check, it's a good idea to load the module, and check that an executable file you know exists there is in fact on your PATH:
	
	```
	[$USER@kl3 ~]$ module reset
	[$USER@kl3 ~]$ module load openmpi/5.0.5/gcc-14.2.0
	[$USER@kl3 ~]$ which mpirun
	/scratch/$USER/openmpi/5.0.5-gcc-14.2.0/bin/mpirun
	```

!!! tip "Quick testing without editing your profile"
    To test a modulefile without first adding it to your `.bash_profile`, use `module use $(pwd)/modules` from the directory containing it, as described in [Installing Software](../../../RHEL9_upgrade/modules/install_software.md#testing-modulefiles).

