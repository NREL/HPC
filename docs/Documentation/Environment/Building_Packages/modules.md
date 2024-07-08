---
layout: default
title: Create a Modulefile
parent: Building Packages
grand_parent: Intermediate
---
# Setting up your module
Modulefiles can be hosted in any location that the user makes known to the Lmod system via the ```module use``` command. By copying the reference modules, we set up into custom locations and modifying to your own tastes, you can host personal, project, or multi-project collections that you have complete control over.

Now that the package has been installed to your preferred location, we can set up an environment module.

## Create a directory for your module
If this is your first package, then you probably need to create a place to collect modulefiles. For example, `mkdir -p /scratch/$USER/modules/default`.

You can look at the systems module collection(s), _e.g._, `/nopt/nrel/apps/modules/default/modulefiles` on Eagle or `/nopt/nrel/apps/modules/default` on Kestrel, to see how modules are organized from a filesystem perspective. 

In short, each library, application, or framework has its own directory in the `modulefiles` directory, and the modulefile itself sits either in this directory, or one level lower to accomodate additional versioning. 

In this example, there is the MPI version (4.1.0), as well as the compiler type and version (GCC 8.4.0) to keep track of. 

So, we'll make a `/scratch/$USER/modules/default/openmpi/4.1.0` directory, and name the file by the compiler version used to build (gcc-8.4.0). 
You're free to modify this scheme to suit your own intentions.

If you want to create module collections that all users on a project can see and use, you can instead create a new directory in a project location (e.g., ```/projects/<your project name>```).

## Copy the original modulefile to your new directory
The modulefiles that we provide are only a starting point. For maximum control, users should copy these files from the locations in /nopt to their own locations for which they have write access.

Users may and should freely copy these example modulefiles to preferred locations and customize them for their own use cases. This can be particularly desirable to preserve a critical workflow as the software environment changes on Eagle or Kestrel, or to change the behavior, e.g., turn off automatic loading of prerequisites. 

In the `openmpi/4.1.0/gcc840` directory you just made, or whatever directory name you chose, copy the actual modulefile. 

It is much easier to copy an example from the system collection than to write one _de novo_. To copy the system modulefile, run:

???+ example "On Eagle"
	```
	cp /nopt/nrel/apps/modules/default/modulefiles/openmpi/4.0.4/gcc-8.4.0.lua /scratch/$USER/modules/default/openmpi/4.1.0/.
	```

???+ example "On Eagle"
	```
	cp /nopt/nrel/apps/modules/default/compilers_mpi/openmpi/4.1.5-gcc /scratch/$USER/modules/default/openmpi/4.1.0/.
	```

???+ warning "OpenMPI modulefile on Kestrel"
	Please note that the OpenMPI modulefile on Kestrel is of TCL type
	It is not necessary for you to know the language to modify our examples.
	
	The Lmod modules system uses the Lua language natively for module code. Tcl modules will also work under Lmod, but don't offer quite as much flexibility.
## Edit the modulefile copy as needed
For this example, (a) the OpenMPI version we're building is 4.1.0 instead of 4.0.4 on Eagle or 4.1.5 on Kestrel, and (b) the location is in `/scratch/$USER`, rather than `/nopt/nrel/apps`. 

So, edit `/scratch/$USER/modules/default/openmpi/4.1.0/gcc-8.4.0.lua` to make the required changes. 

Most of these changes only need to be made at the top of the file; variable definitions take care of the rest.

## Add your new modulefile location to a login script
```$MODULEPATH``` is an environment variable that the system searches to find available modules. In order for the system to be able to find your new module, the module use command may be added to a login script (e.g., ```.bash_profile```) or issued in an interactive shell or job script. In your `$HOME/.bash_profile`, add the following line near the top:

```
module use -a /scratch/$USER/modules/default
```

The -a flag appends the path that follows to environment variable $MODULEPATH; leaving it out will prepend the path. The first module found in searching $MODULEPATH is used, so the search order is important.

If you've built packages before and enabled them this way, you don't have to do this again!

!!! note
	Since new versions of software are periodically added to the system, check current availability with the ```module spider``` command. If a module is needed often, the ```module load <module_name>``` command can also be put in ```.bash_profile``` or other shell startup files.

## Check if your personal modules are available
Now logout, log back in, and you should see your personal modules collection with a brand new module.

```
$ module avail

---------------------------------- /scratch/$USER/modules/default -----------------------------------
openmpi/4.1.0/gcc-8.4.0
```

Notice that the ".lua" extension does not appear--the converse is also true, if the extension is missing it will not appear via module commands!
As a sanity check, it's a good idea to load the module, and check that an executable file you know exists there is in fact on your PATH:

```
$ module load openmpi/4.1.0/gcc-8.4.0
$ which mpirun
/scratch/$USER/openmpi/4.1.0-gcc-8.4.0/bin/mpirun
```