---
layout: default
title: Modules
parent: Swift
grand_parent: Systems
---

# Swift Modules
This describes how to activate and use the modules available on Swift. 
## Source 
Environments are provided with a number of commonly used modules including compilers, common build tools, specific AMD optimized libraries, and some analysis tools. When you first login there is a default set of modules available.  These can be seen by running the command:

 ```
 module avail 
 ```

Since Swift is a new machine we are experimenting with additional environments. The environments are in date stamped subdirectory under in the directory /nopt/nrel/apps.  Each environemnt directory has a file myenv.\*.   If the myenv.\*. is missing from a directory then that environment is a work in progress.   Sourcing myenv.\* file will enable the environment and give you a new set of modules.  

For example to enable the environment /nopt/nrel/apps/210728a source the provided environment file. 

```
source /nopt/nrel/apps/210728a/myenv.2107290127
```

You will now have access to the modules provided. These can be listed using the following: 

```
ml avail 
```

If you want to build applications you can then module load compilers and the like; for example

```
ml gcc openmpi
```

will load gnu 9.4 and openmpi.

Software is installed using a spack hierarchy. It is possible to add software to the hierarchy.  This should be only done by people responsible for installing software for all users.  It is also possible to do a spack install creating a new level of the hierarchy in your personal space.  These procedures are documented in [https://github.nrel.gov/tkaiser2/spackit.git](https://github.nrel.gov/tkaiser2/spackit.git) in the file `Notes03.md` under the sections **Building on the hierarchy** and **Building outside the hierarchy**.  If you want to try this please contact [Tim Kaiser](mailto:tkaiser2@nrel.gov) to walk through the procedure.


 Most environments have an example directory.  You can copy this directory to you own space and compile and run the examples.  The files runintel and runopenmp are
 simple batch scripts.  These also have "module load" lines that you need to run before building with either compiler set.


