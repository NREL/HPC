---
layout: default
title: Modules
parent: Swift
grand_parent: Systems
---

# Swift modules
This describes how to activate and use the modules available on Swift. 

## Source 
Environments are provided with a number of commonly used modules including compilers, common build tools, specific AMD optimized libraries, and some analysis tools. The environments are in date stamped subdirectory under in the directory /nopt/nrel/apps.  Each environemnt directory has a file myenv.\*.   Sourcing that file will enable the environment.

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

Software is installed using a spack hierarchy. It is possible to add software to the hierarchy.  This should be only done by people responsible for installing software for 
all users.  It is also possible to do a spack install creating a new level of the hierarchy in your personal space.  These procedures are documented in https://github.nrel.gov/tkaiser2/spackit.git in the file Notes03.md under the sections **Building on the hierarchy** and **Building outside the hierarchy**.  If you want to try this please contact Tim Kaiser tkaiser2@nrel.gov to walk through the procedure.



## Know issues and solutions
1. While you can compile with IntelMPI programs do not launch correctly.
1. srun and salloc can not be used to get an interactive session.
1. OpenMPI appears to be working properly for a single node but parallel jobs must be launched with mpirun instead of srun
1. Multinode MPI is not working at this point.
1. Don't load the slurm module.  The version of slurm it points to has not been configured.
1. To use the configured version of slurm *export PATH=/nopt/nrel/slurm/bin:$PATH*
1. There are some example slrum scripts in the example directory.  Again, don't use IntelMPI on Swift.
1. There is a very basic version of conda in the "anaconda" directory in each  /nopt/nrel/apps/YYMMDDa directory. However, there is a more complete environment pointed to by the module under /nopt/nrel/apps/modules. This is set up like Eagle.  See: https://www.nrel.gov/hpc/eagle-software-python.html


