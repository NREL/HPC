---
layout: default
title: Modules
parent: Swift
grand_parent: Systems
---

# Swift modules
This describes how to activate and use the modules available on Swift. 

## Source 
Environments are provided with a number of commonly used modules including compilers, common build tools, specific AMD optimized libraries, and some analysis tools. The environments are in date stamped in the directory /nopt/nrel/apps.  Each environemnt directory has a file myenv.\*.   Sourcing that file will enable the environment.

For example to enable the environment /nopt/nrel/apps/210728a source the provided environment file. 

```
source /nopt/nrel/apps/210728a/myenv.2107290127
```

You will now have access to the modules provided. These can be listed using the following: 

```
ml avail 
```

## Know issues and solutions
1. While you can compile with IntelMPI programs do not launch correctly.
2. OpenMPI appears to be working properly but parallel jobs must be launched with mpirun instead of srun
3. Don't load the slurm module.  The version of slurm it points to has not been configured.
4. To use the configured version of slurm *export PATH=/nopt/nrel/slurm/bin:$PATH*
5. There are some example slrum scripts in the example directory.  Again, don't use IntelMPI on Swift.
6. There is a very basic version of conda in the "anaconda" directory.  However, there is a more complete environment pointed to by the module under /nopt/nrel/apps/modules. This is set up like Eagle.  See: https://www.nrel.gov/hpc/eagle-software-python.html


