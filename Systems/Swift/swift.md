---
layout: default
title: Swift
parent: Systems
has_children: true
---

# Swift

Swift is an AMD based cluster. 

## Known issues and solutions
1. To run IntelMPI programs must set I\_MPI\_PMI\_LIBRARY=/nopt/nrel/apps/210728a/level01/gcc-9.4.0/slurm-20-11-5-1/lib/libpmi2.so and use the  --mpi=pmi2 with srun
1. Stack size and max locked memory are very small, ulimit -s 8192 ulimit -l 64. This will break VASP
1. srun and salloc can not be used to get an interactive session.
1. OpenMPI appears to be working properly for a single node but parallel jobs must be launched with mpirun instead of srun
1. Multinode MPI is not working at this point.
1. Don't load the slurm module.  The version of slurm it points to has not been configured.
1. To use the configured version of slurm *export PATH=/nopt/nrel/slurm/bin:$PATH*
1. There are some example slrum scripts in the example directory.  Again, don't use IntelMPI on Swift.
1. There is a very basic version of conda in the "anaconda" directory in each  /nopt/nrel/apps/YYMMDDa directory. However, there is a more complete environment pointed to by the module under /nopt/nrel/apps/modules. This is set up like Eagle.  See: https://www.nrel.gov/hpc/eagle-software-python.html
1. There are no GPU nodes currently available on Swift.

