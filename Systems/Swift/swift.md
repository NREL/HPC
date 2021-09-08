---
layout: default
title: Swift
parent: Systems
has_children: true
---

# Swift

Swift is an AMD based cluster. 

## Known issues and solutions
1. To run IntelMPI programs must
	1. export I\_MPI\_PMI\_LIBRARY=/nopt/nrel/apps/210728a/level01/gcc-9.4.0/slurm-20-11-5-1/lib/libpmi2.so
	1. export UCX_TLS=all
1. OpenMPI appears to be working properly 
1. There are some example slrum scripts in the example directory. 
1. There is a very basic version of conda in the "anaconda" directory in each  /nopt/nrel/apps/YYMMDDa directory. However, there is a more complete environment pointed to by the module under /nopt/nrel/apps/modules. This is set up like Eagle.  See: https://www.nrel.gov/hpc/eagle-software-python.html
1. There are no GPU nodes currently available on Swift.
1. Use `--cpus-per-task` with srun/sbatch otherwise some applications may only utilize a single core. This behavior differs from Eagle. 
1. User accounts have a default set of keys `cluster` and `cluster.pub`. The `config` file will use these even if you generate a new keypair using `ssh-keygen`. If you are adding your keys to Github or elsewhere you should either use `cluster.pub` or will have to modify the `config` file. 
1. Use the following commands to enable slurm:

```
source /nopt/nrel/apps/210728a/myenv.2107290127
module load slurm
```

