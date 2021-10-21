---
layout: default
title: Swift
parent: Systems
has_children: true
---

# Swift

Swift is an AMD based cluster. 

## Known issues and solutions
1. IntelMPI appears to be working properly
1. OpenMPI appears to be working properly 
1. There are some example slrum scripts in the example directory. 
1. There is a very basic version of conda in the "anaconda" directory in each  /nopt/nrel/apps/YYMMDDa directory. However, there is a more complete environment pointed to by the module under /nopt/nrel/apps/modules. This is set up like Eagle.  See: https://www.nrel.gov/hpc/eagle-software-python.html
1. There are no GPU nodes currently available on Swift.
1. Use `--cpus-per-task` with srun/sbatch otherwise some applications may only utilize a single core. This behavior differs from Eagle.
 1. By default, nodes can be shared between users.  To get exclusive access to a node use the `--exclusive` flag in your sbatch script or on the sbatch command line.
 1. User accounts have a default set of keys `cluster` and `cluster.pub`. The `config` file will use these even if you generate a new keypair using `ssh-keygen`. If you are adding your keys to Github or elsewhere you should either use `cluster.pub` or will have to modify the `config` file. 

