## Running VASP

## VASP Documentation

This repo contains the results of two seperate VASP performance studies on Eagle and Swift using the VASP benchmarks in the [benchmarks directory](/HPC/applications/vasp/benchmarks). Each study evaluates performance differentl, as described below, and provides recommendations for running VASP most efficiently in the README files. The READMEs in each directory contain the following information.

Performance Study 1 (VASP6 on Eagle):
- Recommendations for setting LREAL
- Recommendations for setting cpu pinning
- Recommendations for setting NPAR
- Recommendations for setting NSIM
- Instructions for using the OpenMP version of VASP
- Instructions for running multiple VASP jobs on the same nodes (and [scripts to do so](https://github.com/claralarson/HPC/tree/master/applications/vasp/Performance%20Study%201/multi))
- Runtime comparison using VASP5

Performance Study 2 (VASP6 on Eagle and Swift):
- Information on how runtime scales with nodecount
- Recommendations for chosing the most efficient value of cpus/node
- Recommendations for running VASP on Eagle's GPU nodes (and [scripts to do so](https://github.com/claralarson/HPC/tree/master/applications/vasp/Performance%20Study%202/VASP%20scripts))
- Recommendations for chosing Intel MPI or Open MPI (and [scripts for running with both MPIs](https://github.com/claralarson/HPC/tree/master/applications/vasp/Performance%20Study%202/VASP%20scripts))
- Recommendations for setting KPAR
- Recommendations for setting cpu pinning
- Information on k-points scaling
- Instructions for running multiple VASP jobs on the same nodes on Swift (and [scripts to do so](https://github.com/claralarson/HPC/tree/master/applications/vasp/Performance%20Study%202/VASP%20scripts))

For information on running VASP on Vermilion, see the VASP section of [this documentation](https://nrel.github.io/HPC/Documentation/Systems/Vermillion/running/).
