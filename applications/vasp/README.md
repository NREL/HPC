## Running VASP

VASP6 is available on Eagle and Swift. The code below demonstrates how to load different versions of VASP on each system and which accompanying modules to load. 

### On Eagle
Load VASP with Intel MPI:
```
ml vasp
```
Load VASP with Open MPI:
```
source /nopt/nrel/apps/210830a/myenv.2108301742
ml vasp/6.1.1-l2mkbb2
```
Load the GPU build of VASP:
```
module use /nopt/nrel/apps/220511a/modules/lmod/linux-centos7-x86_64/gcc/12.1.0
ml fftw nvhpc
export LD_LIBRARY_PATH=/nopt/nrel/apps/220511a/install/opt/spack/linux-centos7-skylake_avx512/gcc-12.1.0/nvhpc-22.3-c4qk6fly5hls3mjimoxg6vyuy5cc3vti/Linux_x86_64/22.3/compilers/extras/qd/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/nopt/nrel/apps/220511a/install/opt/spack/linux-centos7-skylake_avx512/gcc-12.1.0/nvhpc-22.3-c4qk6fly5hls3mjimoxg6vyuy5cc3vti/Linux_x86_64/22.3/compilers/extras/qd/lib:$LD_LIBRARY_PATH
export PATH=/projects/hpcapps/tkaiser2/vasp/6.3.1/nvhpc_acc:$PATH
```

### On Swift 
Load VASP with Intel MPI:
```
ml vaspintel 
ml slurm/21-08-1-1-o2xw5ti 
ml gcc/9.4.0-v7mri5d 
ml intel-oneapi-compilers/2021.3.0-piz2usr 
ml intel-oneapi-mpi/2021.3.0-hcp2lkf 
ml intel-oneapi-mkl/2021.3.0-giz47h4
```
Load VASP with Open MPI:
```
ml vasp 
ml slurm/21-08-1-1-o2xw5ti 
ml openmpi/4.1.1-6vr2flz
```

### On Vermilion

For information about accessing and running VASP on Vermilion, see the VASP section at the bottom of [this documentation](https://nrel.github.io/HPC/Documentation/Systems/Vermillion/running/). 

## VASP Documentation

This repo contains the results of two seperate VASP performance studies. The first, Performance Study 1, studies VASP performance on Eagle using the input files provided in the directory. The second, Performance Study 2, studies VASP performance on Eagle and Swift using benchmarks from the ESIF benchmarking suite, which can be found [here](https://github.com/NREL/ESIFHPC3/tree/master/VASP) or in the benchmarks folder in the Performance Harness 2 directory. Each study evaluates performance differently, as described below, and provides recommendations for running VASP most efficiently in the README files. The READMEs in each directory contain the following information.

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
