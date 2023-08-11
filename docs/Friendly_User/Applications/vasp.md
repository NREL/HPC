## VASP modules

Section coming soon.

## Running VASP

We have found that it is optimal to run an intel toolchain build of VASP using cray-mpich-abi at runtime. Cray-mpich-abi has several dependencies on cray network modules, so the easiest way to load it is to first load ```PrgEnv-intel``` and then swap the default cray-mpich module for the cray-mpich-abi ```module swap cray-mpich cray-mpich-abi```. You must then load your intel compilers and math libraries, and unload cray's libsci. A sample script showing all of this is in the dropdown below.

!!! Note
    It is necessary to specify the launcher using srun --mpi=pmi2 

??? Sample job script for your own vasp build

    ```
    #!/bin/bash
    #SBATCH --nodes=2
    #SBATCH --tasks-per-node=104
    #SBATCH --time=2:00:00
    #SBATCH --mem=0 # ensures you are given all the memory on a node

    # Load cray-mpich-abi and its dependencies within PrgEnv-intel, intel compilers, mkl, and unload cray's libsci
    source /nopt/nrel/apps/env.sh
    module purge
    module load PrgEnv-intel
    module swap cray-mpich cray-mpich-abi
    module unload cray-libsci/22.10.1.2
    module load intel-oneapi-compilers/2022.1.0  
    module load intel-oneapi-mkl/2023.0.0-intel

    set -x 
    export OMP_NUM_THREADS=1 #turns off multithreading
    export VASP_PATH=/PATH/TO/YOUR/vasp_exe


    srun --mpi=pmi2 ${VASP_PATH}/vasp_std |& tee out

    #Note: it may be optimal to run with more processers per task, especially for heavier gw calculations e.g:
    srun --mpi=pmi2 -ntasks 64 --ntasks-per-node=32 ${VASP_PATH}/vasp_std  |& tee out

    ```


## Building VASP

We reccomend building vasp with a full intel toolchain and launching with the cray-mpich-abi at runtime. Additionally, you should build on a compute node so that you have the same architechture as at runtime:
```
salloc -N 1 -p standard -t TIME [-A account once accounting has been implemented]
```
Then, load appropriate modules for your mpi, compilers, and math packages:
```
module purge
source /nopt/nrel/apps/env.sh  #to access all modules
module load craype-x86-spr #specifies sapphire rapids architecture
module load intel-oneapi-compilers
module load intel-oneapi-mpi
module load intel-oneapi-mkl
```
!!! Note
    On Kestrel, any modules you have loaded on the log in node will be copied to a compute node, and there are many loaded by default for the cray programming environment. Make sure you are using what you intend to. 

Sample makefiles for vasp5 and vasp6 on Kestrel can be found in our [Kestrel Repo](https://github.com/NREL/HPC/tree/master/kestrel) under the vasp folder.
