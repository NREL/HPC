## VASP modules on Kestrel

There are modules for CPU builds of VASP 5.4.4 and VASP 6.3.2 each with solvation, transision state tools, and BEEF-vdW functionals. These modules can be loaded with ```module load vasp/5.4.4``` or ```module load vasp/6.3.2```. A sample job script is shown below.

!!! Note
    It is necessary to specify the launcher using srun --mpi=pmi2 

??? example "Sample job script: using modules"

    ```
    #!/bin/bash
    #SBATCH --nodes=2
    #SBATCH --tasks-per-node=104
    #SBATCH --time=2:00:00
    #SBATCH --account=<your-account-name>
    #SBATCH --job-name=<your-job-name>

    source /nopt/nrel/apps/env.sh  #the need for this will eventually be removed
    module load vasp/6.3.2

    srun --mpi=pmi2 vasp_std |& tee out

    ```

## Compiling VASP yourself

This section has recommendations for toolchains to use for building and running VASP. Please read carefully before compiling on Kestrel's cray architecture.

### Building VASP

We recomend building vasp with a full intel toolchain and launching with the cray-mpich-abi at runtime. Additionally, you should build on a compute node so that you have the same architecture as at runtime:
```
salloc -N 1 -p standard -t TIME -A ACCOUNT
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
    On Kestrel, any modules you have loaded on the login node will be copied to a compute node, and there are many loaded by default for the cray programming environment. Make sure you are using what you intend to. 

Sample makefiles for vasp5 and vasp6 on Kestrel can be found in our [Kestrel Repo](https://github.com/NREL/HPC/tree/master/kestrel) under the vasp folder.

### Running your build

We have found that it is optimal to run an intel toolchain build of VASP using cray-mpich-abi at runtime. Cray-mpich-abi has several dependencies on cray network modules, so the easiest way to load it is to first load ```PrgEnv-intel``` and then swap the default cray-mpich module for the cray-mpich-abi ```module swap cray-mpich cray-mpich-abi```. You must then load your intel compilers and math libraries, and unload cray's libsci. A sample script showing all of this is in the dropdown below.

!!! Note
    It is necessary to specify the launcher using srun --mpi=pmi2 

??? example "Sample job script: using your own build"

    ```
    #!/bin/bash
    #SBATCH --nodes=2
    #SBATCH --tasks-per-node=104
    #SBATCH --time=2:00:00
    #SBATCH --account=<your-account-name>
    #SBATCH --job-name=<your-job-name>

    # Load cray-mpich-abi and its dependencies within PrgEnv-intel, intel compilers, mkl, and unload cray's libsci
    source /nopt/nrel/apps/env.sh
    module purge
    module load PrgEnv-intel
    module swap cray-mpich cray-mpich-abi
    module unload cray-libsci
    module load intel-oneapi-compilers
    module load intel-oneapi-mkl

    export VASP_PATH=/PATH/TO/YOUR/vasp_exe

    srun --mpi=pmi2 ${VASP_PATH}/vasp_std |& tee out

    ```



