# Kestrel Release Notes

*We will update this page with Kestrel release notes after major Kestrel upgrades.*

## Apr. 23 Upgrades
Modules change :

In order to prep for GPUs on Kestrel A CPU module stack was created to separate GPU applications from CPU applications. 
You should note that CPU nodes have sapphire rapids and GPU nodes have H100 GPUs with Zen4 AMD (genoa).

The new CPU stack will be default when you login to Kestrel and can be used on Partitions like: 

The main changes between the new CPU stack and old stack is the following: (Use ctrl-f to search for your application of interest)

Anaconda:
The Anaconda module will be deprecated in the future, please use conda instead
Mamba is also available for a faster installation of envs.

NVHPC:
3 versions are available 21.3,22.7, 23.3
Multiple modulefiles exist for nvhpc, their description is the following:
Nvhpc: this module is meant to be used with Prgenv-nvhpc.
Loading nvhpc will trigger prgenv-nvhpc to be loaded.
Nvhpc-native: this module contains the full nvhpc toolkit.
Nvhpc-no-mpi: this module load nvhpc toolkit without the mpi (openmpi) that ships with nvhpc.
Nvhpc-byo-compiler: this module loads libraries and header files from the nvhpc toolkit, this module does not provide a compiler nor an mpi

Arm-forge downgraded to 22.0.4

Boost upgraded to 1.84.0

CMake: upgraded to 3.27.9      

Curl: upgraded to 8.6.0  

emacs: upgraded to 29.2    

gettext: upgraded to 0.22.4    

Glib upgraded to 2.78.3                             

Hdf5 upgraded to 1.14.3

Hypre added develop and 2.22

intel-oneapi-dal upgraded to 2024.0.0-intel

intel-oneapi-dnn upgraded to 2024.0.0-intel

intel-oneapi-dpl upgraded to 2022.3.0-intel

intel-oneapi-ipp upgraded to 2021.10.0-intel

intel-oneapi-ippcp upgraded to 2021.9.0-intel

intel-oneapi-mkl upgraded to 2024.0.0-intel

intel-oneapi-tbb upgraded to 2021.11.0-intel

Added libnsl

netcdf-fortran upgraded to 4.6.1  

openblas upgraded to 0.3.26

petsc upgraded to 3.20.4

Slepc upgraded to 3.20.1

tar upgraded to 1.34

yaml-cpp downgraded to 0.5.3

go upgraded to 1.22.0       

intel-oneapi-ccl upgraded to 2021.11.2-intel     

openmpi upgraded to 4.1.6-gcc

Openfoam 9 and 11 built against cray-mpich


## Jan. 29 - Feb. 14 Upgrades

1. We have experienced that most previously built software runs without modification (this includes NREL provided modules) and performs at the same level. 

2. Cray programming environment (CPE) 22.10, the default on the system, produces an error with cray-libsci when using PrgEnv-intel and the cc, CC, or ftn compiler wrappers. This error can be overcome either by swapping in a newer revision of cray-libsci, or by loading CPE/22.12. 

    In the first case, you can load PrgEnv-intel then swap to the newer libsci library: 

    ```
    module swap PrgEnv-cray PrgEnv-intel 
    module swap cray-libsci cray-libsci/22.12.1.1 
    ```
    

    In the second case, you can load the newer CPE with PrgEnv-intel by:  

    ```
    module restore system 
    module purge 
    module use /opt/cray/pe/modulefiles/ 
    module load cpe/22.12 
    module load craype-x86-spr 
    module load PrgEnv-cray 
    module swap PrgEnv-cray PrgEnv-intel  
    ```

3. CPE 23.12 is now available on the system but is a work-in-progress. We are still building out the CPE 23 NREL modules.  

    To load CPE 23.12: 

    ```
    module restore system 
    source /nopt/nrel/apps/cpu_stack/env_cpe23.sh
    module purge
    module use /opt/cray/pe/modulefiles/
    module load cpe/23.12
    module load craype-x86-spr
    module load intel-oneapi/2023.0.0
    module load PrgEnv-intel
    ```

    To load our modules built with CPE 23.12, you need to source the following environment. (Note that we are still building/updating these) 

    `source /nopt/nrel/apps/cpu_stack/env_cpe23.sh` 

    NOTE: In CPE 23.12, some modules, when invoked, silently fail to load. We are still working on fixing this. For now, check that your modules have loaded appropriately with `module list`.

 

 
