# Kestrel Release Notes

*We will update this page with Kestrel release notes after major Kestrel upgrades.*

## October 8, 2024

1. Slurm was upgraded from 23.11.7 to 23.11.10. 
1. PrgEnv-gnu/8.5.0 is now loaded by default when you login to Kestrel instead of PrgEnv-cray. 
1. The `module restore` command shouldn't be used. It will load broken modules. 

## August 14, 2024

Jobs running on `debug` GPU nodes are now limited to a total of half of one GPU node's resources across one or two nodes. This is equivalent to 64 CPUs, 2 GPUs, and 180G of RAM on one node or 32 CPUs, 1 GPU, and 90GB of RAM on two nodes. `--exclusive` can no longer be used for GPU debug jobs. 

## August 9, 2024

As of 08/09/2024 we have released new modules for VASP on Kestrel CPUs: 

```
------------ /nopt/nrel/apps/cpu_stack/modules/default/application -------------
   #new modules:
   vasp/5.4.4+tpc     vasp/6.3.2_openMP+tpc    vasp/6.4.2_openMP+tpc
   vasp/5.4.4_base    vasp/6.3.2_openMP        vasp/6.4.2_openMP
   
   #legacy modules will be removed during next system time:
   vasp/5.4.4         vasp/6.3.2               vasp/6.4.2            (D)
```

 What’s new: 
 
 * New modules have been rebuilt with the latest Cray Programming Environment (cpe23), updated compilers, and math libraries.
 * OpenMP capability has been added to VASP 6 builds.
 * Modules that include third-party codes (e.g., libXC, libBEEF, VTST tools, and VASPsol) are now denoted with +tpc. Use `module show vasp/<version>` to see details of a specific version.

## July 29 - July 30, 2024

1. Two [GPU login nodes](../Kestrel/index.md) were added. Use the GPU login nodes for compiling software to run on GPU nodes and for submitting GPU jobs. 
1. GPU compute nodes were made available for general use and additional GPU partitions were added. See [Running on Kestrel](./Running/index.md) for additional information and recommendations.

Module Updates/Changes 

1. Modules are automatically loaded depending on node type, e.g., the GPU module stack is automatically loaded on GPU nodes. 

1. Naming convention for compilers: <br>
    example gcc compiler: 
    * Gcc/version is the compiler used by CPE with Prgenv
    * Gcc-native/version: also meant to be used with Prgenv. The difference gcc-native and gcc is that the former is optimized for the specific architecture
    * Gcc-stdalone/version this gcc is meant to be used outside of CPE. 
    * The same applies to nvhpc and aocc.

1. Intel vs oneapi: <br>
Moving forward the naming -intel in modules e.g. adios/1.13.1-intel-oneapi-mpi-intel will be deprecated in favor of -oneapi e.g. adios/1.13.1-intel-oneapi-mpi-oneapi. <br>
This is implemented for the gpu modules and will be implemented for the CPU in the future. <br>
Oneapi is the new naming convention for intel compilers.

1. compilers-mixed: <br>
In the list of compilers, you’ll see compilers with -mixed e.g. nvhpc-mixed (same applies to intel, gcc, aocc, etc). 
Those are meant to be used with CPE Prgenv, where you can force a mix and match between compilers. 
Example: loading Prgenv-nvhpc and loading gcc-mixed. 
This is not recommended and should only be used if you know what you’re doing. 

1. Nvhpc: <br>
There 5 types of nvhpc modules: <br>
Nvidia module is equivalent to nvhpc and is meant to be used with CPE (Prgenv-nvidia). 
Per HPE’s instruction, only Prgenv-nvhpc should be used and not Prgenv-nvidia
    * Nvhpc which is meant to be used with CPE (Prgenv-nvhpc)
    * Nvhpc-mixed : meant to be used with CPE
    * Nvhpc-stdalone : can be used outside of CPE for your usual compilation will load the compilers and a precompiled openmpi that ships with nvhpc
    * nvhpc-nompi:  Similar to Nvhpc-stdalone but doesn’t load the precompiled ompi
    * nvhpc-byo-compiler: only load libs and header files contained in the nvidia SDK, no compiler or mpi is loaded <br>

1. Cuda: <br>
    * Cuda/11.7 was removed. If you'd like to access cuda as a standalone you can load cuda/12.3, cuda/12.1 was also added (for the gpus)

1.  Intel: <br>
    * Intel, intel-oneapi and intel-classic are modules to be used with CPE. If you want to use standalone intel compilers outside of CPE please use: 
Intel-oneapi-compilers. 
    * intel-oneapi-compilers/2024.1.0 was added.

1. Anaconda: <br>
    * The 2024 version is now added.
 

## April 12 - April 17, 2024

1. The size of the [shared node partition](./Running/index.md#shared-node-partition) was doubled from 32 nodes to 64 nodes. 

2. Cray programming environment (CPE) 23.12 is now the default on the system. 

3. To use node local storage, you will now need to use the `$TMPDIR` environment variable. `$TMPDIR` will now be set to `/tmp/scratch/$JOBID`. Hard-coding `/tmp/scratch` won't work. This change was made to prevent conflicts between multiple users/jobs writing to local disk on shared nodes. As a reminder, writing to `$TMPDIR` will use local disk on the nodes that have one, and RAM (up to 128Gb) on nodes without.

4. `/kfs2/pdatasets` was renamed to `/kfs2/datasets` and a symlink `/datasets` was added. 


## Jan. 29 - Feb. 14, 2024 Upgrades

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

 

 
