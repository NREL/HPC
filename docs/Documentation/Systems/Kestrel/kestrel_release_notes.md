# Kestrel Release Notes

*We will update this page with Kestrel release notes after major Kestrel upgrades.*

## April 12 - April 17

1. The size of the [shared node partition](./running.md#shared-node-partition) was doubled from 32 nodes to 64 nodes. 

2. Cray programming environment (CPE) 23.12 is now the default on the system. 

3. To use node local storage, you will now need to use the `$TMPDIR` environment variable. `$TMPDIR` will now be set to `/tmp/scratch/$JOBID`. Hard-coding `/tmp/scratch` won't work. Writing to `$TMPDIR` will use local disk on the nodes that have one, and RAM on nodes without. This change was made to prevent conflicts between multiple users/jobs writing to local disk on shared nodes. 


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

 

 
