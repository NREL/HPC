# Performance Recommendations

Please note that all of these recommendations are subject to change as we continue to improve the system.

## MPI

Applications running across multiple CPU nodes on Kestrel might experience performance problems. Following your scaling tests, if your application underperforms, consider these performance improvement suggestions.

- For **any application**, use `Cray-MPICH` instead of `OpenMPI` because `Cray-MPICH` is highly optimized to leverge the [interconnect](../../index.md) used on Kestrel. For OpenMPI to Cray MPICH code rebuilding assistance, please contact hpc-help@nrel.gov. If your application was built with an [MPICH ABI-compatible MPI library](https://github.com/pmodels/mpich/blob/main/doc/wiki/testing/ABI_Compatibility_Initiative.md), use `cray-mpich-abi` for optimal performance. The cray-mpich-abi usage involves these steps: load the programming environment module (`PrgEnv-*`) that matches your application's compiler; load the `cray-mpich-abi` module; and run your application with `Slurm` (see example below).
   ```
   $ module load PrgEnv-gnu or module load PrgEnv-intel
   $ module swap cray-mpich/8.1.28 cray-mpich-abi/8.1.28
   $ srun -N $SLURM_NNODES -n $SLURM_NTASKS --distribution=block:block --cpu_bind=rank_ldom ./<application executable>
   ```

- The performance of __latency-sensitive applications__, such as AMR-Wind, Nalu-Wind, and LAMMPS with medium-size input, is impacted by message communication congestion in the interconnect when running jobs on over 8 CPU-nodes using `cray-mpich/8.1.28`. Higher congestion is observed using the standard CPU nodes which each have one NIC, than using the nodes in the `hbw` partition, which each have two NICs per node. The stall feature in `CRAY MPICH version 8.1.30.1` boosts application performance by regulating message injection into the interconnect. To utilize this library, applications must be built using either the `PrgEnv-gnu`, `PrgEnv-cray`, or `PrgEnv-intel` programming environment modules, or an MPICH ABI-compatible MPI library. Your job script needs one of these script snippets: use the first for programming environment builds; the second for MPICH-ABI-compatible builds.
   ```
   # Set the number of NICs to 2 for `hbw`
   export NIC=1
   
   # Load the shared libraries
   export LD_LIBRARY_PATH=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_*:$LD_LIBRARY_PATH
   
   # Enable the stall mechanism
   export MPICH_OFI_CQ_STALL=1

   # Activate the stall library
   export MPICH_OFI_CQ_MIN_PPN_PER_NIC=($SLURM_NTASKS_PER_NODE/NIC)

   # Tune the stall value
   export MPICH_OFI_CQ_STALL_USECS=26
   ```   
   ```
   # Set the number of NIC to 2 for `hbw`
   export NIC=1
   export LD_LIBRARY_PATH=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_*_adj:$LD_LIBRARY_PATH
   export MPICH_OFI_CQ_STALL=1
   export MPICH_OFI_CQ_MIN_PPN_PER_NIC=($SLURM_NTASKS_PER_NODE/NIC)
   export MPICH_OFI_CQ_STALL_USECS=1

   ```
Substitute `*` with the compiler name (e.g., `cray`, `intel`, or `gnu`) used to compile your application. For best performance, experiment with stall (`MPICH_OFI_CQ_STALL_USECS`) values of between 1 and 26 microseconds; the default is 12 microseconds. For example, you may run your application using a stall value from this list: [1, 3, 6, 9, 12, 16, 20, 26]. If you need assistance in using this stall library, please email hpc-help@nrel.gov.
!!! Note
      `Spack`-built applications have hardcoded runtime paths in their executables, necessitating the use of `LD_PRELOAD`. For example, the PrgEnv-intel shared libraries can be loaded as follows: `export LD_PRELOAD=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpi_intel.so.12:/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpifort_intel.so.12`

- For hybrid MPI/OpenMP codes, requesting more threads per task than you tend to request on Eagle. This may yield performance improvements.

## OpenMP

If you are running a code with OpenMP enabled, we recommend manually setting one of the following environment variables:

```
export OMP_PROC_BIND=spread # for non-intel built codes

export KMP_AFFINITY=balanced # for codes built with intel compilers
```

You may need to export these variables even if you are not running your job with threading, i.e., with `OMP_NUM_THREADS=1`
