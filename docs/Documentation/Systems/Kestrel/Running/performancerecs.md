# Performance Recommendations

Please note that all of these recommendations are subject to change as we continue to improve the system.

## OpenMP

If you are running a code with OpenMP enabled, we recommend manually setting one of the following environment variables:

```
export OMP_PROC_BIND=spread # for non-intel built codes

export KMP_AFFINITY=balanced # for codes built with intel compilers
```

You may need to export these variables even if you are not running your job with threading, i.e., with `OMP_NUM_THREADS=1`

## MPI

Currently, some applications on Kestrel are not scaling with the expected performance. We are actively working with the vendor's engineers to resolve these issues. For now, for these applications, we have compiled a set of recommendations that may help with performance. Note that any given recommendation may or may not apply to your specific application. We strongly recommend conducting your own performance and scalability tests on your performance-critical codes.

1. Use Cray MPICH over OpenMPI or Intel MPI. If you need help rebuilding your code so that it uses Cray MPICH, please reach out to hpc-help@nrel.gov

2. For MPI collectives-heavy applications, setting the following environment variables (for Cray MPICH):
```
export MPICH_SHARED_MEM_COLL_OPT=mpi_bcast,mpi_barrier 
export MPICH_COLL_OPT_OFF=mpi_allreduce 
```
These environment variables turn off some collective optimizations that we have seen can cause slowdowns. For more information on these environment variables, visit HPE's documentation site [here](https://cpe.ext.hpe.com/docs/mpt/mpich/intro_mpi_ucx.html).

4. For hybrid MPI/OpenMP codes, requesting more threads per task than you tend to request on Eagle. This may yield performance improvements.
  
### MPI Stall Library
For calculations requesting more than ~10 nodes, you can use the cray mpich stall library. This library can help reduce slowdowns in your calculation runtime caused by congestion in MPI communication, a possible performance bottleneck on Kestrel for calculations using ~10 nodes or more.  To use the library, you must first make sure your code has been compiled within one of the `PrgEnv-gnu`, `PrgEnv-cray`, or  `PrgEnv-intel` programming environments. Then, add the following lines to your sbatch submit script:
   ```
   stall_path=/nopt/nrel/apps/cray-mpich-stall
   export LD_LIBRARY_PATH=$stall_path/libs_mpich_nrel_{PRGENV-NAME}:$LD_LIBRARY_PATH
   export MPICH_OFI_CQ_STALL=1
   ```
  Where {PRGENV-NAME} is replaced with one of `cray`, `intel`, or `gnu`. For example, if you compiled your code within the default `PrgEnv-gnu` environment, then you would export the following lines:
   ```
   stall_path=/nopt/nrel/apps/cray-mpich-stall
   export LD_LIBRARY_PATH=$stall_path/libs_mpich_nrel_gnu:$LD_LIBRARY_PATH
   export MPICH_OFI_CQ_STALL=1
   ```
The default "stall" of the MPI tasks is 12 microseconds, which we recommend trying before manually adjusting the stall time. You can adjust the stall to be longer or shorter with `export MPICH_OFI_CQ_STALL_USECS=[time in microseconds]` e.g. for 6 microseconds, `export MPICH_OFI_CQ_STALL_USECS=6`. A stall time of 0 would be the same as "regular" MPI. As stall time increases, the amount of congestion decreases, up to a calculation-dependent "optimal" stall time. If you need assistance in using this stall library, please email hpc-help@nrel.gov.
