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

6. ONLY if you are running on 10 or more nodes and are experiencing scalability issues, you can try half-packing the nodes you request, i.e., requesting 52 ranks per node instead of 104 ranks per node, then spreading these ranks evenly across the two sockets. This can be accomplished by including the following in your srun command:   
```
--ntasks-per-node=52 --distribution=cyclic:cyclic --cpu_bind=cores
```

