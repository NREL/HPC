# PETSc

**Documentation:** [PETSc](https://petsc.org)

*PETSc is a suite of data structures and routines for the scalable (parallel) solution of scientific applications modeled by partial differential equations.*

On Kestrel, PETSc is provided under multiple toolchains

```
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
  petsc:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
     Versions:
        petsc/3.14.6-cray-mpich-intel
        petsc/3.19.3-intel-oneapi-mpi-intel
        petsc/3.19.3-openmpi-gcc

```

`petsc/3.14.6-cray-mpich-intel` is a PETSc installation that uses HPE provided `PrgEnv-intel`. 
Therefore, the MPI used here is *cray-mpich* and the compiler is *intel/2023*.

`petsc/3.19.3-intel-oneapi-mpi-intel` is a PETSc installation that uses *intel-oneapi-compilers* and *intel-oneapi-mpi* for the compilers and MPI, respectively.

`petsc/3.19.3-openmpi-gcc` is a PETSc installation that uses *gcc/10.1.0* and *openmpi/4.1.5-gcc* for the compilers and MPI, respectively.

## RHEL 9 GPU nodes

On the RHEL 9 H100 GPU nodes, PETSc 3.25.6 is available as `petsc/3.25.6-cuda12.8`.

| Component | Version used for the build |
|-----------|----------------------------|
| Compilers | gcc / g++ / gfortran 14.2.0 (`gcc/14.2.0`) |
| MPI | MPICH 5.0.0 over Slingshot/libfabric (`mpich/5.0.0-gpu`) |
| CUDA | 12.8.1 (`cuda/12.8.1`), built for H100 (`sm_90`) |
| GPU backends | CUDA (cuSPARSE) and Kokkos 5.1.0 / Kokkos-Kernels (built by PETSc) |
| External packages | HYPRE (GPU), SuperLU_dist, METIS/ParMETIS, fblaslapack |
| Precision | real, double, 32-bit indices, optimized (`--with-debugging=0`) |

```bash
module load petsc/3.25.6-cuda12.8
```

The module loads the compilers and CUDA, puts the MPI compilers on `PATH`, and sets `PETSC_DIR` (`PETSC_ARCH` is empty). Build with `mpicc app.c $(pkg-config --cflags --libs PETSc)` or CMake `find_package(PkgConfig)`.

Run one MPI rank per GPU with `srun`, and select GPU types at runtime:

```bash
srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --mpi=pmi2 --network=job_vni ./app \
     -mat_type aijcusparse -vec_type cuda -device_select $SLURM_LOCALID -pc_type gamg
```

!!! note
    - Launch with `srun --mpi=pmi2` and a Slingshot VNI (`--network=single_node_vni` on one node, `--network=job_vni` on two or more). Bare `mpiexec`/`mpirun` fails to start on this network.
    - The module turns GPU-aware MPI **off** (`MPICH_GPU_SUPPORT_ENABLED=0`, `PETSC_OPTIONS=-use_gpu_aware_mpi 0`). Keep it off: with it on, multi-GPU GAMG and HYPRE solves hang or crash, and solves are also slower. If you set `PETSC_OPTIONS` yourself, add your options to it instead of replacing it.
    - Use `--gpus-per-node`, not `--gpu-bind`, `--gpus-per-task`, or a per-rank `CUDA_VISIBLE_DEVICES`.

This build passes all 1,867 PETSc tests that use CUDA or Kokkos. A CG + Jacobi solve with 144M unknowns runs in 6.20 s on 1 GPU, 1.64 s on 4 GPUs (3.8×), and 0.88 s on 8 GPUs across 2 nodes (7.0×). The full guide is at `/nopt/nlr/apps/kestrel-gpu/software/petsc/petsc-3.25.6-guide.md`.
