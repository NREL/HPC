# Kokkos

**Documentation:** [Kokkos](https://kokkos.org/kokkos-core-wiki/)

*Kokkos is a C++ performance-portability library: you write parallel loops and data structures once, and Kokkos runs them on CPUs or GPUs.*

## RHEL 9 GPU nodes

On the RHEL 9 H100 GPU nodes, Kokkos 5.2.2 is available as `kokkos/5.2.2`.

| Component | Version used for the build |
|-----------|----------------------------|
| Compiler | g++ 14.2.0 (`gcc/14.2.0`), C++20, with Kokkos `nvcc_wrapper` |
| CUDA | 13.2 (`cuda/13.2`) |
| Backends | CUDA (default), OpenMP, Serial |
| Architecture | NVIDIA H100 (`HOPPER90`), AMD Genoa host (`ZEN4`) |
| Notes | Built with CUDA `malloc-async` off for Slingshot MPI. Kokkos-Kernels is not included. |

```bash
module load kokkos/5.2.2
```

The module loads `gcc/14.2.0` and `cuda/13.2`, sets `Kokkos_ROOT`, and puts `nvcc_wrapper` on `PATH`. With CMake, use `find_package(Kokkos REQUIRED)` and link `Kokkos::kokkos`, using `nvcc_wrapper` as the C++ compiler:

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=$Kokkos_ROOT/bin/nvcc_wrapper -DCMAKE_CXX_STANDARD=20
```

### Using Kokkos with MPI

Use the GPU-aware MPICH, which is built with the same CUDA 13.2:

```bash
module load gcc/14.2.0 cuda/13.2 kokkos/5.2.2 mpich/5.0.0-gpu
export MPICH_CXX=$Kokkos_ROOT/bin/nvcc_wrapper
mpicxx -O3 -std=c++20 -fopenmp --extended-lambda --expt-relaxed-constexpr \
  app.cpp -o app -lkokkoscore -lkokkoscontainers
```

Run one rank per GPU, and have each rank pick its GPU from `SLURM_LOCALID`:

```cpp
const char* lid = std::getenv("SLURM_LOCALID");
Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(lid ? std::atoi(lid) : 0));
```

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --mpi=pmi2 --network=job_vni ./app
```

!!! note
    - Launch with `srun --mpi=pmi2` and a Slingshot VNI (`--network=single_node_vni` on one node, `--network=job_vni` on two or more).
    - Do **not** use `--gpu-bind=closest`, `--gpus-per-task`, or a per-rank `CUDA_VISIBLE_DEVICES`. Each rank then sees only one GPU, and GPU-aware MPI between ranks on the same node crashes with `gpu_ipc_handle_map failed`.
    - GPU-aware MPI (`MPICH_GPU_SUPPORT_ENABLED=1`) works with this Kokkos build and is faster than staging through host memory.
    - To use Kokkos together with PETSc, build against PETSc's own Kokkos (load only `petsc` and set `Kokkos_ROOT=$PETSC_DIR`). A program can't mix two Kokkos versions.

### Scaling

A Kokkos + MPI 3D stencil benchmark with halo exchange, run with GPU-aware MPI on the RHEL 9 GPU nodes:

| GPUs | Nodes × GPUs per node | Strong scaling, 768³ grid | Weak scaling, 512³ per GPU |
|:---:|:---:|:---:|:---:|
| 1 | 1 × 1 | 5.72 s (1.00×) | 8.40 ms/step (100%) |
| 4 | 1 × 4 | 1.36 s (4.20×) | 8.95 ms/step (94%) |
| 8 | 2 × 4 | 0.80 s (7.20×) | 9.33 ms/step (90%) |

The full guide, including a ready-to-run MPI example, is at `/nopt/nlr/apps/kestrel-gpu/software/kokkos/kokkos-5.2.2-guide.md`.
