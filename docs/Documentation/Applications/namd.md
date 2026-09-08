# NAMD

NAMD is developed by the Theoretical and Computational Biophysics Group at the University of Illinois Urbana-Champaign. It uses the Charm++ parallel programming model with an SMP (Symmetric Multi-Processing) runtime, enabling efficient scaling across both CPU cores and GPU accelerators. Common use cases include protein folding, membrane dynamics, and free energy calculations. For more information and documentation, see the [NAMD website](https://www.ks.uiuc.edu/Research/namd/).

!!! tip "GPU Nodes"
    This page covers the GPU-accelerated NAMD module (`namd/3.0.2-mpi-smp-cuda`). Jobs using this module must be submitted from a GPU login node and must request GPU resources via `--gres=gpu`. CPU-only modules are also available; see [Accessing NAMD on Kestrel](#accessing-namd-on-kestrel) below.

## Accessing NAMD on Kestrel

NAMD is available through the module system on both GPU and CPU nodes.

### GPU module (covered by this guide)

```bash
module load namd/3.0.2-mpi-smp-cuda
```

Loading this module automatically pulls in the required dependencies. Available on both RHEL8 and RHEL9 nodes.

### CPU modules

For CPU-only runs, the following modules are available on RHEL8 standard compute nodes:

```
namd/2.14
namd/2.14_cray
namd/2.14_cray_abi
namd/3.0_cray
namd/3.0_intel
namd/3.0_intel2
namd/3.0_intel_mpich
```

The rest of this guide focuses on the GPU module.

## Running NAMD on Kestrel

!!! warning "Required GPU Configuration Flags"
    Specifying `+devices` on the command line tells NAMD which GPUs are available, but GPU offloading is **not enabled by default**. Without the following lines in your `.namd` configuration file, the simulation will run on CPU only:

    ```
    GPUresident     off
    usePMEGPU       on
    ```

    `GPUresident off` enables the standard GPU force-offloading mode, where non-bonded forces and PME electrostatics are computed on the GPU while integration remains on the CPU. `usePMEGPU on` additionally offloads particle mesh Ewald (long-range electrostatics) to the GPU. Both flags are required together to make effective use of the H100 GPUs on Kestrel.

NAMD uses Charm++ SMP mode: one MPI rank is launched per node, and that rank spawns multiple worker threads (PEs) across the node's CPU cores. The key launch parameters are:

- `--ntasks-per-node=1` — one MPI rank per node
- `--cpus-per-task=<N>` — expose N cores to each rank
- `+ppn <N-1>` — worker PE thread count (one less than `cpus-per-task`, reserving one core for the Charm++ communication thread)
- `+setcpuaffinity` — bind threads to cores for better NUMA locality
- `+devices <list>` — comma-separated list of GPU device indices to use (e.g., `0` for one GPU, `0,1` for two)

### Single-Node, Single-GPU

Allocate a single GPU node interactively:

```bash
salloc -A <account> -t 00:30:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=26 --mem=80G --gres=gpu:1 -p debug
```

Once on the node:

```bash
module load namd/3.0.2-mpi-smp-cuda
cd /path/to/your/simulation

srun -N1 --ntasks-per-node=1 --cpus-per-task=26 \
    namd3 +ppn 25 +setcpuaffinity +devices 0 input.namd
```

### Single-Node, Two GPUs

Using 2 GPUs on a single node can improve throughput for larger systems. Allocate a node with 2 GPUs:

```bash
salloc -A <account> -t 00:30:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=52 --mem=160G --gres=gpu:2 -p debug
```

Once on the node:

```bash
module load namd/3.0.2-mpi-smp-cuda
cd /path/to/your/simulation

srun -N1 --ntasks-per-node=1 --cpus-per-task=52 \
    namd3 +ppn 51 +setcpuaffinity +devices 0,1 input.namd
```

!!! note "When to use multiple GPUs"
    Using 2 GPUs per node is beneficial when your system is large enough to fully utilize both. NAMD distributes work in units called *patches*; if the total number of patches is too small relative to the number of GPUs, performance can decrease rather than improve. As a rule of thumb, aim for at least ~100 patches per GPU. For small systems, using a single GPU is often faster.

### Multi-Node

For larger systems that benefit from more than one node, use `srun` across multiple nodes. NAMD requires one MPI rank per node in SMP mode.

```bash
salloc -A <account> -t 01:00:00 --nodes=2 --ntasks-per-node=1 --cpus-per-task=26 --mem=80G --gres=gpu:1 -p debug
```

Once the allocation is granted:

```bash
module load namd/3.0.2-mpi-smp-cuda
cd /path/to/your/simulation

srun -N2 --ntasks-per-node=1 --cpus-per-task=26 \
    namd3 +ppn 25 +setcpuaffinity +devices 0 input.namd
```

## Sample Slurm Scripts

### Single-Node, Single-GPU Batch Job

```bash
#!/bin/bash
#SBATCH -J namd_1node_1gpu
#SBATCH -A <account>
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -o namd_%j.out

module load namd/3.0.2-mpi-smp-cuda

cd /path/to/your/simulation

srun namd3 +ppn 25 +setcpuaffinity +devices 0 input.namd
```

### Single-Node, Two-GPU Batch Job

```bash
#!/bin/bash
#SBATCH -J namd_1node_2gpu
#SBATCH -A <account>
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH -o namd_%j.out

module load namd/3.0.2-mpi-smp-cuda

cd /path/to/your/simulation

# 51 worker threads + 1 communication thread = 52 cores total
srun namd3 +ppn 51 +setcpuaffinity +devices 0,1 input.namd
```

### Multi-Node Batch Job

```bash
#!/bin/bash
#SBATCH -J namd_multinode
#SBATCH -A <account>
#SBATCH -t 04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -o namd_%j.out

# Loads PrgEnv-gnu, cray-mpich/8.1.28, libfabric, and cuda automatically
module load namd/3.0.2-mpi-smp-cuda

cd /path/to/your/simulation

# 1 MPI rank per node, 25 worker threads per rank, 1 GPU per node
srun namd3 +ppn 25 +setcpuaffinity +devices 0 input.namd
```

## Checking Performance

NAMD prints periodic timing summaries during the run. To extract performance data from the output log:

```bash
grep "PERFORMANCE\|Benchmark time" namd_output.log | tail -10
```

The output will look similar to:

```
Info: Benchmark time: 25 CPUs X.XXXXX s/step X.XXXXX days/ns XXX MB memory
PERFORMANCE: 500  averaging XX.X ns/day, X.XXXXXX sec/step ...
```

Higher `ns/day` and lower `s/step` values indicate better performance.

## Hints and Additional Resources

1. **`+ppn` tuning**: Set `+ppn` to one less than `--cpus-per-task`. This reserves exactly one core per node for Charm++'s communication thread. Using the full core count for worker threads can cause communication stalls and degrade performance.

2. **Multiple GPUs per node**: For multi-node runs, whether `+devices 0,1` helps depends on the size of your system. Enabling 2 GPUs on each node increases the total GPU count, which reduces the number of patches per GPU. If your system is not large enough to saturate all GPUs, this can slow the simulation down. Test on a short run first.

3. **GPU offloading flags**: The two configuration file settings required to actually use the GPU are `GPUresident off` and `usePMEGPU on`. `GPUresident off` enables standard GPU force-offloading (non-bonded forces computed on GPU, integration on CPU). `usePMEGPU on` additionally moves long-range electrostatics (PME) to the GPU. Without both flags, NAMD will run on CPU only even when `+devices` is specified. For fully GPU-resident execution (all computation on GPU, including integration), use `GPUresident on` instead — this can give further speedups on large systems but has some restrictions on supported features.

4. **Input files**: NAMD accepts a `.namd` configuration file that references your PSF, PDB, and parameter files. For generation of input files compatible with NAMD, see [VMD](https://www.ks.uiuc.edu/Research/vmd/) (Visual Molecular Dynamics), which is developed by the same group.

5. For additional documentation, tutorials, and mailing list support, see the [NAMD documentation page](https://www.ks.uiuc.edu/Research/namd/current/ug/) and the [NAMD mailing list](https://www.ks.uiuc.edu/Research/namd/mailing_list/).
