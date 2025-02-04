# Using LAMMPS Software

*Learn how to use LAMMPS software — an open-source, classical molecular dynamics program designed for massively parallel systems. It is distributed by Sandia National Laboratories.*

LAMMPS has numerous built-in potentials for simulations of solid-state, soft matter, and coarse-grained systems. It can be run on a single processor or in parallel using MPI. To learn more, see the [LAMMPS website](https://www.lammps.org/#gsc.tab=0). 

The versions of LAMMPS on Kestrel, Swift, and Vermilion can be checked by running `module avail lammps`. Usually there are two recent stable versions available that were compiled using different compiler and MPI toolchains. The following packages have been installed: asphere, body, bocs, class2, colloid, dielectric, diffraction, dipole, dpd-basic, drude, eff, electrode, extra-fix, extra-pair, fep, granular, h5md, intel, interlayer, kspace, manifold, manybody, mc, meam, misc, molecule, mpiio, openmp, opt, python, phonon, qep, qmmm, reaction, reaxff, replica, rigid, shock, spin, voronoi.

If you need other packages or a certain LAMMPS version, please [contact us](mailto:HPC-Help@nrel.gov). 

## Sample CPU Slurm Script 
A sample Slurm script for running LAMMPS on Kestrel CPU nodes is given below:

```
#!/bin/bash
#SBATCH --job-name cpu-test
#SBATCH --nodes=2 #Request two CPU nodes
#SBATCH --time=1:00:00
#SBATCH --account=[your allocation name]
#SBATCH --error=std.err
#SBATCH --output=std.out
#SBATCH --tasks-per-node=104
#SBATCH --exclusive
#SBATCH -p debug

module load lammps/080223-intel-mpich
module list

run_cmd="srun --mpi=pmi2 "
lmp_path=lmp
name=my_job
$run_cmd $lmp_path -in $name.in >& $name.log
```

where `my_job.in` is the input and `my_job.log` is the output. This runs LAMMPS using two nodes with 208 MPI ranks. 

## Sample GPU Slurm Script 
A sample Slurm script for running LAMMPS on Kestrel GPU nodes is given below:

```
#!/bin/bash
#SBATCH --job-name gpu-test
#SBATCH --nodes=1 #Request one GPU node
#SBATCH --time=1:00:00
#SBATCH --account=[your_allocation_name]
#SBATCH --error=std.err
#SBATCH --output=std.out
#SBATCH --tasks-per-node=8 #Running 8 MPI tasks per node
#SBATCH --mem=16G #Request memory
#SBATCH --gres=gpu:2 #Request 2 GPU per node
#SBATCH -p debug

module load lammps/080223-gpu
module list

export MPICH_GPU_SUPPORT_ENABLED=1
#Request 2 GPU per node
export CUDA_VISIBLE_DEVICES=0,1 

run_cmd="srun --mpi=pmi2 "
lmp_path=lmp
name=medium
#Request 2 GPU per node
gpu_opt="-sf gpu -pk gpu 2"
$run_cmd $lmp_path $gpu_opt -in $name.in >& $name.gpu.log
```

This runs LAMMPS using one nodes with 8 MPI ranks and 2 GPUs. The following information will be printed out in my_job.log file:
```
--------------------------------------------------------------------------
- Using acceleration for pppm:
-  with 4 proc(s) per device.
-  Horizontal vector operations: ENABLED
-  Shared memory system: No
--------------------------------------------------------------------------
Device 0: NVIDIA H100 80GB HBM3, 132 CUs, 77/79 GB, 2 GHZ (Mixed Precision)
Device 1: NVIDIA H100 80GB HBM3, 132 CUs, 2 GHZ (Mixed Precision)
--------------------------------------------------------------------------
```

## Sample High-Bandwidth Partition Slurm Script

When running LAMMPs on more than 10 nodes, it is recommended to run LAMMPs on the [High-Bandwidth Partition (hbw)](../Systems/Kestrel/Running/index.md#high-bandwidth-partition) – this partition consists of nodes that have dual-NICs as part of its hardware architecture, which can significantly improve LAMMPs performance. 

```
#!/bin/bash
#SBATCH --job-name lammps-16nodes-96ranks
#SBATCH --nodes=16
#SBATCH --time=01:30:00
#SBATCH --account=<your allocation name here>
#SBATCH --error=std.err
#SBATCH --output=std.out
#SBATCH --tasks-per-node=96
#SBATCH --exclusive
#SBATCH -p hbw
#SBATCH --array=1-5
#SBATCH --output=lammps-96nodes/lammps-16nodes-96ranks_%a.out


module load lammps/062322-cray-mpich

export OMP_NUM_THREADS=1

CPUBIND='--cpu-bind=map_cpu:0,52,13,65,26,78,39,91,1,53,14,66,27,79,40,92,2,54,15,67,28,80,41,93,3,55,16,68,29,81,42,94,4,56,17,69,30,82,43,95,5,57,18,70,31,83,44,96,6,58,19,71,32,84,45,97,7,59,20,72,33,85,46,98,8,60,21,73,34,86,47,99,9,61,22,74,35,87,48,100,10,62,23,75,36,88,49,101,11,63,24,76,37,89,50,102,12,64,25,77,38,90,51,103'

export MPICH_OFI_NIC_POLICY="NUMA"

#MPI only, no OpenMP
run_cmd="srun --mpi=pmi2 $CPUBIND"
lmp_path=lmp
run_name=(medium)

$run_cmd $lmp_path -in $name.in >& $name.log
```

Please note – the CPU binding and MPICH_OFI_NIC_POLICY being set explicitly allow for extra performance gains on the high-bandwidth partition. If not set, there are still performance gains on the high-bandwidth nodes, just not as much as there would be otherwise. 

## Hints and Additional Resources
1. For calculations requesting more than ~10 nodes, running on the high-bandwidth partition is recommended. Further information on the High-Bandwidth partition can be found here: [High-Bandwidth Partition](../Systems/Kestrel/Running/index.md#high-bandwidth-partition).
2. For CPU runs, especially for multi-nodes runs, the optimal performance for a particular job may be at a tasks-per-node value less than 104. For GPU runs, number of GPUs should also be varied to achieve the optimal performance. Users should investigate those parameters for large jobs by performing some short test runs. Some tasks-per-node values that could be useful to test are: 72, 52, and 48.
3. For instructions on running LAMMPS with OpenMP, see the [HPC Github code repository](https://github.com/NREL/HPC/tree/master/applications/lammps).



