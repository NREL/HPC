# Using LAMMPS Software

*Learn how to use LAMMPS software — an open-source, classical molecular dynamics program designed for massively parallel systems. It is distributed by Sandia National Laboratories.*

LAMMPS has numerous built-in potentials for simulations of solid-state, soft matter, and coarse-grained systems. It can be run on a single processor or in parallel using MPI. To learn more, see the [LAMMPS website](https://www.lammps.org/#gsc.tab=0). 

The versions of LAMMPS on Kestrel, Swift, Vermillion at the time of this page being published are 23Jun22 and 02AUG23 versions. The following packages have been installed: asphere, body, bocs, class2, colloid, dielectric, diffraction, dipole, dpd-basic, drude, eff, electrode, extra-fix, extra-pair, fep, granular, h5md, intel, interlayer, kspace, manifold, manybody, mc, meam, misc, molecule, mpiio, openmp, opt, python, phonon, qep, qmmm, reaction, reaxff, replica, rigid, shock, spin, voronoi.

If you need other packages, please [contact us](mailto:HPC-Help@nrel.gov). 

## Sample CPU Slurm Script 
A sample Slurm script for running LAMMPS on Kestrel CPU nodes is given below:

```
#!/bin/bash
#SBATCH --job-name cpu-test
#SBATCH --nodes=2 #Request two CPU nodes
#SBATCH --time=1:00:00
#SBATCH --account=<your allocation name>
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
A sample Slurm script for running LAMMPS on Kestrel CPU nodes is given below:

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

## Hints and Additional Resources
1. For calculations requesting more than ~10 nodes, the cray mpich stall library is recommended, the details are described at [MPI Stall Library](https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/performancerecs/#mpi-stall-library) and [Improvement of LAMMPS Performance by Using CQ STALL Feature](https://github.nrel.gov/hlong/lammps_stall)
3. For CPU runs, especially for multi-nodes runs, the optimal performance for a particular job may be occurred at a tasks-per-node value less than 104. For GPU runs, number of GPUs should also be varied to achieve the optimal performance. Users should investigate those parameters for large jobs by performing some short test runs.
4. For instructions on running LAMMPS with OpenMP, see the [HPC Github code repository](https://github.com/NREL/HPC/tree/master/applications/lammps).



