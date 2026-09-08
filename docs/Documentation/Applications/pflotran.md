# PFLOTRAN

**Documentation:** [PFLOTRAN](https://pflotran.org/) | [Reference manual](https://documentation.pflotran.org/)

*PFLOTRAN is an open-source, massively parallel subsurface flow and reactive transport code, used for simulating multi-phase flow, heat transport, and biogeochemical reactions in porous media.* The scope of this documentation is limited to the [RHEL9](https://natlabrockies.github.io/HPC/RHEL9_upgrade/) CPU nodes on Kestrel, currently accessible through the `kl3` login node.

## Getting Started

The `pflotran` module builds against **PETSc 3.24.5** (bundled in the
install, including METIS 5.1.0 and ParMETIS 4.0.3 for unstructured grids)
and the Cray PE toolchain (`PrgEnv-gnu`, `cray-mpich`, `hdf5`). All of these
prerequisites are auto-loaded by the module.

```
module use /nopt/nlr/apps/kestrel-cpu/software/pflotran/modules
module load pflotran/6.0
```

`which pflotran` should resolve to
`/nopt/nlr/apps/kestrel-cpu/software/pflotran/6.0/bin/pflotran`.

!!! note
	`pflotran` always calls `MPI_Init`, even for a single rank. Do **not**
	pass `--mpi=none` to `srun` — you'll get a
	`PMI2_Job_GetId returned 14` error. Use `srun --ntasks=1 pflotran ...`
	for single-rank runs.

### Example Job Scripts

??? example "Kestrel CPU — single node"

	```slurm
	#!/bin/bash
	#SBATCH --account=<alloc>
	#SBATCH --partition=standard
	#SBATCH --nodes=1
	#SBATCH --ntasks=64
	#SBATCH --cpus-per-task=1
	#SBATCH --mem=128G
	#SBATCH --time=24:00:00

	module use /nopt/nlr/apps/kestrel-cpu/software/pflotran/modules
	module load pflotran/6.0

	srun pflotran -pflotranin your_input.in
	```

??? example "Kestrel CPU — multi-node"

	```slurm
	#!/bin/bash
	#SBATCH --account=<alloc>
	#SBATCH --nodes=2
	#SBATCH --ntasks-per-node=64
	#SBATCH --cpus-per-task=1
	#SBATCH --mem=128G
	#SBATCH --time=24:00:00

	module use /nopt/nlr/apps/kestrel-cpu/software/pflotran/modules
	module load pflotran/6.0

	srun pflotran -pflotranin your_input.in
	```

!!! note
	Run PFLOTRAN from `/scratch/$USER/…` (or `/projects/…`, `/home/$USER`) —
	not `/tmp`, which is a compute-node-local mount not visible to the
	login node where the input was likely prepared.

## Multi-node scaling

PFLOTRAN scales well across Kestrel-CPU nodes over Slingshot-11. Strong-
scaling data from a 200,000-cell Richards flow benchmark:

| Nodes | Ranks/node | Total ranks | Wall time [s] | Speedup vs 1 rank |
|------:|-----------:|------------:|--------------:|------------------:|
|     1 |          1 |           1 |         50.78 |             1.0×  |
|     1 |          8 |           8 |          5.19 |             9.8×  |
|     1 |         32 |          32 |          2.10 |            24.2×  |
|     1 |         64 |          64 |          0.99 |            51.2×  |
|     1 |        104 |         104 |          0.77 |            66.1×  |
|     2 |         64 |         128 |          0.59 |            86.0×  |
|     4 |         64 |         256 |          0.53 |            95.8×  |

Aim for at least **~1000 cells per rank** for good strong-scaling
efficiency; beyond that, network and partitioning overhead dominates
(e.g. 4 nodes × 104 ranks-per-node was *slower* than 4 × 64 for this case).

## Running the shipped tests

```
SCR=/scratch/$USER/pflotran_check
mkdir -p $SCR && cd $SCR
cp -r $PFLOTRAN_HOME/pflotran/regression_tests/default/543/. .
srun --ntasks=8 pflotran -pflotranin 543_flow_and_tracer-np8.in
```

Use a `-np*` variant when running in parallel — the plain `.in` files often
select a direct (LU) solver, which PFLOTRAN refuses to run in parallel.

## Advanced

Full details on the bundled PETSc build, install recipe, and rebuild
instructions are in the
[maintainer notes](https://documentation.pflotran.org/user_guide/how_to/installation/linux.html)
and the install's own guide at
`/nopt/nlr/apps/kestrel-cpu/software/pflotran/pflotran-6.0-guide.md`.

## Troubleshooting

**`MPIR_pmi_init(115): PMI2_Job_GetId returned 14`, exit 1**
You launched with `srun --mpi=none`. Drop that flag — pflotran is always MPI.

**`ERROR: File: "input.in" not found. Stopping!`**
Either you're running from `/tmp` (use `/scratch/$USER/…` instead), or the
input path passed to `-pflotranin` is wrong relative to the `srun` launch
directory.

**`ERROR: Direct solver (KSPPREONLY + PCLU) not supported when running in parallel`**
The input's `LINEAR_SOLVER` block selects a serial-only direct solver. Use a
`-np*` variant of shipped tests, or switch to an iterative solver.

**`ERROR: HDF5 file "xxx.h5" not found`**
Some test cases have companion `.h5` inputs — copy the whole test directory,
not just the `.in` file.

## Support

- PFLOTRAN home: <https://pflotran.org/>
- Reference manual: <https://documentation.pflotran.org/>
- PETSc: <https://petsc.org/release/> (v3.24.5 shipped in this install)
